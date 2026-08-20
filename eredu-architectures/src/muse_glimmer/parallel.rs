//! Semantic tensor-parallel placement for Muse-Glimmer text and vision components.

use eredu_runtime::{
    MemberSharding, ParallelPlanError, ParameterGroupSpec, ParameterMemberSpec, ParameterRole,
};

use super::DecoderConfig;

/// Derives rank-local text construction geometry from the semantic placement.
pub fn local_decoder_config(
    args: &DecoderConfig,
    layer: usize,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<DecoderConfig, ParallelPlanError> {
    if layer >= args.num_hidden_layers as usize {
        return Err(invalid(format!(
            "Muse-Glimmer layer {layer} is out of range"
        )));
    }
    let root = format!("model.layers.{layer}");
    let tensor = |suffix: &str| {
        layout
            .tensor(&format!("{root}.{suffix}.weight"))
            .or_else(|| layout.tensor(&format!("{root}.{suffix}.inner.weight")))
            .or_else(|| layout.tensor(&format!("{root}.{suffix}")))
            .ok_or_else(|| {
                ParallelPlanError::InvalidTensor(format!(
                    "missing local layout for {root}.{suffix}"
                ))
            })
    };
    let query_width = local_axis(tensor("self_attn.q_proj")?, 0, "query width")?;
    let key_width = local_axis(tensor("self_attn.k_proj")?, 0, "key width")?;
    if query_width % args.head_dim != 0 || key_width % args.head_dim != 0 {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "local Muse-Glimmer attention widths q={query_width}, k={key_width} split head dimension {}",
            args.head_dim
        )));
    }
    let mut local = args.clone();
    local.num_attention_heads = query_width / args.head_dim;
    local.num_key_value_heads = key_width / args.head_dim;
    if args.is_moe() {
        let fused = local_axis(
            tensor("mlp.experts.gate_up_proj")?,
            1,
            "expert gate/up width",
        )?;
        if fused % 2 != 0 {
            return Err(ParallelPlanError::InvalidTensor(format!(
                "local Muse-Glimmer expert gate/up width {fused} is not even"
            )));
        }
        local.moe_intermediate_size = fused / 2;
    } else {
        local.intermediate_size =
            local_axis(tensor("mlp.gate_proj")?, 0, "dense intermediate width")?;
    }
    Ok(local)
}

fn local_axis(
    tensor: &eredu_runtime::LocalTensorLayout,
    axis: usize,
    label: &str,
) -> Result<i32, ParallelPlanError> {
    let value = *tensor.local_shape().get(axis).ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!(
            "local Muse-Glimmer {label} tensor has no axis {axis}"
        ))
    })?;
    i32::try_from(value).map_err(|_| {
        ParallelPlanError::InvalidTensor(format!("local Muse-Glimmer {label} exceeds i32"))
    })
}

/// Declares pinned text embeddings, final normalization, and vocabulary output.
pub fn static_parameter_groups(
    args: &DecoderConfig,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let hidden = dim(args.hidden_size)?;
    let vocabulary = dim(args.vocab_size)?;
    let mut groups = vec![group(
        "model.embed_tokens",
        ParameterRole::Vocabulary,
        [(
            "model.embed_tokens.weight",
            vec![vocabulary, hidden],
            MemberSharding::Balanced { axis: 0 },
        )],
    )?];
    groups.push(replicated(
        "model.norm",
        [("model.norm.weight", vec![hidden])],
    )?);
    if !args.tie_word_embeddings {
        groups.push(group(
            "lm_head",
            ParameterRole::Vocabulary,
            [(
                "lm_head.weight",
                vec![vocabulary, hidden],
                MemberSharding::Balanced { axis: 0 },
            )],
        )?);
    }
    Ok(groups)
}

/// Declares one decoder block's head, dense/expert, and replicated groups.
pub fn layer_parameter_groups(
    args: &DecoderConfig,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    if layer >= args.num_hidden_layers as usize {
        return Err(invalid(format!(
            "Muse-Glimmer layer {layer} is out of range"
        )));
    }
    let hidden = dim(args.hidden_size)?;
    let query_heads = dim(args.num_attention_heads)?;
    let key_value_heads = dim(args.num_key_value_heads)?;
    let head = dim(args.head_dim)?;
    let query_width = query_heads
        .checked_mul(head)
        .ok_or_else(|| invalid("Muse-Glimmer query width overflow"))?;
    let key_value_width = key_value_heads
        .checked_mul(head)
        .ok_or_else(|| invalid("Muse-Glimmer key/value width overflow"))?;
    let root = format!("model.layers.{layer}");
    let attention = format!("{root}.self_attn");
    let mut groups = vec![ParameterGroupSpec::partitioned(
        format!("{attention}.query_heads"),
        ParameterRole::AttentionHeads,
        query_heads,
        [
            member(
                format!("{attention}.q_proj.weight"),
                vec![query_width, hidden],
                partitioned(0),
            ),
            member(
                format!("{attention}.gate_proj.weight"),
                vec![query_width, hidden],
                partitioned(0),
            ),
            member(
                format!("{attention}.o_proj.weight"),
                vec![hidden, query_width],
                partitioned(1),
            ),
        ],
    )?];
    groups.push(ParameterGroupSpec::partitioned(
        format!("{attention}.key_value_heads"),
        ParameterRole::AttentionHeads,
        key_value_heads,
        [
            member(
                format!("{attention}.k_proj.weight"),
                vec![key_value_width, hidden],
                partitioned(0),
            ),
            member(
                format!("{attention}.v_proj.weight"),
                vec![key_value_width, hidden],
                partitioned(0),
            ),
        ],
    )?);
    if args.is_moe() {
        let experts = dim(args.num_experts)?;
        let intermediate = dim(args.moe_intermediate_size)?;
        groups.push(ParameterGroupSpec::partitioned(
            format!("{root}.mlp.experts.intermediate"),
            ParameterRole::ExpertIntermediate,
            intermediate,
            [
                member(
                    format!("{root}.mlp.experts.gate_up_proj"),
                    vec![experts, 2 * intermediate, hidden],
                    MemberSharding::PartitionedSegments {
                        axis: 1,
                        segments: vec![0..intermediate, intermediate..2 * intermediate],
                    },
                ),
                member(
                    format!("{root}.mlp.experts.down_proj"),
                    vec![experts, hidden, intermediate],
                    partitioned(2),
                ),
            ],
        )?);
        groups.push(replicated(
            format!("{root}.mlp.router"),
            [(format!("{root}.mlp.gate.weight"), vec![experts, hidden])],
        )?);
    } else {
        let intermediate = dim(args.intermediate_size)?;
        groups.push(ParameterGroupSpec::partitioned(
            format!("{root}.mlp.intermediate"),
            ParameterRole::FeedForwardIntermediate,
            intermediate,
            [
                member(
                    format!("{root}.mlp.gate_proj.weight"),
                    vec![intermediate, hidden],
                    partitioned(0),
                ),
                member(
                    format!("{root}.mlp.up_proj.weight"),
                    vec![intermediate, hidden],
                    partitioned(0),
                ),
                member(
                    format!("{root}.mlp.down_proj.weight"),
                    vec![hidden, intermediate],
                    partitioned(1),
                ),
            ],
        )?);
    }
    groups.push(replicated(
        format!("{root}.norms"),
        [
            (format!("{root}.input_layernorm.weight"), vec![hidden]),
            (
                format!("{root}.post_attention_layernorm.weight"),
                vec![hidden],
            ),
            (
                format!("{root}.pre_feedforward_layernorm.weight"),
                vec![hidden],
            ),
            (
                format!("{root}.post_feedforward_layernorm.weight"),
                vec![hidden],
            ),
        ],
    )?);
    Ok(groups)
}

/// Declares the patch/position roots, vision blocks, merge adapter, and projection.
pub fn vision_parameter_groups(
    args: &DecoderConfig,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let config = &args.vision_config;
    let hidden = dim(config.hidden_size)?;
    let heads = dim(config.num_heads)?;
    let patch_input = dim(config.temporal_patch_size * 3 * config.patch_size * config.patch_size)?;
    let positions = dim(config.position_height * config.position_width)?;
    let mut groups = vec![ParameterGroupSpec::partitioned(
        "model.vision_tower.patch_channels",
        ParameterRole::Channels,
        hidden,
        [
            member(
                "model.vision_tower.patch_embedder.patch_embedding.weight",
                vec![hidden, patch_input],
                partitioned(0),
            ),
            member(
                "model.vision_tower.patch_embedder.position_embedding_table.weight",
                vec![positions, hidden],
                partitioned(1),
            ),
        ],
    )?];
    for layer in 0..config.layer_count() {
        let root = format!("model.vision_tower.layers.{layer}");
        groups.push(ParameterGroupSpec::partitioned(
            format!("{root}.attention_heads"),
            ParameterRole::AttentionHeads,
            heads,
            [
                member(
                    format!("{root}.attn.q_proj.weight"),
                    vec![hidden, hidden],
                    partitioned(0),
                ),
                member(
                    format!("{root}.attn.q_proj.bias"),
                    vec![hidden],
                    partitioned(0),
                ),
                member(
                    format!("{root}.attn.k_proj.weight"),
                    vec![hidden, hidden],
                    partitioned(0),
                ),
                member(
                    format!("{root}.attn.k_proj.bias"),
                    vec![hidden],
                    partitioned(0),
                ),
                member(
                    format!("{root}.attn.v_proj.weight"),
                    vec![hidden, hidden],
                    partitioned(0),
                ),
                member(
                    format!("{root}.attn.v_proj.bias"),
                    vec![hidden],
                    partitioned(0),
                ),
                member(
                    format!("{root}.attn.proj.weight"),
                    vec![hidden, hidden],
                    partitioned(1),
                ),
                member(
                    format!("{root}.attn.proj.bias"),
                    vec![hidden],
                    MemberSharding::Replicated,
                ),
            ],
        )?);
        let intermediate = dim(config.intermediate_size)?;
        groups.push(ParameterGroupSpec::partitioned(
            format!("{root}.mlp.intermediate"),
            ParameterRole::FeedForwardIntermediate,
            intermediate,
            [
                member(
                    format!("{root}.mlp.fc1.weight"),
                    vec![intermediate, hidden],
                    partitioned(0),
                ),
                member(
                    format!("{root}.mlp.fc1.bias"),
                    vec![intermediate],
                    partitioned(0),
                ),
                member(
                    format!("{root}.mlp.fc2.weight"),
                    vec![hidden, intermediate],
                    partitioned(1),
                ),
                member(
                    format!("{root}.mlp.fc2.bias"),
                    vec![hidden],
                    MemberSharding::Replicated,
                ),
            ],
        )?);
        groups.push(replicated(
            format!("{root}.norms"),
            [
                (format!("{root}.norm1.weight"), vec![hidden]),
                (format!("{root}.norm1.bias"), vec![hidden]),
                (format!("{root}.norm2.weight"), vec![hidden]),
                (format!("{root}.norm2.bias"), vec![hidden]),
            ],
        )?);
    }
    let projector = dim(config.projector_hidden_size)?;
    let shuffled = hidden
        .checked_mul(dim(config.merge_size * config.merge_size)?)
        .ok_or_else(|| invalid("Muse-Glimmer shuffled width overflow"))?;
    groups.push(ParameterGroupSpec::partitioned(
        "model.vision_adapter.intermediate",
        ParameterRole::FeedForwardIntermediate,
        projector,
        [
            member(
                "model.vision_adapter.fc1.weight",
                vec![projector, shuffled],
                partitioned(0),
            ),
            member(
                "model.vision_adapter.fc2.weight",
                vec![projector, projector],
                partitioned(1),
            ),
        ],
    )?);
    groups.push(ParameterGroupSpec::partitioned(
        "model.vision_projection",
        ParameterRole::ColumnProjection,
        dim(config.language_hidden_size)?,
        [member(
            "model.vision_projection.weight",
            vec![dim(config.language_hidden_size)?, projector],
            partitioned(0),
        )],
    )?);
    groups.push(replicated(
        "model.vision_tower.static_norms",
        [
            ("model.vision_tower.ln_pre.weight", vec![hidden]),
            ("model.vision_tower.ln_pre.bias", vec![hidden]),
            ("model.vision_tower.ln_post.weight", vec![hidden]),
            ("model.vision_tower.ln_post.bias", vec![hidden]),
        ],
    )?);
    Ok(groups)
}

fn group(
    name: impl Into<String>,
    role: ParameterRole,
    members: impl IntoIterator<Item = (impl Into<String>, Vec<usize>, MemberSharding)>,
) -> Result<ParameterGroupSpec, ParallelPlanError> {
    ParameterGroupSpec::new(
        name,
        role,
        members
            .into_iter()
            .map(|(name, shape, sharding)| member(name, shape, sharding)),
    )
}

fn replicated(
    name: impl Into<String>,
    members: impl IntoIterator<Item = (impl Into<String>, Vec<usize>)>,
) -> Result<ParameterGroupSpec, ParallelPlanError> {
    group(
        name,
        ParameterRole::Replicated,
        members
            .into_iter()
            .map(|(name, shape)| (name, shape, MemberSharding::Replicated)),
    )
}

fn member(
    name: impl Into<String>,
    shape: Vec<usize>,
    sharding: MemberSharding,
) -> ParameterMemberSpec {
    ParameterMemberSpec::new(name, shape, sharding)
}

const fn partitioned(axis: usize) -> MemberSharding {
    MemberSharding::Partitioned { axis }
}

fn dim(value: i32) -> Result<usize, ParallelPlanError> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| invalid(format!("invalid Muse-Glimmer dimension {value}")))
}

fn invalid(message: impl Into<String>) -> ParallelPlanError {
    ParallelPlanError::InvalidGroup(message.into())
}

#[cfg(test)]
mod tests {
    use eredu_runtime::{LocalModelLayout, LocalTensorLayout, TensorPlacement};

    use super::*;

    fn args() -> DecoderConfig {
        DecoderConfig::from_hf_value(&serde_json::json!({
          "architectures":["MuseGlimmerForConditionalGeneration"],
          "model_type":"muse_glimmer",
          "image_token_id":22,"video_token_id":23,"out_hidden_size":32,"projector_hidden_size":16,
          "text_config":{"model_type":"muse_glimmer_text","hidden_size":16,"num_hidden_layers":1,
            "intermediate_size":0,"moe_intermediate_size":12,"num_experts":4,"num_experts_per_tok":2,
            "norm_topk_prob":true,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":4,
            "rms_norm_eps":0.00001,"post_norm_eps":0.00001,"vocab_size":24,"max_position_embeddings":64,
            "rope_theta":10000.0,"layer_types":["sliding_attention"],"layer_rope_theta":[10000.0],
            "sliding_window":8,"tie_word_embeddings":false,"hidden_act":"silu","attention_dropout":0.0,
            "qk_scale_factor":1.0,"output_multiplier":1.0,"final_logit_softcapping":30.0},
          "vision_config":{"model_type":"muse_glimmer_vision","hidden_size":8,"intermediate_size":12,
            "num_attention_heads":2,"num_hidden_layers":1,"patch_size":2,"patch_temporal":1,"merge_size":2,
            "pos_emb_height":2,"pos_emb_width":2,"max_position_embeddings":4,"layer_norm_eps":0.00001,
            "hidden_act":"gelu","layer_types":["full_attention"],
            "rope_parameters":{"rope_theta":10000.0,"rope_type":"default"}}
        }))
        .unwrap()
    }

    #[test]
    fn distinguishes_text_experts_router_and_vision_heads() {
        let args = args();
        let text = layer_parameter_groups(&args, 0).unwrap();
        assert!(text
            .iter()
            .any(|group| group.role() == ParameterRole::ExpertIntermediate));
        assert!(text
            .iter()
            .any(|group| group.logical_name().ends_with("mlp.router")));
        let vision = vision_parameter_groups(&args).unwrap();
        assert!(vision
            .iter()
            .any(|group| group.logical_name().ends_with("attention_heads")));
    }

    fn local_tensor(shape: Vec<usize>) -> LocalTensorLayout {
        LocalTensorLayout::new(
            "test",
            ParameterRole::AttentionHeads,
            shape.clone(),
            shape,
            TensorPlacement::Local,
            None,
            None,
            false,
        )
    }

    #[test]
    fn local_decoder_geometry_tracks_heads_and_expert_intermediate() {
        let args = args();
        let mut layout = LocalModelLayout::default();
        layout.insert(
            "model.layers.0.self_attn.q_proj.weight".into(),
            local_tensor(vec![8, 16]),
        );
        layout.insert(
            "model.layers.0.self_attn.k_proj.weight".into(),
            local_tensor(vec![4, 16]),
        );
        layout.insert(
            "model.layers.0.mlp.experts.gate_up_proj".into(),
            local_tensor(vec![4, 12, 16]),
        );
        let local = local_decoder_config(&args, 0, &layout).unwrap();
        assert_eq!(local.num_attention_heads, 2);
        assert_eq!(local.num_key_value_heads, 1);
        assert_eq!(local.moe_intermediate_size, 6);
    }
}
