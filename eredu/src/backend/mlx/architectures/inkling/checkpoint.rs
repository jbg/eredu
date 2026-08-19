//! Architecture-owned checkpoint contracts for Inkling text and media weights.

use std::collections::HashMap;

use safemlx::ops::{GgufCheckpoint, GgufMetadataValue};
use serde_json::Value;

use super::model::{self, FeedForwardPolicy, InklingMmprojGguf, ModelArgs};
use crate::backend::mlx::runtime::checkpoint::store::SafetensorsWeightStore;
use eredu_checkpoint::schema::{
    AlternativeLayoutGroup, CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint,
    GgufTypeConstraint, LayoutVariant, SafetensorsCheckpointPlan, SafetensorsTensorConstraint,
    StoredDtypeConstraint, TensorOperation,
};
use eredu_checkpoint::validation;
use eredu_checkpoint::validation::{CheckpointIssue, CheckpointIssueKind, CheckpointValidation};

pub(crate) fn validate_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
) -> CheckpointValidation {
    let args = match model::model_args_from_config_value(config) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let plan = match safetensors_plan(&args) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    validation::validate_safetensors_plan(store, &plan)
}

pub(crate) fn safetensors_plan(args: &ModelArgs) -> Result<SafetensorsCheckpointPlan, String> {
    let text = &args.text_config;
    let hidden = dimension(text.hidden_size, "hidden size")?;
    let vocab = dimension(text.vocab_size, "vocabulary size")?;
    let layers = dimension(text.num_hidden_layers, "layer count")?;
    let mut common = Vec::new();
    let mut groups = Vec::new();

    for (released, canonical, shape) in [
        (
            "model.llm.embed.weight",
            "model.embed_tokens.weight",
            vec![vocab, hidden],
        ),
        (
            "model.llm.embed_norm.weight",
            "model.embed_norm.weight",
            vec![hidden],
        ),
        ("model.llm.norm.weight", "model.norm.weight", vec![hidden]),
        (
            "model.llm.unembed.weight",
            "lm_head.weight",
            vec![vocab, hidden],
        ),
    ] {
        common.push(safe_alias(released, canonical, shape));
    }

    for layer in 0..layers {
        let released = format!("model.llm.layers.{layer}");
        let canonical = format!("model.layers.{layer}");
        let policy = *text
            .layer_policy(layer)
            .ok_or_else(|| format!("Inkling layer {layer} has no normalized policy"))?;
        let local = policy.attention.window().is_some();
        let query_heads = dimension(text.q_heads(local), "query head count")?;
        let kv_heads = dimension(text.kv_heads(local), "key/value head count")?;
        let head = dimension(text.attention_head_dim(local), "attention head width")?;
        let d_rel = dimension(text.d_rel, "relative-attention width")?;
        let relative = policy
            .attention
            .window()
            .map(|window| window.get() as usize)
            .unwrap_or(dimension(text.rel_extent, "relative-attention extent")?);
        let convolution = dimension(text.sconv_kernel_size, "short-convolution width")?;
        let query = checked_mul(query_heads, head, "query projection width")?;
        let key_value = checked_mul(kv_heads, head, "key/value projection width")?;
        let relative_query = checked_mul(query_heads, d_rel, "relative-query projection width")?;
        for (source, target, shape) in [
            (
                format!("{released}.attn_norm.weight"),
                format!("{canonical}.input_layernorm.weight"),
                vec![hidden],
            ),
            (
                format!("{released}.mlp_norm.weight"),
                format!("{canonical}.post_attention_layernorm.weight"),
                vec![hidden],
            ),
            (
                format!("{released}.attn.wq_du.weight"),
                format!("{canonical}.self_attn.q_proj.weight"),
                vec![query, hidden],
            ),
            (
                format!("{released}.attn.wk_dv.weight"),
                format!("{canonical}.self_attn.k_proj.weight"),
                vec![key_value, hidden],
            ),
            (
                format!("{released}.attn.wv_dv.weight"),
                format!("{canonical}.self_attn.v_proj.weight"),
                vec![key_value, hidden],
            ),
            (
                format!("{released}.attn.wr_du.weight"),
                format!("{canonical}.self_attn.r_proj.weight"),
                vec![relative_query, hidden],
            ),
            (
                format!("{released}.attn.wo_ud.weight"),
                format!("{canonical}.self_attn.o_proj.weight"),
                vec![hidden, query],
            ),
            (
                format!("{released}.attn.q_norm.weight"),
                format!("{canonical}.self_attn.q_norm.weight"),
                vec![head],
            ),
            (
                format!("{released}.attn.k_norm.weight"),
                format!("{canonical}.self_attn.k_norm.weight"),
                vec![head],
            ),
            (
                format!("{released}.attn.rel_logits_proj.proj"),
                format!("{canonical}.self_attn.rel_proj"),
                vec![d_rel, relative],
            ),
            (
                format!("{released}.attn.k_sconv.weight"),
                format!("{canonical}.self_attn.k_sconv.weight"),
                vec![key_value, 1, convolution],
            ),
            (
                format!("{released}.attn.v_sconv.weight"),
                format!("{canonical}.self_attn.v_sconv.weight"),
                vec![key_value, 1, convolution],
            ),
            (
                format!("{released}.attn_sconv.weight"),
                format!("{canonical}.attn_sconv.weight"),
                vec![hidden, 1, convolution],
            ),
            (
                format!("{released}.mlp_sconv.weight"),
                format!("{canonical}.mlp_sconv.weight"),
                vec![hidden, 1, convolution],
            ),
        ] {
            common.push(safe_alias(source, target, shape));
        }

        match policy.feed_forward {
            FeedForwardPolicy::Dense => {
                let intermediate =
                    dimension(text.dense_intermediate_size(), "dense intermediate size")?;
                let packed = format!("{released}.mlp.w13_dn.weight");
                let gate = format!("{canonical}.dense.gate_proj.weight");
                let up = format!("{canonical}.dense.up_proj.weight");
                groups.push(AlternativeLayoutGroup {
                    id: format!("{canonical} dense gate/up storage"),
                    required: true,
                    variants: vec![
                        LayoutVariant {
                            id: "canonical split".into(),
                            tensors: vec![
                                safe(&gate, vec![intermediate, hidden]),
                                safe(&up, vec![intermediate, hidden]),
                            ],
                            discriminator_keys: vec![gate, up],
                        },
                        LayoutVariant {
                            id: "released interleaved".into(),
                            tensors: vec![safe(
                                &packed,
                                vec![
                                    checked_mul(2, intermediate, "dense interleaved width")?,
                                    hidden,
                                ],
                            )],
                            discriminator_keys: vec![packed],
                        },
                    ],
                });
                common.extend([
                    safe_alias(
                        format!("{released}.mlp.w2_md.weight"),
                        format!("{canonical}.dense.down_proj.weight"),
                        vec![hidden, intermediate],
                    ),
                    safe_alias(
                        format!("{released}.mlp.global_scale"),
                        format!("{canonical}.dense_global_scale"),
                        vec![1],
                    ),
                ]);
            }
            FeedForwardPolicy::SparseMoe => {
                let routed = dimension(text.n_routed_experts, "routed expert count")?;
                let shared = dimension(text.n_shared_experts, "shared expert count")?;
                let intermediate =
                    dimension(text.moe_intermediate_size(), "MoE intermediate size")?;
                let fused = checked_mul(2, intermediate, "fused expert width")?;
                for (source, target, shape) in [
                    (
                        format!("{released}.mlp.gate.weight"),
                        format!("{canonical}.moe.router.weight"),
                        vec![checked_add(routed, shared, "router width")?, hidden],
                    ),
                    (
                        format!("{released}.mlp.gate.bias"),
                        format!("{canonical}.moe.router.bias"),
                        vec![routed],
                    ),
                    (
                        format!("{released}.mlp.gate.global_scale"),
                        format!("{canonical}.moe.router.global_scale"),
                        vec![1],
                    ),
                    (
                        format!("{released}.mlp.experts.w13_weight"),
                        format!("{canonical}.moe.experts.gate_up_proj"),
                        vec![routed, fused, hidden],
                    ),
                    (
                        format!("{released}.mlp.experts.w2_weight"),
                        format!("{canonical}.moe.experts.down_proj"),
                        vec![routed, hidden, intermediate],
                    ),
                    (
                        format!("{released}.mlp.shared_experts.shared_w13_weight"),
                        format!("{canonical}.moe.shared_experts.gate_up_proj"),
                        vec![shared, fused, hidden],
                    ),
                    (
                        format!("{released}.mlp.shared_experts.shared_w2_weight"),
                        format!("{canonical}.moe.shared_experts.down_proj"),
                        vec![shared, hidden, intermediate],
                    ),
                ] {
                    common.push(safe_alias(source, target, shape));
                }
            }
        }
    }

    add_safetensors_media(args, hidden, &mut common)?;
    add_safetensors_mtp(args, hidden, &mut common)?;
    SafetensorsCheckpointPlan::new(
        "Inkling SafeTensors",
        common,
        groups,
        CatalogPolicy::strict(),
    )
    .map_err(|error| error.to_string())
}

fn add_safetensors_media(
    args: &ModelArgs,
    hidden: usize,
    output: &mut Vec<SafetensorsTensorConstraint>,
) -> Result<(), String> {
    if let Some(audio) = &args.audio_config {
        let codebooks = dimension(audio.num_codebooks, "audio codebook count")?;
        let codebook_size = dimension(audio.codebook_size, "audio codebook size")?;
        output.extend([
            safe_alias(
                "model.audio.encoder.weight",
                "audio.encoder.weight",
                vec![
                    checked_mul(codebooks, codebook_size, "audio vocabulary")?,
                    hidden,
                ],
            ),
            safe_alias(
                "model.audio.final_norm.weight",
                "audio.final_norm.weight",
                vec![hidden],
            ),
        ]);
    }
    if let Some(vision) = &args.vision_config {
        let specs = vision.layer_specs();
        for (layer, (input, output_width, _, _)) in specs.iter().copied().enumerate() {
            let input = dimension(input, "vision layer input width")?;
            let output_width = dimension(output_width, "vision layer output width")?;
            output.push(safe_alias(
                format!("model.visual.layers.linear_{layer}.weight"),
                format!("visual.layers.{layer}.projection.weight"),
                vec![output_width, input],
            ));
            if layer + 1 != specs.len() {
                output.push(safe_alias(
                    format!("model.visual.layers.norm_{layer}.weight"),
                    format!("visual.layers.{layer}.layer_norm.weight"),
                    vec![output_width],
                ));
            }
        }
        output.push(safe_alias(
            "model.visual.final_norm.weight",
            "visual.final_norm.weight",
            vec![hidden],
        ));
    }
    Ok(())
}

fn add_safetensors_mtp(
    args: &ModelArgs,
    hidden: usize,
    output: &mut Vec<SafetensorsTensorConstraint>,
) -> Result<(), String> {
    let Some(mtp) = &args.mtp_config else {
        return Ok(());
    };
    let text = &args.text_config;
    let count = count(mtp.num_nextn_predict_layers, "MTP layer count")?;
    for depth in 0..count {
        let root = format!("model.mtp.layers.{depth}");
        output.extend([
            safe(format!("{root}.hidden_norm.weight"), vec![hidden]),
            safe(format!("{root}.embed_norm.weight"), vec![hidden]),
            safe(
                format!("{root}.input_proj.weight"),
                vec![hidden, checked_mul(2, hidden, "MTP input width")?],
            ),
        ]);
        let local = mtp.local_layer_ids.contains(&depth);
        let query_heads = if local {
            mtp.swa_num_attention_heads
                .or(text.swa_num_attention_heads)
                .unwrap_or(mtp.num_attention_heads.unwrap_or(text.num_attention_heads))
        } else {
            mtp.num_attention_heads.unwrap_or(text.num_attention_heads)
        };
        let kv_heads = if local {
            mtp.swa_num_key_value_heads
                .or(text.swa_num_key_value_heads)
                .unwrap_or(mtp.num_key_value_heads.unwrap_or(text.num_key_value_heads))
        } else {
            mtp.num_key_value_heads.unwrap_or(text.num_key_value_heads)
        };
        let head = if local {
            mtp.swa_head_dim
                .or(text.swa_head_dim)
                .unwrap_or(mtp.head_dim.unwrap_or(text.head_dim))
        } else {
            mtp.head_dim.unwrap_or(text.head_dim)
        };
        let query_heads = dimension(query_heads, "MTP query head count")?;
        let kv_heads = dimension(kv_heads, "MTP key/value head count")?;
        let head = dimension(head, "MTP attention head width")?;
        let d_rel = dimension(mtp.d_rel.unwrap_or(text.d_rel), "MTP relative width")?;
        let relative = if local {
            text.layer_schedule
                .iter()
                .find_map(|policy| policy.attention.window())
                .map(|window| window.get() as usize)
                .unwrap_or(dimension(
                    mtp.rel_extent.unwrap_or(text.rel_extent),
                    "MTP relative extent",
                )?)
        } else {
            dimension(
                mtp.rel_extent.unwrap_or(text.rel_extent),
                "MTP relative extent",
            )?
        };
        let convolution = dimension(
            mtp.sconv_kernel_size.unwrap_or(text.sconv_kernel_size),
            "MTP convolution width",
        )?;
        let intermediate = dimension(
            mtp.dense_intermediate_size
                .or(text.dense_intermediate_size)
                .unwrap_or(mtp.intermediate_size.unwrap_or(text.intermediate_size)),
            "MTP intermediate size",
        )?;
        let query = checked_mul(query_heads, head, "MTP query projection width")?;
        let key_value = checked_mul(kv_heads, head, "MTP key/value projection width")?;
        let relative_query = checked_mul(query_heads, d_rel, "MTP relative-query width")?;
        let block = format!("{root}.transformer_block");
        for (suffix, shape) in [
            ("attn_norm.weight", vec![hidden]),
            ("mlp_norm.weight", vec![hidden]),
            ("attn.wq_du.weight", vec![query, hidden]),
            ("attn.wk_dv.weight", vec![key_value, hidden]),
            ("attn.wv_dv.weight", vec![key_value, hidden]),
            ("attn.wr_du.weight", vec![relative_query, hidden]),
            ("attn.wo_ud.weight", vec![hidden, query]),
            ("attn.q_norm.weight", vec![head]),
            ("attn.k_norm.weight", vec![head]),
            ("attn.rel_logits_proj.proj", vec![d_rel, relative]),
            ("attn.k_sconv.weight", vec![key_value, 1, convolution]),
            ("attn.v_sconv.weight", vec![key_value, 1, convolution]),
            ("attn_sconv.weight", vec![hidden, 1, convolution]),
            ("mlp_sconv.weight", vec![hidden, 1, convolution]),
            (
                "mlp.w13_dn.weight",
                vec![checked_mul(2, intermediate, "MTP fused MLP width")?, hidden],
            ),
            ("mlp.w2_md.weight", vec![hidden, intermediate]),
            ("mlp.global_scale", vec![1]),
        ] {
            output.push(safe(format!("{block}.{suffix}"), shape));
        }
    }
    if mtp.chain_hidden_post_norm {
        output.push(safe("model.mtp.chain_norm.weight", vec![hidden]));
    }
    Ok(())
}

fn safe(key: impl Into<String>, shape: Vec<usize>) -> SafetensorsTensorConstraint {
    SafetensorsTensorConstraint::required(key, shape, StoredDtypeConstraint::Floating)
}

fn safe_alias(
    released: impl Into<String>,
    canonical: impl Into<String>,
    shape: Vec<usize>,
) -> SafetensorsTensorConstraint {
    safe(released, shape).with_aliases([canonical])
}

pub(crate) fn validate_gguf(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> CheckpointValidation {
    if let Err(error) = checkpoint
        .catalog()
        .translated_outputs(model::translate_gguf_weight_name)
    {
        return conflicting_layout(error.to_string());
    }
    let args = match model::args_from_gguf_catalog(metadata) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    if args.text_config.num_hidden_layers as usize > checkpoint.catalog().physical_tensor_count() {
        return invalid_geometry(format!(
            "configured layer count {} exceeds the entire {}-tensor Inkling GGUF catalog",
            args.text_config.num_hidden_layers,
            checkpoint.catalog().physical_tensor_count()
        ));
    }
    let plan = match gguf_plan(&args) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    let mut issues = issues(validation::validate_gguf_plan(checkpoint, &plan));
    issues.extend(validation::validate_dense_or_matching_gguf_encodings(
        checkpoint,
        args.text_config
            .layer_schedule
            .iter()
            .enumerate()
            .filter(|(_, policy)| policy.feed_forward == FeedForwardPolicy::SparseMoe)
            .flat_map(|(layer, _)| {
                [
                    (
                        format!("blk.{layer}.ffn_gate_exps.weight"),
                        format!("blk.{layer}.ffn_up_exps.weight"),
                    ),
                    (
                        format!("blk.{layer}.ffn_gate_shexp.weight"),
                        format!("blk.{layer}.ffn_up_shexp.weight"),
                    ),
                ]
            }),
        "Inkling",
    ));
    CheckpointValidation::from_issues(issues)
}

pub(crate) fn gguf_plan(args: &ModelArgs) -> Result<GgufCheckpointPlan, String> {
    let text = &args.text_config;
    let hidden = dimension(text.hidden_size, "hidden size")?;
    let vocab = dimension(text.vocab_size, "vocabulary size")?;
    let layers = dimension(text.num_hidden_layers, "layer count")?;
    let mut tensors = vec![
        gguf(
            "token_embd.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
        ),
        gguf(
            "token_embd_norm.weight",
            vec![hidden],
            TensorOperation::Vector,
        ),
        gguf("output_norm.weight", vec![hidden], TensorOperation::Vector),
        gguf(
            "output.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
        ),
    ];
    for layer in 0..layers {
        let root = format!("blk.{layer}");
        let policy = *text
            .layer_policy(layer)
            .ok_or_else(|| format!("Inkling layer {layer} has no normalized policy"))?;
        let local = policy.attention.window().is_some();
        let query_heads = dimension(text.q_heads(local), "query head count")?;
        let kv_heads = dimension(text.kv_heads(local), "key/value head count")?;
        let head = dimension(text.attention_head_dim(local), "attention head width")?;
        let d_rel = dimension(text.d_rel, "relative-attention width")?;
        let relative = policy
            .attention
            .window()
            .map(|window| window.get() as usize)
            .unwrap_or(dimension(text.rel_extent, "relative-attention extent")?);
        let convolution = dimension(text.sconv_kernel_size, "short-convolution width")?;
        let query = checked_mul(query_heads, head, "query projection width")?;
        let key_value = checked_mul(kv_heads, head, "key/value projection width")?;
        tensors.extend([
            gguf(
                format!("{root}.attn_norm.weight"),
                vec![hidden],
                TensorOperation::Vector,
            ),
            gguf(
                format!("{root}.ffn_norm.weight"),
                vec![hidden],
                TensorOperation::Vector,
            ),
            gguf(
                format!("{root}.attn_q.weight"),
                vec![query, hidden],
                TensorOperation::Matrix,
            ),
            gguf(
                format!("{root}.attn_k.weight"),
                vec![key_value, hidden],
                TensorOperation::Matrix,
            ),
            gguf(
                format!("{root}.attn_v.weight"),
                vec![key_value, hidden],
                TensorOperation::Matrix,
            ),
            gguf(
                format!("{root}.attn_r.weight"),
                vec![
                    checked_mul(query_heads, d_rel, "relative-query width")?,
                    hidden,
                ],
                TensorOperation::Matrix,
            ),
            gguf(
                format!("{root}.attn_output.weight"),
                vec![hidden, query],
                TensorOperation::Matrix,
            ),
            gguf(
                format!("{root}.attn_q_norm.weight"),
                vec![head],
                TensorOperation::Vector,
            ),
            gguf(
                format!("{root}.attn_k_norm.weight"),
                vec![head],
                TensorOperation::Vector,
            ),
            gguf(
                format!("{root}.attn_rel_proj.weight"),
                vec![d_rel, relative],
                TensorOperation::Dense,
            )
            .with_aliases([format!("{root}.attn_rel_proj")]),
        ]);
        for (name, channels) in [
            ("shortconv_k", key_value),
            ("shortconv_v", key_value),
            ("shortconv_attn", hidden),
            ("shortconv_mlp", hidden),
        ] {
            tensors.push(
                gguf(
                    format!("{root}.{name}.weight"),
                    vec![channels, convolution],
                    TensorOperation::Dense,
                )
                .with_alternate_shapes([vec![channels, 1, convolution]]),
            );
        }
        match policy.feed_forward {
            FeedForwardPolicy::Dense => {
                let intermediate =
                    dimension(text.dense_intermediate_size(), "dense intermediate size")?;
                tensors.extend([
                    gguf(
                        format!("{root}.ffn_gate.weight"),
                        vec![intermediate, hidden],
                        TensorOperation::Matrix,
                    ),
                    gguf(
                        format!("{root}.ffn_up.weight"),
                        vec![intermediate, hidden],
                        TensorOperation::Matrix,
                    ),
                    gguf(
                        format!("{root}.ffn_down.weight"),
                        vec![hidden, intermediate],
                        TensorOperation::Matrix,
                    ),
                    gguf(
                        format!("{root}.ffn_gscale"),
                        vec![1],
                        TensorOperation::Vector,
                    )
                    .with_aliases([format!("{root}.ffn_gscale.weight")]),
                ]);
            }
            FeedForwardPolicy::SparseMoe => {
                let routed = dimension(text.n_routed_experts, "routed expert count")?;
                let shared = dimension(text.n_shared_experts, "shared expert count")?;
                let intermediate =
                    dimension(text.moe_intermediate_size(), "MoE intermediate size")?;
                tensors.extend([
                    gguf(
                        format!("{root}.ffn_gate_inp.weight"),
                        vec![checked_add(routed, shared, "router width")?, hidden],
                        TensorOperation::Dense,
                    ),
                    gguf(
                        format!("{root}.exp_probs_b.bias"),
                        vec![routed],
                        TensorOperation::Vector,
                    )
                    .with_aliases([
                        format!("{root}.ffn_exp_probs_b.bias"),
                        format!("{root}.ffn_exp_probs_b"),
                    ]),
                    gguf(
                        format!("{root}.ffn_gscale"),
                        vec![1],
                        TensorOperation::Vector,
                    )
                    .with_aliases([format!("{root}.ffn_gscale.weight")]),
                    gguf(
                        format!("{root}.ffn_gate_exps.weight"),
                        vec![routed, intermediate, hidden],
                        TensorOperation::Matrix,
                    ),
                    gguf(
                        format!("{root}.ffn_up_exps.weight"),
                        vec![routed, intermediate, hidden],
                        TensorOperation::Matrix,
                    ),
                    gguf(
                        format!("{root}.ffn_down_exps.weight"),
                        vec![routed, hidden, intermediate],
                        TensorOperation::Matrix,
                    ),
                    gguf(
                        format!("{root}.ffn_gate_shexp.weight"),
                        vec![shared, intermediate, hidden],
                        TensorOperation::Matrix,
                    ),
                    gguf(
                        format!("{root}.ffn_up_shexp.weight"),
                        vec![shared, intermediate, hidden],
                        TensorOperation::Matrix,
                    ),
                    gguf(
                        format!("{root}.ffn_down_shexp.weight"),
                        vec![shared, hidden, intermediate],
                        TensorOperation::Matrix,
                    ),
                ]);
            }
        }
    }
    GgufCheckpointPlan::new("Inkling GGUF", tensors, Vec::new(), CatalogPolicy::strict())
        .map_err(|error| error.to_string())
}

pub(crate) fn validate_mmproj_gguf(
    model_metadata: &HashMap<String, GgufMetadataValue>,
    mmproj: &InklingMmprojGguf,
) -> CheckpointValidation {
    if let Err(error) = mmproj
        .checkpoint
        .catalog()
        .translated_outputs(model::translate_mmproj_weight_name)
    {
        return conflicting_layout(error.to_string());
    }
    let mut args = match model::args_from_gguf_catalog(model_metadata) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    if let Err(error) = model::apply_mmproj_args(&mut args, model_metadata, mmproj) {
        return invalid_geometry(error.to_string());
    }
    let plan = match mmproj_gguf_plan(&args) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    validation::validate_gguf_plan(&mmproj.checkpoint, &plan)
}

pub(crate) fn mmproj_gguf_plan(args: &ModelArgs) -> Result<GgufCheckpointPlan, String> {
    let audio = args
        .audio_config
        .as_ref()
        .ok_or_else(|| "Inkling mmproj has no audio geometry".to_string())?;
    let vision = args
        .vision_config
        .as_ref()
        .ok_or_else(|| "Inkling mmproj has no vision geometry".to_string())?;
    let hidden = dimension(args.text_config.hidden_size, "hidden size")?;
    let codebooks = dimension(audio.num_codebooks, "audio codebook count")?;
    let codebook_size = dimension(audio.codebook_size, "audio codebook size")?;
    let mut tensors = vec![
        gguf(
            "a.dmel.embedding.weight",
            vec![
                checked_mul(codebooks, codebook_size, "audio vocabulary")?,
                hidden,
            ],
            TensorOperation::Matrix,
        ),
        gguf(
            "a.dmel.final_norm.weight",
            vec![hidden],
            TensorOperation::Vector,
        ),
    ];
    let specs = vision.layer_specs();
    for (layer, (input, output, _, _)) in specs.iter().copied().enumerate() {
        let input = dimension(input, "vision layer input width")?;
        let output = dimension(output, "vision layer output width")?;
        tensors.push(gguf(
            format!("v.hmlp.{layer}.linear.weight"),
            vec![output, input],
            TensorOperation::Matrix,
        ));
        if layer + 1 != specs.len() {
            tensors.push(gguf(
                format!("v.hmlp.{layer}.norm.weight"),
                vec![output],
                TensorOperation::Vector,
            ));
        }
    }
    tensors.push(gguf(
        "v.hmlp.final_norm.weight",
        vec![hidden],
        TensorOperation::Vector,
    ));
    GgufCheckpointPlan::new(
        "Inkling mmproj GGUF",
        tensors,
        Vec::new(),
        CatalogPolicy::strict(),
    )
    .map_err(|error| error.to_string())
}

fn gguf(
    key: impl Into<String>,
    shape: Vec<usize>,
    operation: TensorOperation,
) -> GgufTensorConstraint {
    GgufTensorConstraint::required(key, shape, GgufTypeConstraint::OperationClass(operation))
}

fn issues(validation: CheckpointValidation) -> Vec<CheckpointIssue> {
    match validation {
        CheckpointValidation::Exact => Vec::new(),
        CheckpointValidation::Invalid(issues) => issues,
        CheckpointValidation::Unverified(issue) => vec![issue],
    }
}

fn dimension(value: i32, name: &str) -> Result<usize, String> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("Inkling {name} must be positive, got {value}"))
}

fn count(value: i32, name: &str) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| format!("Inkling {name} must be non-negative, got {value}"))
}

fn checked_add(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_add(right)
        .ok_or_else(|| format!("Inkling {name} geometry overflows"))
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("Inkling {name} geometry overflows"))
}

fn conflicting_layout(detail: String) -> CheckpointValidation {
    CheckpointValidation::Invalid(vec![CheckpointIssue {
        kind: CheckpointIssueKind::ConflictingLayout,
        detail,
        tensor_name: None,
        tensor_type_code: None,
        metadata_key: None,
    }])
}

fn invalid_geometry(detail: String) -> CheckpointValidation {
    CheckpointValidation::Invalid(vec![CheckpointIssue {
        kind: CheckpointIssueKind::InvalidGeometry,
        detail,
        tensor_name: None,
        tensor_type_code: None,
        metadata_key: None,
    }])
}
