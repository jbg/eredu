//! Backend-neutral Inkling checkpoint name normalization and derived recipes.

use std::collections::{BTreeMap, HashMap};

use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeCatalog},
    schema::{
        AlternativeLayoutGroup, CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint,
        GgufTypeConstraint, LayoutVariant, SafetensorsCheckpointPlan, SafetensorsTensorConstraint,
        StoredDtypeConstraint, TensorOperation,
    },
    store::TensorSelection,
    WeightQuantization,
};

use super::{FeedForwardPolicy, ModelArgs};

/// Derives an Inkling configuration whose text matrix formats reflect
/// load-time quantization instead of checkpoint-specific selections.
pub fn load_time_quantization(
    args: &ModelArgs,
    quantization: WeightQuantization,
) -> Result<ModelArgs, String> {
    quantization.validate().map_err(|error| error.to_string())?;
    let mut target = args.clone();
    target.text_config.weight_quantization = Some(quantization);
    target.text_config.quantized_weight_configs = None;
    target.validate().map_err(|error| error.to_string())?;
    Ok(target)
}

/// Applies canonical checkpoint format metadata to a complete Inkling
/// configuration.
pub fn with_checkpoint_formats(
    args: &ModelArgs,
    mut formats: HashMap<String, WeightQuantization>,
) -> Result<ModelArgs, String> {
    normalize_gguf_weight_formats(args, &mut formats)?;
    let mut target = args.clone();
    target.text_config.quantized_weight_configs = (!formats.is_empty()).then_some(formats);
    target.validate().map_err(|error| error.to_string())?;
    Ok(target)
}

/// Canonical target and released source identity.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ParameterAlias {
    /// Released checkpoint tensor name.
    pub source: String,
    /// Stable neutral parameter identity.
    pub target: String,
}

/// Gate and up projections derived from one row-interleaved `w13` tensor.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DenseW13Recipes {
    /// Even rows forming the gate projection.
    pub gate: DerivedWeightRecipe,
    /// Odd rows forming the up projection.
    pub up: DerivedWeightRecipe,
}

/// Rehomes split GGUF expert formats onto canonical fused routed and shared banks.
pub fn normalize_gguf_weight_formats(
    args: &ModelArgs,
    formats: &mut HashMap<String, WeightQuantization>,
) -> Result<(), String> {
    let layers = dimension(args.text_config.num_hidden_layers, "layer count")?;
    for layer in 0..layers {
        if args
            .text_config
            .layer_policy(layer)
            .is_none_or(|policy| policy.feed_forward != FeedForwardPolicy::SparseMoe)
        {
            continue;
        }
        for bank in ["experts", "shared_experts"] {
            let root = format!("model.layers.{layer}.moe.{bank}");
            let gate = formats.get(&format!("{root}.gate_proj")).copied();
            let up = formats.get(&format!("{root}.up_proj")).copied();
            match (gate, up) {
                (Some(gate), Some(up)) if gate == up => {
                    formats.insert(format!("{root}.gate_up_proj"), gate);
                }
                (None, None) => {}
                (gate, up) => {
                    return Err(format!(
                        "Inkling GGUF fused expert bank {root:?} requires matching gate/up formats, got {gate:?} and {up:?}"
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Builds the complete released SafeTensors catalog, including native media
/// towers and embedded prediction depths.
pub fn safetensors_plan(args: &ModelArgs) -> Result<SafetensorsCheckpointPlan, String> {
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
        let policy = text
            .layer_policy(layer)
            .ok_or_else(|| format!("Inkling layer {layer} has no normalized policy"))?;
        let local = policy.attention.window().is_some();
        let query_heads = dimension(text.query_heads(local), "query head count")?;
        let key_value_heads = dimension(text.key_value_heads(local), "key/value head count")?;
        let head = dimension(text.attention_head_dim(local), "attention head width")?;
        let d_rel = dimension(text.d_rel, "relative-attention width")?;
        let relative = policy
            .attention
            .window()
            .map(|window| window.get() as usize)
            .unwrap_or(dimension(text.rel_extent, "relative-attention extent")?);
        let convolution = dimension(text.sconv_kernel_size, "short-convolution width")?;
        let query = checked_mul(query_heads, head, "query projection width")?;
        let key_value = checked_mul(key_value_heads, head, "key/value projection width")?;
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
        add_feed_forward_layout(
            text,
            policy.feed_forward,
            &released,
            &canonical,
            hidden,
            &mut common,
            &mut groups,
        )?;
    }
    add_safetensors_media(args, hidden, &mut common)?;
    add_safetensors_mtp(args, hidden, &mut common, &mut groups)?;
    SafetensorsCheckpointPlan::new(
        "Inkling SafeTensors",
        common,
        groups,
        CatalogPolicy::strict(),
    )
    .map_err(|error| error.to_string())
}

/// Builds the complete text GGUF catalog admitted for Inkling.
pub fn gguf_plan(args: &ModelArgs) -> Result<GgufCheckpointPlan, String> {
    let text = &args.text_config;
    let hidden = dimension(text.hidden_size, "hidden size")?;
    let vocabulary = dimension(text.vocab_size, "vocabulary size")?;
    let layers = dimension(text.num_hidden_layers, "layer count")?;
    let mut tensors = vec![
        gguf(
            "token_embd.weight",
            vec![vocabulary, hidden],
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
            vec![vocabulary, hidden],
            TensorOperation::Matrix,
        ),
    ];
    for layer in 0..layers {
        let root = format!("blk.{layer}");
        let policy = text
            .layer_policy(layer)
            .ok_or_else(|| format!("Inkling layer {layer} has no normalized policy"))?;
        let local = policy.attention.window().is_some();
        let query_heads = dimension(text.query_heads(local), "query head count")?;
        let key_value_heads = dimension(text.key_value_heads(local), "key/value head count")?;
        let head = dimension(text.attention_head_dim(local), "attention head width")?;
        let d_rel = dimension(text.d_rel, "relative-attention width")?;
        let relative = policy
            .attention
            .window()
            .map(|window| window.get() as usize)
            .unwrap_or(dimension(text.rel_extent, "relative-attention extent")?);
        let convolution = dimension(text.sconv_kernel_size, "short-convolution width")?;
        let query = checked_mul(query_heads, head, "query projection width")?;
        let key_value = checked_mul(key_value_heads, head, "key/value projection width")?;
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

/// Builds the sibling audio/vision projector GGUF catalog.
pub fn mmproj_gguf_plan(args: &ModelArgs) -> Result<GgufCheckpointPlan, String> {
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

#[allow(clippy::too_many_arguments)]
fn add_feed_forward_layout(
    text: &super::TextArgs,
    policy: FeedForwardPolicy,
    released: &str,
    canonical: &str,
    hidden: usize,
    common: &mut Vec<SafetensorsTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
) -> Result<(), String> {
    match policy {
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
            let intermediate = dimension(text.moe_intermediate_size(), "MoE intermediate size")?;
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
    Ok(())
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
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
) -> Result<(), String> {
    let Some(mtp) = &args.mtp_config else {
        return Ok(());
    };
    let text = &args.text_config;
    let count = nonnegative_count(mtp.num_nextn_predict_layers, "MTP layer count")?;
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
        let key_value_heads = if local {
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
        let key_value_heads = dimension(key_value_heads, "MTP key/value head count")?;
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
        let key_value = checked_mul(key_value_heads, head, "MTP key/value projection width")?;
        let relative_query = checked_mul(query_heads, d_rel, "MTP relative-query width")?;
        let released = format!("{root}.transformer_block");
        let canonical = released.clone();
        for (source, target, shape) in [
            ("attn_norm.weight", "input_layernorm.weight", vec![hidden]),
            (
                "mlp_norm.weight",
                "post_attention_layernorm.weight",
                vec![hidden],
            ),
            (
                "attn.wq_du.weight",
                "self_attn.q_proj.weight",
                vec![query, hidden],
            ),
            (
                "attn.wk_dv.weight",
                "self_attn.k_proj.weight",
                vec![key_value, hidden],
            ),
            (
                "attn.wv_dv.weight",
                "self_attn.v_proj.weight",
                vec![key_value, hidden],
            ),
            (
                "attn.wr_du.weight",
                "self_attn.r_proj.weight",
                vec![relative_query, hidden],
            ),
            (
                "attn.wo_ud.weight",
                "self_attn.o_proj.weight",
                vec![hidden, query],
            ),
            ("attn.q_norm.weight", "self_attn.q_norm.weight", vec![head]),
            ("attn.k_norm.weight", "self_attn.k_norm.weight", vec![head]),
            (
                "attn.rel_logits_proj.proj",
                "self_attn.rel_proj",
                vec![d_rel, relative],
            ),
            (
                "attn.k_sconv.weight",
                "self_attn.k_sconv.weight",
                vec![key_value, 1, convolution],
            ),
            (
                "attn.v_sconv.weight",
                "self_attn.v_sconv.weight",
                vec![key_value, 1, convolution],
            ),
            (
                "attn_sconv.weight",
                "attn_sconv.weight",
                vec![hidden, 1, convolution],
            ),
            (
                "mlp_sconv.weight",
                "mlp_sconv.weight",
                vec![hidden, 1, convolution],
            ),
        ] {
            output.push(safe_alias(
                format!("{released}.{source}"),
                format!("{canonical}.{target}"),
                shape,
            ));
        }
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
                        vec![checked_mul(2, intermediate, "MTP fused MLP width")?, hidden],
                    )],
                    discriminator_keys: vec![packed],
                },
            ],
        });
        output.extend([
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
    if mtp.chain_hidden_post_norm {
        output.push(safe("model.mtp.chain_norm.weight", vec![hidden]));
    }
    Ok(())
}

/// Returns the complete direct-alias catalog for released SafeTensors names.
/// Interleaved dense/expert `w13` sources are intentionally represented by
/// recipe helpers instead of aliases.
pub fn safetensors_aliases(args: &ModelArgs) -> Result<Vec<ParameterAlias>, String> {
    let mut aliases = vec![
        alias("model.llm.embed.weight", "model.embed_tokens.weight"),
        alias("model.llm.embed_norm.weight", "model.embed_norm.weight"),
        alias("model.llm.norm.weight", "model.norm.weight"),
        alias("model.llm.unembed.weight", "lm_head.weight"),
    ];
    for layer in 0..args.text_config.num_hidden_layers as usize {
        let released = format!("model.llm.layers.{layer}");
        let canonical = format!("model.layers.{layer}");
        aliases.extend(block_aliases(&released, &canonical));
        match args
            .text_config
            .layer_policy(layer)
            .ok_or_else(|| format!("Inkling checkpoint layer {layer} is out of range"))?
            .feed_forward
        {
            FeedForwardPolicy::Dense => aliases.extend([
                alias(
                    format!("{released}.mlp.w2_md.weight"),
                    format!("{canonical}.dense.down_proj.weight"),
                ),
                alias(
                    format!("{released}.mlp.global_scale"),
                    format!("{canonical}.dense_global_scale"),
                ),
            ]),
            FeedForwardPolicy::SparseMoe => aliases.extend([
                alias(
                    format!("{released}.mlp.gate.weight"),
                    format!("{canonical}.moe.router.weight"),
                ),
                alias(
                    format!("{released}.mlp.gate.bias"),
                    format!("{canonical}.moe.router.bias"),
                ),
                alias(
                    format!("{released}.mlp.gate.global_scale"),
                    format!("{canonical}.moe.router.global_scale"),
                ),
                alias(
                    format!("{released}.mlp.experts.w2_weight"),
                    format!("{canonical}.moe.experts.down_proj"),
                ),
                alias(
                    format!("{released}.mlp.shared_experts.shared_w2_weight"),
                    format!("{canonical}.moe.shared_experts.down_proj"),
                ),
            ]),
        }
    }
    if let Some(audio) = &args.audio_config {
        let _ = audio;
        aliases.extend([
            alias("model.audio.encoder.weight", "audio.encoder.weight"),
            alias("model.audio.final_norm.weight", "audio.final_norm.weight"),
        ]);
    }
    if let Some(vision) = &args.vision_config {
        for layer in 0..vision.layer_specs().len() {
            aliases.push(alias(
                format!("model.visual.layers.linear_{layer}.weight"),
                format!("visual.layers.{layer}.projection.weight"),
            ));
            if layer + 1 != vision.layer_specs().len() {
                aliases.push(alias(
                    format!("model.visual.layers.norm_{layer}.weight"),
                    format!("visual.layers.{layer}.layer_norm.weight"),
                ));
            }
        }
        aliases.push(alias(
            "model.visual.final_norm.weight",
            "visual.final_norm.weight",
        ));
    }
    if let Some(mtp) = &args.mtp_config {
        for depth in 0..mtp.num_nextn_predict_layers as usize {
            let root = format!("model.mtp.layers.{depth}");
            aliases.extend([
                alias(
                    format!("{root}.hidden_norm.weight"),
                    format!("{root}.hidden_norm.weight"),
                ),
                alias(
                    format!("{root}.embed_norm.weight"),
                    format!("{root}.embed_norm.weight"),
                ),
                alias(
                    format!("{root}.input_proj.weight"),
                    format!("{root}.input_proj.weight"),
                ),
            ]);
            aliases.extend(block_aliases(
                &format!("{root}.transformer_block"),
                &format!("{root}.transformer_block"),
            ));
            aliases.extend([
                alias(
                    format!("{root}.transformer_block.mlp.w2_md.weight"),
                    format!("{root}.transformer_block.dense.down_proj.weight"),
                ),
                alias(
                    format!("{root}.transformer_block.mlp.global_scale"),
                    format!("{root}.transformer_block.dense_global_scale"),
                ),
            ]);
        }
        if mtp.chain_hidden_post_norm {
            aliases.push(alias(
                "model.mtp.chain_norm.weight",
                "model.mtp.chain_norm.weight",
            ));
        }
    }
    Ok(aliases)
}

/// Resolves the complete released SafeTensors layout into canonical
/// architecture parameter recipes.
///
/// The catalog covers direct aliases, two-dimensional released convolution
/// kernels, interleaved dense projections, routed/shared fused experts, and
/// embedded MTP blocks. Architecture selectors partition these outputs into
/// complete static and execution-unit groups before backend binding.
pub fn safetensors_recipes<C: RecipeCatalog + ?Sized>(
    args: &ModelArgs,
    catalog: &C,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let mut recipes = BTreeMap::new();
    for alias in safetensors_aliases(args)? {
        if has_tensor(catalog, &alias.source) {
            recipes.insert(
                alias.target,
                DerivedWeightRecipe::source(alias.source, TensorSelection::Full),
            );
        }
    }

    for layer in 0..nonnegative_count(args.text_config.num_hidden_layers, "layer count")? {
        let canonical = format!("model.layers.{layer}");
        let released = format!("model.llm.layers.{layer}");
        add_convolution_recipes(catalog, &released, &canonical, &mut recipes)?;
        match args
            .text_config
            .layer_policy(layer)
            .ok_or_else(|| format!("Inkling checkpoint layer {layer} is out of range"))?
            .feed_forward
        {
            FeedForwardPolicy::Dense => {
                add_dense_recipes(catalog, &released, &canonical, &mut recipes)?;
            }
            FeedForwardPolicy::SparseMoe => {
                add_expert_recipes(catalog, &released, &canonical, &mut recipes)?;
            }
        }
    }

    if let Some(mtp) = &args.mtp_config {
        for depth in
            0..nonnegative_count(mtp.num_nextn_predict_layers, "MTP prediction depth count")?
        {
            let root = format!("model.mtp.layers.{depth}.transformer_block");
            add_convolution_recipes(catalog, &root, &root, &mut recipes)?;
            add_dense_recipes(catalog, &root, &root, &mut recipes)?;
        }
    }
    Ok(recipes)
}

/// Selects the complete architecture-owned SafeTensors recipe group for pinned modules.
///
/// The returned recipes are not intersected with a backend module. Consumers must bind the
/// complete group and reject any recipe that remains unconsumed.
pub fn static_safetensors_recipes<C: RecipeCatalog + ?Sized>(
    args: &ModelArgs,
    catalog: &C,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    select_safetensors_recipe_group(args, catalog, None)
}

/// Selects the complete architecture-owned SafeTensors recipe group for one execution unit.
///
/// `group` and `index` use the canonical Inkling execution layout: vision is group zero and the
/// text decoder is group one. The returned group is deliberately complete so a backend binding
/// mismatch fails on leftover recipes instead of being silently filtered.
pub fn unit_safetensors_recipes<C: RecipeCatalog + ?Sized>(
    args: &ModelArgs,
    catalog: &C,
    group: usize,
    index: usize,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let available = match group {
        0 => args
            .vision_config
            .as_ref()
            .map_or(0, |vision| vision.num_hidden_layers as usize),
        1 => nonnegative_count(args.text_config.num_hidden_layers, "layer count")?,
        _ => {
            return Err(format!(
                "Inkling recipe group {group} is outside two groups"
            ))
        }
    };
    if index >= available {
        return Err(format!(
            "Inkling recipe unit {index} is outside group {group} with {available} units"
        ));
    }
    select_safetensors_recipe_group(args, catalog, Some((group, index)))
}

fn select_safetensors_recipe_group<C: RecipeCatalog + ?Sized>(
    args: &ModelArgs,
    catalog: &C,
    selected: Option<(usize, usize)>,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let vision_layers = args
        .vision_config
        .as_ref()
        .map_or(0, |vision| vision.num_hidden_layers as usize);
    let text_layers = nonnegative_count(args.text_config.num_hidden_layers, "layer count")?;
    let mut selected_recipes = BTreeMap::new();
    for (target, recipe) in safetensors_recipes(args, catalog)? {
        let owner = if let Some(index) = indexed_recipe_target(&target, "visual.layers.")? {
            if index >= vision_layers {
                return Err(format!(
                    "Inkling recipe {target:?} names vision unit {index}, but only {vision_layers} exist"
                ));
            }
            Some((0, index))
        } else if let Some(index) = indexed_recipe_target(&target, "model.layers.")? {
            if index >= text_layers {
                return Err(format!(
                    "Inkling recipe {target:?} names text unit {index}, but only {text_layers} exist"
                ));
            }
            Some((1, index))
        } else {
            None
        };
        if owner == selected {
            selected_recipes.insert(target, recipe);
        }
    }
    Ok(selected_recipes)
}

fn indexed_recipe_target(target: &str, root: &str) -> Result<Option<usize>, String> {
    let Some(rest) = target.strip_prefix(root) else {
        return Ok(None);
    };
    let Some((index, parameter)) = rest.split_once('.') else {
        return Err(format!(
            "architecture recipe target {target:?} has no parameter below its execution unit"
        ));
    };
    if parameter.is_empty() {
        return Err(format!(
            "architecture recipe target {target:?} has an empty unit parameter"
        ));
    }
    index
        .parse::<usize>()
        .map(Some)
        .map_err(|_| format!("architecture recipe target {target:?} has a non-numeric unit index"))
}

/// Builds the complete architecture-owned schedule for independent expert residency.
pub fn expert_residency_catalog<C: RecipeCatalog + ?Sized>(
    args: &ModelArgs,
    catalog: &C,
) -> Result<crate::ExpertResidencyCatalog, String> {
    let layers = nonnegative_count(args.text_config.num_hidden_layers, "layer count")?;
    let recipes = safetensors_recipes(args, catalog)?;
    let owner_group =
        eredu_runtime::ExecutionGroupId::new("text_decoder").map_err(|error| error.to_string())?;
    let mut units = Vec::new();
    for layer in 0..layers {
        if args
            .text_config
            .layer_policy(layer)
            .is_none_or(|policy| policy.feed_forward != FeedForwardPolicy::SparseMoe)
        {
            continue;
        }
        let unit_path = format!("model.layers.{layer}");
        let groups =
            super::layer_parameter_groups(args, layer).map_err(|error| error.to_string())?;
        for (bank, cache_layer, count, distribution) in [
            (
                "experts",
                layer,
                args.text_config.n_routed_experts,
                crate::ExpertResidencyDistribution::ExpertParallel,
            ),
            (
                "shared_experts",
                layers + layer,
                args.text_config.n_shared_experts,
                crate::ExpertResidencyDistribution::Replicated,
            ),
        ] {
            let count = nonnegative_count(count, "expert count")?;
            let bank_root = format!("{unit_path}.moe.{bank}");
            let logical_group = format!("{bank_root}.intermediate");
            let group = groups
                .iter()
                .find(|group| {
                    group.role() == eredu_runtime::ParameterRole::ExpertIntermediate
                        && group.logical_name() == logical_group
                })
                .ok_or_else(|| {
                    format!(
                        "Inkling sparse layer {layer} is missing expert group {logical_group:?}"
                    )
                })?;
            for expert in 0..count {
                let selection = TensorSelection::Range {
                    axis: 0,
                    start: expert,
                    end: expert + 1,
                };
                let has_member = |binding: &str| {
                    let target = format!("{bank_root}.{binding}");
                    group
                        .members()
                        .iter()
                        .any(|member| member.target() == target)
                };
                let gate_up_quantizable =
                    !has_member("gate_up_proj_scales") && !has_member("gate_up_proj_biases");
                let down_quantizable =
                    !has_member("down_proj_scales") && !has_member("down_proj_biases");
                let parameters = group
                    .members()
                    .iter()
                    .map(|member| {
                        let target = member.target();
                        let binding = target
                            .strip_prefix(&bank_root)
                            .and_then(|suffix| suffix.strip_prefix('.'))
                            .ok_or_else(|| {
                                format!(
                                    "Inkling expert target {target:?} is outside bank {bank_root:?}"
                                )
                            })?;
                        let bank_recipe = recipes.get(target).cloned().unwrap_or_else(|| {
                            DerivedWeightRecipe::source(target, TensorSelection::Full)
                        });
                        let recipe = DerivedWeightRecipe::Select {
                            input: Box::new(bank_recipe),
                            selection: selection.clone(),
                        };
                        let role = match binding {
                            "gate_up_proj" if gate_up_quantizable => {
                                crate::ExpertParameterRole::quantizable_projection(
                                    "gate_up_proj_scales",
                                    "gate_up_proj_biases",
                                )
                            }
                            "down_proj" if down_quantizable => {
                                crate::ExpertParameterRole::quantizable_projection(
                                    "down_proj_scales",
                                    "down_proj_biases",
                                )
                            }
                            _ => crate::ExpertParameterRole::Preserved,
                        };
                        crate::ExpertParameterRecipe::new(binding, target, recipe, role)
                            .map_err(|error| error.to_string())
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                units.push(
                    crate::ExpertResidencyUnit::new(
                        eredu_runtime::ExpertIdentity::new(cache_layer, expert),
                        owner_group.clone(),
                        layer,
                        &unit_path,
                        distribution,
                        parameters,
                    )
                    .map_err(|error| error.to_string())?,
                );
            }
        }
    }
    crate::ExpertResidencyCatalog::new(units).map_err(|error| error.to_string())
}

fn add_convolution_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    released: &str,
    canonical: &str,
    recipes: &mut BTreeMap<String, DerivedWeightRecipe>,
) -> Result<(), String> {
    for (source_suffix, target_suffix) in [
        ("attn.k_sconv.weight", "self_attn.k_sconv.weight"),
        ("attn.v_sconv.weight", "self_attn.v_sconv.weight"),
        ("attn_sconv.weight", "attn_sconv.weight"),
        ("mlp_sconv.weight", "mlp_sconv.weight"),
    ] {
        let released_source = format!("{released}.{source_suffix}");
        let target = format!("{canonical}.{target_suffix}");
        let source = if has_tensor(catalog, &released_source) {
            released_source
        } else if has_tensor(catalog, &target) {
            target.clone()
        } else {
            continue;
        };
        let metadata = catalog
            .tensor_metadata(&source)
            .map_err(|error| error.to_string())?;
        let input = DerivedWeightRecipe::source(&source, TensorSelection::Full);
        let recipe = match metadata.logical_shape.as_slice() {
            [channels, kernel] => DerivedWeightRecipe::Reshape {
                input: Box::new(input),
                shape: vec![*channels, 1, *kernel],
            },
            _ => input,
        };
        recipes.insert(target, recipe);
    }
    Ok(())
}

fn add_dense_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    released: &str,
    canonical: &str,
    recipes: &mut BTreeMap<String, DerivedWeightRecipe>,
) -> Result<(), String> {
    let source = format!("{released}.mlp.w13_dn.weight");
    if !has_tensor(catalog, &source) {
        return Ok(());
    }
    let split = dense_w13_recipes(catalog, &source)?;
    recipes.insert(format!("{canonical}.dense.gate_proj.weight"), split.gate);
    recipes.insert(format!("{canonical}.dense.up_proj.weight"), split.up);
    Ok(())
}

fn add_expert_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    released: &str,
    canonical: &str,
    recipes: &mut BTreeMap<String, DerivedWeightRecipe>,
) -> Result<(), String> {
    for (released_bank, canonical_bank) in [
        ("mlp.experts.w13_weight", "moe.experts"),
        ("mlp.shared_experts.shared_w13_weight", "moe.shared_experts"),
    ] {
        let source = format!("{released}.{released_bank}");
        let target = format!("{canonical}.{canonical_bank}.gate_up_proj");
        if has_tensor(catalog, &source) {
            recipes.insert(target, expert_w13_recipe(catalog, &source)?);
            continue;
        }
        let gate = format!("{canonical}.{canonical_bank}.gate_proj");
        let up = format!("{canonical}.{canonical_bank}.up_proj");
        if has_tensor(catalog, &gate) && has_tensor(catalog, &up) {
            recipes.insert(
                target,
                DerivedWeightRecipe::Concatenate {
                    axis: 1,
                    inputs: vec![
                        DerivedWeightRecipe::source(gate, TensorSelection::Full),
                        DerivedWeightRecipe::source(up, TensorSelection::Full),
                    ],
                },
            );
        }
    }
    Ok(())
}

fn has_tensor<C: RecipeCatalog + ?Sized>(catalog: &C, name: &str) -> bool {
    catalog.tensor_metadata(name).is_ok()
}

fn block_aliases(released: &str, canonical: &str) -> Vec<ParameterAlias> {
    [
        ("attn_norm.weight", "input_layernorm.weight"),
        ("mlp_norm.weight", "post_attention_layernorm.weight"),
        ("attn.wq_du.weight", "self_attn.q_proj.weight"),
        ("attn.wk_dv.weight", "self_attn.k_proj.weight"),
        ("attn.wv_dv.weight", "self_attn.v_proj.weight"),
        ("attn.wr_du.weight", "self_attn.r_proj.weight"),
        ("attn.wo_ud.weight", "self_attn.o_proj.weight"),
        ("attn.q_norm.weight", "self_attn.q_norm.weight"),
        ("attn.k_norm.weight", "self_attn.k_norm.weight"),
        ("attn.rel_logits_proj.proj", "self_attn.rel_proj"),
        ("attn.k_sconv.weight", "self_attn.k_sconv.weight"),
        ("attn.v_sconv.weight", "self_attn.v_sconv.weight"),
        ("attn_sconv.weight", "attn_sconv.weight"),
        ("mlp_sconv.weight", "mlp_sconv.weight"),
    ]
    .into_iter()
    .map(|(source, target)| {
        alias(
            format!("{released}.{source}"),
            format!("{canonical}.{target}"),
        )
    })
    .collect()
}

fn alias(source: impl Into<String>, target: impl Into<String>) -> ParameterAlias {
    ParameterAlias {
        source: source.into(),
        target: target.into(),
    }
}

/// Splits a released row-interleaved dense gate/up tensor into canonical
/// projections using bounded index selections.
pub fn dense_w13_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    source: &str,
) -> Result<DenseW13Recipes, String> {
    let metadata = catalog
        .tensor_metadata(source)
        .map_err(|error| error.to_string())?;
    let rows = metadata
        .logical_shape
        .first()
        .copied()
        .ok_or_else(|| format!("Inkling dense w13 tensor {source} has no row axis"))?;
    if rows == 0 || rows % 2 != 0 {
        return Err(format!(
            "Inkling dense w13 tensor {source} has invalid interleaved row count {rows}"
        ));
    }
    let selected = |parity| {
        DerivedWeightRecipe::source(
            source,
            TensorSelection::Indices {
                axis: 0,
                indices: (parity..rows).step_by(2).collect(),
            },
        )
    };
    Ok(DenseW13Recipes {
        gate: selected(0),
        up: selected(1),
    })
}

/// Converts a released interleaved expert `w13` bank into the canonical fused
/// gate-then-up layout by selecting parity rows and concatenating them.
pub fn expert_w13_recipe<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    source: &str,
) -> Result<DerivedWeightRecipe, String> {
    let metadata = catalog
        .tensor_metadata(source)
        .map_err(|error| error.to_string())?;
    let rows = metadata
        .logical_shape
        .get(1)
        .copied()
        .ok_or_else(|| format!("Inkling expert w13 tensor {source} has no row axis"))?;
    if rows == 0 || rows % 2 != 0 {
        return Err(format!(
            "Inkling expert w13 tensor {source} has invalid interleaved row count {rows}"
        ));
    }
    let selected = |parity| {
        DerivedWeightRecipe::source(
            source,
            TensorSelection::Indices {
                axis: 1,
                indices: (parity..rows).step_by(2).collect(),
            },
        )
    };
    Ok(DerivedWeightRecipe::Concatenate {
        axis: 1,
        inputs: vec![selected(0), selected(1)],
    })
}

/// Translates one Inkling text GGUF tensor into its neutral parameter identity.
pub fn translate_gguf_weight_name(name: &str) -> String {
    for (source, target) in [
        ("token_embd", "model.embed_tokens"),
        ("token_embd_norm", "model.embed_norm"),
        ("output_norm", "model.norm"),
        ("output", "lm_head"),
    ] {
        if name == source || name.starts_with(&format!("{source}.")) {
            return name.replacen(source, target, 1);
        }
    }
    let Some(rest) = name.strip_prefix("blk.") else {
        return name.to_owned();
    };
    let Some((layer, parameter)) = rest.split_once('.') else {
        return name.to_owned();
    };
    for (source, target) in [
        ("ffn_gate_exps", "moe.experts.gate_proj"),
        ("ffn_up_exps", "moe.experts.up_proj"),
        ("ffn_down_exps", "moe.experts.down_proj"),
        ("ffn_gate_shexp", "moe.shared_experts.gate_proj"),
        ("ffn_up_shexp", "moe.shared_experts.up_proj"),
        ("ffn_down_shexp", "moe.shared_experts.down_proj"),
    ] {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            let suffix = match parameter.strip_prefix(source).unwrap_or_default() {
                ".weight" => "",
                ".scales" => "_scales",
                ".biases" => "_biases",
                other => other,
            };
            return format!("model.layers.{layer}.{target}{suffix}");
        }
    }
    for (source, target) in [
        ("attn_norm", "input_layernorm"),
        ("attn_q", "self_attn.q_proj"),
        ("attn_k", "self_attn.k_proj"),
        ("attn_v", "self_attn.v_proj"),
        ("attn_r", "self_attn.r_proj"),
        ("attn_output", "self_attn.o_proj"),
        ("attn_q_norm", "self_attn.q_norm"),
        ("attn_k_norm", "self_attn.k_norm"),
        ("shortconv_k", "self_attn.k_sconv"),
        ("shortconv_v", "self_attn.v_sconv"),
        ("shortconv_attn", "attn_sconv"),
        ("shortconv_mlp", "mlp_sconv"),
        ("ffn_norm", "post_attention_layernorm"),
        ("ffn_gate", "dense.gate_proj"),
        ("ffn_up", "dense.up_proj"),
        ("ffn_down", "dense.down_proj"),
        ("ffn_gate_inp", "moe.router.weight"),
    ] {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            let mut translated = parameter.replacen(source, target, 1);
            if source == "ffn_gate_inp" && translated.ends_with(".weight") {
                translated.truncate(translated.len() - ".weight".len());
            }
            return format!("model.layers.{layer}.{translated}");
        }
    }
    if parameter == "attn_rel_proj.weight" || parameter == "attn_rel_proj" {
        return format!("model.layers.{layer}.self_attn.rel_proj");
    }
    if parameter == "ffn_gscale" || parameter == "ffn_gscale.weight" {
        return format!("model.layers.{layer}.dense_global_scale");
    }
    if matches!(
        parameter,
        "exp_probs_b.bias" | "ffn_exp_probs_b.bias" | "ffn_exp_probs_b"
    ) {
        return format!("model.layers.{layer}.moe.router.bias");
    }
    name.to_owned()
}

/// Translates an Inkling GGUF tensor using the layer schedule to disambiguate
/// the shared `ffn_gscale` spelling used by dense and sparse blocks.
pub fn translate_gguf_weight_name_for_model(name: &str, args: &ModelArgs) -> String {
    let translated = translate_gguf_weight_name(name);
    let Some(rest) = translated.strip_prefix("model.layers.") else {
        return translated;
    };
    let Some((layer, parameter)) = rest.split_once('.') else {
        return translated;
    };
    if parameter != "dense_global_scale" {
        return translated;
    }
    let Ok(layer) = layer.parse::<usize>() else {
        return translated;
    };
    if args
        .text_config
        .layer_policy(layer)
        .is_some_and(|policy| policy.feed_forward == FeedForwardPolicy::SparseMoe)
    {
        format!("model.layers.{layer}.moe.router.global_scale")
    } else {
        translated
    }
}

/// Translates one sibling projector GGUF tensor into its neutral identity.
pub fn translate_mmproj_weight_name(name: &str) -> String {
    for (source, target) in [
        ("a.dmel.embedding", "audio.encoder"),
        ("a.dmel.final_norm", "audio.final_norm"),
        ("v.hmlp.final_norm", "visual.final_norm"),
    ] {
        if name == source || name.starts_with(&format!("{source}.")) {
            return name.replacen(source, target, 1);
        }
    }
    if let Some(rest) = name.strip_prefix("v.hmlp.") {
        if let Some((layer, parameter)) = rest.split_once('.') {
            if layer.parse::<usize>().is_ok() {
                let parameter =
                    parameter
                        .replacen("linear", "projection", 1)
                        .replacen("norm", "layer_norm", 1);
                return format!("visual.layers.{layer}.{parameter}");
            }
        }
    }
    name.to_owned()
}

/// Partitions translated sibling-projector weight formats by the component
/// that owns each canonical parameter.
pub(super) fn partition_mmproj_weight_formats<T>(
    formats: HashMap<String, T>,
) -> Result<(HashMap<String, T>, HashMap<String, T>), String> {
    let mut audio = HashMap::new();
    let mut vision = HashMap::new();
    for (name, format) in formats {
        if matches!(
            name.as_str(),
            "audio.encoder.weight" | "audio.final_norm.weight"
        ) {
            audio.insert(name, format);
        } else if name == "visual.final_norm.weight" || is_visual_layer_parameter(&name) {
            vision.insert(name, format);
        } else {
            return Err(format!(
                "Inkling mmproj format map contains unknown canonical parameter {name:?}"
            ));
        }
    }
    Ok((audio, vision))
}

fn is_visual_layer_parameter(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("visual.layers.") else {
        return false;
    };
    let Some((layer, parameter)) = rest.split_once('.') else {
        return false;
    };
    layer.parse::<usize>().is_ok() && matches!(parameter, "projection.weight" | "layer_norm.weight")
}

fn gguf(
    key: impl Into<String>,
    shape: Vec<usize>,
    operation: TensorOperation,
) -> GgufTensorConstraint {
    GgufTensorConstraint::required(key, shape, GgufTypeConstraint::OperationClass(operation))
}

fn safe(key: impl Into<String>, shape: Vec<usize>) -> SafetensorsTensorConstraint {
    SafetensorsTensorConstraint::required(key, shape, StoredDtypeConstraint::Floating)
}

fn safe_alias(
    released: impl Into<String>,
    canonical: impl Into<String>,
    shape: Vec<usize>,
) -> SafetensorsTensorConstraint {
    let released = released.into();
    let canonical = canonical.into();
    let tensor = safe(&released, shape);
    if released == canonical {
        tensor
    } else {
        tensor.with_aliases([canonical])
    }
}

fn dimension(value: i32, name: &str) -> Result<usize, String> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("Inkling {name} must be positive, got {value}"))
}

fn nonnegative_count(value: i32, name: &str) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| format!("Inkling {name} cannot be negative, got {value}"))
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("Inkling {name} geometry overflows"))
}

fn checked_add(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_add(right)
        .ok_or_else(|| format!("Inkling {name} geometry overflows"))
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};

    use eredu_checkpoint::{
        recipe::{DerivedWeightRecipe, RecipeCatalog},
        store::{StoreError, TensorMetadata, TensorSelection},
        AffineQuantization, StoredDtype, WeightQuantization,
    };

    use super::{
        dense_w13_recipes, expert_residency_catalog, expert_w13_recipe, gguf_plan,
        mmproj_gguf_plan, normalize_gguf_weight_formats, partition_mmproj_weight_formats,
        safetensors_plan, safetensors_recipes, static_safetensors_recipes,
        translate_gguf_weight_name, translate_gguf_weight_name_for_model,
        translate_mmproj_weight_name, unit_safetensors_recipes,
    };
    use crate::inkling::ModelArgs;

    struct Catalog(BTreeMap<String, TensorMetadata>);

    impl RecipeCatalog for Catalog {
        fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
            self.0
                .get(key)
                .cloned()
                .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })
        }
    }

    fn metadata(name: &str, shape: Vec<usize>) -> TensorMetadata {
        TensorMetadata {
            name: name.into(),
            logical_shape: shape.clone(),
            physical_shape: shape.clone(),
            encoded_byte_len: shape.iter().product::<usize>() as u64 * 2,
            stored_dtype: StoredDtype::F16,
            backing_shard: None,
        }
    }

    #[test]
    fn dense_and_expert_interleaving_are_normalized_by_recipe() {
        let catalog = Catalog(BTreeMap::from([
            ("dense".into(), metadata("dense", vec![8, 3])),
            ("experts".into(), metadata("experts", vec![2, 8, 3])),
        ]));
        let dense = dense_w13_recipes(&catalog, "dense").unwrap();
        assert!(matches!(
            dense.gate,
            DerivedWeightRecipe::Source {
                selection: TensorSelection::Indices { axis: 0, ref indices },
                ..
            } if indices == &[0, 2, 4, 6]
        ));
        let expert = expert_w13_recipe(&catalog, "experts").unwrap();
        let inferred = expert.infer(&catalog).unwrap();
        assert_eq!(inferred.shape(), &[2, 8, 3]);
    }

    #[test]
    fn gguf_format_normalization_owns_fused_expert_names_and_compatibility() {
        let args = ModelArgs::from_hf_json(
            br#"{
              "model_type":"inkling_mm_model",
              "text_config":{"hidden_size":16,"num_hidden_layers":2,"vocab_size":64,
                "num_attention_heads":4,"num_key_value_heads":2,"head_dim":4,
                "sliding_window_size":8,"layer_types":["sliding_attention","full_attention"],
                "mlp_layer_types":["dense","moe"],"sconv_kernel_size":4,
                "d_rel":2,"rel_extent":16,"intermediate_size":32,
                "n_routed_experts":4,"num_experts_per_tok":2,"n_shared_experts":1}
            }"#,
        )
        .unwrap();
        let routed = "model.layers.1.moe.experts";
        let shared = "model.layers.1.moe.shared_experts";
        let affine = WeightQuantization::Affine(AffineQuantization::new(16, 4).unwrap());
        let mut formats = HashMap::from([
            (format!("{routed}.gate_proj"), WeightQuantization::MxFp4),
            (format!("{routed}.up_proj"), WeightQuantization::MxFp4),
            (format!("{shared}.gate_proj"), affine),
            (format!("{shared}.up_proj"), affine),
        ]);

        normalize_gguf_weight_formats(&args, &mut formats).unwrap();
        assert_eq!(
            formats.get(&format!("{routed}.gate_up_proj")),
            Some(&WeightQuantization::MxFp4)
        );
        assert_eq!(
            formats.get(&format!("{shared}.gate_up_proj")),
            Some(&affine)
        );

        formats.insert(format!("{shared}.up_proj"), WeightQuantization::MxFp4);
        assert!(normalize_gguf_weight_formats(&args, &mut formats)
            .unwrap_err()
            .contains("requires matching gate/up formats"));
    }

    #[test]
    fn safetensors_plan_covers_media_mtp_and_alternative_dense_storage() {
        let args = ModelArgs::from_hf_json(
            br#"{
              "model_type":"inkling_mm_model","image_token_id":60,"audio_token_id":61,
              "text_config":{"hidden_size":16,"num_hidden_layers":2,"vocab_size":64,
                "num_attention_heads":4,"num_key_value_heads":2,"head_dim":4,
                "sliding_window_size":8,"layer_types":["sliding_attention","full_attention"],
                "mlp_layer_types":["dense","moe"],"sconv_kernel_size":4,
                "d_rel":2,"rel_extent":16,"intermediate_size":32,
                "n_routed_experts":4,"num_experts_per_tok":2,"n_shared_experts":1},
              "mtp_config":{"num_nextn_predict_layers":2,"local_layer_ids":[1]},
              "audio_config":{"text_hidden_size":16,"num_codebooks":4,"codebook_size":8},
              "vision_config":{"text_hidden_size":16,"patch_size":40,"temporal_patch_size":2,
                "num_channels":3,"num_hidden_layers":4}
            }"#,
        )
        .unwrap();
        let plan = safetensors_plan(&args).unwrap();
        assert!(plan.catalog_policy.strict);
        assert_eq!(plan.layout_groups.len(), 3);
        assert!(plan.common_tensors.iter().any(|tensor| {
            tensor.key == "model.audio.encoder.weight" && tensor.aliases == ["audio.encoder.weight"]
        }));
        assert!(plan.common_tensors.iter().any(|tensor| {
            tensor.key == "model.visual.layers.linear_0.weight"
                && tensor.aliases == ["visual.layers.0.projection.weight"]
        }));
        assert!(plan.layout_groups.iter().all(|group| {
            group
                .variants
                .iter()
                .any(|variant| variant.id == "released interleaved")
        }));
        let gguf = gguf_plan(&args).unwrap();
        let projector = mmproj_gguf_plan(&args).unwrap();
        assert!(gguf.catalog_policy.strict && projector.catalog_policy.strict);
        assert_eq!(
            translate_gguf_weight_name("blk.1.ffn_gate_exps.weight"),
            "model.layers.1.moe.experts.gate_proj"
        );
        assert_eq!(
            translate_gguf_weight_name_for_model("blk.1.ffn_gscale", &args),
            "model.layers.1.moe.router.global_scale"
        );
        assert_eq!(
            translate_mmproj_weight_name("v.hmlp.0.linear.weight"),
            "visual.layers.0.projection.weight"
        );
        let (audio, vision) = partition_mmproj_weight_formats(HashMap::from([
            ("audio.encoder.weight".into(), 1),
            ("audio.final_norm.weight".into(), 2),
            ("visual.layers.0.projection.weight".into(), 3),
            ("visual.final_norm.weight".into(), 4),
        ]))
        .unwrap();
        assert_eq!(audio.len(), 2);
        assert_eq!(vision.len(), 2);
        assert!(
            partition_mmproj_weight_formats(HashMap::from([("decoder.weight".into(), 1,)]))
                .is_err()
        );
    }

    #[test]
    fn safetensors_recipes_own_convolution_and_fused_projection_equations() {
        let args = ModelArgs::from_hf_json(
            br#"{
              "model_type":"inkling_mm_model",
              "text_config":{"hidden_size":16,"num_hidden_layers":2,"vocab_size":64,
                "num_attention_heads":4,"num_key_value_heads":2,"head_dim":4,
                "sliding_window_size":8,"layer_types":["sliding_attention","full_attention"],
                "mlp_layer_types":["dense","moe"],"sconv_kernel_size":4,
                "d_rel":2,"rel_extent":16,"intermediate_size":32,
                "n_routed_experts":4,"num_experts_per_tok":2,"n_shared_experts":1}
            }"#,
        )
        .unwrap();
        let convolution = "model.llm.layers.0.attn_sconv.weight";
        let dense = "model.llm.layers.0.mlp.w13_dn.weight";
        let routed = "model.llm.layers.1.mlp.experts.w13_weight";
        let shared = "model.llm.layers.1.mlp.shared_experts.shared_w13_weight";
        let catalog = Catalog(BTreeMap::from([
            (
                "model.llm.embed.weight".into(),
                metadata("model.llm.embed.weight", vec![64, 16]),
            ),
            (convolution.into(), metadata(convolution, vec![16, 4])),
            (dense.into(), metadata(dense, vec![64, 16])),
            (routed.into(), metadata(routed, vec![4, 64, 16])),
            (shared.into(), metadata(shared, vec![1, 64, 16])),
        ]));
        let recipes = safetensors_recipes(&args, &catalog).unwrap();
        assert!(matches!(
            recipes.get("model.layers.0.attn_sconv.weight"),
            Some(DerivedWeightRecipe::Reshape { shape, .. }) if shape == &[16, 1, 4]
        ));
        assert!(matches!(
            recipes.get("model.layers.0.dense.gate_proj.weight"),
            Some(DerivedWeightRecipe::Source {
                selection: TensorSelection::Indices { axis: 0, indices },
                ..
            }) if indices.len() == 32
        ));
        assert!(matches!(
            recipes.get("model.layers.1.moe.experts.gate_up_proj"),
            Some(DerivedWeightRecipe::Concatenate { axis: 1, .. })
        ));
        assert!(matches!(
            recipes.get("model.layers.1.moe.shared_experts.gate_up_proj"),
            Some(DerivedWeightRecipe::Concatenate { axis: 1, .. })
        ));

        let pinned = static_safetensors_recipes(&args, &catalog).unwrap();
        assert_eq!(
            pinned.keys().map(String::as_str).collect::<Vec<_>>(),
            ["model.embed_tokens.weight"]
        );
        let dense_unit = unit_safetensors_recipes(&args, &catalog, 1, 0).unwrap();
        assert!(!dense_unit.is_empty());
        assert!(dense_unit
            .keys()
            .all(|name| name.starts_with("model.layers.0.")));
        let sparse_unit = unit_safetensors_recipes(&args, &catalog, 1, 1).unwrap();
        assert!(sparse_unit
            .keys()
            .all(|name| name.starts_with("model.layers.1.")));
        assert!(unit_safetensors_recipes(&args, &catalog, 2, 0).is_err());
        assert!(unit_safetensors_recipes(&args, &catalog, 1, 2).is_err());
    }

    #[test]
    fn residency_catalog_owns_sparse_and_shared_expert_schedule() {
        let args = ModelArgs::from_hf_json(
            br#"{
              "model_type":"inkling_mm_model",
              "text_config":{"hidden_size":16,"num_hidden_layers":2,"vocab_size":64,
                "num_attention_heads":4,"num_key_value_heads":2,"head_dim":4,
                "sliding_window_size":8,"layer_types":["sliding_attention","full_attention"],
                "mlp_layer_types":["dense","moe"],"sconv_kernel_size":4,
                "d_rel":2,"rel_extent":16,"intermediate_size":32,
                "n_routed_experts":4,"num_experts_per_tok":2,"n_shared_experts":1}
            }"#,
        )
        .unwrap();
        let routed_gate = "model.llm.layers.1.mlp.experts.w13_weight";
        let routed_down = "model.llm.layers.1.mlp.experts.w2_weight";
        let shared_gate = "model.llm.layers.1.mlp.shared_experts.shared_w13_weight";
        let shared_down = "model.llm.layers.1.mlp.shared_experts.shared_w2_weight";
        let catalog = Catalog(BTreeMap::from([
            (routed_gate.into(), metadata(routed_gate, vec![4, 64, 16])),
            (routed_down.into(), metadata(routed_down, vec![4, 16, 32])),
            (shared_gate.into(), metadata(shared_gate, vec![1, 64, 16])),
            (shared_down.into(), metadata(shared_down, vec![1, 16, 32])),
        ]));
        let catalog = expert_residency_catalog(&args, &catalog).unwrap();
        assert_eq!(catalog.units().len(), 5);
        assert_eq!(catalog.units()[0].unit_path(), "model.layers.1");
        assert_eq!(
            catalog.units()[0].identity(),
            eredu_runtime::ExpertIdentity::new(1, 0)
        );
        assert_eq!(
            catalog.units()[4].identity(),
            eredu_runtime::ExpertIdentity::new(3, 0)
        );
        assert_eq!(catalog.units()[4].owner_group().as_str(), "text_decoder");
        assert_eq!(catalog.units()[4].owner_unit(), 1);
        assert_eq!(
            catalog.units()[4].distribution(),
            crate::ExpertResidencyDistribution::Replicated
        );
        assert_eq!(
            catalog.units()[4]
                .parameters()
                .iter()
                .map(|parameter| (parameter.binding_name(), parameter.logical_target()))
                .collect::<Vec<_>>(),
            [
                (
                    "gate_up_proj",
                    "model.layers.1.moe.shared_experts.gate_up_proj"
                ),
                ("down_proj", "model.layers.1.moe.shared_experts.down_proj"),
            ]
        );
    }
}
