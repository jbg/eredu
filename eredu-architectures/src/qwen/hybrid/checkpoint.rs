//! SafeTensors contracts and fused-projection recipes for the hybrid decoder.

use eredu_checkpoint::{
    recipe::{AtomicRecipeSet, DerivedWeightRecipe, RecipeCatalog},
    schema::{
        matrix_for_linear_format, AlternativeLayoutGroup, CatalogPolicy, GgufCheckpointPlan,
        GgufTensorConstraint, GgufTypeConstraint, LayoutVariant, MatrixScaleNames,
        SafetensorsCheckpointPlan, SafetensorsTensorConstraint, StoredDtypeConstraint,
        TensorOperation,
    },
    store::TensorSelection,
};

use super::{
    fp8_block_row_widths, fused_projection_widths, HybridConfig, HybridLayerPolicy, HybridVariant,
};

/// Builds the strict hybrid text SafeTensors catalog.
pub fn safetensors_plan(config: &HybridConfig) -> Result<SafetensorsCheckpointPlan, String> {
    config.validate().map_err(|error| error.to_string())?;
    let hidden = dim(config.hidden_size, "hidden_size")?;
    let vocab = dim(config.vocab_size, "vocab_size")?;
    let mut common = Vec::new();
    let mut groups = Vec::new();
    matrix(
        config,
        &mut common,
        "model.embed_tokens.weight",
        vec![vocab, hidden],
    )?;
    vector(&mut common, "model.norm.weight", hidden);
    if !config.tie_word_embeddings {
        matrix(config, &mut common, "lm_head.weight", vec![vocab, hidden])?;
    }
    for (layer, policy) in config.layer_schedule.iter().copied().enumerate() {
        add_block(config, layer, policy, &mut common, &mut groups)?;
    }
    let mtp_layers = usize::try_from(config.mtp_num_hidden_layers)
        .map_err(|_| "mtp_num_hidden_layers must be non-negative")?;
    if mtp_layers > 0 {
        vector(&mut common, "mtp.pre_fc_norm_hidden.weight", hidden);
        vector(&mut common, "mtp.pre_fc_norm_embedding.weight", hidden);
        common.extend(
            matrix_for_linear_format(
                "mtp.fc.weight",
                Vec::<String>::new(),
                vec![hidden, mul(2, hidden)?],
                eredu_checkpoint::LinearFormat::Dense,
                None,
            )
            .map_err(|error| error.to_string())?,
        );
        vector(&mut common, "mtp.norm.weight", hidden);
        for depth in 0..mtp_layers {
            add_block_at(
                config,
                config.num_hidden_layers as usize + depth,
                &format!("mtp.layers.{depth}"),
                HybridLayerPolicy::SelfAttention(eredu_core::attention::AttentionPolicy::Full),
                &mut common,
                &mut groups,
            )?;
        }
    }
    SafetensorsCheckpointPlan::new(
        format!("{} SafeTensors", config.model_type),
        common,
        groups,
        CatalogPolicy::strict(),
    )
    .map_err(|error| error.to_string())
}

/// Builds the canonical llama.cpp Qwen3-Next/Qwen3.5 tensor catalog.
pub fn gguf_plan(config: &HybridConfig) -> Result<GgufCheckpointPlan, String> {
    config.validate().map_err(|error| error.to_string())?;
    let hidden = dim(config.hidden_size, "hidden_size")?;
    let vocab = dim(config.vocab_size, "vocab_size")?;
    let query = mul(
        dim(config.num_attention_heads, "num_attention_heads")?,
        dim(config.head_dim, "head_dim")?,
    )?;
    let key_value = mul(
        dim(config.num_key_value_heads, "num_key_value_heads")?,
        dim(config.head_dim, "head_dim")?,
    )?;
    let key = mul(
        dim(config.linear_num_key_heads, "linear_num_key_heads")?,
        dim(config.linear_key_head_dim, "linear_key_head_dim")?,
    )?;
    let value = mul(
        dim(config.linear_num_value_heads, "linear_num_value_heads")?,
        dim(config.linear_value_head_dim, "linear_value_head_dim")?,
    )?;
    let value_heads = dim(config.linear_num_value_heads, "linear_num_value_heads")?;
    let mut tensors = vec![
        gguf_tensor(
            "token_embd.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
        ),
        gguf_tensor("output_norm.weight", vec![hidden], TensorOperation::Vector),
    ];
    if !config.tie_word_embeddings {
        tensors.push(gguf_tensor(
            "output.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
        ));
    }
    let mut groups = Vec::new();
    for (layer, policy) in config.layer_schedule.iter().copied().enumerate() {
        let root = format!("blk.{layer}");
        tensors.extend([
            gguf_tensor(
                format!("{root}.attn_norm.weight"),
                vec![hidden],
                TensorOperation::Vector,
            ),
            gguf_tensor(
                format!("{root}.post_attention_norm.weight"),
                vec![hidden],
                TensorOperation::Vector,
            ),
        ]);
        match policy {
            HybridLayerPolicy::SelfAttention(_) => {
                tensors.extend([
                    gguf_tensor(
                        format!("{root}.attn_q.weight"),
                        vec![mul(2, query)?, hidden],
                        TensorOperation::Matrix,
                    ),
                    gguf_tensor(
                        format!("{root}.attn_k.weight"),
                        vec![key_value, hidden],
                        TensorOperation::Matrix,
                    ),
                    gguf_tensor(
                        format!("{root}.attn_v.weight"),
                        vec![key_value, hidden],
                        TensorOperation::Matrix,
                    ),
                    gguf_tensor(
                        format!("{root}.attn_output.weight"),
                        vec![hidden, query],
                        TensorOperation::Matrix,
                    ),
                    gguf_tensor(
                        format!("{root}.attn_q_norm.weight"),
                        vec![dim(config.head_dim, "head_dim")?],
                        TensorOperation::Vector,
                    ),
                    gguf_tensor(
                        format!("{root}.attn_k_norm.weight"),
                        vec![dim(config.head_dim, "head_dim")?],
                        TensorOperation::Vector,
                    ),
                ]);
                if config.attention_bias {
                    tensors.extend([
                        gguf_tensor(
                            format!("{root}.attn_q.bias"),
                            vec![mul(2, query)?],
                            TensorOperation::Vector,
                        ),
                        gguf_tensor(
                            format!("{root}.attn_k.bias"),
                            vec![key_value],
                            TensorOperation::Vector,
                        ),
                        gguf_tensor(
                            format!("{root}.attn_v.bias"),
                            vec![key_value],
                            TensorOperation::Vector,
                        ),
                        gguf_tensor(
                            format!("{root}.attn_output.bias"),
                            vec![hidden],
                            TensorOperation::Vector,
                        ),
                    ]);
                }
            }
            HybridLayerPolicy::LinearAttention => {
                if config.variant == HybridVariant::Qwen3Next {
                    tensors.extend([
                        gguf_tensor(
                            format!("{root}.attn_qkvz.weight"),
                            vec![add(mul(2, key)?, mul(2, value)?)?, hidden],
                            TensorOperation::Matrix,
                        ),
                        gguf_tensor(
                            format!("{root}.ssm_ba.weight"),
                            vec![mul(2, value_heads)?, hidden],
                            TensorOperation::Matrix,
                        ),
                    ]);
                } else {
                    tensors.extend([
                        gguf_tensor(
                            format!("{root}.attn_qkv.weight"),
                            vec![add(mul(2, key)?, value)?, hidden],
                            TensorOperation::Matrix,
                        ),
                        gguf_tensor(
                            format!("{root}.attn_gate.weight"),
                            vec![value, hidden],
                            TensorOperation::Matrix,
                        ),
                        gguf_tensor(
                            format!("{root}.ssm_beta.weight"),
                            vec![value_heads, hidden],
                            TensorOperation::Matrix,
                        ),
                        gguf_tensor(
                            format!("{root}.ssm_alpha.weight"),
                            vec![value_heads, hidden],
                            TensorOperation::Matrix,
                        ),
                    ]);
                }
                tensors.extend([
                    gguf_tensor(
                        format!("{root}.ssm_conv1d.weight"),
                        vec![
                            add(mul(2, key)?, value)?,
                            dim(config.linear_conv_kernel_dim, "linear_conv_kernel_dim")?,
                        ],
                        TensorOperation::Dense,
                    ),
                    gguf_tensor(
                        format!("{root}.ssm_dt.bias"),
                        vec![value_heads],
                        TensorOperation::Vector,
                    ),
                    gguf_tensor(
                        format!("{root}.ssm_a"),
                        vec![value_heads],
                        TensorOperation::Dense,
                    ),
                    gguf_tensor(
                        format!("{root}.ssm_norm.weight"),
                        vec![dim(config.linear_value_head_dim, "linear_value_head_dim")?],
                        TensorOperation::Vector,
                    ),
                    gguf_tensor(
                        format!("{root}.ssm_out.weight"),
                        vec![hidden, value],
                        TensorOperation::Matrix,
                    ),
                ]);
            }
        }
        if config.is_moe() {
            let experts = dim(config.num_experts, "num_experts")?;
            let intermediate = dim(config.moe_intermediate_size, "moe_intermediate_size")?;
            let shared = dim(
                config.shared_expert_intermediate_size,
                "shared_expert_intermediate_size",
            )?;
            tensors.extend([
                gguf_tensor(
                    format!("{root}.ffn_gate_inp.weight"),
                    vec![experts, hidden],
                    TensorOperation::Matrix,
                ),
                gguf_tensor(
                    format!("{root}.ffn_gate_shexp.weight"),
                    vec![shared, hidden],
                    TensorOperation::Matrix,
                ),
                gguf_tensor(
                    format!("{root}.ffn_up_shexp.weight"),
                    vec![shared, hidden],
                    TensorOperation::Matrix,
                ),
                gguf_tensor(
                    format!("{root}.ffn_down_shexp.weight"),
                    vec![hidden, shared],
                    TensorOperation::Matrix,
                ),
                gguf_tensor(
                    format!("{root}.ffn_gate_exps.weight"),
                    vec![experts, intermediate, hidden],
                    TensorOperation::Matrix,
                ),
                gguf_tensor(
                    format!("{root}.ffn_up_exps.weight"),
                    vec![experts, intermediate, hidden],
                    TensorOperation::Matrix,
                ),
                gguf_tensor(
                    format!("{root}.ffn_down_exps.weight"),
                    vec![experts, hidden, intermediate],
                    TensorOperation::Matrix,
                ),
            ]);
            let shared_gate = format!("{root}.ffn_gate_inp_shexp.weight");
            groups.push(AlternativeLayoutGroup {
                id: format!("{root} shared-expert gate rank"),
                required: true,
                variants: vec![
                    LayoutVariant {
                        id: "matrix".into(),
                        tensors: vec![gguf_tensor(
                            shared_gate.clone(),
                            vec![1, hidden],
                            TensorOperation::Matrix,
                        )],
                        discriminator_keys: vec![shared_gate.clone()],
                    },
                    LayoutVariant {
                        id: "vector".into(),
                        tensors: vec![gguf_tensor(
                            shared_gate.clone(),
                            vec![hidden],
                            TensorOperation::Vector,
                        )],
                        discriminator_keys: vec![shared_gate],
                    },
                ],
            });
        } else {
            let intermediate = dim(config.intermediate_size, "intermediate_size")?;
            tensors.extend([
                gguf_tensor(
                    format!("{root}.ffn_gate.weight"),
                    vec![intermediate, hidden],
                    TensorOperation::Matrix,
                ),
                gguf_tensor(
                    format!("{root}.ffn_up.weight"),
                    vec![intermediate, hidden],
                    TensorOperation::Matrix,
                ),
                gguf_tensor(
                    format!("{root}.ffn_down.weight"),
                    vec![hidden, intermediate],
                    TensorOperation::Matrix,
                ),
            ]);
        }
    }
    let mut catalog = CatalogPolicy::non_strict();
    catalog.allowed_prefixes.push("rope_freqs.".into());
    GgufCheckpointPlan::new(
        format!("{} GGUF", config.model_type),
        tensors,
        groups,
        catalog,
    )
    .map_err(|error| error.to_string())
}

fn gguf_tensor(
    key: impl Into<String>,
    shape: Vec<usize>,
    operation: TensorOperation,
) -> GgufTensorConstraint {
    GgufTensorConstraint::required(key, shape, GgufTypeConstraint::OperationClass(operation))
}

/// Translates llama.cpp Qwen hybrid tensor identities to canonical parameters.
pub fn translate_gguf_weight_name(name: &str) -> String {
    for (source, target) in [
        ("token_embd", "model.embed_tokens"),
        ("output_norm", "model.norm"),
        ("output", "lm_head"),
    ] {
        if name == source || name.starts_with(&format!("{source}.")) {
            return name.replacen(source, target, 1);
        }
    }
    let Some((layer, parameter)) = name
        .strip_prefix("blk.")
        .and_then(|rest| rest.split_once('.'))
    else {
        return name.to_owned();
    };
    for (source, target) in [
        ("attn_norm", "input_layernorm"),
        ("post_attention_norm", "post_attention_layernorm"),
        ("attn_q_norm", "self_attn.q_norm"),
        ("attn_k_norm", "self_attn.k_norm"),
        ("attn_q", "self_attn.q_proj"),
        ("attn_k", "self_attn.k_proj"),
        ("attn_v", "self_attn.v_proj"),
        ("attn_output", "self_attn.o_proj"),
        ("attn_qkvz", "linear_attn.in_proj_qkvz"),
        ("attn_qkv", "linear_attn.in_proj_qkv"),
        ("attn_gate", "linear_attn.in_proj_z"),
        ("ssm_beta", "linear_attn.in_proj_b"),
        ("ssm_alpha", "linear_attn.in_proj_a"),
        ("ssm_ba", "linear_attn.in_proj_ba"),
        ("ssm_conv1d", "linear_attn.conv1d"),
        ("ssm_dt.bias", "linear_attn.dt_bias"),
        ("ssm_a", "linear_attn.A_log"),
        ("ssm_norm", "linear_attn.norm"),
        ("ssm_out", "linear_attn.out_proj"),
        ("ffn_gate_inp_shexp", "mlp.shared_expert_gate"),
        ("ffn_gate_inp", "mlp.gate"),
        ("ffn_gate_shexp", "mlp.shared_expert.gate_proj"),
        ("ffn_up_shexp", "mlp.shared_expert.up_proj"),
        ("ffn_down_shexp", "mlp.shared_expert.down_proj"),
        ("ffn_gate_exps", "mlp.experts.gate_proj"),
        ("ffn_up_exps", "mlp.experts.up_proj"),
        ("ffn_down_exps", "mlp.experts.down_proj"),
        ("ffn_gate", "mlp.gate_proj"),
        ("ffn_up", "mlp.up_proj"),
        ("ffn_down", "mlp.down_proj"),
        ("rope_freqs", "rope_freqs"),
    ] {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            let mut suffix = parameter.strip_prefix(source).unwrap_or_default();
            if target.starts_with("mlp.experts.") {
                suffix = match suffix {
                    ".weight" => "",
                    ".scales" => "_scales",
                    ".biases" => "_biases",
                    other => other,
                };
            }
            return format!("model.layers.{layer}.{target}{suffix}");
        }
    }
    name.to_owned()
}

fn add_block(
    config: &HybridConfig,
    layer: usize,
    policy: HybridLayerPolicy,
    common: &mut Vec<SafetensorsTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
) -> Result<(), String> {
    add_block_at(
        config,
        layer,
        &format!("model.layers.{layer}"),
        policy,
        common,
        groups,
    )
}

fn add_block_at(
    config: &HybridConfig,
    _layer: usize,
    root: &str,
    policy: HybridLayerPolicy,
    common: &mut Vec<SafetensorsTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
) -> Result<(), String> {
    let hidden = dim(config.hidden_size, "hidden_size")?;
    vector(common, format!("{root}.input_layernorm.weight"), hidden);
    vector(
        common,
        format!("{root}.post_attention_layernorm.weight"),
        hidden,
    );
    match policy {
        HybridLayerPolicy::SelfAttention(_) => add_attention(config, root, common)?,
        HybridLayerPolicy::LinearAttention => add_linear_attention(config, root, common, groups)?,
    }
    add_feed_forward(config, root, common, groups)
}

fn add_attention(
    config: &HybridConfig,
    root: &str,
    common: &mut Vec<SafetensorsTensorConstraint>,
) -> Result<(), String> {
    let hidden = dim(config.hidden_size, "hidden_size")?;
    let head = dim(config.head_dim, "head_dim")?;
    let query = mul(
        dim(config.num_attention_heads, "num_attention_heads")?,
        head,
    )?;
    let kv = mul(
        dim(config.num_key_value_heads, "num_key_value_heads")?,
        head,
    )?;
    let prefix = format!("{root}.self_attn");
    for (field, shape) in [
        ("q_proj", vec![mul(2, query)?, hidden]),
        ("k_proj", vec![kv, hidden]),
        ("v_proj", vec![kv, hidden]),
        ("o_proj", vec![hidden, query]),
    ] {
        matrix(config, common, format!("{prefix}.{field}.weight"), shape)?;
    }
    vector(common, format!("{prefix}.q_norm.weight"), head);
    vector(common, format!("{prefix}.k_norm.weight"), head);
    if config.attention_bias {
        for (field, width) in [
            ("q_proj", mul(2, query)?),
            ("k_proj", kv),
            ("v_proj", kv),
            ("o_proj", hidden),
        ] {
            vector(common, format!("{prefix}.{field}.bias"), width);
        }
    }
    Ok(())
}

fn add_linear_attention(
    config: &HybridConfig,
    root: &str,
    common: &mut Vec<SafetensorsTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
) -> Result<(), String> {
    let hidden = dim(config.hidden_size, "hidden_size")?;
    let key = mul(
        dim(config.linear_num_key_heads, "linear_num_key_heads")?,
        dim(config.linear_key_head_dim, "linear_key_head_dim")?,
    )?;
    let value = mul(
        dim(config.linear_num_value_heads, "linear_num_value_heads")?,
        dim(config.linear_value_head_dim, "linear_value_head_dim")?,
    )?;
    let heads = dim(config.linear_num_value_heads, "linear_num_value_heads")?;
    let prefix = format!("{root}.linear_attn");
    let split = layout(
        config,
        "split",
        [
            (
                format!("{prefix}.in_proj_qkv.weight"),
                vec![add(mul(2, key)?, value)?, hidden],
            ),
            (format!("{prefix}.in_proj_z.weight"), vec![value, hidden]),
            (format!("{prefix}.in_proj_b.weight"), vec![heads, hidden]),
            (format!("{prefix}.in_proj_a.weight"), vec![heads, hidden]),
        ],
    )?;
    let mut variants = vec![split];
    if config.variant == HybridVariant::Qwen3Next {
        variants.push(layout(
            config,
            "fused",
            [
                (
                    format!("{prefix}.in_proj_qkvz.weight"),
                    vec![add(mul(2, key)?, mul(2, value)?)?, hidden],
                ),
                (
                    format!("{prefix}.in_proj_ba.weight"),
                    vec![mul(2, heads)?, hidden],
                ),
            ],
        )?);
    }
    groups.push(AlternativeLayoutGroup {
        id: format!("{root} linear-attention inputs"),
        required: true,
        variants,
    });
    common.push(SafetensorsTensorConstraint::required(
        format!("{prefix}.conv1d.weight"),
        vec![
            add(mul(2, key)?, value)?,
            1,
            dim(config.linear_conv_kernel_dim, "linear_conv_kernel_dim")?,
        ],
        StoredDtypeConstraint::Floating,
    ));
    vector(common, format!("{prefix}.dt_bias"), heads);
    vector(common, format!("{prefix}.A_log"), heads);
    vector(
        common,
        format!("{prefix}.norm.weight"),
        dim(config.linear_value_head_dim, "linear_value_head_dim")?,
    );
    matrix(
        config,
        common,
        format!("{prefix}.out_proj.weight"),
        vec![hidden, value],
    )
}

fn add_feed_forward(
    config: &HybridConfig,
    root: &str,
    common: &mut Vec<SafetensorsTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
) -> Result<(), String> {
    let hidden = dim(config.hidden_size, "hidden_size")?;
    let prefix = format!("{root}.mlp");
    if !config.is_moe() {
        let intermediate = dim(config.intermediate_size, "intermediate_size")?;
        for (field, shape) in [
            ("gate_proj", vec![intermediate, hidden]),
            ("up_proj", vec![intermediate, hidden]),
            ("down_proj", vec![hidden, intermediate]),
        ] {
            matrix(config, common, format!("{prefix}.{field}.weight"), shape)?;
        }
        return Ok(());
    }
    let experts = dim(config.num_experts, "num_experts")?;
    let intermediate = dim(config.moe_intermediate_size, "moe_intermediate_size")?;
    let shared = dim(
        config.shared_expert_intermediate_size,
        "shared_expert_intermediate_size",
    )?;
    for (field, shape) in [
        ("gate", vec![experts, hidden]),
        ("shared_expert.gate_proj", vec![shared, hidden]),
        ("shared_expert.up_proj", vec![shared, hidden]),
        ("shared_expert.down_proj", vec![hidden, shared]),
        ("shared_expert_gate", vec![1, hidden]),
    ] {
        matrix(config, common, format!("{prefix}.{field}.weight"), shape)?;
    }
    let expert_prefix = format!("{prefix}.experts");
    let packed = layout(
        config,
        "packed",
        [
            (
                format!("{expert_prefix}.gate_up_proj"),
                vec![experts, mul(2, intermediate)?, hidden],
            ),
            (
                format!("{expert_prefix}.down_proj"),
                vec![experts, hidden, intermediate],
            ),
        ],
    )?;
    let mut independent = Vec::with_capacity(mul(experts, 3)?);
    for expert in 0..experts {
        independent.extend([
            (
                format!("{expert_prefix}.{expert}.gate_proj.weight"),
                vec![intermediate, hidden],
            ),
            (
                format!("{expert_prefix}.{expert}.up_proj.weight"),
                vec![intermediate, hidden],
            ),
            (
                format!("{expert_prefix}.{expert}.down_proj.weight"),
                vec![hidden, intermediate],
            ),
        ]);
    }
    groups.push(AlternativeLayoutGroup {
        id: format!("{expert_prefix} storage"),
        required: true,
        variants: vec![packed, layout(config, "independent", independent)?],
    });
    Ok(())
}

/// Resolves a Qwen3-Next fused QKVZ/BA family into canonical split parameters.
///
/// Group-major physical rows are gathered into component-major canonical rows.
/// Weight, affine companions, and FP8 inverse scales become visible atomically.
pub fn qwen3_next_fused_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    config: &HybridConfig,
    layer: usize,
) -> Result<AtomicRecipeSet, String> {
    if config.variant != HybridVariant::Qwen3Next {
        return Err("fused hybrid recipes require Qwen3-Next policy".into());
    }
    let prefix = format!("model.layers.{layer}.linear_attn");
    let (widths, ba_width) = fused_projection_widths(config).map_err(|error| error.to_string())?;
    let groups = dim(config.linear_num_key_heads, "linear_num_key_heads")?;
    let mut outputs = Vec::new();
    for suffix in ["weight", "scales", "biases"] {
        let qkvz = format!("{prefix}.in_proj_qkvz.{suffix}");
        if catalog.tensor_metadata(&qkvz).is_ok() {
            add_grouped_outputs(
                catalog,
                &mut outputs,
                &qkvz,
                groups,
                &widths.map(|width| usize::try_from(width).unwrap()),
                [
                    (format!("{prefix}.in_proj_qkv.{suffix}"), vec![0, 1, 2]),
                    (format!("{prefix}.in_proj_z.{suffix}"), vec![3]),
                ],
            )?;
        }
        let ba = format!("{prefix}.in_proj_ba.{suffix}");
        if catalog.tensor_metadata(&ba).is_ok() {
            let width = usize::try_from(ba_width).map_err(|_| "invalid BA width")?;
            add_grouped_outputs(
                catalog,
                &mut outputs,
                &ba,
                groups,
                &[width, width],
                [
                    (format!("{prefix}.in_proj_b.{suffix}"), vec![0]),
                    (format!("{prefix}.in_proj_a.{suffix}"), vec![1]),
                ],
            )?;
        }
    }
    let scale = format!("{prefix}.in_proj_qkvz.weight_scale_inv");
    if catalog.tensor_metadata(&scale).is_ok() {
        let block_widths = fp8_block_row_widths(&widths)
            .map_err(|error| error.to_string())?
            .into_iter()
            .map(|width| usize::try_from(width).map_err(|_| "invalid FP8 block width"))
            .collect::<Result<Vec<_>, _>>()?;
        add_grouped_outputs(
            catalog,
            &mut outputs,
            &scale,
            groups,
            &block_widths,
            [
                (
                    format!("{prefix}.in_proj_qkv.weight_scale_inv"),
                    vec![0, 1, 2],
                ),
                (format!("{prefix}.in_proj_z.weight_scale_inv"), vec![3]),
            ],
        )?;
    }
    if catalog
        .tensor_metadata(&format!("{prefix}.in_proj_ba.weight_scale_inv"))
        .is_ok()
    {
        return Err("Qwen3-Next fused BA must remain dense and cannot carry inverse scales".into());
    }
    if outputs.is_empty() {
        return Err(format!("no fused projections found for layer {layer}"));
    }
    AtomicRecipeSet::new(catalog, outputs).map_err(|error| error.to_string())
}

fn add_grouped_outputs<C, const N: usize>(
    catalog: &C,
    outputs: &mut Vec<(String, DerivedWeightRecipe)>,
    source: &str,
    groups: usize,
    widths: &[usize],
    targets: [(String, Vec<usize>); N],
) -> Result<(), String>
where
    C: RecipeCatalog + ?Sized,
{
    let group_width = widths.iter().try_fold(0usize, |sum, width| {
        sum.checked_add(*width).ok_or("grouped row width overflow")
    })?;
    for (target, components) in targets {
        if catalog.tensor_metadata(&target).is_ok() {
            return Err(format!(
                "fused output {target:?} collides with a physical tensor"
            ));
        }
        let mut indices = Vec::new();
        for component in components {
            let start = widths
                .get(..component)
                .ok_or("invalid grouped component")?
                .iter()
                .sum::<usize>();
            let width = *widths.get(component).ok_or("invalid grouped component")?;
            for group in 0..groups {
                let base = group
                    .checked_mul(group_width)
                    .and_then(|base| base.checked_add(start))
                    .ok_or("grouped row index overflow")?;
                indices.extend(base..base + width);
            }
        }
        outputs.push((
            target,
            DerivedWeightRecipe::source(source, TensorSelection::Indices { axis: 0, indices }),
        ));
    }
    Ok(())
}

fn layout(
    config: &HybridConfig,
    id: &str,
    tensors: impl IntoIterator<Item = (String, Vec<usize>)>,
) -> Result<LayoutVariant<SafetensorsTensorConstraint>, String> {
    let tensors = tensors.into_iter().collect::<Vec<_>>();
    let discriminator_keys = tensors.iter().map(|(name, _)| name.clone()).collect();
    let mut constraints = Vec::new();
    for (name, shape) in tensors {
        matrix(config, &mut constraints, name, shape)?;
    }
    Ok(LayoutVariant {
        id: id.into(),
        discriminator_keys,
        tensors: constraints,
    })
}

fn matrix(
    config: &HybridConfig,
    output: &mut Vec<SafetensorsTensorConstraint>,
    name: impl Into<String>,
    shape: Vec<usize>,
) -> Result<(), String> {
    let name = name.into();
    let aliases = aliases(&name);
    let format = config.linear_format(&name);
    let scale = matches!(format, eredu_checkpoint::LinearFormat::E4M3BlockFp8(_)).then(|| {
        MatrixScaleNames {
            key: format!("{name}_scale_inv"),
            aliases: aliases
                .iter()
                .map(|name| format!("{name}_scale_inv"))
                .collect(),
        }
    });
    output.extend(
        matrix_for_linear_format(&name, aliases, shape, format, scale)
            .map_err(|error| error.to_string())?,
    );
    Ok(())
}

fn vector(output: &mut Vec<SafetensorsTensorConstraint>, name: impl Into<String>, width: usize) {
    let name = name.into();
    output.push(
        SafetensorsTensorConstraint::required(&name, vec![width], StoredDtypeConstraint::Floating)
            .with_aliases(aliases(&name)),
    );
}

fn aliases(name: &str) -> Vec<String> {
    name.strip_prefix("model.")
        .map(|rest| {
            vec![
                format!("model.language_model.{rest}"),
                format!("language_model.{rest}"),
                format!("model.model.{rest}"),
            ]
        })
        .unwrap_or_default()
}

fn dim(value: i32, name: &str) -> Result<usize, String> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("{name} must be positive"))
}

fn mul(left: usize, right: usize) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| "checkpoint dimension overflow".into())
}

fn add(left: usize, right: usize) -> Result<usize, String> {
    left.checked_add(right)
        .ok_or_else(|| "checkpoint dimension overflow".into())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use eredu_checkpoint::{
        recipe::RecipeCatalog,
        store::{StoreError, TensorMetadata},
        StoredDtype,
    };
    use serde_json::json;

    use super::*;
    use crate::qwen::hybrid::model_args_from_config_value;

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
            encoded_byte_len: shape.iter().product::<usize>() as u64 * 2,
            logical_shape: shape.clone(),
            physical_shape: shape,
            stored_dtype: StoredDtype::F16,
            backing_shard: None,
        }
    }

    fn config() -> HybridConfig {
        model_args_from_config_value(&json!({
            "model_type": "qwen3_next",
            "vocab_size": 64,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 128,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "intermediate_size": 48,
            "layer_types": ["linear_attention", "full_attention"]
        }))
        .unwrap()
        .text
    }

    #[test]
    fn plan_freezes_fused_or_split_recurrent_and_gated_attention_shapes() {
        let plan = safetensors_plan(&config()).unwrap();
        let recurrent = plan
            .layout_groups
            .iter()
            .find(|group| group.id == "model.layers.0 linear-attention inputs")
            .unwrap();
        assert_eq!(recurrent.variants.len(), 2);
        assert!(recurrent.variants.iter().any(|variant| {
            variant
                .discriminator_keys
                .contains(&"model.layers.0.linear_attn.in_proj_qkvz.weight".into())
                && variant
                    .discriminator_keys
                    .contains(&"model.layers.0.linear_attn.in_proj_ba.weight".into())
        }));
        let query = plan
            .common_tensors
            .iter()
            .find(|tensor| tensor.key == "model.layers.1.self_attn.q_proj.weight")
            .unwrap();
        assert_eq!(query.shape, [64, 32]);
    }

    #[test]
    fn grouped_fused_recipes_restore_component_major_rows_atomically() {
        let config = config();
        let prefix = "model.layers.0.linear_attn";
        let mut tensors = BTreeMap::new();
        for (name, shape) in [
            (format!("{prefix}.in_proj_qkvz.weight"), vec![96, 32]),
            (format!("{prefix}.in_proj_ba.weight"), vec![8, 32]),
        ] {
            tensors.insert(name.clone(), metadata(&name, shape));
        }
        let catalog = Catalog(tensors);
        let recipes = qwen3_next_fused_recipes(&catalog, &config, 0).unwrap();
        assert_eq!(recipes.iter().count(), 4);
        assert_eq!(
            recipes
                .get(&format!("{prefix}.in_proj_qkv.weight"))
                .unwrap()
                .infer(&catalog)
                .unwrap()
                .shape(),
            &[64, 32]
        );
        assert_eq!(
            recipes
                .get(&format!("{prefix}.in_proj_z.weight"))
                .unwrap()
                .infer(&catalog)
                .unwrap()
                .shape(),
            &[32, 32]
        );
    }

    #[test]
    fn fused_recipes_reject_physical_split_collision() {
        let config = config();
        let prefix = "model.layers.0.linear_attn";
        let mut tensors = BTreeMap::new();
        for (name, shape) in [
            (format!("{prefix}.in_proj_qkvz.weight"), vec![96, 32]),
            (format!("{prefix}.in_proj_qkv.weight"), vec![64, 32]),
        ] {
            tensors.insert(name.clone(), metadata(&name, shape));
        }
        assert!(qwen3_next_fused_recipes(&Catalog(tensors), &config, 0).is_err());
    }
}
