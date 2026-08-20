//! Pure Kimi Linear SafeTensors/GGUF checkpoint plans.

use eredu_checkpoint::schema::{
    AlternativeLayoutGroup, CatalogPolicy, DepthwiseConvolutionSchema, GgufCheckpointPlan,
    GgufTensorConstraint, GgufTypeConstraint, LayoutVariant, SafetensorsCheckpointPlan,
    SafetensorsTensorConstraint, StoredDtypeConstraint, TensorOperation,
};
use eredu_checkpoint::{StoredDtype, WeightQuantization};

use super::{AttentionKind, FeedForwardPolicy, ModelArgs};

/// Builds the strict SafeTensors catalog plan.
pub fn safetensors_plan(args: &ModelArgs) -> Result<SafetensorsCheckpointPlan, String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let vocab = dimension(args.vocab_size, "vocabulary size")?;
    let mut common = Vec::new();
    let mut groups = Vec::new();
    add_safe_matrix(
        args,
        &mut common,
        "model.embed_tokens.weight",
        vec![vocab, hidden],
    )?;
    common.push(safe_aliases("model.norm.weight", vec![hidden]));
    if !args.tie_word_embeddings {
        add_safe_matrix(args, &mut common, "lm_head.weight", vec![vocab, hidden])?;
    }

    for (layer, policy) in args.layer_schedule.iter().enumerate() {
        let root = format!("model.layers.{layer}");
        common.extend([
            safe_aliases(&format!("{root}.input_layernorm.weight"), vec![hidden]),
            safe_aliases(
                &format!("{root}.post_attention_layernorm.weight"),
                vec![hidden],
            ),
        ]);
        match policy.attention {
            AttentionKind::Kda => add_safe_kda(args, &root, &mut common)?,
            AttentionKind::Mla => add_safe_mla(args, &root, &mut common)?,
        }
        match policy.feed_forward {
            FeedForwardPolicy::Dense => {
                let intermediate = dimension(args.intermediate_size, "dense intermediate size")?;
                for (local, shape) in [
                    ("gate_proj.weight", vec![intermediate, hidden]),
                    ("up_proj.weight", vec![intermediate, hidden]),
                    ("down_proj.weight", vec![hidden, intermediate]),
                ] {
                    add_safe_matrix(args, &mut common, &format!("{root}.mlp.{local}"), shape)?;
                }
            }
            FeedForwardPolicy::SparseMoe => {
                add_safe_moe(args, layer, &root, &mut common, &mut groups)?;
            }
        }
    }
    let mut policy = CatalogPolicy::strict();
    policy.allowed_prefixes.push("model.mtp.".into());
    SafetensorsCheckpointPlan::new("Kimi Linear SafeTensors", common, groups, policy)
        .map_err(|error| error.to_string())
}

fn add_safe_kda(
    args: &ModelArgs,
    root: &str,
    output: &mut Vec<SafetensorsTensorConstraint>,
) -> Result<(), String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let heads = dimension(args.kda_config.num_heads, "KDA head count")?;
    let head = dimension(args.kda_config.head_dim, "KDA head width")?;
    let kernel = dimension(
        args.kda_config.short_conv_kernel_size,
        "KDA convolution width",
    )?;
    let projection = checked_mul(heads, head, "KDA projection width")?;
    let prefix = format!("{root}.self_attn");
    for (local, shape) in [
        ("q_proj.weight", vec![projection, hidden]),
        ("k_proj.weight", vec![projection, hidden]),
        ("v_proj.weight", vec![projection, hidden]),
        ("f_a_proj.weight", vec![head, hidden]),
        ("f_b_proj.weight", vec![projection, head]),
        ("b_proj.weight", vec![heads, hidden]),
        ("g_a_proj.weight", vec![head, hidden]),
        ("g_b_proj.weight", vec![projection, head]),
        ("o_proj.weight", vec![hidden, projection]),
    ] {
        add_safe_matrix(args, output, &format!("{prefix}.{local}"), shape)?;
    }
    output.extend([
        safe_aliases(&format!("{prefix}.dt_bias"), vec![projection]),
        safe_aliases(&format!("{prefix}.o_norm.weight"), vec![head]),
    ]);
    let convolution_elements = checked_mul(projection, kernel, "KDA convolution elements")?;
    let convolution = DepthwiseConvolutionSchema::new(projection, kernel, false)?;
    for name in ["q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight"] {
        output.push(
            safe_aliases(&format!("{prefix}.{name}"), convolution.storage_shape())
                .with_element_count(convolution_elements),
        );
    }
    output.push(
        safe_aliases(&format!("{prefix}.A_log"), vec![1, 1, heads, 1]).with_element_count(heads),
    );
    Ok(())
}

fn add_safe_mla(
    args: &ModelArgs,
    root: &str,
    output: &mut Vec<SafetensorsTensorConstraint>,
) -> Result<(), String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let heads = dimension(args.num_attention_heads, "attention head count")?;
    let nope = dimension(args.qk_nope_head_dim, "MLA no-PE width")?;
    let rope = dimension(args.qk_rope_head_dim, "MLA RoPE width")?;
    let value = dimension(args.v_head_dim, "MLA value width")?;
    let kv_rank = dimension(args.kv_lora_rank, "KV LoRA rank")?;
    let query_head = checked_add(nope, rope, "MLA query head width")?;
    let prefix = format!("{root}.self_attn");
    if let Some(rank) = args.q_lora_rank {
        let rank = dimension(rank, "query LoRA rank")?;
        add_safe_matrix(
            args,
            output,
            &format!("{prefix}.q_a_proj.weight"),
            vec![rank, hidden],
        )?;
        output.push(safe_aliases(
            &format!("{prefix}.q_a_layernorm.weight"),
            vec![rank],
        ));
        add_safe_matrix(
            args,
            output,
            &format!("{prefix}.q_b_proj.weight"),
            vec![
                checked_mul(heads, query_head, "query projection width")?,
                rank,
            ],
        )?;
    } else {
        add_safe_matrix(
            args,
            output,
            &format!("{prefix}.q_proj.weight"),
            vec![
                checked_mul(heads, query_head, "query projection width")?,
                hidden,
            ],
        )?;
    }
    add_safe_matrix(
        args,
        output,
        &format!("{prefix}.kv_a_proj_with_mqa.weight"),
        vec![checked_add(kv_rank, rope, "KV-A width")?, hidden],
    )?;
    output.push(safe_aliases(
        &format!("{prefix}.kv_a_layernorm.weight"),
        vec![kv_rank],
    ));
    if args.split_kv_b {
        add_safe_matrix(
            args,
            output,
            &format!("{prefix}.k_b_proj.weight"),
            vec![checked_mul(heads, nope, "K-B width")?, kv_rank],
        )?;
        add_safe_matrix(
            args,
            output,
            &format!("{prefix}.v_b_proj.weight"),
            vec![checked_mul(heads, value, "V-B width")?, kv_rank],
        )?;
    } else {
        add_safe_matrix(
            args,
            output,
            &format!("{prefix}.kv_b_proj.weight"),
            vec![
                checked_mul(
                    heads,
                    checked_add(nope, value, "KV-B head width")?,
                    "KV-B width",
                )?,
                kv_rank,
            ],
        )?;
    }
    add_safe_matrix(
        args,
        output,
        &format!("{prefix}.o_proj.weight"),
        vec![hidden, checked_mul(heads, value, "attention output width")?],
    )
}

fn add_safe_moe(
    args: &ModelArgs,
    layer: usize,
    root: &str,
    common: &mut Vec<SafetensorsTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
) -> Result<(), String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let experts = dimension(args.num_experts, "expert count")?;
    let intermediate = dimension(args.moe_intermediate_size, "MoE intermediate size")?;
    for (local, shape, matrix) in [
        ("gate.weight", vec![experts, hidden], true),
        ("gate.e_score_correction_bias", vec![experts], false),
        (
            "shared_experts.gate_proj.weight",
            vec![intermediate, hidden],
            true,
        ),
        (
            "shared_experts.up_proj.weight",
            vec![intermediate, hidden],
            true,
        ),
        (
            "shared_experts.down_proj.weight",
            vec![hidden, intermediate],
            true,
        ),
    ] {
        let name = format!("{root}.mlp.{local}");
        if matrix {
            add_safe_matrix(args, common, &name, shape)?;
        } else {
            common.push(safe_aliases(&name, shape));
        }
    }

    let prefix = format!("{root}.mlp.experts");
    let gate_up = format!("{prefix}.gate_up_proj");
    let down = format!("{prefix}.down_proj");
    let mut packed_tensors = safe_matrix_constraints(
        &gate_up,
        vec![
            experts,
            checked_mul(2, intermediate, "fused expert width")?,
            hidden,
        ],
        args.weight_quantization_for(&gate_up),
    )?;
    packed_tensors.extend(safe_matrix_constraints(
        &down,
        vec![experts, hidden, intermediate],
        args.weight_quantization_for(&down),
    )?);
    let mut split_tensors = Vec::new();
    let mut split_discriminators = Vec::new();
    for expert in 0..experts {
        for (projection, shape) in [
            ("w1", vec![intermediate, hidden]),
            ("w2", vec![hidden, intermediate]),
            ("w3", vec![intermediate, hidden]),
        ] {
            let name = format!("{prefix}.{expert}.{projection}.weight");
            split_discriminators.push(name.clone());
            split_tensors.push(safe_aliases(&name, shape));
        }
    }
    groups.push(AlternativeLayoutGroup {
        id: format!("Kimi Linear layer {layer} expert storage"),
        required: true,
        variants: vec![
            LayoutVariant {
                id: "packed".into(),
                tensors: packed_tensors,
                discriminator_keys: vec![gate_up, down],
            },
            LayoutVariant {
                id: "split".into(),
                tensors: split_tensors,
                discriminator_keys: split_discriminators,
            },
        ],
    });
    Ok(())
}

fn add_safe_matrix(
    args: &ModelArgs,
    output: &mut Vec<SafetensorsTensorConstraint>,
    name: &str,
    shape: Vec<usize>,
) -> Result<(), String> {
    output.extend(safe_matrix_constraints(
        name,
        shape,
        args.weight_quantization_for(name),
    )?);
    Ok(())
}

fn safe_matrix_constraints(
    name: &str,
    shape: Vec<usize>,
    quantization: Option<WeightQuantization>,
) -> Result<Vec<SafetensorsTensorConstraint>, String> {
    let Some(quantization) = quantization else {
        return Ok(vec![safe_aliases(name, shape)]);
    };
    let input = *shape
        .last()
        .ok_or_else(|| format!("Kimi Linear matrix {name:?} has scalar shape"))?;
    let bits = dimension(quantization.bits(), "affine bit width")?;
    let group = dimension(quantization.group_size(), "affine group size")?;
    let packed_bits = checked_mul(input, bits, "affine packing")?;
    if !input.is_multiple_of(group) || !input.is_multiple_of(32) || !packed_bits.is_multiple_of(32)
    {
        return Err(format!(
            "Kimi Linear quantized tensor {name:?} input dimension {input} is incompatible with group size {group} and {bits}-bit packing"
        ));
    }
    let mut packed = shape.clone();
    *packed.last_mut().expect("matrix shape") = packed_bits / 32;
    let mut companion_shape = shape;
    *companion_shape.last_mut().expect("matrix shape") = input / group;
    let canonical_prefix = outer_prefix(name);
    let aliases_for = |suffix: &str| {
        let canonical = format!("{canonical_prefix}.{suffix}");
        physical_names(name)
            .into_iter()
            .map(|candidate| format!("{}.{suffix}", outer_prefix(&candidate)))
            .filter(|candidate| candidate != &canonical)
            .collect::<Vec<_>>()
    };
    let companion_dtype = || {
        StoredDtypeConstraint::OneOf(vec![
            StoredDtype::F16,
            StoredDtype::BF16,
            StoredDtype::F32,
            StoredDtype::U8,
        ])
    };
    let mut constraints = vec![safe_aliases_with_dtype(
        name,
        packed,
        StoredDtypeConstraint::Exact(StoredDtype::U32),
    )];
    constraints.push(
        SafetensorsTensorConstraint::required(
            format!("{canonical_prefix}.scales"),
            companion_shape.clone(),
            companion_dtype(),
        )
        .with_aliases(aliases_for("scales"))
        .companion(),
    );
    if quantization.has_biases() {
        constraints.push(
            SafetensorsTensorConstraint::required(
                format!("{canonical_prefix}.biases"),
                companion_shape,
                companion_dtype(),
            )
            .with_aliases(aliases_for("biases"))
            .companion(),
        );
    }
    Ok(constraints)
}

fn safe_aliases(key: &str, shape: Vec<usize>) -> SafetensorsTensorConstraint {
    safe_aliases_with_dtype(key, shape, StoredDtypeConstraint::Floating)
}

fn safe_aliases_with_dtype(
    key: &str,
    shape: Vec<usize>,
    dtype: StoredDtypeConstraint,
) -> SafetensorsTensorConstraint {
    SafetensorsTensorConstraint::required(key, shape, dtype).with_aliases(
        physical_names(key)
            .into_iter()
            .filter(|candidate| candidate != key),
    )
}

fn physical_names(canonical: &str) -> Vec<String> {
    let mut names = vec![canonical.to_string()];
    if canonical.contains(".mlp.") {
        names.push(canonical.replace(".mlp.", ".block_sparse_moe."));
    }
    for name in names.clone() {
        if let Some(prefix) = name.strip_suffix(".weight") {
            names.push(format!("{prefix}.inner.weight"));
        }
    }
    names.sort();
    names.dedup();
    names
}

fn outer_prefix(name: &str) -> &str {
    name.strip_suffix(".inner.weight")
        .or_else(|| name.strip_suffix(".weight"))
        .unwrap_or(name)
}

/// Builds the GGUF physical catalog plan.
pub fn gguf_plan(args: &ModelArgs) -> Result<GgufCheckpointPlan, String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let vocab = dimension(args.vocab_size, "vocabulary size")?;
    let mut tensors = vec![
        gguf(
            "token_embd.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
        ),
        gguf("output_norm.weight", vec![hidden], TensorOperation::Vector),
    ];
    if !args.tie_word_embeddings {
        tensors.push(gguf(
            "output.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
        ));
    }
    for (layer, policy) in args.layer_schedule.iter().enumerate() {
        let block = format!("blk.{layer}");
        tensors.extend([
            gguf(
                format!("{block}.attn_norm.weight"),
                vec![hidden],
                TensorOperation::Vector,
            ),
            gguf(
                format!("{block}.ffn_norm.weight"),
                vec![hidden],
                TensorOperation::Vector,
            ),
        ]);
        match policy.attention {
            AttentionKind::Kda => add_gguf_kda(args, &block, &mut tensors)?,
            AttentionKind::Mla => add_gguf_mla(args, &block, &mut tensors)?,
        }
        match policy.feed_forward {
            FeedForwardPolicy::Dense => {
                let intermediate = dimension(args.intermediate_size, "dense intermediate size")?;
                for (local, shape) in [
                    ("ffn_gate.weight", vec![intermediate, hidden]),
                    ("ffn_up.weight", vec![intermediate, hidden]),
                    ("ffn_down.weight", vec![hidden, intermediate]),
                ] {
                    tensors.push(gguf(
                        format!("{block}.{local}"),
                        shape,
                        TensorOperation::Matrix,
                    ));
                }
            }
            FeedForwardPolicy::SparseMoe => {
                let experts = dimension(args.num_experts, "expert count")?;
                let intermediate = dimension(args.moe_intermediate_size, "MoE intermediate size")?;
                for (local, shape, operation) in [
                    (
                        "ffn_gate_inp.weight",
                        vec![experts, hidden],
                        TensorOperation::Matrix,
                    ),
                    ("exp_probs_b.bias", vec![experts], TensorOperation::Vector),
                    (
                        "ffn_gate_shexp.weight",
                        vec![intermediate, hidden],
                        TensorOperation::Matrix,
                    ),
                    (
                        "ffn_up_shexp.weight",
                        vec![intermediate, hidden],
                        TensorOperation::Matrix,
                    ),
                    (
                        "ffn_down_shexp.weight",
                        vec![hidden, intermediate],
                        TensorOperation::Matrix,
                    ),
                    (
                        "ffn_gate_exps.weight",
                        vec![experts, intermediate, hidden],
                        TensorOperation::Matrix,
                    ),
                    (
                        "ffn_up_exps.weight",
                        vec![experts, intermediate, hidden],
                        TensorOperation::Matrix,
                    ),
                    (
                        "ffn_down_exps.weight",
                        vec![experts, hidden, intermediate],
                        TensorOperation::Matrix,
                    ),
                ] {
                    tensors.push(gguf(format!("{block}.{local}"), shape, operation));
                }
            }
        }
    }
    GgufCheckpointPlan::new(
        "Kimi Linear GGUF",
        tensors,
        Vec::new(),
        CatalogPolicy::non_strict(),
    )
    .map_err(|error| error.to_string())
}

fn add_gguf_kda(
    args: &ModelArgs,
    block: &str,
    output: &mut Vec<GgufTensorConstraint>,
) -> Result<(), String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let heads = dimension(args.kda_config.num_heads, "KDA head count")?;
    let head = dimension(args.kda_config.head_dim, "KDA head width")?;
    let kernel = dimension(
        args.kda_config.short_conv_kernel_size,
        "KDA convolution width",
    )?;
    let projection = checked_mul(heads, head, "KDA projection width")?;
    for (local, shape, operation) in [
        (
            "attn_q.weight",
            vec![projection, hidden],
            TensorOperation::Matrix,
        ),
        (
            "attn_k.weight",
            vec![projection, hidden],
            TensorOperation::Matrix,
        ),
        (
            "attn_v.weight",
            vec![projection, hidden],
            TensorOperation::Matrix,
        ),
        (
            "ssm_f_a.weight",
            vec![head, hidden],
            TensorOperation::Matrix,
        ),
        (
            "ssm_f_b.weight",
            vec![projection, head],
            TensorOperation::Matrix,
        ),
        (
            "ssm_beta.weight",
            vec![heads, hidden],
            TensorOperation::Matrix,
        ),
        (
            "ssm_g_a.weight",
            vec![head, hidden],
            TensorOperation::Matrix,
        ),
        (
            "ssm_g_b.weight",
            vec![projection, head],
            TensorOperation::Matrix,
        ),
        ("ssm_dt.bias", vec![projection], TensorOperation::Vector),
        ("ssm_norm.weight", vec![head], TensorOperation::Vector),
        (
            "attn_output.weight",
            vec![hidden, projection],
            TensorOperation::Matrix,
        ),
    ] {
        output.push(gguf(format!("{block}.{local}"), shape, operation));
    }
    let convolution_elements = checked_mul(projection, kernel, "KDA convolution elements")?;
    for local in [
        "ssm_conv1d_q.weight",
        "ssm_conv1d_k.weight",
        "ssm_conv1d_v.weight",
    ] {
        output.push(
            gguf(
                format!("{block}.{local}"),
                vec![projection, 1, kernel],
                TensorOperation::Dense,
            )
            .with_element_count(convolution_elements),
        );
    }
    output.push(
        gguf(
            format!("{block}.ssm_a"),
            vec![heads],
            TensorOperation::Vector,
        )
        .with_aliases([format!("{block}.ssm_a.weight")])
        .with_element_count(heads),
    );
    Ok(())
}

fn add_gguf_mla(
    args: &ModelArgs,
    block: &str,
    output: &mut Vec<GgufTensorConstraint>,
) -> Result<(), String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let heads = dimension(args.num_attention_heads, "attention head count")?;
    let nope = dimension(args.qk_nope_head_dim, "MLA no-PE width")?;
    let rope = dimension(args.qk_rope_head_dim, "MLA RoPE width")?;
    let value = dimension(args.v_head_dim, "MLA value width")?;
    let kv_rank = dimension(args.kv_lora_rank, "KV LoRA rank")?;
    let query_head = checked_add(nope, rope, "MLA query head width")?;
    if let Some(rank) = args.q_lora_rank {
        let rank = dimension(rank, "query LoRA rank")?;
        for (local, shape, operation) in [
            (
                "attn_q_a.weight",
                vec![rank, hidden],
                TensorOperation::Matrix,
            ),
            ("attn_q_a_norm.weight", vec![rank], TensorOperation::Vector),
            (
                "attn_q_b.weight",
                vec![checked_mul(heads, query_head, "query width")?, rank],
                TensorOperation::Matrix,
            ),
        ] {
            output.push(gguf(format!("{block}.{local}"), shape, operation));
        }
    } else {
        output.push(gguf(
            format!("{block}.attn_q.weight"),
            vec![checked_mul(heads, query_head, "query width")?, hidden],
            TensorOperation::Matrix,
        ));
    }
    output.extend([
        gguf(
            format!("{block}.attn_kv_a_mqa.weight"),
            vec![checked_add(kv_rank, rope, "KV-A width")?, hidden],
            TensorOperation::Matrix,
        ),
        gguf(
            format!("{block}.attn_kv_a_norm.weight"),
            vec![kv_rank],
            TensorOperation::Vector,
        ),
    ]);
    if args.split_kv_b {
        output.extend([
            gguf(
                format!("{block}.attn_k_b.weight"),
                vec![checked_mul(heads, nope, "K-B width")?, kv_rank],
                TensorOperation::Matrix,
            ),
            gguf(
                format!("{block}.attn_v_b.weight"),
                vec![checked_mul(heads, value, "V-B width")?, kv_rank],
                TensorOperation::Matrix,
            ),
        ]);
    } else {
        output.push(gguf(
            format!("{block}.attn_kv_b.weight"),
            vec![
                checked_mul(
                    heads,
                    checked_add(nope, value, "KV-B head width")?,
                    "KV-B width",
                )?,
                kv_rank,
            ],
            TensorOperation::Matrix,
        ));
    }
    output.push(gguf(
        format!("{block}.attn_output.weight"),
        vec![hidden, checked_mul(heads, value, "attention output width")?],
        TensorOperation::Matrix,
    ));
    Ok(())
}

fn gguf(
    key: impl Into<String>,
    shape: Vec<usize>,
    operation: TensorOperation,
) -> GgufTensorConstraint {
    GgufTensorConstraint::required(key, shape, GgufTypeConstraint::OperationClass(operation))
}

fn dimension(value: i32, name: &str) -> Result<usize, String> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("Kimi Linear {name} must be positive, got {value}"))
}

fn checked_add(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_add(right)
        .ok_or_else(|| format!("Kimi Linear {name} geometry overflows"))
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("Kimi Linear {name} geometry overflows"))
}

/// Translates a physical GGUF tensor name to canonical Kimi parameter identity.
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
    let Some(rest) = name.strip_prefix("blk.") else {
        return name.to_owned();
    };
    let Some((layer, parameter)) = rest.split_once('.') else {
        return name.to_owned();
    };
    for (source, target) in [
        ("ffn_gate_exps", "mlp.experts.gate_proj"),
        ("ffn_up_exps", "mlp.experts.up_proj"),
        ("ffn_down_exps", "mlp.experts.down_proj"),
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
    if matches!(parameter, "exp_probs_b.bias" | "ffn_exp_probs_b.bias") {
        return format!("model.layers.{layer}.mlp.gate.e_score_correction_bias");
    }
    for (source, target) in [
        ("attn_q", "self_attn.q_proj"),
        ("attn_k", "self_attn.k_proj"),
        ("attn_v", "self_attn.v_proj"),
        ("attn_q_a", "self_attn.q_a_proj"),
        ("attn_q_b", "self_attn.q_b_proj"),
        ("attn_kv_a_mqa", "self_attn.kv_a_proj_with_mqa"),
        ("attn_kv_b", "self_attn.kv_b_proj"),
        ("attn_k_b", "self_attn.k_b_proj"),
        ("attn_v_b", "self_attn.v_b_proj"),
        ("attn_q_a_norm", "self_attn.q_a_layernorm"),
        ("attn_kv_a_norm", "self_attn.kv_a_layernorm"),
        ("attn_output", "self_attn.o_proj"),
        ("ssm_conv1d_q", "self_attn.q_conv1d"),
        ("ssm_conv1d_k", "self_attn.k_conv1d"),
        ("ssm_conv1d_v", "self_attn.v_conv1d"),
        ("ssm_f_a", "self_attn.f_a_proj"),
        ("ssm_f_b", "self_attn.f_b_proj"),
        ("ssm_beta", "self_attn.b_proj"),
        ("ssm_g_a", "self_attn.g_a_proj"),
        ("ssm_g_b", "self_attn.g_b_proj"),
        ("ssm_norm", "self_attn.o_norm"),
        ("attn_norm", "input_layernorm"),
        ("ffn_norm", "post_attention_layernorm"),
        ("ffn_gate", "mlp.gate_proj"),
        ("ffn_up", "mlp.up_proj"),
        ("ffn_down", "mlp.down_proj"),
        ("ffn_gate_shexp", "mlp.shared_experts.gate_proj"),
        ("ffn_up_shexp", "mlp.shared_experts.up_proj"),
        ("ffn_down_shexp", "mlp.shared_experts.down_proj"),
        ("ffn_gate_inp", "mlp.gate"),
    ] {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            return format!(
                "model.layers.{layer}.{}",
                parameter.replacen(source, target, 1)
            );
        }
    }
    if parameter == "ssm_a" || parameter.starts_with("ssm_a.") {
        let suffix = parameter.strip_prefix("ssm_a").unwrap_or_default();
        let suffix = if suffix == ".weight" { "" } else { suffix };
        return format!("model.layers.{layer}.self_attn.A_log{suffix}");
    }
    if parameter == "ssm_dt.bias" || parameter == "ssm_dt" {
        return format!("model.layers.{layer}.self_attn.dt_bias");
    }
    name.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(split_kv_b: bool, q_lora_rank: Option<i32>) -> ModelArgs {
        let mut value = serde_json::json!({
            "model_type":"kimi_linear", "vocab_size":16, "hidden_size":12,
            "num_hidden_layers":2, "num_attention_heads":3, "num_key_value_heads":3,
            "intermediate_size":17, "head_dim":4, "model_max_length":64,
            "linear_attn_config":{"kda_layers":[1],"full_attn_layers":[2],"num_heads":3,"head_dim":4,"short_conv_kernel_size":3},
            "num_experts":2, "moe_intermediate_size":9, "kv_lora_rank":6,
            "qk_nope_head_dim":4, "qk_rope_head_dim":2, "v_head_dim":4,
            "mla_use_nope":true, "num_experts_per_token":1, "num_shared_experts":1,
            "routed_scaling_factor":1.0, "first_k_dense_replace":1,
            "num_expert_group":1, "topk_group":1, "tie_word_embeddings":false,
            "split_kv_b": split_kv_b
        });
        if let Some(rank) = q_lora_rank {
            value["q_lora_rank"] = rank.into();
        }
        let mut args = super::super::config::model_args_from_config_value(&value).unwrap();
        args.split_kv_b = split_kv_b;
        args
    }

    #[test]
    fn plans_kda_mla_low_rank_and_alternative_expert_artifacts() {
        for (split, query_rank) in [(false, None), (true, Some(5))] {
            let plan = safetensors_plan(&fixture(split, query_rank)).unwrap();
            let names = plan
                .common_tensors
                .iter()
                .map(|tensor| tensor.key.as_str())
                .collect::<Vec<_>>();
            assert!(names.contains(&"model.layers.0.self_attn.q_conv1d.weight"));
            assert!(names.contains(&"model.layers.0.self_attn.A_log"));
            assert_eq!(
                names.contains(&"model.layers.1.self_attn.kv_b_proj.weight"),
                !split
            );
            assert_eq!(
                names.contains(&"model.layers.1.self_attn.k_b_proj.weight"),
                split
            );
            assert_eq!(
                names.contains(&"model.layers.1.self_attn.q_a_proj.weight"),
                query_rank.is_some()
            );
            let experts = plan
                .layout_groups
                .iter()
                .find(|group| group.id.contains("expert storage"))
                .unwrap();
            assert_eq!(
                experts
                    .variants
                    .iter()
                    .map(|variant| variant.id.as_str())
                    .collect::<Vec<_>>(),
                ["packed", "split"]
            );
        }
    }

    #[test]
    fn gguf_plan_and_translation_cover_hybrid_physical_names() {
        let plan = gguf_plan(&fixture(false, None)).unwrap();
        let names = plan
            .common_tensors
            .iter()
            .map(|tensor| tensor.key.as_str())
            .collect::<Vec<_>>();
        assert!(names.contains(&"blk.0.ssm_conv1d_q.weight"));
        assert!(names.contains(&"blk.1.attn_kv_b.weight"));
        assert!(names.contains(&"blk.1.ffn_gate_exps.weight"));
        assert_eq!(
            translate_gguf_weight_name("blk.0.ssm_a.weight"),
            "model.layers.0.self_attn.A_log"
        );
        assert_eq!(
            translate_gguf_weight_name("blk.1.ffn_gate_exps.scales"),
            "model.layers.1.mlp.experts.gate_proj_scales"
        );
    }
}
