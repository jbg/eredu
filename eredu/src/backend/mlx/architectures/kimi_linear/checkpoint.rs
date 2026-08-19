//! Architecture-owned checkpoint contracts for Kimi Linear.

//!
//! Kimi Linear owns its alternating KDA/MLA geometry, expert layouts, aliases,
//! and quantization policy. The generic runtime only evaluates these physical
//! constraints and materializes architecture-supplied recipes.

use eredu_checkpoint::{StoredDtype, WeightQuantization};

use std::collections::HashMap;

use safemlx::ops::{GgufCheckpoint, GgufMetadataValue};
use serde_json::Value;

use super::model::{self, AttentionKind, FeedForwardPolicy, ModelArgs};
use crate::backend::mlx::runtime::checkpoint::{
    store::{SafetensorsWeightStore, WeightStore},
    validation,
};
use eredu_checkpoint::schema::{
    AlternativeLayoutGroup, CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint,
    GgufTypeConstraint, LayoutVariant, SafetensorsCheckpointPlan, SafetensorsTensorConstraint,
    StoredDtypeConstraint, TensorOperation,
};
use eredu_checkpoint::validation::{CheckpointIssue, CheckpointIssueKind, CheckpointValidation};

pub(crate) fn validate_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
) -> CheckpointValidation {
    let args = match model::model_args_from_config_value(config) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    if args.num_hidden_layers as usize > store.keys().len() {
        return invalid_geometry(format!(
            "configured layer count {} exceeds the entire {}-tensor checkpoint catalog",
            args.num_hidden_layers,
            store.keys().len()
        ));
    }
    let plan = match safetensors_plan(&args) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    let mut issues = Vec::new();
    if args.quantization.is_some() {
        for (layer, policy) in args.layer_schedule.iter().enumerate() {
            if policy.feed_forward != FeedForwardPolicy::SparseMoe {
                continue;
            }
            let split = format!("model.layers.{layer}.mlp.experts.0.w1.weight");
            if let Some(raw) = store
                .keys()
                .into_iter()
                .find(|key| canonical_name(key) == split)
            {
                issues.push(CheckpointIssue {
                    kind: CheckpointIssueKind::ConflictingLayout,
                    detail: format!(
                        "checkpoint-native quantized Kimi Linear layer {layer} requires packed expert banks"
                    ),
                    tensor_name: Some(raw),
                    tensor_type_code: None,
                    metadata_key: Some("quantization".into()),
                });
            }
        }
    }
    append_validation(
        validation::validate_safetensors_plan(store, &plan),
        &mut issues,
    );
    finish(issues)
}

pub(crate) fn safetensors_plan(args: &ModelArgs) -> Result<SafetensorsCheckpointPlan, String> {
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
    for name in ["q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight"] {
        output.push(
            safe_aliases(&format!("{prefix}.{name}"), vec![projection, 1, kernel])
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

fn canonical_name(name: &str) -> String {
    name.replace(".block_sparse_moe.", ".mlp.")
        .replace(".inner.weight", ".weight")
}

pub(crate) fn validate_gguf(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> CheckpointValidation {
    if let Err(error) = checkpoint
        .catalog()
        .translated_outputs(model::translate_gguf_weight_name)
    {
        return invalid_geometry(error.to_string());
    }
    let args = match model::model_args_from_gguf_catalog(checkpoint, metadata) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    if let Err(error) = args.validate() {
        return invalid_geometry(error.to_string());
    }
    let plan = match gguf_plan(&args) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    let mut issues = Vec::new();
    append_validation(
        validation::validate_gguf_plan(checkpoint, &plan),
        &mut issues,
    );
    issues.extend(validation::validate_matching_gguf_encodings(
        checkpoint,
        (0..args.num_hidden_layers as usize).map(|layer| {
            (
                format!("blk.{layer}.ffn_gate_exps.weight"),
                format!("blk.{layer}.ffn_up_exps.weight"),
            )
        }),
        "Kimi Linear GGUF",
    ));
    finish(issues)
}

pub(crate) fn gguf_plan(args: &ModelArgs) -> Result<GgufCheckpointPlan, String> {
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

fn append_validation(validation: CheckpointValidation, issues: &mut Vec<CheckpointIssue>) {
    match validation {
        CheckpointValidation::Exact => {}
        CheckpointValidation::Invalid(mut found) => issues.append(&mut found),
        CheckpointValidation::Unverified(issue) => issues.push(issue),
    }
}

fn finish(issues: Vec<CheckpointIssue>) -> CheckpointValidation {
    if issues.is_empty() {
        CheckpointValidation::Exact
    } else {
        CheckpointValidation::Invalid(issues)
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_checkpoint::AffineQuantization;

    #[test]
    fn affine_matrix_constraints_keep_packed_storage_and_runtime_geometry_distinct() {
        let quantization = WeightQuantization::Affine(AffineQuantization::new(32, 4).unwrap());
        let constraints = safe_matrix_constraints(
            "model.layers.0.mlp.gate_proj.weight",
            vec![8, 64],
            Some(quantization),
        )
        .unwrap();
        assert_eq!(constraints[0].shape, [8, 8]);
        assert_eq!(
            constraints[0].dtype,
            StoredDtypeConstraint::Exact(StoredDtype::U32)
        );
        assert_eq!(constraints[1].shape, [8, 2]);
        assert_eq!(
            constraints[1].role,
            eredu_checkpoint::schema::TensorRole::Companion
        );
        assert!(constraints[0]
            .aliases
            .contains(&"model.layers.0.block_sparse_moe.gate_proj.inner.weight".into()));
    }
}
