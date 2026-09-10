//! Exact K2 Horizon source names, tensor geometry and packed expert axes.

use eredu_checkpoint::schema::{
    matrix_for_linear_format, CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint,
    GgufTypeConstraint, SafetensorsCheckpointPlan, SafetensorsTensorConstraint,
    StoredDtypeConstraint, TensorOperation,
};

use super::ModelArgs;

/// Declares all floating logical parameters. GGUF packs each independent bank
/// along its own leading expert axis; SafeTensors stores individual experts.
pub fn parameter_shapes(
    args: &ModelArgs,
    packed_experts: bool,
) -> Result<Vec<(String, Vec<usize>)>, String> {
    args.validate().map_err(|e| e.to_string())?;
    let hidden = args.hidden_size as usize;
    let query = args.num_attention_heads as usize * args.head_dim as usize;
    let kv = args.num_key_value_heads as usize * args.head_dim as usize;
    let mut output = vec![
        (
            "model.embed_tokens.weight".into(),
            vec![args.vocab_size as usize, hidden],
        ),
        ("model.norm.weight".into(), vec![hidden]),
    ];
    if !args.tie_word_embeddings {
        output.push((
            "lm_head.weight".into(),
            vec![args.vocab_size as usize, hidden],
        ));
    }
    for layer in 0..args.num_hidden_layers as usize {
        let prefix = format!("model.layers.{layer}");
        output.extend([
            (format!("{prefix}.input_layernorm.weight"), vec![hidden]),
            (
                format!("{prefix}.post_attention_layernorm.weight"),
                vec![hidden],
            ),
        ]);
        for (projection, rows, columns) in [
            ("q_proj", query, hidden),
            ("k_proj", kv, hidden),
            ("o_proj", hidden, query),
        ] {
            output.push((
                format!("{prefix}.self_attn.{projection}.weight"),
                vec![rows, columns],
            ));
            if args.attention_bias {
                output.push((format!("{prefix}.self_attn.{projection}.bias"), vec![rows]));
            }
        }
        if args.attention_gate_func.is_some() {
            output.push((
                format!("{prefix}.self_attn.gate_proj.weight"),
                vec![query, hidden],
            ));
        }
        if args.query_key_norm {
            output.push((format!("{prefix}.self_attn.q_norm.weight"), vec![query]));
            output.push((format!("{prefix}.self_attn.k_norm.weight"), vec![kv]));
        }
        if args.is_mova_layer(layer) {
            output.push((
                format!("{prefix}.self_attn.v_router.weight"),
                vec![args.mova_num_experts as usize, hidden],
            ));
            if args.moe_gate_bias {
                output.push((
                    format!("{prefix}.self_attn.v_router.bias"),
                    vec![args.mova_num_experts as usize],
                ));
            }
            if packed_experts {
                output.push((
                    format!("{prefix}.self_attn.v_experts.weight"),
                    vec![args.mova_num_experts as usize, kv, hidden],
                ));
            } else {
                for expert in 0..args.mova_num_experts {
                    output.push((
                        format!("{prefix}.self_attn.v_experts.{expert}.weight"),
                        vec![kv, hidden],
                    ));
                }
            }
        } else {
            output.push((
                format!("{prefix}.self_attn.v_proj.weight"),
                vec![kv, hidden],
            ));
            if args.attention_bias {
                output.push((format!("{prefix}.self_attn.v_proj.bias"), vec![kv]));
            }
        }
        let mut mlp = |root: &str, width: usize, count: Option<usize>| {
            for (projection, rows, columns) in [
                ("gate_proj", width, hidden),
                ("up_proj", width, hidden),
                ("down_proj", hidden, width),
            ] {
                let mut shape = count.into_iter().collect::<Vec<_>>();
                shape.extend([rows, columns]);
                output.push((format!("{root}.{projection}.weight"), shape));
            }
        };
        if args.is_sparse_layer(layer) {
            if packed_experts {
                mlp(
                    &format!("{prefix}.mlp.experts"),
                    args.moe_intermediate_size as usize,
                    Some(args.num_experts as usize),
                );
            } else {
                for expert in 0..args.num_experts {
                    mlp(
                        &format!("{prefix}.mlp.experts.{expert}"),
                        args.moe_intermediate_size as usize,
                        None,
                    );
                }
            }
            if args.num_shared_experts > 0 {
                mlp(
                    &format!("{prefix}.mlp.shared_experts"),
                    (args.moe_intermediate_size * args.num_shared_experts) as usize,
                    None,
                );
            }
            output.push((
                format!("{prefix}.mlp.gate.weight"),
                vec![args.num_experts as usize, hidden],
            ));
            if args.moe_gate_bias {
                output.push((
                    format!("{prefix}.mlp.gate.bias"),
                    vec![args.num_experts as usize],
                ));
            }
        } else {
            mlp(
                &format!("{prefix}.mlp"),
                args.intermediate_size as usize,
                None,
            );
        }
    }
    Ok(output)
}

/// Strict official SafeTensors admission, using stored encodings rather than
/// the descriptive `dtype` field in config.json.
pub fn safetensors_plan(args: &ModelArgs) -> Result<SafetensorsCheckpointPlan, String> {
    let mut tensors = Vec::new();
    for (name, shape) in parameter_shapes(args, false)? {
        if shape.len() >= 2 {
            tensors.extend(
                matrix_for_linear_format(
                    &name,
                    Vec::<String>::new(),
                    shape,
                    args.linear_format_for(&name),
                    if matches!(
                        args.linear_format_for(&name),
                        eredu_checkpoint::LinearFormat::E4M3BlockFp8(_)
                    ) {
                        let canonical =
                            format!("{}.weight_scale_inv", name.trim_end_matches(".weight"));
                        let source = args
                            .fp8_metadata()
                            .expect("FP8 format has normalized metadata")
                            .source_scale_name(&name)?;
                        Some(eredu_checkpoint::schema::MatrixScaleNames {
                            key: canonical.clone(),
                            aliases: if source == canonical {
                                vec![]
                            } else {
                                vec![source]
                            },
                        })
                    } else {
                        None
                    },
                )
                .map_err(|e| e.to_string())?,
            );
        } else {
            tensors.push(SafetensorsTensorConstraint::required(
                name,
                shape,
                StoredDtypeConstraint::Floating,
            ));
        }
    }
    SafetensorsCheckpointPlan::new(
        "K2 Horizon SafeTensors",
        tensors,
        Vec::new(),
        CatalogPolicy::strict(),
    )
    .map_err(|e| e.to_string())
}

const FIELDS: &[(&str, &str)] = &[
    ("attn_q", "self_attn.q_proj"),
    ("attn_k", "self_attn.k_proj"),
    ("attn_v", "self_attn.v_proj"),
    ("attn_output", "self_attn.o_proj"),
    ("attn_gate", "self_attn.gate_proj"),
    ("attn_q_norm", "self_attn.q_norm"),
    ("attn_k_norm", "self_attn.k_norm"),
    ("attn_v_gate", "self_attn.v_router"),
    ("attn_v_exps", "self_attn.v_experts"),
    ("attn_norm", "input_layernorm"),
    ("ffn_norm", "post_attention_layernorm"),
    ("ffn_gate", "mlp.gate_proj"),
    ("ffn_up", "mlp.up_proj"),
    ("ffn_down", "mlp.down_proj"),
    ("ffn_gate_inp", "mlp.gate"),
    ("exp_probs_b", "mlp.gate"),
    ("ffn_gate_exps", "mlp.experts.gate_proj"),
    ("ffn_up_exps", "mlp.experts.up_proj"),
    ("ffn_down_exps", "mlp.experts.down_proj"),
    ("ffn_gate_shexp", "mlp.shared_experts.gate_proj"),
    ("ffn_up_shexp", "mlp.shared_experts.up_proj"),
    ("ffn_down_shexp", "mlp.shared_experts.down_proj"),
];

/// Exact publisher tensor translation. Q/K rows keep their published order.
pub fn translate_gguf_weight_name(name: &str) -> String {
    match name {
        "token_embd.weight" => return "model.embed_tokens.weight".into(),
        "output_norm.weight" => return "model.norm.weight".into(),
        "output.weight" => return "lm_head.weight".into(),
        _ => {}
    }
    let Some(rest) = name.strip_prefix("blk.") else {
        return name.into();
    };
    let Some((layer, rest)) = rest.split_once('.') else {
        return name.into();
    };
    let Some((field, suffix)) = rest.rsplit_once('.') else {
        return name.into();
    };
    FIELDS
        .iter()
        .find(|(source, _)| *source == field)
        .map_or_else(
            || name.into(),
            |(_, target)| format!("model.layers.{layer}.{target}.{suffix}"),
        )
}

fn gguf_name(name: &str) -> Result<String, String> {
    match name {
        "model.embed_tokens.weight" => return Ok("token_embd.weight".into()),
        "model.norm.weight" => return Ok("output_norm.weight".into()),
        "lm_head.weight" => return Ok("output.weight".into()),
        _ => {}
    }
    let (layer, rest) = name
        .strip_prefix("model.layers.")
        .and_then(|s| s.split_once('.'))
        .ok_or_else(|| format!("invalid K2 parameter {name}"))?;
    let (field, suffix) = rest
        .rsplit_once('.')
        .ok_or_else(|| format!("invalid K2 parameter {name}"))?;
    let source = if field == "mlp.gate" && suffix == "bias" {
        "exp_probs_b"
    } else {
        FIELDS
            .iter()
            .find(|(_, target)| *target == field)
            .map(|(source, _)| *source)
            .ok_or_else(|| format!("unmapped K2 parameter {name}"))?
    };
    Ok(format!("blk.{layer}.{source}.{suffix}"))
}

/// Strict publisher GGUF catalog, including separately stacked value/MLP banks.
pub fn gguf_plan(args: &ModelArgs) -> Result<GgufCheckpointPlan, String> {
    let tensors = parameter_shapes(args, true)?
        .into_iter()
        .map(|(name, shape)| {
            let operation = if shape.len() == 1 {
                TensorOperation::Vector
            } else {
                TensorOperation::Matrix
            };
            Ok(GgufTensorConstraint::required(
                gguf_name(&name)?,
                shape,
                GgufTypeConstraint::OperationClass(operation),
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    GgufCheckpointPlan::new(
        "K2 Horizon GGUF",
        tensors,
        Vec::new(),
        CatalogPolicy::strict(),
    )
    .map_err(|e| e.to_string())
}
