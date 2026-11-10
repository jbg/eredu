//! Exact K2 Horizon source names, tensor geometry and packed expert axes.

use eredu_checkpoint::schema::{
    matrix_for_linear_format, CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint,
    GgufTypeConstraint, SafetensorsCheckpointPlan, SafetensorsTensorConstraint,
    StoredDtypeConstraint, TensorOperation,
};

use super::Declaration;
use eredu_nn::DeclareLogicalParameters;

/// Published source organization for independently routed parameter banks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceLayout {
    /// Each expert has separate projection tensors.
    Safetensors,
    /// Expert axes are packed, with separate gate and up matrices.
    Gguf,
}

/// A format tensor linked to its authoritative logical construction slot.
#[derive(Debug, Clone)]
pub struct SourceParameter {
    /// Exact ordinary source identity before encoding companions are expanded.
    pub name: String,
    /// Exact logical source axes before scalar encoding.
    pub shape: Vec<usize>,
    /// Owning logical slot, including packed and fused expert matrices.
    pub logical: eredu_nn::ParameterId,
    /// Meaning inherited from the owning neural slot.
    pub role: eredu_nn::LogicalParameterRole,
}
impl SourceParameter {
    fn projected(
        p: &eredu_nn::LogicalParameterDeclaration<'_>,
        name: String,
        shape: Vec<usize>,
    ) -> Self {
        Self {
            name,
            shape,
            logical: p.parameter.id.clone(),
            role: p.role,
        }
    }
    fn direct(p: eredu_nn::LogicalParameterDeclaration<'_>) -> Self {
        Self {
            name: p.parameter.id.to_string(),
            shape: p.shape,
            logical: p.parameter.id.clone(),
            role: p.role,
        }
    }
}

/// Projects ordinary source identities and geometry from the actual construction declarations.
/// Encoding companions, aliases and dtype constraints are separate format extensions.
pub fn source_parameters(
    declaration: &Declaration,
    layout: SourceLayout,
) -> Result<Vec<SourceParameter>, String> {
    let blocks = declaration.blocks().map_err(|e| e.to_string())?;
    let mut output = declaration
        .common()
        .static_construction()
        .logical_parameters()
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(SourceParameter::direct)
        .collect::<Vec<_>>();
    for block in &blocks {
        output.extend(
            block
                .common
                .logical_parameters()
                .map_err(|e| e.to_string())?
                .into_iter()
                .map(SourceParameter::direct),
        );
        if let Some(values) = &block.values {
            output.extend(
                values
                    .router
                    .logical_parameters()
                    .map_err(|e| e.to_string())?
                    .into_iter()
                    .map(SourceParameter::direct),
            );
            for p in values
                .experts
                .logical_parameters()
                .map_err(|e| e.to_string())?
            {
                match layout {
                    SourceLayout::Gguf => output.push(SourceParameter::direct(p)),
                    SourceLayout::Safetensors => {
                        let root = p
                            .parameter
                            .id
                            .as_str()
                            .strip_suffix(".weight")
                            .ok_or("value bank has no declared weight slot")?;
                        for expert in 0..p.shape[0] {
                            output.push(SourceParameter::projected(
                                &p,
                                format!("{root}.{expert}.weight"),
                                p.shape[1..].to_vec(),
                            ));
                        }
                    }
                }
            }
        }
        if let Some(routed) = &block.routed {
            let source = routed
                .parameters
                .experts
                .packed_parameters()
                .map_err(|e| e.to_string())?;
            for (p, fields) in [
                (
                    source.gate_up(),
                    vec![
                        ("gate_proj", source.unfused_shape()),
                        ("up_proj", source.unfused_shape()),
                    ],
                ),
                (
                    source.down(),
                    vec![("down_proj", source.down().shape.clone())],
                ),
            ] {
                let root = p
                    .parameter
                    .id
                    .as_str()
                    .strip_suffix(if p.parameter.id == source.gate_up().parameter.id {
                        ".gate_up_proj"
                    } else {
                        ".down_proj"
                    })
                    .ok_or("unresolved K2 packed expert projection")?;
                for (field, shape) in fields {
                    match layout {
                        SourceLayout::Gguf => output.push(SourceParameter::projected(
                            p,
                            format!("{root}.{field}.weight"),
                            shape,
                        )),
                        SourceLayout::Safetensors => {
                            for expert in 0..shape[0] {
                                output.push(SourceParameter::projected(
                                    p,
                                    format!("{root}.{expert}.{field}.weight"),
                                    shape[1..].to_vec(),
                                ));
                            }
                        }
                    }
                }
            }
            if let Some(shared) = &routed.shared {
                output.extend(
                    shared
                        .logical_parameters()
                        .map_err(|e| e.to_string())?
                        .into_iter()
                        .map(SourceParameter::direct),
                );
            }
            output.extend(
                routed
                    .parameters
                    .selector
                    .logical_parameters()
                    .map_err(|e| e.to_string())?
                    .into_iter()
                    .map(SourceParameter::direct),
            );
        }
    }
    Ok(output)
}

/// Strict official SafeTensors admission, using stored encodings rather than
/// the descriptive `dtype` field in config.json.
pub fn safetensors_plan(declaration: &Declaration) -> Result<SafetensorsCheckpointPlan, String> {
    let args = declaration.args();
    let mut tensors = Vec::new();
    for parameter in source_parameters(declaration, SourceLayout::Safetensors)? {
        let SourceParameter { name, shape, .. } = parameter;
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
pub fn gguf_plan(declaration: &Declaration) -> Result<GgufCheckpointPlan, String> {
    let tensors = source_parameters(declaration, SourceLayout::Gguf)?
        .into_iter()
        .map(|SourceParameter { name, shape, .. }| {
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
