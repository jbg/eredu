//! Released source-format extensions over retained assistant parameter declarations.

use super::integration::MuseAssistantConfig;
use crate::external_assistant::AssistantParameterDeclarations;
use eredu_checkpoint::{
    schema::{
        matrix_for_linear_format, CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint,
        GgufTypeConstraint, SafetensorsCheckpointPlan, SafetensorsTensorConstraint,
        StoredDtypeConstraint, TensorOperation,
    },
    LinearFormat, StoredDtype, WeightQuantization,
};
use eredu_nn::{DeclareLogicalParameters, LogicalParameterRole};
use std::collections::HashMap;

pub(super) fn safetensors_plan(
    parameters: &AssistantParameterDeclarations,
    config: &MuseAssistantConfig,
) -> Result<SafetensorsCheckpointPlan, String> {
    let glimmer = matches!(config, MuseAssistantConfig::Glimmer(_));
    let mut tensors = Vec::new();
    for parameter in parameters.logical_parameters().map_err(|e| e.to_string())? {
        let name = parameter.parameter.id.as_str();
        let encoding = parameter.linear_format.map(|f| f.encoding());
        if let Some(format) = encoding.filter(|f| glimmer || *f != LinearFormat::Dense) {
            tensors.extend(
                matrix_for_linear_format(
                    name,
                    std::iter::empty::<String>(),
                    parameter.shape,
                    format,
                    None,
                )
                .map_err(|e| e.to_string())?,
            );
            continue;
        }
        let dtype = if glimmer {
            if parameter.role != LogicalParameterRole::Normalization {
                return Err(format!("unsupported DFlash source parameter {name}"));
            }
            StoredDtypeConstraint::Floating
        } else {
            StoredDtypeConstraint::Exact(if parameter.role == LogicalParameterRole::TokenMapping {
                StoredDtype::I64
            } else {
                StoredDtype::BF16
            })
        };
        let mut tensor = SafetensorsTensorConstraint::required(name, parameter.shape, dtype);
        tensor.linear_format = encoding;
        tensors.push(tensor);
    }
    let name = match config {
        MuseAssistantConfig::Glimmer(_) => "Muse-Glimmer DFlash SafeTensors",
        MuseAssistantConfig::Spark(c) => c.released().identity().architecture(),
    };
    SafetensorsCheckpointPlan::new(name, tensors, Vec::new(), CatalogPolicy::strict())
        .map_err(|e| e.to_string())
}

pub(super) fn gguf_plan(
    parameters: &AssistantParameterDeclarations,
    config: &MuseAssistantConfig,
) -> Result<GgufCheckpointPlan, String> {
    if !matches!(config, MuseAssistantConfig::Glimmer(_)) {
        return Err("no released Spark assistant GGUF format is declared".into());
    }
    let mut tensors = Vec::new();
    for parameter in parameters.logical_parameters().map_err(|e| e.to_string())? {
        let name = parameter.parameter.id.as_str();
        let key = dflash_gguf_source_name(name)?;
        let operation = match parameter.role {
            LogicalParameterRole::Normalization => TensorOperation::Vector,
            LogicalParameterRole::LinearWeight => TensorOperation::Matrix,
            _ => return Err(format!("unsupported DFlash GGUF parameter {name}")),
        };
        tensors.push(GgufTensorConstraint::required(
            key,
            parameter.shape,
            GgufTypeConstraint::OperationClass(operation),
        ));
    }
    GgufCheckpointPlan::new(
        "Muse-Glimmer DFlash GGUF",
        tensors,
        Vec::new(),
        CatalogPolicy::strict(),
    )
    .map_err(|e| e.to_string())
}

const DFLASH_GGUF_STATICS: [(&str, &str); 3] = [
    ("fc.weight", "encoder.fc.weight"),
    ("enc.output_norm.weight", "encoder.output_norm_enc.weight"),
    ("output_norm.weight", "norm.weight"),
];
const DFLASH_GGUF_BLOCK_FIELDS: [(&str, &str); 11] = [
    ("attn_norm", "input_layernorm"),
    ("ffn_norm", "post_attention_layernorm"),
    ("attn_q_norm", "self_attn.q_norm"),
    ("attn_k_norm", "self_attn.k_norm"),
    ("attn_q", "self_attn.q_proj"),
    ("attn_k", "self_attn.k_proj"),
    ("attn_v", "self_attn.v_proj"),
    ("attn_output", "self_attn.o_proj"),
    ("ffn_gate", "mlp.gate_proj"),
    ("ffn_up", "mlp.up_proj"),
    ("ffn_down", "mlp.down_proj"),
];
fn dflash_gguf_source_name(target: &str) -> Result<String, String> {
    if let Some((source, _)) = DFLASH_GGUF_STATICS
        .iter()
        .find(|(_, canonical)| *canonical == target)
    {
        return Ok((*source).into());
    }
    if let Some((layer, field)) = target
        .strip_prefix("layers.")
        .and_then(|rest| rest.split_once('.'))
    {
        for (source, canonical) in DFLASH_GGUF_BLOCK_FIELDS {
            if field == format!("{canonical}.weight") {
                return Ok(format!("blk.{layer}.{source}.weight"));
            }
        }
    }
    Err(format!(
        "DFlash parameter {target} has no GGUF source extension"
    ))
}
/// Translates one released DFlash GGUF tensor to neutral parameter identity.
pub fn translate_dflash_gguf_weight_name(name: &str) -> String {
    if let Some((_, target)) = DFLASH_GGUF_STATICS
        .iter()
        .find(|(source, _)| *source == name)
    {
        return (*target).into();
    }
    let Some((layer, parameter)) = name
        .strip_prefix("blk.")
        .and_then(|rest| rest.split_once('.'))
    else {
        return name.into();
    };
    for (source, target) in DFLASH_GGUF_BLOCK_FIELDS {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            return format!("layers.{layer}.{}", parameter.replacen(source, target, 1));
        }
    }
    name.into()
}

/// Checks the encodings actually retained after complete assistant projection.
pub(super) fn validate_selected_formats(
    parameters: &impl DeclareLogicalParameters,
    formats: &HashMap<String, WeightQuantization>,
) -> Result<(), String> {
    validate_formats(parameters, formats)?;
    crate::format_references::validate_matrices(
        parameters.logical_parameters().map_err(|e| e.to_string())?,
        formats
            .iter()
            .map(|(name, format)| (name.as_str(), Some((*format).into()))),
    )
    .map_err(|e| e.to_string())
}

pub(super) fn validate_formats(
    parameters: &impl DeclareLogicalParameters,
    formats: &HashMap<String, WeightQuantization>,
) -> Result<(), String> {
    let parameters = parameters.logical_parameters().map_err(|e| e.to_string())?;
    for (name, format) in formats {
        let parameter = parameters
            .iter()
            .find(|p| p.parameter.id.as_str() == name && p.linear_format.is_some())
            .ok_or_else(|| format!("{name} is not a declared assistant matrix"))?;
        format.validate().map_err(|e| e.to_string())?;
        eredu_checkpoint::schema::matrix_for_linear_format(
            name,
            std::iter::empty::<String>(),
            parameter.shape.clone(),
            (*format).into(),
            None,
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}
