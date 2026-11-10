//! Architecture-owned checkpoint contracts for Llama, Mistral and Phi-4 decoders.
//!
//! This module projects logical decoder declarations into physical tensor names,
//! affine SafeTensors companions and GGUF encodings. It retains the dtype rules
//! and RoPE catalog exclusions accepted by Llama-compatible loaders. Generic
//! checkpoint code evaluates the resulting declarative constraints.

use std::{collections::HashMap, fmt};

use eredu_checkpoint::schema::{
    CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint, GgufTypeConstraint,
    SafetensorsCheckpointPlan, SafetensorsTensorConstraint, StoredDtypeConstraint, TensorOperation,
};
use eredu_checkpoint::WeightQuantization;

use super::ModelArgs;

/// Derives a Llama configuration whose physical matrix formats reflect
/// load-time quantization instead of checkpoint-specific format selections.
pub fn load_time_quantization(
    args: &ModelArgs,
    quantization: WeightQuantization,
) -> Result<ModelArgs, String> {
    quantization.validate().map_err(|error| error.to_string())?;
    let mut target = args.clone();
    target.quantization = Some(quantization);
    target.quantized_weights = None;
    target.quantized_weight_configs = None;
    target.validate().map_err(|error| error.to_string())?;
    Ok(target)
}

/// Applies canonical checkpoint format metadata to a complete Llama
/// configuration.
pub fn with_checkpoint_formats(
    args: &ModelArgs,
    formats: HashMap<String, WeightQuantization>,
) -> Result<ModelArgs, String> {
    let mut target = args.clone();
    target.quantized_weights = Some(formats.keys().cloned().collect());
    target.quantization = None;
    target.quantized_weight_configs = Some(formats);
    target.validate().map_err(|error| error.to_string())?;
    Ok(target)
}

// Physical storage names and canonical construction references share one map.
const GGUF_PARAMETER_NAMES: &[(&str, &str)] = &[
    ("blk.", "model.layers."),
    ("ffn_gate", "mlp.gate_proj"),
    ("ffn_down", "mlp.down_proj"),
    ("ffn_up", "mlp.up_proj"),
    ("attn_q", "self_attn.q_proj"),
    ("attn_k", "self_attn.k_proj"),
    ("attn_v", "self_attn.v_proj"),
    ("attn_output", "self_attn.o_proj"),
    ("attn_norm", "input_layernorm"),
    ("ffn_norm", "post_attention_layernorm"),
    ("token_embd", "model.embed_tokens"),
    ("output_norm", "model.norm"),
    ("output", "lm_head"),
];

/// Translates one llama.cpp GGUF tensor name into the canonical HF layout.
pub fn translate_gguf_weight_name(name: &str) -> String {
    GGUF_PARAMETER_NAMES
        .iter()
        .fold(name.to_owned(), |name, (physical, canonical)| {
            name.replace(physical, canonical)
        })
}

/// Translates the physical fused Phi layout or the ordinary split layout.
pub(crate) fn translate_gguf_weight_name_for_model(name: &str, args: &ModelArgs) -> String {
    if args.model_type == "phi3" {
        translate_gguf_weight_name(
            &name
                .replace("attn_qkv", "self_attn.qkv_proj")
                .replace("ffn_up", "mlp.gate_up_proj"),
        )
    } else {
        translate_gguf_weight_name(name)
    }
}

/// Builds the complete SafeTensors catalog expected by this configuration.
pub fn safetensors_plan(
    args: &ModelArgs,
    declaration: &crate::decoder::DecoderDeclaration,
) -> Result<SafetensorsCheckpointPlan, SafetensorsPlanError> {
    let declarations = declaration
        .parameter_description(args, crate::decoder::ParameterEncoding::Logical)
        .map_err(|error| SafetensorsPlanError::Geometry(error.to_string()))?;
    safetensors_plan_from_parameters(
        declarations
            .groups()
            .iter()
            .flat_map(|group| group.members()),
    )
}

pub(crate) fn safetensors_plan_from_parameters<'a>(
    parameters: impl IntoIterator<Item = &'a eredu_runtime::ParameterMemberSpec>,
) -> Result<SafetensorsCheckpointPlan, SafetensorsPlanError> {
    let mut tensors = Vec::new();
    for parameter in parameters {
        let quantization = parameter
            .logical_linear_format()
            .and_then(|format| format.encoding().weight_quantization());
        let constraints = if let Some(quantization) = quantization {
            eredu_checkpoint::schema::matrix_for_packed_words(
                parameter.target(),
                Vec::<String>::new(),
                parameter.global_shape().to_vec(),
                quantization,
                None,
            )
            .map_err(|error| SafetensorsPlanError::Companion {
                name: parameter.target().into(),
                detail: error.to_string(),
            })?
        } else {
            vec![safe(parameter.target(), parameter.global_shape().to_vec())]
        };
        tensors.extend(constraints);
    }
    let mut policy = CatalogPolicy::strict();
    policy.allowed_prefixes.push("rope_freqs.".into());
    policy.allowed_suffixes.push(".rotary_emb.inv_freq".into());
    SafetensorsCheckpointPlan::new("Llama SafeTensors", tensors, Vec::new(), policy)
        .map_err(|error| SafetensorsPlanError::Geometry(error.to_string()))
}

fn safe(key: impl Into<String>, shape: Vec<usize>) -> SafetensorsTensorConstraint {
    SafetensorsTensorConstraint::required(key, shape, StoredDtypeConstraint::Floating)
}

/// Failure to derive a physically valid SafeTensors catalog.
#[derive(Debug)]
pub enum SafetensorsPlanError {
    /// Model dimensions cannot describe a valid tensor layout.
    Geometry(String),
    /// A quantized tensor's required companion layout is invalid.
    Companion {
        /// Canonical quantized weight name.
        name: String,
        /// Physical-layout mismatch.
        detail: String,
    },
}

impl fmt::Display for SafetensorsPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Geometry(detail) | Self::Companion { detail, .. } => formatter.write_str(detail),
        }
    }
}

impl From<String> for SafetensorsPlanError {
    fn from(detail: String) -> Self {
        Self::Geometry(detail)
    }
}

/// Builds the complete GGUF catalog expected by this configuration.
pub fn gguf_plan(
    args: &ModelArgs,
    declaration: &crate::decoder::DecoderDeclaration,
) -> Result<GgufCheckpointPlan, String> {
    let declarations = declaration
        .parameter_description(args, crate::decoder::ParameterEncoding::Logical)
        .map_err(|error| error.to_string())?;
    gguf_plan_from_parameters(
        args,
        declarations
            .groups()
            .iter()
            .flat_map(|group| group.members()),
    )
}

pub(crate) fn gguf_plan_from_parameters<'a>(
    args: &ModelArgs,
    parameters: impl IntoIterator<Item = &'a eredu_runtime::ParameterMemberSpec>,
) -> Result<GgufCheckpointPlan, String> {
    let mut tensors = Vec::new();
    for parameter in parameters {
        tensors.push(gguf(
            gguf_parameter_name(parameter.target(), args),
            parameter.global_shape().to_vec(),
            if parameter.logical_linear_format().is_some() {
                TensorOperation::Matrix
            } else {
                TensorOperation::Vector
            },
        ));
    }
    let mut policy = CatalogPolicy::strict();
    policy.allowed_prefixes.push("rope_freqs.".into());
    GgufCheckpointPlan::new("Llama GGUF", tensors, Vec::new(), policy)
        .map_err(|error| error.to_string())
}

fn gguf(
    key: impl Into<String>,
    shape: Vec<usize>,
    operation: TensorOperation,
) -> GgufTensorConstraint {
    GgufTensorConstraint::required(key, shape, GgufTypeConstraint::OperationClass(operation))
}

/// Physical GGUF names reference the canonical construction identities.
pub(crate) fn gguf_parameter_name(name: &str, args: &ModelArgs) -> String {
    let name = GGUF_PARAMETER_NAMES
        .iter()
        .fold(name.to_owned(), |name, (physical, canonical)| {
            name.replace(canonical, physical)
        });
    if args.model_type == "phi3" {
        name.replace("self_attn.qkv_proj", "attn_qkv")
            .replace("mlp.gate_up_proj", "ffn_up")
    } else {
        name
    }
}
