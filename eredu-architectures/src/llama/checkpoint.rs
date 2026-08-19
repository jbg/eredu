//! Architecture-owned checkpoint contracts for Llama and Mistral decoders.
//!
//! This module owns physical tensor names, geometry, bias policy, affine
//! SafeTensors companions, GGUF encodings, and the small RoPE catalog
//! exclusions accepted by the Llama-compatible loaders. Generic checkpoint
//! code only evaluates the resulting declarative constraints.

use std::fmt;

use eredu_checkpoint::schema::{
    CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint, GgufTypeConstraint,
    SafetensorsCheckpointPlan, SafetensorsTensorConstraint, StoredDtypeConstraint, TensorOperation,
};
use eredu_checkpoint::{StoredDtype, WeightQuantization};

use super::ModelArgs;

/// Translates one llama.cpp GGUF tensor name into the canonical HF layout.
pub fn translate_gguf_weight_name(name: &str) -> String {
    name.replace("blk.", "model.layers.")
        .replace("ffn_gate", "mlp.gate_proj")
        .replace("ffn_down", "mlp.down_proj")
        .replace("ffn_up", "mlp.up_proj")
        .replace("attn_q", "self_attn.q_proj")
        .replace("attn_k", "self_attn.k_proj")
        .replace("attn_v", "self_attn.v_proj")
        .replace("attn_output", "self_attn.o_proj")
        .replace("attn_norm", "input_layernorm")
        .replace("ffn_norm", "post_attention_layernorm")
        .replace("token_embd", "model.embed_tokens")
        .replace("output_norm", "model.norm")
        .replace("output", "lm_head")
}

/// Builds the complete SafeTensors catalog expected by this configuration.
pub fn safetensors_plan(
    args: &ModelArgs,
) -> Result<SafetensorsCheckpointPlan, SafetensorsPlanError> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let vocab = dimension(args.vocab_size, "vocabulary size")?;
    let intermediate = dimension(args.intermediate_size, "intermediate size")?;
    let query = checked_mul(
        dimension(args.num_attention_heads, "attention head count")?,
        dimension(args.head_dim, "attention head dimension")?,
        "query projection width",
    )?;
    let key_value = checked_mul(
        dimension(args.num_key_value_heads, "key/value head count")?,
        dimension(args.head_dim, "attention head dimension")?,
        "key/value projection width",
    )?;
    let mut tensors = Vec::new();
    add_safe(
        args,
        &mut tensors,
        "model.embed_tokens.weight",
        vec![vocab, hidden],
        TensorOperation::Matrix,
    )?;
    tensors.push(safe("model.norm.weight", vec![hidden]));
    if !args.tie_word_embeddings {
        add_safe(
            args,
            &mut tensors,
            "lm_head.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
        )?;
    }
    for layer in 0..dimension(args.num_hidden_layers, "layer count")? {
        let block = format!("model.layers.{layer}");
        tensors.extend([
            safe(format!("{block}.input_layernorm.weight"), vec![hidden]),
            safe(
                format!("{block}.post_attention_layernorm.weight"),
                vec![hidden],
            ),
        ]);
        for (local, shape) in [
            ("self_attn.q_proj.weight", vec![query, hidden]),
            ("self_attn.k_proj.weight", vec![key_value, hidden]),
            ("self_attn.v_proj.weight", vec![key_value, hidden]),
            ("self_attn.o_proj.weight", vec![hidden, query]),
            ("mlp.gate_proj.weight", vec![intermediate, hidden]),
            ("mlp.up_proj.weight", vec![intermediate, hidden]),
            ("mlp.down_proj.weight", vec![hidden, intermediate]),
        ] {
            add_safe(
                args,
                &mut tensors,
                &format!("{block}.{local}"),
                shape,
                TensorOperation::Matrix,
            )?;
        }
        if args.attention_bias {
            for (local, size) in [
                ("q_proj.bias", query),
                ("k_proj.bias", key_value),
                ("v_proj.bias", key_value),
                ("o_proj.bias", hidden),
            ] {
                tensors.push(safe(format!("{block}.self_attn.{local}"), vec![size]));
            }
        }
        if args.mlp_bias {
            for (local, size) in [
                ("gate_proj.bias", intermediate),
                ("up_proj.bias", intermediate),
                ("down_proj.bias", hidden),
            ] {
                tensors.push(safe(format!("{block}.mlp.{local}"), vec![size]));
            }
        }
    }
    SafetensorsCheckpointPlan::new(
        "Llama SafeTensors",
        tensors,
        Vec::new(),
        CatalogPolicy::non_strict(),
    )
    .map_err(|error| SafetensorsPlanError::Geometry(error.to_string()))
}

fn add_safe(
    args: &ModelArgs,
    output: &mut Vec<SafetensorsTensorConstraint>,
    name: &str,
    shape: Vec<usize>,
    operation: TensorOperation,
) -> Result<(), SafetensorsPlanError> {
    let quantization = (operation == TensorOperation::Matrix)
        .then(|| args.weight_quantization_for(name))
        .flatten();
    output.extend(safe_matrix_constraints(name, shape, quantization)?);
    Ok(())
}

fn safe_matrix_constraints(
    name: &str,
    shape: Vec<usize>,
    quantization: Option<WeightQuantization>,
) -> Result<Vec<SafetensorsTensorConstraint>, SafetensorsPlanError> {
    let Some(quantization) = quantization else {
        return Ok(vec![safe(name, shape)]);
    };
    let input = *shape
        .last()
        .ok_or_else(|| format!("quantized Llama matrix {name:?} has scalar shape"))?;
    let bits = dimension(quantization.bits(), "quantization bit width")?;
    let group = dimension(quantization.group_size(), "quantization group size")?;
    let packed_bits = input
        .checked_mul(bits)
        .ok_or_else(|| SafetensorsPlanError::Companion {
            name: name.into(),
            detail: format!("quantized tensor {name:?} packing geometry overflows"),
        })?;
    if !input.is_multiple_of(group) || !input.is_multiple_of(32) || !packed_bits.is_multiple_of(32)
    {
        return Err(SafetensorsPlanError::Companion {
            name: name.into(),
            detail: format!(
                "quantized tensor {name:?} input dimension {input} is incompatible with group size {group} and {bits}-bit packing"
            ),
        });
    }
    let mut packed = shape.clone();
    *packed.last_mut().expect("matrix shape") = packed_bits / 32;
    let mut companion = shape;
    *companion.last_mut().expect("matrix shape") = input / group;
    let prefix = name.strip_suffix(".weight").unwrap_or(name);
    let dtype = || {
        StoredDtypeConstraint::OneOf(vec![
            StoredDtype::F16,
            StoredDtype::BF16,
            StoredDtype::F32,
            StoredDtype::U8,
        ])
    };
    let mut constraints = vec![SafetensorsTensorConstraint::required(
        name,
        packed,
        StoredDtypeConstraint::Exact(StoredDtype::U32),
    )
    .with_aliases([format!("{prefix}.inner.weight")])];
    constraints.push(
        SafetensorsTensorConstraint::required(
            format!("{prefix}.scales"),
            companion.clone(),
            dtype(),
        )
        .companion(),
    );
    if quantization.has_biases() {
        constraints.push(
            SafetensorsTensorConstraint::required(format!("{prefix}.biases"), companion, dtype())
                .companion(),
        );
    }
    Ok(constraints)
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
pub fn gguf_plan(args: &ModelArgs) -> Result<GgufCheckpointPlan, String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let vocab = dimension(args.vocab_size, "vocabulary size")?;
    let intermediate = dimension(args.intermediate_size, "intermediate size")?;
    let query = checked_mul(
        dimension(args.num_attention_heads, "attention head count")?,
        dimension(args.head_dim, "attention head dimension")?,
        "query projection width",
    )?;
    let key_value = checked_mul(
        dimension(args.num_key_value_heads, "key/value head count")?,
        dimension(args.head_dim, "attention head dimension")?,
        "key/value projection width",
    )?;
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
    for layer in 0..dimension(args.num_hidden_layers, "layer count")? {
        let block = format!("blk.{layer}");
        for (local, shape, operation) in [
            ("attn_norm.weight", vec![hidden], TensorOperation::Vector),
            ("ffn_norm.weight", vec![hidden], TensorOperation::Vector),
            (
                "attn_q.weight",
                vec![query, hidden],
                TensorOperation::Matrix,
            ),
            (
                "attn_k.weight",
                vec![key_value, hidden],
                TensorOperation::Matrix,
            ),
            (
                "attn_v.weight",
                vec![key_value, hidden],
                TensorOperation::Matrix,
            ),
            (
                "attn_output.weight",
                vec![hidden, query],
                TensorOperation::Matrix,
            ),
            (
                "ffn_gate.weight",
                vec![intermediate, hidden],
                TensorOperation::Matrix,
            ),
            (
                "ffn_up.weight",
                vec![intermediate, hidden],
                TensorOperation::Matrix,
            ),
            (
                "ffn_down.weight",
                vec![hidden, intermediate],
                TensorOperation::Matrix,
            ),
        ] {
            tensors.push(gguf(format!("{block}.{local}"), shape, operation));
        }
        if args.attention_bias {
            for (local, size) in [
                ("attn_q.bias", query),
                ("attn_k.bias", key_value),
                ("attn_v.bias", key_value),
                ("attn_output.bias", hidden),
            ] {
                tensors.push(gguf(
                    format!("{block}.{local}"),
                    vec![size],
                    TensorOperation::Vector,
                ));
            }
        }
        if args.mlp_bias {
            for (local, size) in [
                ("ffn_gate.bias", intermediate),
                ("ffn_up.bias", intermediate),
                ("ffn_down.bias", hidden),
            ] {
                tensors.push(gguf(
                    format!("{block}.{local}"),
                    vec![size],
                    TensorOperation::Vector,
                ));
            }
        }
    }
    GgufCheckpointPlan::new(
        "Llama GGUF",
        tensors,
        Vec::new(),
        CatalogPolicy::non_strict(),
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

fn dimension(value: i32, name: &str) -> Result<usize, String> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("Llama {name} must be positive, got {value}"))
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("Llama {name} geometry overflows"))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    fn args() -> ModelArgs {
        crate::llama::model_args_from_config_value(&serde_json::json!({
            "model_type": "llama", "hidden_size": 32, "num_hidden_layers": 1,
            "intermediate_size": 64, "num_attention_heads": 4,
            "num_key_value_heads": 2, "rms_norm_eps": 1e-5, "vocab_size": 32,
            "max_position_embeddings": 128, "rope_theta": 10000.0,
            "attention_bias": true, "mlp_bias": true, "tie_word_embeddings": false
        }))
        .unwrap()
    }

    #[test]
    fn plans_output_and_bias_geometry() {
        let mut args = args();
        let plan = safetensors_plan(&args).unwrap();
        let names = plan
            .common_tensors
            .iter()
            .map(|tensor| tensor.key.as_str())
            .collect::<BTreeSet<_>>();
        assert!(names.contains("lm_head.weight"));
        assert!(names.contains("model.layers.0.self_attn.o_proj.bias"));
        assert!(names.contains("model.layers.0.mlp.down_proj.bias"));

        args.tie_word_embeddings = true;
        assert!(!safetensors_plan(&args)
            .unwrap()
            .common_tensors
            .iter()
            .any(|tensor| tensor.key == "lm_head.weight"));
    }

    #[test]
    fn translates_gguf_names_without_backend_code() {
        assert_eq!(
            translate_gguf_weight_name("blk.3.attn_q.weight"),
            "model.layers.3.self_attn.q_proj.weight"
        );
    }
}
