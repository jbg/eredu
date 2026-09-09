//! Gemma 2 dense decoder policy over the shared portable decoder.

use crate::decoder::{AttentionProjection, BlockParameterFields, Config};
use eredu_checkpoint::{
    schema::{
        GgufCheckpointPlan, GgufTensorConstraint, GgufTypeConstraint, SafetensorsCheckpointPlan,
        SafetensorsTensorConstraint, StoredDtypeConstraint, TensorOperation,
    },
    WeightQuantization,
};
use eredu_core::{
    cache::derive_prompt_cache_architecture_fingerprint, AttentionPolicy, LayerSchedule,
};
use eredu_gguf::MetadataValue;
use eredu_nn::{Error, GatedProductPolicy, RotarySpec};
use serde::Deserialize;
use serde_json::Value;
use std::{collections::HashMap, io::Read};

/// Invalid Gemma 2 configuration or unsupported equation.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// Malformed JSON or field type.
    #[error("invalid Gemma 2 configuration: {0}")]
    Json(#[from] serde_json::Error),
    /// Invalid geometry or equation policy.
    #[error("invalid Gemma 2 configuration: {0}")]
    Invalid(String),
}

/// Validated geometry and equations, independent of any backend.
#[derive(Debug, Clone)]
pub struct ModelArgs {
    dense: crate::llama::ModelArgs,
    query_pre_attn_scalar: f32,
    attn_logit_softcapping: Option<f32>,
    final_logit_softcapping: Option<f32>,
    // llama.cpp stores RMS scales after adding one; HF stores the offsets.
    normalization_offset: f32,
}

impl ModelArgs {
    /// Shared dense geometry and checkpoint encoding policy.
    pub fn dense_config(&self) -> &crate::llama::ModelArgs {
        &self.dense
    }
    /// Exact physical encoding of one canonical checkpoint parameter.
    pub fn weight_quantization_for(&self, name: &str) -> Option<WeightQuantization> {
        self.dense.weight_quantization_for(name)
    }
    /// Validates all geometry and numerical policies.
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.dense
            .validate()
            .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        for (name, value) in [
            ("rms_norm_eps", self.dense.rms_norm_eps),
            ("rope_theta", self.dense.rope_theta),
            ("query_pre_attn_scalar", self.query_pre_attn_scalar),
        ] {
            positive(name, value)?;
        }
        for (name, cap) in [
            ("attn_logit_softcapping", self.attn_logit_softcapping),
            ("final_logit_softcapping", self.final_logit_softcapping),
        ] {
            if let Some(cap) = cap {
                positive(name, cap)?;
            }
        }
        if self.dense.head_dim % 2 != 0 {
            return Err(invalid("head_dim must be even"));
        }
        if self.dense.mlp_bias {
            return Err(invalid("Gemma 2 feed-forward projections have no biases"));
        }
        Ok(())
    }
}
fn invalid(detail: impl Into<String>) -> ConfigError {
    ConfigError::Invalid(detail.into())
}
fn positive(name: &str, value: f32) -> Result<(), ConfigError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(invalid(format!("{name} must be finite and positive")))
    }
}
fn default_query_scale() -> f32 {
    256.0
}
fn default_attention_cap() -> Option<f32> {
    Some(50.0)
}
fn default_output_cap() -> Option<f32> {
    Some(30.0)
}
fn default_window() -> Option<u32> {
    Some(4096)
}
fn default_activation() -> String {
    "gelu_pytorch_tanh".into()
}
#[derive(Deserialize)]
struct EquationConfig {
    #[serde(default = "default_query_scale")]
    query_pre_attn_scalar: f32,
    #[serde(default = "default_attention_cap")]
    attn_logit_softcapping: Option<f32>,
    #[serde(default = "default_output_cap")]
    final_logit_softcapping: Option<f32>,
    #[serde(default = "default_window")]
    sliding_window: Option<u32>,
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    #[serde(default = "default_activation")]
    hidden_activation: String,
    #[serde(default)]
    use_bidirectional_attention: Option<bool>,
}

/// Reads a Hugging Face configuration.
pub fn model_args_from_config_reader(reader: impl Read) -> Result<ModelArgs, ConfigError> {
    model_args_from_config_value(&serde_json::from_reader(reader)?)
}
/// Normalizes Gemma 2 equations and the full/sliding attention schedule.
pub fn model_args_from_config_value(value: &Value) -> Result<ModelArgs, ConfigError> {
    if value.get("model_type").and_then(Value::as_str) != Some("gemma2") {
        return Err(invalid("model_type must be gemma2"));
    }
    let policy: EquationConfig = serde_json::from_value(value.clone())?;
    if policy.hidden_activation != "gelu_pytorch_tanh" {
        return Err(invalid("hidden_activation must be gelu_pytorch_tanh"));
    }
    if policy.use_bidirectional_attention == Some(true) {
        return Err(invalid(
            "bidirectional attention cannot use causal decoder sessions",
        ));
    }
    for field in ["head_dim", "num_key_value_heads", "max_position_embeddings"] {
        if value
            .get(field)
            .is_some_and(|v| !v.as_i64().is_some_and(|n| n > 0))
        {
            return Err(invalid(format!("{field} must be a positive integer")));
        }
    }
    let mut common = value.clone();
    common["model_type"] = "llama".into();
    common["sliding_window"] = Value::Null;
    let defaults = serde_json::json!({"hidden_size":2304,"intermediate_size":9216,
        "num_hidden_layers":26,"num_attention_heads":8,"num_key_value_heads":4,
        "head_dim":256,"vocab_size":256000,"rms_norm_eps":1e-6,
        "max_position_embeddings":8192,"tie_word_embeddings":true});
    for (key, default) in defaults.as_object().unwrap() {
        common
            .as_object_mut()
            .unwrap()
            .entry(key)
            .or_insert_with(|| default.clone());
    }
    if let Some(rope) = value.get("rope_parameters").filter(|r| !r.is_null()) {
        let base = rope
            .get("rope_theta")
            .ok_or_else(|| invalid("rope_parameters requires rope_theta"))?;
        if value.get("rope_theta").is_some_and(|v| v != base) {
            return Err(invalid("conflicting rope_theta declarations"));
        }
        common["rope_theta"] = base.clone();
        if rope
            .get("rope_type")
            .and_then(Value::as_str)
            .unwrap_or("default")
            != "default"
        {
            common["rope_scaling"] = rope.clone();
        }
    }
    let mut dense =
        crate::llama::model_args_from_config_value(&common).map_err(|e| invalid(e.to_string()))?;
    if dense.rope_traditional {
        return Err(invalid("Gemma 2 uses split-half rotary encoding"));
    }
    if policy
        .sliding_window
        .is_some_and(|w| w == 0 || w > i32::MAX as u32)
    {
        return Err(invalid("sliding_window must be positive and fit i32"));
    }
    let layers = dense.num_hidden_layers as usize;
    let kinds = policy.layer_types.unwrap_or_else(|| {
        (0..layers)
            .map(|i| {
                if i % 2 == 0 {
                    "sliding_attention"
                } else {
                    "full_attention"
                }
                .into()
            })
            .collect()
    });
    if kinds.len() != layers {
        return Err(invalid("layer_types must contain one policy per layer"));
    }
    let schedule = kinds
        .iter()
        .map(|kind| match kind.as_str() {
            "full_attention" => Ok(AttentionPolicy::Full),
            "sliding_attention" => AttentionPolicy::sliding(
                policy
                    .sliding_window
                    .ok_or_else(|| invalid("sliding attention requires sliding_window"))?,
            )
            .map_err(|e| invalid(e.to_string())),
            _ => Err(invalid(format!("unknown layer type {kind:?}"))),
        })
        .collect::<Result<Vec<_>, _>>()?;
    dense.attention_schedule =
        LayerSchedule::new(layers, schedule).map_err(|e| invalid(e.to_string()))?;
    let args = ModelArgs {
        dense,
        query_pre_attn_scalar: policy.query_pre_attn_scalar,
        attn_logit_softcapping: policy.attn_logit_softcapping,
        final_logit_softcapping: policy.final_logit_softcapping,
        normalization_offset: 1.0,
    };
    args.validate()?;
    Ok(args)
}

/// Normalizes the published llama.cpp Gemma 2 metadata and scale convention.
pub fn model_args_from_gguf_catalog(
    arrays: &impl crate::GgufTensorCatalog,
    metadata: &HashMap<String, MetadataValue>,
) -> Result<ModelArgs, ConfigError> {
    if metadata
        .get("general.architecture")
        .and_then(MetadataValue::as_str)
        != Some("gemma2")
    {
        return Err(invalid("GGUF architecture must be gemma2"));
    }
    let mut common = metadata
        .iter()
        .map(|(k, v)| (k.replacen("gemma2.", "llama.", 1), v.clone()))
        .collect::<HashMap<_, _>>();
    common.insert(
        "general.architecture".into(),
        MetadataValue::String("llama".into()),
    );
    common.remove("llama.attention.sliding_window");
    let mut dense = crate::llama::model_args_from_gguf_catalog(arrays, &common)
        .map_err(|e| invalid(e.to_string()))?;
    if !dense.tie_word_embeddings || dense.attention_bias || dense.mlp_bias {
        return Err(invalid(
            "published Gemma 2 GGUF requires tied embeddings and bias-free projections",
        ));
    }
    let number = |key: &str, default: f32| -> Result<f32, ConfigError> {
        metadata
            .get(&format!("gemma2.{key}"))
            .map_or(Ok(default), |v| {
                v.as_f32()
                    .ok_or_else(|| invalid(format!("invalid GGUF {key}")))
            })
    };
    let integer = |key: &str, default: u32| -> Result<u32, ConfigError> {
        metadata
            .get(&format!("gemma2.{key}"))
            .map_or(Ok(default), |v| {
                v.as_i64()
                    .and_then(|v| u32::try_from(v).ok())
                    .ok_or_else(|| invalid(format!("invalid GGUF {key}")))
            })
    };
    for field in ["attention.value_length", "rope.dimension_count"] {
        if integer(field, dense.head_dim as u32)? != dense.head_dim as u32 {
            return Err(invalid(format!(
                "{field} must equal the full attention head dimension"
            )));
        }
    }
    if number("rope.freq_base_swa", dense.rope_theta)? != dense.rope_theta {
        return Err(invalid(
            "Gemma 2 uses the same rotary base in sliding and full layers",
        ));
    }
    let window = integer("attention.sliding_window", 4096)?;
    if window == 0 || window > i32::MAX as u32 {
        return Err(invalid("sliding window must be positive and fit i32"));
    }
    let period = integer("attention.sliding_window_pattern", 2)?;
    if period == 0 {
        return Err(invalid("sliding window period must be positive"));
    }
    dense.attention_schedule = LayerSchedule::new(
        dense.num_hidden_layers as usize,
        (0..dense.num_hidden_layers as u32)
            .map(|i| {
                if (i + 1) % period == 0 {
                    Ok(AttentionPolicy::Full)
                } else {
                    AttentionPolicy::sliding(window).map_err(|e| invalid(e.to_string()))
                }
            })
            .collect::<Result<Vec<_>, _>>()?,
    )
    .map_err(|e| invalid(e.to_string()))?;
    dense.rope_traditional = false;
    let scalar = if dense.num_hidden_layers == 46 {
        (dense.hidden_size / dense.num_attention_heads) as f32
    } else {
        dense.head_dim as f32
    };
    let args = ModelArgs {
        dense,
        query_pre_attn_scalar: scalar,
        attn_logit_softcapping: Some(number("attn_logit_softcapping", 50.0)?),
        final_logit_softcapping: Some(number("final_logit_softcapping", 30.0)?),
        normalization_offset: 0.0,
    };
    args.validate()?;
    Ok(args)
}

impl Config for ModelArgs {
    fn model_family(&self) -> &'static str {
        "gemma2"
    }
    fn model_identity(&self) -> &str {
        "gemma2"
    }
    fn architecture_fingerprint(&self) -> String {
        prompt_cache_architecture_fingerprint(self)
    }
    fn validate_config(&self) -> Result<(), Error> {
        self.validate().map_err(Error::backend)
    }
    fn hidden_size(&self) -> i32 {
        self.dense.hidden_size
    }
    fn num_hidden_layers(&self) -> i32 {
        self.dense.num_hidden_layers
    }
    fn intermediate_size(&self) -> i32 {
        self.dense.intermediate_size
    }
    fn num_attention_heads(&self) -> i32 {
        self.dense.num_attention_heads
    }
    fn num_key_value_heads(&self) -> i32 {
        self.dense.num_key_value_heads
    }
    fn head_dim(&self) -> i32 {
        self.dense.head_dim
    }
    fn rms_norm_epsilon(&self) -> f32 {
        self.dense.rms_norm_eps
    }
    fn vocabulary_size(&self) -> i32 {
        self.dense.vocab_size
    }
    fn attention_bias(&self, _: AttentionProjection) -> bool {
        self.dense.attention_bias
    }
    fn mlp_bias(&self) -> bool {
        false
    }
    fn tie_word_embeddings(&self) -> bool {
        self.dense.tie_word_embeddings
    }
    fn attention_schedule(&self) -> &LayerSchedule<AttentionPolicy> {
        &self.dense.attention_schedule
    }
    fn weight_quantization(&self, name: &str) -> Option<WeightQuantization> {
        self.weight_quantization_for(name)
    }
    fn rotary_spec(&self, dimensions: i32) -> RotarySpec {
        self.dense.rotary_spec(dimensions)
    }
    fn block_parameter_fields(&self) -> BlockParameterFields<'_> {
        BlockParameterFields {
            post_attention_norm: "pre_feedforward_layernorm",
            ..Default::default()
        }
    }
    fn normalization_offset(&self) -> f32 {
        self.normalization_offset
    }
    fn attention_output_normalization(&self, layer: usize) -> Option<String> {
        Some(format!(
            "model.layers.{layer}.post_attention_layernorm.weight"
        ))
    }
    fn feed_forward_output_normalization(&self, layer: usize) -> Option<String> {
        Some(format!(
            "model.layers.{layer}.post_feedforward_layernorm.weight"
        ))
    }
    fn embedding_scale(&self) -> f32 {
        (self.dense.hidden_size as f32).sqrt()
    }
    fn output_softcap(&self) -> Option<f32> {
        self.final_logit_softcapping
    }
    fn attention_scale(&self) -> f32 {
        self.query_pre_attn_scalar.sqrt().recip()
    }
    fn attention_softcap(&self) -> Option<f32> {
        self.attn_logit_softcapping
    }
    fn gated_product_policy(&self) -> Option<GatedProductPolicy> {
        Some(GatedProductPolicy::ordinary_gelu_approximate())
    }
}
impl crate::decoder::PartitionedConfig for ModelArgs {
    fn set_local_geometry(&mut self, query: i32, kv: i32, intermediate: i32) -> Result<(), Error> {
        self.dense.set_local_geometry(query, kv, intermediate)
    }
}
/// Complete shared-decoder execution, including bounded residency.
pub type LayeredModel<B> = crate::decoder::LayeredModel<B, ModelArgs>;
/// TP/PP execution using precisely owned blocks and rank-local parameters.
pub type PartitionedLayeredModel<B> = crate::decoder::PartitionedLayeredModel<B, ModelArgs>;
pub use crate::decoder::{dense_parameter_description, state_layout};

/// Identity includes all equation, schedule, format and normalization policies.
pub fn prompt_cache_architecture_fingerprint(args: &ModelArgs) -> String {
    derive_prompt_cache_architecture_fingerprint(
        "gemma2",
        [(
            "equations",
            format!(
                "gemma2:v1:{}:{:08x}:{:?}:{:?}:{:08x}",
                crate::llama::prompt_cache_architecture_fingerprint(&args.dense),
                args.query_pre_attn_scalar.to_bits(),
                args.attn_logit_softcapping.map(f32::to_bits),
                args.final_logit_softcapping.map(f32::to_bits),
                args.normalization_offset.to_bits()
            ),
        )],
    )
}
/// Applies the exact selected matrix encodings without changing model equations.
pub fn with_checkpoint_formats(
    args: &ModelArgs,
    formats: HashMap<String, WeightQuantization>,
) -> Result<ModelArgs, String> {
    let mut target = args.clone();
    target.dense = crate::llama::with_checkpoint_formats(&args.dense, formats)?;
    Ok(target)
}
/// Exact SafeTensors catalog, reusing dense projection and quantization contracts.
pub fn safetensors_plan(args: &ModelArgs) -> Result<SafetensorsCheckpointPlan, String> {
    let mut plan = crate::llama::safetensors_plan(&args.dense).map_err(|e| e.to_string())?;
    for tensor in &mut plan.common_tensors {
        tensor.key = tensor
            .key
            .replace(".post_attention_layernorm.", ".pre_feedforward_layernorm.");
    }
    for layer in 0..args.dense.num_hidden_layers {
        for field in ["post_attention_layernorm", "post_feedforward_layernorm"] {
            plan.common_tensors
                .push(SafetensorsTensorConstraint::required(
                    format!("model.layers.{layer}.{field}.weight"),
                    vec![args.dense.hidden_size as usize],
                    StoredDtypeConstraint::Floating,
                ));
        }
    }
    SafetensorsCheckpointPlan::new(
        "Gemma 2 SafeTensors",
        plan.common_tensors,
        plan.layout_groups,
        plan.catalog_policy,
    )
    .map_err(|e| e.to_string())
}
/// Exact published GGUF tensor catalog, including all four sublayer norms.
pub fn gguf_plan(args: &ModelArgs) -> Result<GgufCheckpointPlan, String> {
    let mut plan = crate::llama::gguf_plan(&args.dense)?;
    for layer in 0..args.dense.num_hidden_layers {
        for field in ["attn_post_norm", "ffn_post_norm"] {
            plan.common_tensors.push(GgufTensorConstraint::required(
                format!("blk.{layer}.{field}.weight"),
                vec![args.dense.hidden_size as usize],
                GgufTypeConstraint::OperationClass(TensorOperation::Vector),
            ));
        }
    }
    GgufCheckpointPlan::new(
        "Gemma 2 GGUF",
        plan.common_tensors,
        plan.layout_groups,
        plan.catalog_policy,
    )
    .map_err(|e| e.to_string())
}
/// Translates only the architecture-owned checkpoint vocabulary.
pub fn translate_gguf_weight_name(name: &str) -> String {
    let name = name
        .replace("attn_post_norm", "post_attention_layernorm")
        .replace("ffn_post_norm", "post_feedforward_layernorm")
        .replace("ffn_norm", "pre_feedforward_layernorm");
    crate::llama::translate_gguf_weight_name(&name)
}

#[cfg(test)]
mod tests;
