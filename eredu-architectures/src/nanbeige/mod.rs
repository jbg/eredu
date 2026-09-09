//! Nanbeige 4.2 dense, shared-weight looped decoder.
//!
//! Checkpoint blocks use ordinary Llama-style GQA and SwiGLU geometry. Each
//! loop revisits those same parameters with independent attention state.

use std::{collections::HashMap, io::Read};

use eredu_checkpoint::WeightQuantization;
use eredu_core::{
    cache::derive_prompt_cache_architecture_fingerprint, AttentionPolicy, LayerSchedule,
};
use eredu_nn::{Error, RotarySpec};
use serde::Deserialize;
use serde_json::Value;

use crate::decoder::{AttentionProjection, Config};

#[cfg(test)]
mod tests;

/// Invalid or unsupported Nanbeige inference configuration.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// Invalid JSON encoding or field type.
    #[error("invalid Nanbeige configuration: {0}")]
    Json(#[from] serde_json::Error),
    /// Invalid geometry or an unsupported equation variant.
    #[error("invalid Nanbeige configuration: {0}")]
    Invalid(String),
}

/// Validated Nanbeige geometry and loop policy.
#[derive(Debug, Clone)]
pub struct ModelArgs {
    // Reuse the conventional dense block geometry and checkpoint encodings.
    // Its identity is internal; public model and cache identities are Nanbeige.
    dense: crate::llama::ModelArgs,
    num_loops: usize,
    skip_loop_final_norm: bool,
    attention_schedule: LayerSchedule<AttentionPolicy>,
}

impl ModelArgs {
    /// Geometry of the physical, shared decoder blocks.
    pub(crate) fn dense_config(&self) -> &crate::llama::ModelArgs {
        &self.dense
    }
    /// Number of physical checkpoint blocks.
    pub fn physical_layer_count(&self) -> usize {
        self.dense.num_hidden_layers as usize
    }
    /// Number of complete passes through the shared block stack.
    pub fn num_loops(&self) -> usize {
        self.num_loops
    }
    /// Whether normalization is deferred until after the last pass.
    pub fn skip_loop_final_norm(&self) -> bool {
        self.skip_loop_final_norm
    }
    /// Number of independent logical attention states.
    pub fn state_layer_count(&self) -> usize {
        self.dense.num_hidden_layers as usize * self.num_loops
    }
    /// Physical encoding of one checkpoint parameter.
    pub fn weight_quantization_for(&self, name: &str) -> Option<WeightQuantization> {
        self.dense.weight_quantization_for(name).or_else(|| {
            self.dense
                .weight_quantization_for(&crate::decoder::repeated::source_name(
                    "model",
                    self.physical_layer_count(),
                    name,
                ))
        })
    }
}

#[derive(Deserialize)]
struct LoopConfig {
    #[serde(default = "one")]
    num_loops: usize,
    #[serde(default)]
    skip_loop_final_norm: bool,
    #[serde(default)]
    loop_loss_weights: Vec<f64>,
}
fn one() -> usize {
    1
}

/// Reads a released Hugging Face configuration.
pub fn model_args_from_config_reader(reader: impl Read) -> Result<ModelArgs, ConfigError> {
    model_args_from_config_value(&serde_json::from_reader(reader)?)
}

/// Parses the supported dense looped Nanbeige inference architecture.
pub fn model_args_from_config_value(value: &Value) -> Result<ModelArgs, ConfigError> {
    let invalid = |message: String| ConfigError::Invalid(message);
    if value.get("model_type").and_then(Value::as_str) != Some("nanbeige") {
        return Err(invalid("model_type must be nanbeige".into()));
    }
    let loops: LoopConfig = serde_json::from_value(value.clone())?;
    if loops.num_loops == 0 {
        return Err(invalid("num_loops must be positive".into()));
    }
    for field in ["head_dim", "num_key_value_heads", "max_position_embeddings"] {
        if value
            .get(field)
            .is_some_and(|v| !v.as_i64().is_some_and(|n| n > 0))
        {
            return Err(invalid(format!("{field} must be a positive integer")));
        }
    }
    if !loops.loop_loss_weights.is_empty() {
        return Err(invalid(
            "nonempty loop_loss_weights changes inference loop selection and is unsupported".into(),
        ));
    }
    // These upstream experiments change equations, parameters or cache sharing;
    // accepting them as the released dense family would silently misexecute.
    for field in [
        "qk_layernorm",
        "enable_double_loop_split",
        "loop_share_kv",
        "mhc_diff_for_loop",
        "enable_hyper_connection",
        "enable_mhc",
        "enable_h_res_identity",
        "mhc_identity_nohresparam",
        "enable_depth_attention",
        "ngram_insert_all_layers",
    ] {
        if let Some(v) = value.get(field) {
            if v.as_bool() != Some(false) {
                return Err(invalid(format!(
                    "{field} must be false for the supported dense architecture"
                )));
            }
        }
    }
    for field in [
        "emb_neighbor_num",
        "emb_split_num",
        "ngram_vocab_size_ratio",
        "ngram_embedding_hidden_size",
        "insert_ngram_layer_idx",
        "loop_middle_layers",
        "mhc_double_stream_position_for_loop",
    ] {
        if value.get(field).is_some_and(|v| !v.is_null()) {
            return Err(invalid(format!("{field} is unsupported")));
        }
    }
    if value
        .get("hidden_act")
        .is_some_and(|v| v.as_str() != Some("silu"))
    {
        return Err(invalid("hidden_act must be silu".into()));
    }
    if value
        .get("pretraining_tp")
        .is_some_and(|v| v.as_u64() != Some(1))
    {
        return Err(invalid("pretraining_tp must be 1".into()));
    }
    if value.get("sliding_window").is_some_and(|v| !v.is_null()) {
        return Err(invalid("sliding_window is unsupported".into()));
    }
    if value
        .get("rope_traditional")
        .is_some_and(|v| v.as_bool() != Some(false))
    {
        return Err(invalid("Nanbeige uses split-half rotary encoding".into()));
    }
    if let Some(rope) = value.get("rope_scaling").filter(|v| !v.is_null()) {
        if rope
            .get("type")
            .or_else(|| rope.get("rope_type"))
            .and_then(Value::as_str)
            != Some("linear")
        {
            return Err(invalid(
                "only unscaled or linear Nanbeige RoPE is supported".into(),
            ));
        }
    }
    let mut dense_value = value.clone();
    dense_value["model_type"] = Value::String("llama".into());
    let object = dense_value.as_object_mut().expect("model_type was present");
    object
        .entry("tie_word_embeddings")
        .or_insert(Value::Bool(false));
    let dense = crate::llama::model_args_from_config_value(&dense_value)
        .map_err(|e| invalid(e.to_string()))?;
    let state_count = (dense.num_hidden_layers as usize)
        .checked_mul(loops.num_loops)
        .filter(|&n| n <= i32::MAX as usize)
        .ok_or_else(|| invalid("num_hidden_layers * num_loops exceeds i32".into()))?;
    if state_count == 0 || dense.head_dim % 2 != 0 {
        return Err(invalid(
            "head_dim must be even and state count positive".into(),
        ));
    }
    if !dense.rms_norm_eps.is_finite()
        || dense.rms_norm_eps <= 0.0
        || !dense.rope_theta.is_finite()
        || dense.rope_theta <= 0.0
    {
        return Err(invalid(
            "rms_norm_eps and rope_theta must be finite and positive".into(),
        ));
    }
    if value
        .get("kv_channels")
        .is_some_and(|v| v.as_i64() != Some(i64::from(dense.head_dim)))
    {
        return Err(invalid("kv_channels must agree with head_dim".into()));
    }
    let attention_schedule =
        crate::decoder::repeated::attention_schedule(&dense.attention_schedule, loops.num_loops)
            .map_err(invalid)?;
    Ok(ModelArgs {
        attention_schedule,
        dense,
        num_loops: loops.num_loops,
        skip_loop_final_norm: loops.skip_loop_final_norm,
    })
}

/// Reads the publisher's GGUF format: physical block_count plus num_loops.
pub fn model_args_from_gguf_catalog(
    arrays: &impl crate::GgufTensorCatalog,
    metadata: &HashMap<String, eredu_gguf::MetadataValue>,
) -> Result<ModelArgs, ConfigError> {
    use eredu_gguf::MetadataValue;
    if metadata
        .get("general.architecture")
        .and_then(MetadataValue::as_str)
        != Some("nanbeige")
    {
        return Err(ConfigError::Invalid(
            "expected nanbeige GGUF architecture".into(),
        ));
    }
    let num_loops = match metadata.get("nanbeige.num_loops") {
        None => 1,
        Some(v) => v
            .as_i64()
            .and_then(|n| usize::try_from(n).ok())
            .filter(|&n| n > 0)
            .ok_or_else(|| ConfigError::Invalid("GGUF num_loops must be positive".into()))?,
    };
    let skip_loop_final_norm = match metadata.get("nanbeige.skip_loop_final_norm") {
        None => false,
        Some(MetadataValue::Bool(value)) => *value,
        _ => {
            return Err(ConfigError::Invalid(
                "GGUF skip_loop_final_norm must be boolean".into(),
            ))
        }
    };
    let mut dense_metadata = metadata.clone();
    for (key, value) in metadata {
        if let Some(suffix) = key.strip_prefix("nanbeige.") {
            dense_metadata.insert(format!("llama.{suffix}"), value.clone());
        }
    }
    dense_metadata.insert(
        "general.architecture".into(),
        MetadataValue::String("llama".into()),
    );
    let dense = crate::llama::model_args_from_gguf_catalog(arrays, &dense_metadata)
        .map_err(|e| ConfigError::Invalid(e.to_string()))?;
    // The publisher permutes Q/K rows in GGUF, matching adjacent-pair RoPE.
    let attention_schedule =
        crate::decoder::repeated::attention_schedule(&dense.attention_schedule, num_loops)
            .map_err(ConfigError::Invalid)?;
    Ok(ModelArgs {
        dense,
        num_loops,
        skip_loop_final_norm,
        attention_schedule,
    })
}

/// Exact physical checkpoint schema; shared blocks occur only once.
pub fn safetensors_plan(
    args: &ModelArgs,
) -> Result<eredu_checkpoint::schema::SafetensorsCheckpointPlan, crate::llama::SafetensorsPlanError>
{
    crate::llama::safetensors_plan(&args.dense)
}

/// Selects exact matrix encodings without changing the loop or cache policy.
pub fn with_checkpoint_formats(
    args: &ModelArgs,
    formats: HashMap<String, WeightQuantization>,
) -> Result<ModelArgs, String> {
    let mut result = args.clone();
    result.dense = crate::llama::with_checkpoint_formats(&args.dense, formats)?;
    Ok(result)
}

/// Stable cache identity including every pass and inter-pass normalization.
pub fn prompt_cache_architecture_fingerprint(args: &ModelArgs) -> String {
    derive_prompt_cache_architecture_fingerprint(
        "nanbeige",
        [
            (
                "dense",
                crate::llama::prompt_cache_architecture_fingerprint(&args.dense),
            ),
            ("num_loops", args.num_loops.to_string()),
            (
                "skip_loop_final_norm",
                args.skip_loop_final_norm.to_string(),
            ),
        ],
    )
}

impl Config for ModelArgs {
    fn model_family(&self) -> &'static str {
        "nanbeige"
    }
    fn model_identity(&self) -> &str {
        "nanbeige"
    }
    fn architecture_fingerprint(&self) -> String {
        prompt_cache_architecture_fingerprint(self)
    }
    fn validate_config(&self) -> Result<(), Error> {
        self.dense.validate_config()
    }
    fn hidden_size(&self) -> i32 {
        self.dense.hidden_size()
    }
    fn num_hidden_layers(&self) -> i32 {
        self.state_layer_count() as i32
    }
    fn block_output_normalization(&self, layer: usize) -> Option<String> {
        (!self.skip_loop_final_norm
            && (layer + 1) % self.physical_layer_count() == 0
            && layer + 1 < self.state_layer_count())
        .then(|| format!("model.layers.{layer}.output_norm.weight"))
    }
    fn intermediate_size(&self) -> i32 {
        self.dense.intermediate_size()
    }
    fn num_attention_heads(&self) -> i32 {
        self.dense.num_attention_heads()
    }
    fn num_key_value_heads(&self) -> i32 {
        self.dense.num_key_value_heads()
    }
    fn head_dim(&self) -> i32 {
        self.dense.head_dim()
    }
    fn rms_norm_epsilon(&self) -> f32 {
        self.dense.rms_norm_epsilon()
    }
    fn vocabulary_size(&self) -> i32 {
        self.dense.vocabulary_size()
    }
    fn attention_bias(&self, projection: AttentionProjection) -> bool {
        self.dense.attention_bias(projection)
    }
    fn mlp_bias(&self) -> bool {
        self.dense.mlp_bias()
    }
    fn tie_word_embeddings(&self) -> bool {
        self.dense.tie_word_embeddings()
    }
    fn attention_schedule(&self) -> &LayerSchedule<AttentionPolicy> {
        &self.attention_schedule
    }
    fn weight_quantization(&self, name: &str) -> Option<WeightQuantization> {
        self.weight_quantization_for(name)
    }
    fn rotary_spec(&self, dimensions: i32) -> RotarySpec {
        self.dense.rotary_spec(dimensions)
    }
}

impl crate::decoder::PartitionedConfig for ModelArgs {
    fn set_local_geometry(
        &mut self,
        query_heads: i32,
        key_value_heads: i32,
        intermediate: i32,
    ) -> Result<(), Error> {
        self.dense
            .set_local_geometry(query_heads, key_value_heads, intermediate)
    }
}

/// Shared dense execution with logical block invocations.
pub type LayeredModel<B> = crate::decoder::LayeredModel<B, ModelArgs>;

/// Independent attention state for every logical layer invocation.
pub fn state_layout(args: &ModelArgs) -> Result<eredu_runtime::StateLayout, Error> {
    eredu_runtime::StateLayout::new(crate::decoder::cache_layout(args)?).map_err(Error::backend)
}
