use std::collections::{HashMap, HashSet};
use std::io::Read;

use eredu_checkpoint::WeightQuantization;
use eredu_core::cache::derive_prompt_cache_architecture_fingerprint;
use eredu_core::{AttentionPolicy, LayerSchedule};
use eredu_gguf::MetadataValue;
use eredu_nn::{RopeValue, RotarySpec};
use serde::Deserialize;
use serde_json::Value;

use super::Config;

/// Invalid Llama/Mistral model configuration.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// Configuration JSON could not be read or decoded.
    #[error("invalid Llama-compatible configuration: {0}")]
    Json(#[from] serde_json::Error),
    /// Configuration describes unsupported or inconsistent geometry.
    #[error("{0}")]
    Invalid(String),
}

/// Normalized Llama/Mistral decoder configuration used by every backend.
#[derive(Debug, Clone)]
pub struct ModelArgs {
    /// Model family identifier.
    pub model_type: String,
    /// Transformer hidden size.
    pub hidden_size: i32,
    /// Number of decoder layers.
    pub num_hidden_layers: i32,
    /// SwiGLU intermediate width.
    pub intermediate_size: i32,
    /// Number of query attention heads.
    pub num_attention_heads: i32,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// Token vocabulary size.
    pub vocab_size: i32,
    /// Number of key/value attention heads.
    pub num_key_value_heads: i32,
    /// Maximum configured position count.
    pub max_position_embeddings: i32,
    /// RoPE base frequency.
    pub rope_theta: f32,
    /// Whether RoPE rotates adjacent pairs.
    pub rope_traditional: bool,
    /// Per-head attention width.
    pub head_dim: i32,
    /// Whether output projection shares the embedding table.
    pub tie_word_embeddings: bool,
    /// Whether attention projections own biases.
    pub attention_bias: bool,
    /// Whether MLP projections own biases.
    pub mlp_bias: bool,
    /// Optional normalized RoPE scaling metadata.
    pub rope_scaling: Option<HashMap<String, RopeValue>>,
    /// Exact ordered attention policy for every decoder layer.
    pub attention_schedule: LayerSchedule<AttentionPolicy>,
    /// Preferred model-wide stored weight encoding.
    pub quantization: Option<WeightQuantization>,
    /// Hugging Face-compatible alias for `quantization`.
    pub quantization_config: Option<WeightQuantization>,
    /// Exact canonical names using the model-wide encoding.
    pub quantized_weights: Option<HashSet<String>>,
    /// Exact per-parameter encodings for mixed checkpoints.
    pub quantized_weight_configs: Option<HashMap<String, WeightQuantization>>,
}

#[derive(Debug, Clone, Deserialize)]
struct ModelArgsSource {
    model_type: String,
    hidden_size: i32,
    num_hidden_layers: i32,
    intermediate_size: i32,
    num_attention_heads: i32,
    rms_norm_eps: f32,
    vocab_size: i32,
    #[serde(default)]
    num_key_value_heads: i32,
    #[serde(default)]
    max_position_embeddings: i32,
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    #[serde(default)]
    rope_traditional: bool,
    #[serde(default)]
    head_dim: i32,
    #[serde(default = "default_true")]
    tie_word_embeddings: bool,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    mlp_bias: bool,
    rope_scaling: Option<HashMap<String, RopeValue>>,
    #[serde(default)]
    sliding_window: Option<Value>,
    #[serde(default)]
    quantization: Option<WeightQuantization>,
    #[serde(default)]
    quantization_config: Option<WeightQuantization>,
}

impl ModelArgs {
    /// Validates normalized geometry and supported RoPE configuration.
    pub fn validate(&self) -> Result<(), ConfigError> {
        validate_model_args(self)
    }

    /// Returns the model-wide checkpoint encoding, if packed.
    pub fn weight_quantization(&self) -> Option<WeightQuantization> {
        self.quantization.or(self.quantization_config)
    }

    /// Returns the physical encoding for one canonical parameter name.
    pub fn weight_quantization_for(&self, weight_name: &str) -> Option<WeightQuantization> {
        if let Some(config) = self
            .quantized_weight_configs
            .as_ref()
            .and_then(|configs| configs.get(weight_name))
        {
            return Some(*config);
        }
        let encoding = self.weight_quantization()?;
        match &self.quantized_weights {
            Some(names) if !names.contains(weight_name) => None,
            _ => Some(encoding),
        }
    }
}

impl Config for ModelArgs {
    fn model_identity(&self) -> &str {
        &self.model_type
    }
    fn architecture_fingerprint(&self) -> String {
        prompt_cache_architecture_fingerprint(self)
    }
    fn validate_config(&self) -> Result<(), eredu_nn::Error> {
        self.validate().map_err(eredu_nn::Error::backend)
    }
    fn hidden_size(&self) -> i32 {
        self.hidden_size
    }
    fn num_hidden_layers(&self) -> i32 {
        self.num_hidden_layers
    }
    fn intermediate_size(&self) -> i32 {
        self.intermediate_size
    }
    fn num_attention_heads(&self) -> i32 {
        self.num_attention_heads
    }
    fn num_key_value_heads(&self) -> i32 {
        self.num_key_value_heads
    }
    fn head_dim(&self) -> i32 {
        self.head_dim
    }
    fn rms_norm_epsilon(&self) -> f32 {
        self.rms_norm_eps
    }
    fn vocabulary_size(&self) -> i32 {
        self.vocab_size
    }
    fn attention_bias(&self, _projection: super::AttentionProjection) -> bool {
        self.attention_bias
    }
    fn mlp_bias(&self) -> bool {
        self.mlp_bias
    }
    fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings
    }
    fn attention_schedule(&self) -> &LayerSchedule<AttentionPolicy> {
        &self.attention_schedule
    }
    fn weight_quantization(&self, name: &str) -> Option<WeightQuantization> {
        self.weight_quantization_for(name)
    }
    fn rotary_spec(&self, dimensions: i32) -> RotarySpec<'_> {
        RotarySpec {
            dimensions,
            base: self.rope_theta,
            traditional: self.rope_traditional,
            max_positions: self.max_position_embeddings,
            scaling: self.rope_scaling.as_ref(),
        }
    }
}

/// Reads and normalizes a Hugging Face configuration document.
pub fn model_args_from_config_reader(reader: impl Read) -> Result<ModelArgs, ConfigError> {
    normalize_model_args(serde_json::from_reader(reader)?)
}

/// Parses and normalizes a Hugging Face configuration value.
pub fn model_args_from_config_value(config: &Value) -> Result<ModelArgs, ConfigError> {
    normalize_model_args(serde_json::from_value(config.clone())?)
}

/// Header-only GGUF tensor catalog used to infer optional architecture fields.
pub trait GgufTensorCatalog {
    /// Whether one exact physical tensor name is present.
    fn contains(&self, name: &str) -> bool;
    /// Whether any physical tensor name satisfies a predicate.
    fn any(&self, predicate: &mut dyn FnMut(&str) -> bool) -> bool;
}

/// Parses normalized Llama/Mistral arguments from a backend-neutral GGUF catalog.
pub fn model_args_from_gguf_catalog(
    arrays: &impl GgufTensorCatalog,
    metadata: &HashMap<String, MetadataValue>,
) -> Result<ModelArgs, ConfigError> {
    let architecture = gguf_string(metadata, "general.architecture")?;
    if !matches!(architecture.as_str(), "llama" | "mistral") {
        return Err(invalid(format!(
            "GGUF architecture {architecture:?}; expected llama or mistral"
        )));
    }
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let hidden_size = gguf_i32(metadata, &key("embedding_length"))?;
    let num_attention_heads = gguf_i32(metadata, &key("attention.head_count"))?;
    if num_attention_heads <= 0 {
        return Err(invalid(format!(
            "GGUF attention head count must be positive, got {num_attention_heads}"
        )));
    }
    let num_key_value_heads = gguf_optional_i64(metadata, &key("attention.head_count_kv"))?
        .map(i32::try_from)
        .transpose()
        .map_err(|_| invalid("GGUF KV-head count exceeds i32"))?
        .unwrap_or(num_attention_heads);
    let head_dim = gguf_optional_i64(metadata, &key("attention.key_length"))?
        .map(i32::try_from)
        .transpose()
        .map_err(|_| invalid("GGUF attention key length exceeds i32"))?
        .unwrap_or(hidden_size / num_attention_heads);
    let rope_theta = gguf_optional_f32(metadata, &key("rope.freq_base"))?.unwrap_or(10_000.0);
    let rope_scaling = gguf_rope_scaling(metadata, &architecture)?;
    let sliding_window = gguf_optional_i64(metadata, &key("attention.sliding_window"))?;
    let vocab_size = match metadata
        .get("tokenizer.ggml.tokens")
        .and_then(MetadataValue::as_strings)
    {
        Some(tokens) => i32::try_from(tokens.len())
            .map_err(|_| invalid("GGUF tokenizer vocabulary exceeds i32"))?,
        None if metadata.contains_key("tokenizer.ggml.tokens") => {
            return Err(invalid(
                "GGUF tokenizer.ggml.tokens metadata has the wrong type",
            ));
        }
        None => gguf_i32(metadata, &key("vocab_size"))?,
    };
    let num_hidden_layers = gguf_i32(metadata, &key("block_count"))?;
    let layer_count = usize::try_from(num_hidden_layers).map_err(|_| {
        invalid(format!(
            "GGUF block count must be positive, got {num_hidden_layers}"
        ))
    })?;
    let attention_schedule = match sliding_window {
        None | Some(0) => LayerSchedule::all_full(layer_count),
        Some(window) => LayerSchedule::all_sliding(
            layer_count,
            u32::try_from(window).map_err(|_| {
                invalid(format!(
                    "GGUF sliding-window size must be positive and fit u32, got {window}"
                ))
            })?,
        ),
    }
    .map_err(|error| invalid(error.to_string()))?;
    let mut attention_bias = |name: &str| {
        name.starts_with("blk.")
            && matches!(
                name.rsplit_once('.'),
                Some((prefix, "bias")) if prefix.ends_with("attn_q")
                    || prefix.ends_with("attn_k")
                    || prefix.ends_with("attn_v")
                    || prefix.ends_with("attn_output")
            )
    };
    let mut mlp_bias = |name: &str| {
        name.starts_with("blk.")
            && matches!(
                name.rsplit_once('.'),
                Some((prefix, "bias")) if prefix.ends_with("ffn_gate")
                    || prefix.ends_with("ffn_down")
                    || prefix.ends_with("ffn_up")
            )
    };
    let args = ModelArgs {
        model_type: architecture.clone(),
        hidden_size,
        num_hidden_layers,
        intermediate_size: gguf_i32(metadata, &key("feed_forward_length"))?,
        num_attention_heads,
        rms_norm_eps: gguf_f32(metadata, &key("attention.layer_norm_rms_epsilon"))?,
        vocab_size,
        num_key_value_heads,
        max_position_embeddings: gguf_i32(metadata, &key("context_length"))?,
        rope_theta,
        rope_traditional: true,
        head_dim,
        tie_word_embeddings: !arrays.contains("output.weight"),
        attention_bias: arrays.any(&mut attention_bias),
        mlp_bias: arrays.any(&mut mlp_bias),
        rope_scaling,
        attention_schedule,
        quantization: None,
        quantization_config: None,
        quantized_weights: None,
        quantized_weight_configs: None,
    };
    args.validate()?;
    Ok(args)
}

fn gguf_rope_scaling(
    metadata: &HashMap<String, MetadataValue>,
    architecture: &str,
) -> Result<Option<HashMap<String, RopeValue>>, ConfigError> {
    let scaling_type_key = format!("{architecture}.rope.scaling.type");
    let Some(scaling_type) = gguf_optional_string(metadata, &scaling_type_key)? else {
        return Ok(None);
    };
    match scaling_type.as_str() {
        "none" | "default" => Ok(None),
        "linear" => {
            let factor_key = format!("{architecture}.rope.scaling.factor");
            let factor = gguf_optional_f32(metadata, &factor_key)?.ok_or_else(|| {
                invalid(format!("linear GGUF RoPE scaling is missing {factor_key}"))
            })?;
            Ok(Some(HashMap::from([
                ("rope_type".into(), RopeValue::String("linear".into())),
                ("factor".into(), RopeValue::Float(factor)),
            ])))
        }
        other => Err(invalid(format!(
            "GGUF RoPE scaling type {other:?} is unsupported"
        ))),
    }
}

fn gguf_string(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<String, ConfigError> {
    gguf_optional_string(metadata, key)?
        .ok_or_else(|| invalid(format!("GGUF metadata is missing required key {key:?}")))
}

fn gguf_optional_string(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<String>, ConfigError> {
    match metadata.get(key) {
        Some(MetadataValue::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(invalid(format!(
            "GGUF metadata key {key:?} has the wrong type"
        ))),
        None => Ok(None),
    }
}

fn gguf_i32(metadata: &HashMap<String, MetadataValue>, key: &str) -> Result<i32, ConfigError> {
    i32::try_from(gguf_i64(metadata, key)?)
        .map_err(|_| invalid(format!("GGUF metadata value {key:?} exceeds i32")))
}

fn gguf_i64(metadata: &HashMap<String, MetadataValue>, key: &str) -> Result<i64, ConfigError> {
    gguf_optional_i64(metadata, key)?
        .ok_or_else(|| invalid(format!("GGUF metadata is missing required key {key:?}")))
}

fn gguf_optional_i64(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<i64>, ConfigError> {
    match metadata.get(key) {
        Some(value) => value
            .as_i64()
            .map(Some)
            .ok_or_else(|| invalid(format!("GGUF metadata key {key:?} has the wrong type"))),
        None => Ok(None),
    }
}

fn gguf_f32(metadata: &HashMap<String, MetadataValue>, key: &str) -> Result<f32, ConfigError> {
    gguf_optional_f32(metadata, key)?
        .ok_or_else(|| invalid(format!("GGUF metadata is missing required key {key:?}")))
}

fn gguf_optional_f32(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<f32>, ConfigError> {
    match metadata.get(key) {
        Some(value) => value
            .as_f32()
            .map(Some)
            .ok_or_else(|| invalid(format!("GGUF metadata key {key:?} has the wrong type"))),
        None => Ok(None),
    }
}

fn normalize_model_args(mut source: ModelArgsSource) -> Result<ModelArgs, ConfigError> {
    if source.num_key_value_heads == 0 {
        source.num_key_value_heads = source.num_attention_heads;
    }
    if source.head_dim == 0 {
        if source.num_attention_heads <= 0 {
            return Err(invalid(format!(
                "num_attention_heads must be positive, got {}",
                source.num_attention_heads
            )));
        }
        source.head_dim = source.hidden_size / source.num_attention_heads;
    }
    if source.max_position_embeddings == 0 {
        source.max_position_embeddings = 2048;
    }
    let layer_count = usize::try_from(source.num_hidden_layers).map_err(|_| {
        invalid(format!(
            "num_hidden_layers must be positive, got {}",
            source.num_hidden_layers
        ))
    })?;
    let attention_schedule = match normalize_hf_sliding_window(source.sliding_window)? {
        None => LayerSchedule::all_full(layer_count),
        Some(window) => LayerSchedule::all_sliding(layer_count, window),
    }
    .map_err(|error| invalid(error.to_string()))?;
    let args = ModelArgs {
        model_type: source.model_type,
        hidden_size: source.hidden_size,
        num_hidden_layers: source.num_hidden_layers,
        intermediate_size: source.intermediate_size,
        num_attention_heads: source.num_attention_heads,
        rms_norm_eps: source.rms_norm_eps,
        vocab_size: source.vocab_size,
        num_key_value_heads: source.num_key_value_heads,
        max_position_embeddings: source.max_position_embeddings,
        rope_theta: source.rope_theta,
        rope_traditional: source.rope_traditional,
        head_dim: source.head_dim,
        tie_word_embeddings: source.tie_word_embeddings,
        attention_bias: source.attention_bias,
        mlp_bias: source.mlp_bias,
        rope_scaling: source.rope_scaling,
        attention_schedule,
        quantization: source.quantization,
        quantization_config: source.quantization_config,
        quantized_weights: None,
        quantized_weight_configs: None,
    };
    validate_model_args(&args)?;
    Ok(args)
}

fn normalize_hf_sliding_window(value: Option<Value>) -> Result<Option<u32>, ConfigError> {
    let Some(value) = value else { return Ok(None) };
    let Value::Number(number) = value else {
        return Err(invalid(format!(
            "sliding_window must be a positive integer or null, got {value}"
        )));
    };
    if let Some(window) = number.as_u64() {
        let window = u32::try_from(window)
            .map_err(|_| invalid(format!("sliding_window exceeds u32: {window}")))?;
        if window == 0 {
            return Err(invalid("sliding_window must be positive, got 0"));
        }
        return Ok(Some(window));
    }
    Err(invalid(format!(
        "sliding_window must be positive and use an integer encoding, got {number}"
    )))
}

fn validate_model_args(args: &ModelArgs) -> Result<(), ConfigError> {
    if !matches!(args.model_type.as_str(), "llama" | "mistral") {
        return Err(invalid(format!(
            "unsupported model type {:?}",
            args.model_type
        )));
    }
    for (name, value) in [
        ("hidden_size", args.hidden_size),
        ("num_hidden_layers", args.num_hidden_layers),
        ("intermediate_size", args.intermediate_size),
        ("num_attention_heads", args.num_attention_heads),
        ("num_key_value_heads", args.num_key_value_heads),
        ("vocab_size", args.vocab_size),
        ("max_position_embeddings", args.max_position_embeddings),
        ("head_dim", args.head_dim),
    ] {
        if value <= 0 {
            return Err(invalid(format!("{name} must be positive, got {value}")));
        }
    }
    if args.num_attention_heads % args.num_key_value_heads != 0 {
        return Err(invalid(format!(
            "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
            args.num_attention_heads, args.num_key_value_heads
        )));
    }
    for (name, heads) in [
        ("query projection", args.num_attention_heads),
        ("key/value projection", args.num_key_value_heads),
    ] {
        heads.checked_mul(args.head_dim).ok_or_else(|| {
            invalid(format!(
                "{name} width overflows i32: {heads} heads x head_dim {}",
                args.head_dim
            ))
        })?;
    }
    if args.attention_schedule.len() != args.num_hidden_layers as usize {
        return Err(invalid(format!(
            "Llama attention schedule has {} layers, expected {}",
            args.attention_schedule.len(),
            args.num_hidden_layers
        )));
    }
    if let Some(config) = &args.rope_scaling {
        validate_rope_scaling(config)?;
    }
    Ok(())
}

fn validate_rope_scaling(config: &HashMap<String, RopeValue>) -> Result<(), ConfigError> {
    let Some(value) = config.get("type").or_else(|| config.get("rope_type")) else {
        return Ok(());
    };
    let RopeValue::String(kind) = value else {
        return Err(invalid(
            "RoPE scaling field type or rope_type must be a string",
        ));
    };
    match kind.as_str() {
        "default" | "linear" | "llama3" | "proportional" | "yarn" => Ok(()),
        "longrope" => Err(invalid(
            "RoPE scaling type \"longrope\" is unsupported; LongRoPE is not implemented",
        )),
        other => Err(invalid(format!(
            "RoPE scaling type {other:?} is unsupported"
        ))),
    }
}

/// Returns the stable cache-compatibility fingerprint for this configuration.
pub fn prompt_cache_architecture_fingerprint(args: &ModelArgs) -> String {
    let rope_scaling = args.rope_scaling.as_ref().map_or_else(
        || "none".to_string(),
        |config| {
            let mut entries = config.iter().collect::<Vec<_>>();
            entries.sort_unstable_by_key(|(key, _)| key.as_str());
            entries
                .into_iter()
                .map(|(key, value)| {
                    let value = match value {
                        RopeValue::Float(value) => format!("f32:{:08x}", value.to_bits()),
                        RopeValue::String(value) => format!("string:{value}"),
                        RopeValue::Bool(value) => format!("bool:{value}"),
                    };
                    format!("{key}={value}")
                })
                .collect::<Vec<_>>()
                .join(";")
        },
    );
    let mut quantized_weights = args
        .quantized_weights
        .as_ref()
        .map(|weights| weights.iter().cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    quantized_weights.sort_unstable();
    let mut quantized_weight_configs = args
        .quantized_weight_configs
        .as_ref()
        .map(|configs| {
            configs
                .iter()
                .map(|(name, config)| format!("{name}={config:?}"))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    quantized_weight_configs.sort_unstable();
    derive_prompt_cache_architecture_fingerprint(
        "llama",
        [
            ("model_type", args.model_type.clone()),
            ("hidden_size", args.hidden_size.to_string()),
            ("num_hidden_layers", args.num_hidden_layers.to_string()),
            ("intermediate_size", args.intermediate_size.to_string()),
            ("num_attention_heads", args.num_attention_heads.to_string()),
            ("num_key_value_heads", args.num_key_value_heads.to_string()),
            ("head_dim", args.head_dim.to_string()),
            (
                "rms_norm_eps",
                format!("{:08x}", args.rms_norm_eps.to_bits()),
            ),
            ("vocab_size", args.vocab_size.to_string()),
            (
                "max_position_embeddings",
                args.max_position_embeddings.to_string(),
            ),
            ("rope_theta", format!("{:08x}", args.rope_theta.to_bits())),
            ("rope_traditional", args.rope_traditional.to_string()),
            ("rope_scaling", rope_scaling),
            (
                "attention_schedule",
                args.attention_schedule.fingerprint_component(),
            ),
            ("tie_word_embeddings", args.tie_word_embeddings.to_string()),
            ("attention_bias", args.attention_bias.to_string()),
            ("mlp_bias", args.mlp_bias.to_string()),
            ("quantization", format!("{:?}", args.weight_quantization())),
            ("quantized_weights", quantized_weights.join(";")),
            (
                "quantized_weight_configs",
                quantized_weight_configs.join(";"),
            ),
        ],
    )
}

fn invalid(message: impl Into<String>) -> ConfigError {
    ConfigError::Invalid(message.into())
}

const fn default_true() -> bool {
    true
}
const fn default_rope_theta() -> f32 {
    10_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Catalog(Vec<String>);

    impl GgufTensorCatalog for Catalog {
        fn contains(&self, name: &str) -> bool {
            self.0.iter().any(|candidate| candidate == name)
        }

        fn any(&self, predicate: &mut dyn FnMut(&str) -> bool) -> bool {
            self.0.iter().any(|name| predicate(name))
        }
    }

    #[test]
    fn normalizes_minimal_llama_config() {
        let value = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "rms_norm_eps": 0.00001,
            "vocab_size": 64
        });
        let args = model_args_from_config_value(&value).unwrap();
        assert_eq!(args.num_key_value_heads, 4);
        assert_eq!(args.head_dim, 4);
        assert_eq!(args.attention_schedule.len(), 2);
        assert_eq!(
            args.weight_quantization_for("model.layers.0.mlp.up_proj.weight"),
            None
        );
    }

    #[test]
    fn preparation_ignores_unrelated_nested_draft_field() {
        let value = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "rms_norm_eps": 0.00001,
            "vocab_size": 64,
            "unrelated": {"num_nextn_predict_layers": 7}
        });

        let capabilities =
            crate::preparation::safetensors_capabilities(crate::ModelKind::Llama, &value).unwrap();

        assert_eq!(capabilities.embedded_draft_layers(), Some(0));
    }

    #[test]
    fn normalizes_gguf_without_backend_types() {
        let metadata = HashMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String("llama".into()),
            ),
            ("llama.embedding_length".into(), MetadataValue::Uint32(16)),
            (
                "llama.attention.head_count".into(),
                MetadataValue::Uint32(4),
            ),
            (
                "llama.attention.head_count_kv".into(),
                MetadataValue::Uint32(2),
            ),
            ("llama.block_count".into(), MetadataValue::Uint32(2)),
            (
                "llama.feed_forward_length".into(),
                MetadataValue::Uint32(32),
            ),
            (
                "llama.attention.layer_norm_rms_epsilon".into(),
                MetadataValue::Float32(1e-5),
            ),
            ("llama.vocab_size".into(), MetadataValue::Uint32(64)),
            ("llama.context_length".into(), MetadataValue::Uint32(128)),
            (
                "llama.attention.sliding_window".into(),
                MetadataValue::Uint32(32),
            ),
        ]);
        let catalog = Catalog(vec!["output.weight".into(), "blk.0.attn_q.bias".into()]);
        let args = model_args_from_gguf_catalog(&catalog, &metadata).unwrap();
        assert_eq!(args.head_dim, 4);
        assert!(args.attention_bias);
        assert!(!args.mlp_bias);
        assert!(!args.tie_word_embeddings);
        assert_eq!(
            args.attention_schedule
                .get(0)
                .unwrap()
                .window()
                .unwrap()
                .get(),
            32
        );
    }
}
