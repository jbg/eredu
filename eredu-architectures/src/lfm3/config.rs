//! Strict, backend-independent LFM3 configuration normalization.

use std::{
    collections::{HashMap, HashSet},
    io::Read,
};

use crate::GgufTensorCatalog;
use eredu_checkpoint::WeightQuantization;
use eredu_core::{
    cache::derive_prompt_cache_architecture_fingerprint, AttentionPolicy, LayerSchedule,
};
use eredu_gguf::MetadataValue;
use serde::Deserialize;
use serde_json::Value;

const DEFAULT_ROPE_THETA: f32 = 1_000_000.0;

fn default_true() -> bool {
    true
}

fn default_rope_theta() -> f32 {
    DEFAULT_ROPE_THETA
}

fn default_conv_l_cache() -> i32 {
    7
}

fn default_norm_eps() -> f32 {
    1e-5
}

/// Invalid or unsupported LFM3 configuration.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// Configuration JSON could not be decoded.
    #[error("invalid LFM3 configuration: {0}")]
    Json(#[from] serde_json::Error),
    /// Configuration changes unsupported execution semantics or has invalid geometry.
    #[error("{0}")]
    Invalid(String),
}

/// Stateful token-mixing operator used by one LFM3 layer.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum OperatorPolicy {
    /// Gated causal depthwise short convolution.
    CausalConvolution,
    /// Grouped-query self attention with its exact retention policy.
    SelfAttention(AttentionPolicy),
}

impl OperatorPolicy {
    fn parse(value: &str, window: u32) -> Result<Self, ConfigError> {
        match value {
            "conv" => Ok(Self::CausalConvolution),
            "full_attention" => Ok(Self::SelfAttention(AttentionPolicy::Full)),
            "sliding_attention" => AttentionPolicy::sliding(window)
                .map(Self::SelfAttention)
                .map_err(|e| invalid(e.to_string())),
            other => Err(invalid(format!("LFM3 layer type {other:?} is unsupported"))),
        }
    }
}

/// Feed-forward operator used by one LFM3 layer.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum FeedForwardPolicy {
    /// Dense SwiGLU feed-forward block.
    Dense,
}

/// Complete execution policy for one LFM3 decoder layer.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub struct LayerPolicy {
    /// Stateful token-mixing operator.
    pub operator: OperatorPolicy,
    /// Feed-forward operator.
    pub feed_forward: FeedForwardPolicy,
}

/// Canonical rotary-position configuration used by LFM3 attention layers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeConfig {
    /// RoPE frequency base.
    pub theta: f32,
    /// Fraction of global attention frequency pairs with nonzero rotation.
    pub partial_rotary_factor: f32,
    /// Independent frequency base for sliding attention.
    pub sliding_theta: f32,
}

/// Validated LFM3 dense tri-hybrid decoder configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelArgs {
    /// Hugging Face model type (`lfm3`).
    pub model_type: String,
    /// Vocabulary size.
    pub vocab_size: i32,
    /// Hidden dimension.
    pub hidden_size: i32,
    /// Exact intermediate size used by dense feed-forward layers.
    pub dense_intermediate_size: i32,
    /// Number of decoder layers.
    pub num_hidden_layers: i32,
    /// Number of query heads.
    pub num_attention_heads: i32,
    /// Number of key/value heads.
    pub num_key_value_heads: i32,
    /// Maximum configured context length.
    pub max_position_embeddings: i32,
    /// RMSNorm epsilon.
    pub norm_eps: f32,
    /// Authoritative operator and feed-forward policy in decoder-layer order.
    pub layer_schedule: LayerSchedule<LayerPolicy>,
    /// Causal convolution kernel width.
    pub conv_l_cache: i32,
    /// Whether convolution projections and kernels include biases.
    pub conv_bias: bool,
    /// Whether logits use tied input embeddings.
    pub tie_word_embeddings: bool,
    /// Canonical rotary-position configuration.
    pub rope: RopeConfig,
    /// Apply output normalization before each residual addition.
    pub use_sandwich_norm: bool,
    /// Checkpoint-wide weight quantization, when present.
    pub weight_quantization: Option<WeightQuantization>,
    /// Exact mixed-quantization weight names populated by GGUF loading.
    pub quantized_weights: Option<HashSet<String>>,
    /// Per-weight affine layouts populated by GGUF loading.
    pub quantized_weight_configs: Option<HashMap<String, WeightQuantization>>,
}

#[derive(Deserialize)]
struct ModelArgsSource {
    model_type: String,
    vocab_size: i32,
    hidden_size: i32,
    intermediate_size: i32,
    num_hidden_layers: i32,
    num_attention_heads: i32,
    num_key_value_heads: i32,
    max_position_embeddings: i32,
    #[serde(default = "default_norm_eps")]
    norm_eps: f32,
    layer_types: Vec<String>,
    #[serde(default = "default_conv_l_cache", rename = "conv_L_cache")]
    conv_l_cache: i32,
    #[serde(default)]
    conv_bias: bool,
    #[serde(default = "default_true")]
    tie_word_embeddings: bool,
    rope_parameters: Value,
    #[serde(default = "default_true")]
    use_sandwich_norm: bool,
    #[serde(default)]
    sliding_window: Option<u32>,
    #[serde(default)]
    head_dim: Option<i32>,
    #[serde(default = "default_true")]
    is_causal: bool,
    #[serde(default)]
    quantization: Option<WeightQuantization>,
    #[serde(default)]
    quantization_config: Option<WeightQuantization>,
}
impl ModelArgsSource {
    fn normalize(self, causal: bool) -> Result<ModelArgs, ConfigError> {
        if self.is_causal != causal {
            return Err(invalid(
                "LFM3 attention direction differs from the declared task",
            ));
        }
        let layer_count = usize::try_from(self.num_hidden_layers)
            .map_err(|_| invalid("invalid LFM3 layer count"))?;
        let policies = self
            .layer_types
            .iter()
            .map(|value| {
                Ok(LayerPolicy {
                    operator: OperatorPolicy::parse(
                        value,
                        self.sliding_window.unwrap_or_default(),
                    )?,
                    feed_forward: FeedForwardPolicy::Dense,
                })
            })
            .collect::<Result<Vec<_>, ConfigError>>()?;
        let layer_schedule =
            LayerSchedule::new(layer_count, policies).map_err(|e| invalid(e.to_string()))?;
        if self.head_dim.is_some_and(|head| {
            self.num_attention_heads.checked_mul(head) != Some(self.hidden_size)
        }) {
            return Err(invalid(
                "LFM3 explicit head_dim conflicts with attention geometry",
            ));
        }
        let weight_quantization = match (self.quantization, self.quantization_config) {
            (Some(a), Some(b)) if a != b => {
                return Err(invalid("LFM3 quantization declarations disagree"))
            }
            (Some(a), _) | (_, Some(a)) => Some(a),
            _ => None,
        };
        Ok(ModelArgs {
            model_type: self.model_type,
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            dense_intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            max_position_embeddings: self.max_position_embeddings,
            norm_eps: self.norm_eps,
            layer_schedule,
            conv_l_cache: self.conv_l_cache,
            conv_bias: self.conv_bias,
            tie_word_embeddings: self.tie_word_embeddings,
            rope: parse_rope(Some(&self.rope_parameters))?,
            use_sandwich_norm: self.use_sandwich_norm,
            weight_quantization,
            quantized_weights: None,
            quantized_weight_configs: None,
        })
    }
}

fn parse_rope(value: Option<&Value>) -> Result<RopeConfig, ConfigError> {
    let value = value.ok_or_else(|| invalid("LFM3 requires layer-specific rope_parameters"))?;
    let parameters = value
        .as_object()
        .ok_or_else(|| invalid("LFM3 rope_parameters must be an object"))?;
    if parameters
        .keys()
        .any(|key| !matches!(key.as_str(), "full_attention" | "sliding_attention"))
    {
        return Err(invalid("unsupported LFM3 rotary layer policy"));
    }
    let parse = |kind: &str, default_fraction: f32| -> Result<(f32, f32), ConfigError> {
        let v = value
            .get(kind)
            .ok_or_else(|| invalid(format!("LFM3 requires {kind} rotary parameters")))?;
        let parameters = v
            .as_object()
            .ok_or_else(|| invalid("LFM3 rotary layer policy must be an object"))?;
        if parameters.keys().any(|key| {
            !matches!(
                key.as_str(),
                "rope_type" | "rope_theta" | "partial_rotary_factor"
            )
        }) {
            return Err(invalid("unsupported LFM3 rotary scaling policy"));
        }
        if v.get("rope_type").and_then(Value::as_str) != Some("default") {
            return Err(invalid(
                "LFM3 rotary parameters require default frequency progression",
            ));
        }
        let theta = v
            .get("rope_theta")
            .and_then(Value::as_f64)
            .ok_or_else(|| invalid("LFM3 requires rope_theta"))? as f32;
        let fraction = v
            .get("partial_rotary_factor")
            .and_then(Value::as_f64)
            .unwrap_or(f64::from(default_fraction)) as f32;
        if !theta.is_finite()
            || theta <= 0.0
            || !fraction.is_finite()
            || fraction <= 0.0
            || fraction > 1.0
        {
            return Err(invalid("invalid LFM3 rotary frequency geometry"));
        }
        Ok((theta, fraction))
    };
    let (theta, partial_rotary_factor) = parse("full_attention", 1.0)?;
    let sliding_theta = if value.get("sliding_attention").is_some() {
        let (theta, fraction) = parse("sliding_attention", 1.0)?;
        if fraction != 1.0 {
            return Err(invalid("LFM3 sliding attention requires full RoPE"));
        }
        theta
    } else {
        10000.0
    };
    Ok(RopeConfig {
        theta,
        partial_rotary_factor,
        sliding_theta,
    })
}

impl ModelArgs {
    /// Validates all normalized geometry and supported execution policy.
    pub fn validate(&self) -> Result<(), ConfigError> {
        validate_args(self)
    }

    /// Returns one validated layer policy without an out-of-range fallback.
    pub fn layer_policy(&self, layer: usize) -> Option<&LayerPolicy> {
        self.layer_schedule.get(layer)
    }

    /// Returns a stable ordered representation of the complete layer schedule.
    pub fn layer_schedule_fingerprint(&self) -> String {
        self.layer_schedule
            .iter()
            .map(|policy| {
                let operator = match policy.operator {
                    OperatorPolicy::CausalConvolution => "c".to_string(),
                    OperatorPolicy::SelfAttention(AttentionPolicy::Full) => "af".to_string(),
                    OperatorPolicy::SelfAttention(AttentionPolicy::Sliding { window }) => {
                        format!("as{}", window.get())
                    }
                };
                let feed_forward = match policy.feed_forward {
                    FeedForwardPolicy::Dense => "d",
                };
                format!("{operator}{feed_forward}")
            })
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Returns the physical encoding for one canonical parameter identity.
    pub fn weight_quantization_for(&self, weight_name: &str) -> Option<WeightQuantization> {
        if let Some(config) = self
            .quantized_weight_configs
            .as_ref()
            .and_then(|configs| configs.get(weight_name))
        {
            return Some(*config);
        }
        let quantization = self.weight_quantization?;
        match &self.quantized_weights {
            Some(names) if !names.contains(weight_name) => None,
            _ => Some(quantization),
        }
    }
}

/// Reads and validates a Hugging Face LFM3 configuration.
pub fn model_args_from_config_reader(reader: impl Read) -> Result<ModelArgs, ConfigError> {
    let value = serde_json::from_reader(reader)?;
    model_args_from_config_value(&value)
}

/// Normalizes and validates Hugging Face configuration.
pub fn model_args_from_config_value(config: &Value) -> Result<ModelArgs, ConfigError> {
    model_args_for_attention(config, true)
}

pub(super) fn model_args_for_attention(
    config: &Value,
    causal: bool,
) -> Result<ModelArgs, ConfigError> {
    if config
        .get("architectures")
        .and_then(Value::as_array)
        .is_some_and(|architectures| {
            architectures
                .iter()
                .any(|name| !matches!(name.as_str(), Some("Lfm3ForCausalLM" | "Lfm3Model")))
        })
    {
        return Err(invalid(
            "LFM3 text admission requires a text decoder architecture",
        ));
    }
    if ["num_experts", "num_experts_per_tok"].iter().any(|key| {
        config
            .get(*key)
            .is_some_and(|value| value.as_i64() != Some(0))
    }) {
        return Err(invalid(
            "LFM3 text admission requires dense feed-forward layers",
        ));
    }
    let source: ModelArgsSource = serde_json::from_value(config.clone())?;
    let args = source.normalize(causal)?;
    args.validate()?;
    Ok(args)
}

/// Parses normalized LFM3 arguments from pure GGUF catalog metadata.
pub fn model_args_from_gguf_catalog(
    arrays: &impl GgufTensorCatalog,
    metadata: &HashMap<String, MetadataValue>,
) -> Result<ModelArgs, ConfigError> {
    let architecture = gguf_string(metadata, "general.architecture")?;
    if !matches!(architecture.as_str(), "lfm3") {
        return Err(invalid(format!(
            "GGUF architecture {architecture:?}; expected lfm3"
        )));
    }
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let num_hidden_layers = gguf_i32(metadata, &key("block_count"))?;
    if num_hidden_layers <= 0 {
        return Err(invalid("LFM3 requires a positive block count"));
    }
    let kv_heads = expand_layer_values(
        &key("attention.head_count_kv"),
        gguf_i64_values(metadata, &key("attention.head_count_kv"))?,
        num_hidden_layers,
    )?;
    let num_key_value_heads = unique_nonzero(&key("attention.head_count_kv"), &kv_heads)?;
    let sliding = match metadata.get(&key("attention.sliding_window_pattern")) {
        Some(MetadataValue::Array(eredu_gguf::MetadataArray::Bool(values))) => values.clone(),
        _ => return Err(missing(&key("attention.sliding_window_pattern"))),
    };
    if sliding.len() != num_hidden_layers as usize {
        return Err(invalid("LFM3 sliding schedule length mismatch"));
    }
    if kv_heads
        .iter()
        .zip(&sliding)
        .any(|(heads, sliding)| *heads < 0 || (*heads == 0 && *sliding))
    {
        return Err(invalid(
            "LFM3 GGUF has conflicting mixer and attention schedules",
        ));
    }
    let window = gguf_i32(metadata, &key("attention.sliding_window"))?;
    let sliding_policy = AttentionPolicy::sliding(
        u32::try_from(window).map_err(|_| invalid("invalid LFM3 sliding window"))?,
    )
    .map_err(|e| invalid(e.to_string()))?;
    let layer_schedule = LayerSchedule::new(
        num_hidden_layers as usize,
        kv_heads
            .iter()
            .enumerate()
            .map(|(layer, heads)| LayerPolicy {
                operator: if *heads == 0 {
                    OperatorPolicy::CausalConvolution
                } else {
                    OperatorPolicy::SelfAttention(if sliding[layer] {
                        sliding_policy
                    } else {
                        AttentionPolicy::Full
                    })
                },
                feed_forward: FeedForwardPolicy::Dense,
            })
            .collect(),
    )
    .map_err(|error| invalid(format!("LFM3 GGUF {error}")))?;
    let vocab_key = key("vocab_size");
    let vocab_size = if metadata.contains_key(&vocab_key) {
        gguf_i32(metadata, &vocab_key)?
    } else {
        let tokens = metadata
            .get("tokenizer.ggml.tokens")
            .and_then(MetadataValue::as_strings)
            .ok_or_else(|| missing("tokenizer.ggml.tokens"))?;
        i32::try_from(tokens.len()).map_err(|_| invalid("LFM3 GGUF vocabulary is too large"))?
    };
    let args = ModelArgs {
        model_type: "lfm3".into(),
        vocab_size,
        hidden_size: gguf_i32(metadata, &key("embedding_length"))?,
        dense_intermediate_size: gguf_i32(metadata, &key("feed_forward_length"))?,
        num_hidden_layers,
        num_attention_heads: gguf_i32(metadata, &key("attention.head_count"))?,
        num_key_value_heads,
        max_position_embeddings: gguf_i32(metadata, &key("context_length"))?,
        norm_eps: gguf_f32(metadata, &key("attention.layer_norm_rms_epsilon"))?,
        layer_schedule,
        conv_l_cache: gguf_i32(metadata, &key("shortconv.l_cache"))?,
        conv_bias: arrays.any(|name| name.contains("shortconv") && name.ends_with(".bias")),
        tie_word_embeddings: !arrays.contains("output.weight"),
        rope: RopeConfig {
            theta: gguf_optional_f32(metadata, &key("rope.freq_base"))?
                .unwrap_or_else(default_rope_theta),
            partial_rotary_factor: 0.5,
            sliding_theta: gguf_f32(metadata, &key("rope.freq_base_swa"))?,
        },
        use_sandwich_norm: arrays.any(|name| name.contains("post_attention_norm")),
        weight_quantization: None,
        quantized_weights: None,
        quantized_weight_configs: None,
    };
    args.validate()?;
    let head = args.hidden_size / args.num_attention_heads;
    for suffix in [
        "attention.key_length",
        "attention.value_length",
        "rope.dimension_count",
    ] {
        if gguf_optional_i64(metadata, &key(suffix))?.is_some_and(|width| width != i64::from(head))
        {
            return Err(invalid(format!(
                "LFM3 GGUF {suffix} conflicts with its head geometry"
            )));
        }
    }
    Ok(args)
}

/// Derives the canonical cache-relevant architecture fingerprint.
pub fn prompt_cache_architecture_fingerprint(args: &ModelArgs) -> String {
    derive_prompt_cache_architecture_fingerprint(
        "lfm3",
        [
            ("model_type", args.model_type.clone()),
            ("hidden_size", args.hidden_size.to_string()),
            ("layers", args.num_hidden_layers.to_string()),
            ("layer_schedule", args.layer_schedule_fingerprint()),
            ("query_heads", args.num_attention_heads.to_string()),
            ("kv_heads", args.num_key_value_heads.to_string()),
            (
                "head_dim",
                (args.hidden_size / args.num_attention_heads).to_string(),
            ),
            ("max_positions", args.max_position_embeddings.to_string()),
            ("rope_theta", format!("{:08x}", args.rope.theta.to_bits())),
            (
                "rope_partial",
                format!("{:08x}", args.rope.partial_rotary_factor.to_bits()),
            ),
            (
                "rope_sliding",
                format!("{:08x}", args.rope.sliding_theta.to_bits()),
            ),
            ("sandwich", args.use_sandwich_norm.to_string()),
            ("norm_eps", format!("{:08x}", args.norm_eps.to_bits())),
            ("conv_history", args.conv_l_cache.to_string()),
            ("conv_bias", args.conv_bias.to_string()),
            ("quantization", format!("{:?}", args.weight_quantization)),
            (
                "quantized_weights",
                crate::cache_identity::string_set(args.quantized_weights.as_ref()),
            ),
            (
                "quantized_weight_configs",
                crate::cache_identity::debug_map(args.quantized_weight_configs.as_ref()),
            ),
        ],
    )
}

fn validate_args(args: &ModelArgs) -> Result<(), ConfigError> {
    if !matches!(args.model_type.as_str(), "lfm3") {
        return Err(invalid(format!(
            "LFM3 loader received model_type {:?}",
            args.model_type
        )));
    }
    for (name, value) in [
        ("vocab_size", args.vocab_size),
        ("hidden_size", args.hidden_size),
        ("dense_intermediate_size", args.dense_intermediate_size),
        ("num_hidden_layers", args.num_hidden_layers),
        ("num_attention_heads", args.num_attention_heads),
        ("num_key_value_heads", args.num_key_value_heads),
        ("conv_L_cache", args.conv_l_cache),
    ] {
        if value <= 0 {
            return Err(invalid(format!(
                "LFM3 {name} must be positive, got {value}"
            )));
        }
    }
    if args.layer_schedule.len() != args.num_hidden_layers as usize {
        return Err(invalid(format!(
            "LFM3 layer schedule has {} entries, expected {}",
            args.layer_schedule.len(),
            args.num_hidden_layers
        )));
    }
    if args.hidden_size % args.num_attention_heads != 0
        || args.num_attention_heads % args.num_key_value_heads != 0
        || (args.hidden_size / args.num_attention_heads) % 2 != 0
    {
        return Err(invalid(
            "LFM3 attention head counts do not divide hidden/query dimensions",
        ));
    }
    if !args.rope.theta.is_finite() || args.rope.theta <= 0.0 {
        return Err(invalid(format!(
            "LFM3 RoPE theta must be finite and positive, got {}",
            args.rope.theta
        )));
    }
    if args.max_position_embeddings <= 0 || !args.norm_eps.is_finite() || args.norm_eps <= 0.0 {
        return Err(invalid(
            "LFM3 context and normalization epsilon must be positive",
        ));
    }
    for theta in [args.rope.theta, args.rope.sliding_theta] {
        if !theta.is_finite() || theta <= 0.0 {
            return Err(invalid("invalid LFM3 RoPE base"));
        }
    }
    eredu_nn::RotaryAlgorithm::ZeroTail {
        rotary_fraction: args.rope.partial_rotary_factor,
    }
    .validate()
    .map_err(|e| invalid(e.to_string()))?;
    Ok(())
}

fn gguf_string(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<String, ConfigError> {
    match metadata.get(key) {
        Some(MetadataValue::String(value)) => Ok(value.clone()),
        Some(_) => Err(wrong_type(key)),
        None => Err(missing(key)),
    }
}

fn gguf_i64_values(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Vec<i64>, ConfigError> {
    metadata
        .get(key)
        .and_then(MetadataValue::to_i64_vec)
        .ok_or_else(|| wrong_type(key))
}

fn gguf_i32(metadata: &HashMap<String, MetadataValue>, key: &str) -> Result<i32, ConfigError> {
    let value = gguf_optional_i64(metadata, key)?.ok_or_else(|| missing(key))?;
    i32::try_from(value).map_err(|_| invalid(format!("GGUF metadata value {key:?} exceeds i32")))
}

fn gguf_optional_i64(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<i64>, ConfigError> {
    match metadata.get(key) {
        Some(value) => value.as_i64().map(Some).ok_or_else(|| wrong_type(key)),
        None => Ok(None),
    }
}

fn gguf_f32(metadata: &HashMap<String, MetadataValue>, key: &str) -> Result<f32, ConfigError> {
    gguf_optional_f32(metadata, key)?.ok_or_else(|| missing(key))
}

fn gguf_optional_f32(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<f32>, ConfigError> {
    match metadata.get(key) {
        Some(value) => value.as_f32().map(Some).ok_or_else(|| wrong_type(key)),
        None => Ok(None),
    }
}

fn expand_layer_values(key: &str, values: Vec<i64>, layers: i32) -> Result<Vec<i32>, ConfigError> {
    let values = if values.len() == 1 {
        vec![values[0]; layers as usize]
    } else if values.len() == layers as usize {
        values
    } else {
        return Err(invalid(format!(
            "GGUF metadata key {key:?} has {} values for {layers} layers",
            values.len()
        )));
    };
    values
        .into_iter()
        .map(|value| {
            i32::try_from(value)
                .map_err(|_| invalid(format!("GGUF metadata value {key:?} exceeds i32")))
        })
        .collect()
}

fn unique_nonzero(key: &str, values: &[i32]) -> Result<i32, ConfigError> {
    let Some(value) = values.iter().copied().find(|value| *value > 0) else {
        return Err(invalid(format!(
            "GGUF metadata key {key:?} has no attention-layer value"
        )));
    };
    if values.iter().any(|other| *other > 0 && *other != value) {
        return Err(invalid(format!(
            "GGUF metadata key {key:?} has non-uniform attention-layer values"
        )));
    }
    Ok(value)
}

fn invalid(reason: impl Into<String>) -> ConfigError {
    ConfigError::Invalid(reason.into())
}

fn missing(key: &str) -> ConfigError {
    invalid(format!("GGUF metadata is missing required key {key:?}"))
}

fn wrong_type(key: &str) -> ConfigError {
    invalid(format!("GGUF metadata key {key:?} has the wrong type"))
}
