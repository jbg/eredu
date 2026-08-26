//! Strict, backend-independent LFM2 configuration normalization.

use std::{
    collections::{HashMap, HashSet},
    io::Read,
};

use crate::{rotary::RopeValue, GgufTensorCatalog};
use eredu_checkpoint::WeightQuantization;
use eredu_core::{
    cache::{
        derive_prompt_cache_architecture_fingerprint, LayerCachePolicy, MutableStateResidency,
        StateTensorDimension, StateTensorDtype, StateTensorPolicy, StateTensorRole,
    },
    AttentionPolicy, LayerSchedule,
};
use eredu_gguf::MetadataValue;
use eredu_runtime::StateLayout;
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
    3
}

fn default_block_multiple_of() -> i32 {
    256
}

fn default_block_ffn_dim_multiplier() -> f32 {
    1.0
}

fn default_norm_eps() -> f32 {
    1e-5
}

fn default_routed_scaling_factor() -> f32 {
    1.0
}

/// Invalid or unsupported LFM2 configuration.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// Configuration JSON could not be decoded.
    #[error("invalid LFM2 configuration: {0}")]
    Json(#[from] serde_json::Error),
    /// Configuration changes unsupported execution semantics or has invalid geometry.
    #[error("{0}")]
    Invalid(String),
}

/// Stateful token-mixing operator used by one LFM2 layer.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum OperatorPolicy {
    /// Gated causal depthwise short convolution.
    CausalConvolution,
    /// Grouped-query self attention with its exact retention policy.
    SelfAttention(AttentionPolicy),
}

impl OperatorPolicy {
    fn parse(value: &str) -> Result<Self, ConfigError> {
        match value {
            "conv" => Ok(Self::CausalConvolution),
            "full_attention" => Ok(Self::SelfAttention(AttentionPolicy::Full)),
            other => Err(invalid(format!("LFM2 layer type {other:?} is unsupported"))),
        }
    }
}

/// Feed-forward operator used by one LFM2 layer.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum FeedForwardPolicy {
    /// Dense SwiGLU feed-forward block.
    Dense,
    /// Routed sparse mixture-of-experts block.
    SparseMoe,
}

/// Complete execution policy for one LFM2 decoder layer.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub struct LayerPolicy {
    /// Stateful token-mixing operator.
    pub operator: OperatorPolicy,
    /// Feed-forward operator.
    pub feed_forward: FeedForwardPolicy,
}

/// Canonical rotary-position configuration used by LFM2 attention layers.
#[derive(Debug, Clone, Copy)]
pub struct RopeConfig {
    /// RoPE frequency base.
    pub theta: f32,
}

/// Validated LFM2/LFM2-MoE decoder configuration.
#[derive(Debug, Clone)]
pub struct ModelArgs {
    /// Hugging Face model type (`lfm2` or `lfm2_moe`).
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
    /// Per-expert intermediate size for MoE checkpoints.
    pub moe_intermediate_size: i32,
    /// Number of routed experts.
    pub num_experts: i32,
    /// Experts selected per token.
    pub num_experts_per_tok: i32,
    /// Whether selected route weights are normalized.
    pub norm_topk_prob: bool,
    /// Router output scale.
    pub routed_scaling_factor: f32,
    /// Whether MoE selection uses the checkpoint expert bias.
    pub use_expert_bias: bool,
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
    #[serde(default = "default_block_multiple_of")]
    block_multiple_of: i32,
    #[serde(default = "default_block_ffn_dim_multiplier")]
    block_ffn_dim_multiplier: f32,
    #[serde(default = "default_true")]
    block_auto_adjust_ff_dim: bool,
    #[serde(default)]
    block_dim: Option<i32>,
    #[serde(default)]
    block_ff_dim: Option<i32>,
    #[serde(default = "default_true", alias = "tie_embedding")]
    tie_word_embeddings: bool,
    #[serde(default)]
    rope_theta: Option<f32>,
    #[serde(default)]
    rope_parameters: Option<HashMap<String, RopeValue>>,
    #[serde(default)]
    moe_intermediate_size: i32,
    #[serde(default)]
    num_dense_layers: i32,
    #[serde(default)]
    num_experts: i32,
    #[serde(default)]
    num_experts_per_tok: i32,
    #[serde(default)]
    norm_topk_prob: bool,
    #[serde(default = "default_routed_scaling_factor")]
    routed_scaling_factor: f32,
    #[serde(default)]
    use_expert_bias: bool,
    #[serde(default)]
    quantization: Option<WeightQuantization>,
    #[serde(default)]
    quantization_config: Option<WeightQuantization>,
}

impl ModelArgsSource {
    fn normalize(self) -> Result<ModelArgs, ConfigError> {
        let layer_count = usize::try_from(self.num_hidden_layers).map_err(|_| {
            invalid(format!(
                "LFM2 num_hidden_layers must be positive, got {}",
                self.num_hidden_layers
            ))
        })?;
        let operators = self
            .layer_types
            .iter()
            .map(|value| OperatorPolicy::parse(value))
            .collect::<Result<Vec<_>, _>>()?;
        if self.model_type == "lfm2" && self.num_dense_layers != 0 {
            return Err(invalid(format!(
                "LFM2 dense config conflicts with num_dense_layers={}",
                self.num_dense_layers
            )));
        }
        if self.model_type == "lfm2_moe"
            && (self.num_dense_layers < 0 || self.num_dense_layers > self.num_hidden_layers)
        {
            return Err(invalid(format!(
                "LFM2 MoE num_dense_layers must be between 0 and {}, got {}",
                self.num_hidden_layers, self.num_dense_layers
            )));
        }
        let policies = operators
            .into_iter()
            .enumerate()
            .map(|(layer, operator)| LayerPolicy {
                operator,
                feed_forward: if self.model_type == "lfm2_moe"
                    && layer >= self.num_dense_layers as usize
                {
                    FeedForwardPolicy::SparseMoe
                } else {
                    FeedForwardPolicy::Dense
                },
            })
            .collect();
        let layer_schedule = LayerSchedule::new(layer_count, policies)
            .map_err(|error| invalid(format!("LFM2 {error}")))?;
        if self
            .block_dim
            .is_some_and(|value| value != self.hidden_size)
        {
            return Err(invalid(format!(
                "LFM2 block_dim must equal hidden_size {}, got {}",
                self.hidden_size,
                self.block_dim.expect("checked above")
            )));
        }
        let dense_intermediate_size = if self.model_type == "lfm2_moe" {
            if self
                .block_ff_dim
                .is_some_and(|value| value != self.intermediate_size)
            {
                return Err(invalid(format!(
                    "LFM2 MoE block_ff_dim must equal intermediate_size {}, got {}",
                    self.intermediate_size,
                    self.block_ff_dim.expect("checked above")
                )));
            }
            self.intermediate_size
        } else {
            let mut size = i64::from(self.block_ff_dim.unwrap_or(self.intermediate_size));
            if self.block_auto_adjust_ff_dim {
                if self.block_multiple_of <= 0
                    || !self.block_ffn_dim_multiplier.is_finite()
                    || self.block_ffn_dim_multiplier <= 0.0
                {
                    return Err(invalid(
                        "LFM2 dense FFN adjustment requires a positive rounding multiple and finite positive multiplier",
                    ));
                }
                size = 2 * size / 3;
                size = (self.block_ffn_dim_multiplier * size as f32) as i64;
                let multiple = i64::from(self.block_multiple_of);
                size = multiple * ((size + multiple - 1) / multiple);
            }
            i32::try_from(size).map_err(|_| {
                invalid(format!(
                    "LFM2 adjusted dense intermediate size {size} exceeds i32"
                ))
            })?
        };
        let rope_theta = parse_rope_theta(self.rope_theta, self.rope_parameters)?;
        let weight_quantization = match (self.quantization, self.quantization_config) {
            (Some(first), Some(second)) if first != second => {
                return Err(invalid(
                    "LFM2 quantization and quantization_config disagree",
                ));
            }
            (Some(value), _) | (_, Some(value)) => Some(value),
            (None, None) => None,
        };
        Ok(ModelArgs {
            model_type: self.model_type,
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            dense_intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            max_position_embeddings: self.max_position_embeddings,
            norm_eps: self.norm_eps,
            layer_schedule,
            conv_l_cache: self.conv_l_cache,
            conv_bias: self.conv_bias,
            tie_word_embeddings: self.tie_word_embeddings,
            rope: RopeConfig { theta: rope_theta },
            moe_intermediate_size: self.moe_intermediate_size,
            num_experts: self.num_experts,
            num_experts_per_tok: self.num_experts_per_tok,
            norm_topk_prob: self.norm_topk_prob,
            routed_scaling_factor: self.routed_scaling_factor,
            use_expert_bias: self.use_expert_bias,
            weight_quantization,
            quantized_weights: None,
            quantized_weight_configs: None,
        })
    }
}

fn parse_rope_theta(
    top_level: Option<f32>,
    parameters: Option<HashMap<String, RopeValue>>,
) -> Result<f32, ConfigError> {
    let Some(parameters) = parameters else {
        return Ok(top_level.unwrap_or_else(default_rope_theta));
    };
    for key in parameters.keys() {
        if !matches!(key.as_str(), "rope_theta" | "rope_type") {
            return Err(invalid(format!(
                "LFM2 rope_parameters key {key:?} is unsupported"
            )));
        }
    }
    if let Some(rope_type) = parameters.get("rope_type") {
        let RopeValue::String(rope_type) = rope_type else {
            return Err(invalid(
                "LFM2 rope_parameters.rope_type must be \"default\"",
            ));
        };
        if rope_type != "default" {
            return Err(invalid(format!(
                "LFM2 rope type {rope_type:?} is unsupported"
            )));
        }
    }
    let nested = match parameters.get("rope_theta") {
        Some(RopeValue::Float(value)) => Some(*value),
        Some(RopeValue::String(value)) => Some(value.parse::<f32>().map_err(|_| {
            invalid(format!(
                "LFM2 rope_parameters.rope_theta {value:?} is not a float"
            ))
        })?),
        Some(RopeValue::Bool(_)) => {
            return Err(invalid("LFM2 rope_parameters.rope_theta must be a float"));
        }
        None => None,
    };
    if let (Some(top), Some(nested)) = (top_level, nested) {
        if top != nested {
            return Err(invalid(format!(
                "LFM2 rope_theta {top} conflicts with rope_parameters.rope_theta {nested}"
            )));
        }
    }
    Ok(nested.or(top_level).unwrap_or_else(default_rope_theta))
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

    /// Returns the canonical routed observation point for one sparse layer.
    pub fn routed_observation_point(
        &self,
        unit_path: &str,
        layer: usize,
    ) -> Option<eredu_runtime::RoutedObservationPoint> {
        (self.layer_policy(layer)?.feed_forward == FeedForwardPolicy::SparseMoe).then(|| {
            eredu_runtime::RoutedObservationPoint::new(
                format!("{unit_path}.feed_forward"),
                self.num_experts,
            )
        })
    }

    /// Returns whether any layer contains routed experts.
    pub fn has_sparse_moe_layers(&self) -> bool {
        self.layer_schedule
            .iter()
            .any(|policy| policy.feed_forward == FeedForwardPolicy::SparseMoe)
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
                    FeedForwardPolicy::SparseMoe => "e",
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

/// Reads and validates a Hugging Face LFM2 configuration.
pub fn model_args_from_config_reader(reader: impl Read) -> Result<ModelArgs, ConfigError> {
    let value = serde_json::from_reader(reader)?;
    model_args_from_config_value(&value)
}

/// Normalizes and validates Hugging Face configuration.
pub fn model_args_from_config_value(config: &Value) -> Result<ModelArgs, ConfigError> {
    let source: ModelArgsSource = serde_json::from_value(config.clone())?;
    let args = source.normalize()?;
    args.validate()?;
    Ok(args)
}

/// Parses normalized LFM2 arguments from pure GGUF catalog metadata.
pub fn model_args_from_gguf_catalog(
    arrays: &impl GgufTensorCatalog,
    metadata: &HashMap<String, MetadataValue>,
) -> Result<ModelArgs, ConfigError> {
    let architecture = gguf_string(metadata, "general.architecture")?;
    if !matches!(architecture.as_str(), "lfm2" | "lfm2moe") {
        return Err(invalid(format!(
            "GGUF architecture {architecture:?}; expected lfm2 or lfm2moe"
        )));
    }
    let is_moe = architecture == "lfm2moe";
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let num_hidden_layers = gguf_i32(metadata, &key("block_count"))?;
    if num_hidden_layers <= 0 {
        return Err(invalid(format!(
            "LFM2 GGUF block count must be positive, got {num_hidden_layers}"
        )));
    }
    let kv_heads = expand_layer_values(
        &key("attention.head_count_kv"),
        gguf_i64_values(metadata, &key("attention.head_count_kv"))?,
        num_hidden_layers,
    )?;
    let num_key_value_heads = unique_nonzero(&key("attention.head_count_kv"), &kv_heads)?;
    let num_dense_layers = if is_moe {
        gguf_i32(metadata, &key("leading_dense_block_count"))?
    } else {
        0
    };
    if num_dense_layers < 0 || num_dense_layers > num_hidden_layers {
        return Err(invalid(format!(
            "LFM2 MoE GGUF leading_dense_block_count must be between 0 and {num_hidden_layers}, got {num_dense_layers}"
        )));
    }
    let layer_schedule = LayerSchedule::new(
        num_hidden_layers as usize,
        kv_heads
            .iter()
            .enumerate()
            .map(|(layer, heads)| LayerPolicy {
                operator: if *heads == 0 {
                    OperatorPolicy::CausalConvolution
                } else {
                    OperatorPolicy::SelfAttention(AttentionPolicy::Full)
                },
                feed_forward: if is_moe && layer >= num_dense_layers as usize {
                    FeedForwardPolicy::SparseMoe
                } else {
                    FeedForwardPolicy::Dense
                },
            })
            .collect(),
    )
    .map_err(|error| invalid(format!("LFM2 GGUF {error}")))?;
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
    let expert_bias_name =
        |name: &str| name.contains("ffn_exp_probs_b") || name.contains("exp_probs_b");
    let args = ModelArgs {
        model_type: if is_moe { "lfm2_moe" } else { "lfm2" }.into(),
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
        },
        moe_intermediate_size: if is_moe {
            gguf_i32(metadata, &key("expert_feed_forward_length"))?
        } else {
            0
        },
        num_experts: if is_moe {
            gguf_i32(metadata, &key("expert_count"))?
        } else {
            0
        },
        num_experts_per_tok: if is_moe {
            gguf_i32(metadata, &key("expert_used_count"))?
        } else {
            0
        },
        norm_topk_prob: if is_moe {
            gguf_optional_i64(metadata, &key("expert_weights_norm"))?.unwrap_or(0) != 0
        } else {
            false
        },
        routed_scaling_factor: gguf_optional_f32(metadata, &key("expert_weights_scale"))?
            .unwrap_or_else(default_routed_scaling_factor),
        use_expert_bias: arrays.any(expert_bias_name),
        weight_quantization: None,
        quantized_weights: None,
        quantized_weight_configs: None,
    };
    args.validate()?;
    Ok(args)
}

/// Rank-local state geometry for one heterogeneous LFM2 layer.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct LayerCacheGeometry {
    /// Rank-local key/value heads for an attention layer.
    pub kv_heads: Option<i32>,
    /// Rank-local convolution channels for a short-convolution layer.
    pub convolution_channels: Option<i32>,
}

/// Declares state geometry using the global replicated parameter layout.
pub fn state_layout(args: &ModelArgs) -> Result<StateLayout, ConfigError> {
    let geometry = args
        .layer_schedule
        .iter()
        .map(|policy| match policy.operator {
            OperatorPolicy::CausalConvolution => LayerCacheGeometry {
                kv_heads: None,
                convolution_channels: Some(args.hidden_size),
            },
            OperatorPolicy::SelfAttention(_) => LayerCacheGeometry {
                kv_heads: Some(args.num_key_value_heads),
                convolution_channels: None,
            },
        })
        .collect::<Vec<_>>();
    state_layout_with_geometry(args, &geometry)
}

/// Declares exact state geometry from a resolved rank-local parameter layout.
pub fn state_layout_with_geometry(
    args: &ModelArgs,
    geometry: &[LayerCacheGeometry],
) -> Result<StateLayout, ConfigError> {
    if geometry.len() != args.layer_schedule.len() {
        return Err(invalid(format!(
            "LFM2 cache geometry has {} layers, expected {}",
            geometry.len(),
            args.layer_schedule.len()
        )));
    }
    let history = args
        .conv_l_cache
        .checked_sub(1)
        .ok_or_else(|| invalid("invalid LFM2 convolution width"))?;
    let fixed =
        |value| StateTensorDimension::fixed(value).map_err(|error| invalid(error.to_string()));
    let policies = args
        .layer_schedule
        .iter()
        .zip(geometry)
        .map(|(policy, geometry)| {
            match (
                policy.operator,
                geometry.kv_heads,
                geometry.convolution_channels,
            ) {
                (OperatorPolicy::CausalConvolution, None, Some(_)) if history == 0 => {
                    Ok(LayerCachePolicy::NoState)
                }
                (OperatorPolicy::CausalConvolution, None, Some(channels)) => {
                    LayerCachePolicy::fixed_only(vec![StateTensorPolicy::new(
                        StateTensorRole::Convolution { slot: 0 },
                        vec![
                            StateTensorDimension::Batch,
                            fixed(history)?,
                            fixed(channels)?,
                        ],
                        StateTensorDtype::Floating,
                        MutableStateResidency::AlwaysDeviceMutable,
                    )
                    .map_err(|error| invalid(error.to_string()))?])
                    .map_err(|error| invalid(error.to_string()))
                }
                (OperatorPolicy::SelfAttention(attention), Some(kv_heads), None) => {
                    LayerCachePolicy::key_value(
                        attention,
                        kv_heads,
                        args.hidden_size / args.num_attention_heads,
                    )
                    .map_err(|error| invalid(error.to_string()))
                }
                (OperatorPolicy::CausalConvolution, _, _) => Err(invalid(
                    "LFM2 convolution layer requires only convolution-channel cache geometry",
                )),
                (OperatorPolicy::SelfAttention(_), _, _) => Err(invalid(
                    "LFM2 attention layer requires only KV-head cache geometry",
                )),
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let schedule = LayerSchedule::new(args.layer_schedule.len(), policies)
        .map_err(|error| invalid(error.to_string()))?;
    StateLayout::new(schedule).map_err(|error| invalid(error.to_string()))
}

/// Derives the canonical cache-relevant architecture fingerprint.
pub fn prompt_cache_architecture_fingerprint(args: &ModelArgs) -> String {
    derive_prompt_cache_architecture_fingerprint(
        "lfm2",
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
    if !matches!(args.model_type.as_str(), "lfm2" | "lfm2_moe") {
        return Err(invalid(format!(
            "LFM2 loader received model_type {:?}",
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
                "LFM2 {name} must be positive, got {value}"
            )));
        }
    }
    if args.layer_schedule.len() != args.num_hidden_layers as usize {
        return Err(invalid(format!(
            "LFM2 layer schedule has {} entries, expected {}",
            args.layer_schedule.len(),
            args.num_hidden_layers
        )));
    }
    if args.layer_schedule.iter().any(|policy| {
        matches!(
            policy.operator,
            OperatorPolicy::SelfAttention(AttentionPolicy::Sliding { .. })
        )
    }) {
        return Err(invalid("LFM2 supports only full self-attention policies"));
    }
    if args.hidden_size % args.num_attention_heads != 0
        || args.num_attention_heads % args.num_key_value_heads != 0
    {
        return Err(invalid(
            "LFM2 attention head counts do not divide hidden/query dimensions",
        ));
    }
    if !args.rope.theta.is_finite() || args.rope.theta <= 0.0 {
        return Err(invalid(format!(
            "LFM2 RoPE theta must be finite and positive, got {}",
            args.rope.theta
        )));
    }
    if args.has_sparse_moe_layers()
        && (args.moe_intermediate_size <= 0
            || args.num_experts <= 0
            || args.num_experts_per_tok <= 0
            || args.num_experts_per_tok > args.num_experts)
    {
        return Err(invalid("LFM2 MoE expert configuration is invalid"));
    }
    if args.model_type == "lfm2" && args.has_sparse_moe_layers() {
        return Err(invalid(
            "LFM2 dense config contains a sparse-MoE layer policy",
        ));
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::cache::{StateComponentRole, StateResidencyClass};

    struct Catalog(Vec<String>);

    impl GgufTensorCatalog for Catalog {
        fn contains(&self, name: &str) -> bool {
            self.0.iter().any(|candidate| candidate == name)
        }

        fn any(&self, predicate: impl FnMut(&str) -> bool) -> bool {
            self.0.iter().map(String::as_str).any(predicate)
        }
    }

    fn dense_fixture() -> Value {
        serde_json::json!({
            "model_type": "lfm2",
            "vocab_size": 128,
            "hidden_size": 16,
            "intermediate_size": 33,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 4096,
            "layer_types": ["conv", "conv", "full_attention", "conv"],
            "conv_L_cache": 3,
            "block_multiple_of": 8,
            "block_ffn_dim_multiplier": 1.0,
            "block_auto_adjust_ff_dim": true,
            "tie_word_embeddings": false
        })
    }

    #[test]
    fn normalizes_dense_width_and_exact_hybrid_schedule() {
        let args = model_args_from_config_value(&dense_fixture()).unwrap();
        assert_eq!(args.dense_intermediate_size, 24);
        assert_eq!(args.layer_schedule_fingerprint(), "cd,cd,afd,cd");
        assert!(!args.tie_word_embeddings);
    }

    #[test]
    fn prompt_cache_fingerprint_includes_load_time_quantization() {
        let dense = model_args_from_config_value(&dense_fixture()).unwrap();
        let quantized = crate::lfm2::load_time_quantization(
            &dense,
            eredu_checkpoint::AffineQuantization::new(16, 4)
                .unwrap()
                .into(),
        )
        .unwrap();

        assert_ne!(
            prompt_cache_architecture_fingerprint(&dense),
            prompt_cache_architecture_fingerprint(&quantized)
        );
    }

    #[test]
    fn normalizes_moe_dense_prefix_and_selection_bias() {
        let mut fixture = dense_fixture();
        fixture["model_type"] = serde_json::json!("lfm2_moe");
        fixture["intermediate_size"] = serde_json::json!(32);
        fixture["num_dense_layers"] = serde_json::json!(2);
        fixture["moe_intermediate_size"] = serde_json::json!(8);
        fixture["num_experts"] = serde_json::json!(4);
        fixture["num_experts_per_tok"] = serde_json::json!(2);
        fixture["use_expert_bias"] = serde_json::json!(true);
        let args = model_args_from_config_value(&fixture).unwrap();
        assert_eq!(args.layer_schedule_fingerprint(), "cd,cd,afe,ce");
        assert!(args.use_expert_bias);
        assert!(args.routed_observation_point("model.layers.0", 0).is_none());
        let point = args.routed_observation_point("model.layers.2", 2).unwrap();
        assert_eq!(point.path(), "model.layers.2.feed_forward");
        assert_eq!(point.expert_count(), 4);
    }

    #[test]
    fn state_layout_freezes_bounded_and_append_only_components() {
        let args = model_args_from_config_value(&dense_fixture()).unwrap();
        let layout = state_layout(&args).unwrap();
        assert_eq!(layout.len(), 4);
        assert_eq!(
            layout.components(0).unwrap()[0].role,
            StateComponentRole::Fixed(StateTensorRole::Convolution { slot: 0 })
        );
        assert_eq!(
            layout.components(0).unwrap()[0].residency,
            StateResidencyClass::AlwaysDeviceMutable
        );
        assert_eq!(
            layout
                .components(2)
                .unwrap()
                .iter()
                .map(|component| component.role)
                .collect::<Vec<_>>(),
            [
                StateComponentRole::AttentionKeys,
                StateComponentRole::AttentionValues
            ]
        );
    }

    #[test]
    fn rejects_schedule_and_rotary_policy_changes() {
        let mut fixture = dense_fixture();
        fixture["layer_types"] = serde_json::json!(["conv", "sliding_attention"]);
        assert!(model_args_from_config_value(&fixture).is_err());

        let mut fixture = dense_fixture();
        fixture["rope_parameters"] = serde_json::json!({"rope_type": "yarn"});
        assert!(model_args_from_config_value(&fixture).is_err());
    }

    #[test]
    fn parses_per_layer_gguf_operator_schedule_without_backend_types() {
        let metadata = HashMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String("lfm2".into()),
            ),
            ("lfm2.block_count".into(), MetadataValue::Uint32(3)),
            (
                "lfm2.attention.head_count_kv".into(),
                MetadataValue::Array(eredu_gguf::MetadataArray::Int32(vec![0, 2, 0])),
            ),
            ("lfm2.embedding_length".into(), MetadataValue::Uint32(16)),
            ("lfm2.feed_forward_length".into(), MetadataValue::Uint32(32)),
            ("lfm2.attention.head_count".into(), MetadataValue::Uint32(4)),
            ("lfm2.context_length".into(), MetadataValue::Uint32(4096)),
            (
                "lfm2.attention.layer_norm_rms_epsilon".into(),
                MetadataValue::Float32(1e-5),
            ),
            ("lfm2.shortconv.l_cache".into(), MetadataValue::Uint32(3)),
            ("lfm2.vocab_size".into(), MetadataValue::Uint32(128)),
        ]);
        let args = model_args_from_gguf_catalog(&Catalog(vec!["output.weight".into()]), &metadata)
            .unwrap();
        assert_eq!(args.layer_schedule_fingerprint(), "cd,afd,cd");
        assert!(!args.tie_word_embeddings);
    }
}
