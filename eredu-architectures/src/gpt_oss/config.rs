//! Strict, backend-independent GPT-OSS configuration normalization.

use std::{collections::HashMap, io::Read};

use crate::rotary::RopeValue;
use eredu_checkpoint::WeightQuantization;
use eredu_core::{
    cache::{derive_prompt_cache_architecture_fingerprint, PromptCacheTopology},
    AttentionPolicy, LayerSchedule,
};
use eredu_gguf::MetadataValue;
use eredu_nn::{GatedProductActivation, GatedProductPolicy, RotarySpec};
use eredu_runtime::{ModelStateIdentity, StateLayout};
use serde::Deserialize;
use serde_json::Value;

use crate::decoder::{self, AttentionProjection, Config};

const DEFAULT_HEAD_DIM: i32 = 64;
const DEFAULT_SLIDING_WINDOW: i64 = 128;
const DEFAULT_ROPE_THETA: f32 = 150_000.0;
const DEFAULT_GATE_BOUND: f32 = 7.0;
const GPT_OSS_SIGMOID_MULTIPLIER: f32 = 1.702;
const GPT_OSS_UP_OFFSET: f32 = 1.0;
const MXFP4_GROUP_SIZE: i32 = 32;

fn default_head_dim() -> i32 {
    DEFAULT_HEAD_DIM
}

fn default_sliding_window() -> i64 {
    DEFAULT_SLIDING_WINDOW
}

fn default_rope_theta() -> f32 {
    DEFAULT_ROPE_THETA
}

fn default_gate_bound() -> f32 {
    DEFAULT_GATE_BOUND
}

/// Invalid or unsupported GPT-OSS configuration.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// Configuration JSON could not be decoded.
    #[error("invalid GPT-OSS configuration: {0}")]
    Json(#[from] serde_json::Error),
    /// Configuration changes unsupported semantics or has invalid geometry.
    #[error("{0}")]
    Invalid(String),
}

/// Published GPT-OSS native expert quantization metadata.
#[derive(Debug, Clone, Eq, PartialEq, Deserialize)]
pub struct MxFp4Config {
    /// Must be `mxfp4`.
    pub quant_method: String,
}

#[derive(Debug, Deserialize)]
struct ModelArgsSource {
    model_type: String,
    hidden_size: i32,
    intermediate_size: i32,
    num_hidden_layers: i32,
    num_attention_heads: i32,
    num_key_value_heads: i32,
    #[serde(default = "default_head_dim")]
    head_dim: i32,
    vocab_size: i32,
    num_local_experts: i32,
    num_experts_per_tok: i32,
    rms_norm_eps: f32,
    #[serde(default = "default_sliding_window")]
    sliding_window: i64,
    max_position_embeddings: i32,
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    #[serde(default)]
    rope_scaling: Option<HashMap<String, RopeValue>>,
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    quantization_config: MxFp4Config,
    #[serde(default)]
    quantization: Option<WeightQuantization>,
    #[serde(default = "default_gate_bound")]
    swiglu_limit: f32,
}

/// Validated GPT-OSS decoder geometry and execution policy.
#[derive(Debug, Clone)]
pub struct ModelArgs {
    /// Canonical Hugging Face model type.
    pub model_type: String,
    /// Canonical decoder parameter namespace.
    pub parameter_root: String,
    /// Transformer hidden width.
    pub hidden_size: i32,
    /// Per-expert hidden width.
    pub intermediate_size: i32,
    /// Number of transformer blocks.
    pub num_hidden_layers: i32,
    /// Number of query heads.
    pub num_attention_heads: i32,
    /// Number of key/value heads.
    pub num_key_value_heads: i32,
    /// Per-head query/key/value width.
    pub head_dim: i32,
    /// Token vocabulary size.
    pub vocab_size: i32,
    /// Number of routed local experts.
    pub num_local_experts: i32,
    /// Exact number of experts selected per token.
    pub num_experts_per_tok: i32,
    /// RMS normalization epsilon.
    pub rms_norm_eps: f32,
    /// Maximum configured context length.
    pub max_position_embeddings: i32,
    /// Rotary frequency base.
    pub rope_theta: f32,
    /// Canonical validated YaRN metadata, when configured.
    pub rope_scaling: Option<HashMap<String, RopeValue>>,
    /// Ordered full/sliding policy for every decoder block.
    pub attention_schedule: LayerSchedule<AttentionPolicy>,
    /// Native MXFP4 expert contract.
    pub quantization_config: MxFp4Config,
    /// Optional model-wide encoding for ordinary dense weights.
    pub quantization: Option<WeightQuantization>,
    /// Exact per-weight dense encodings populated by checkpoint planning.
    pub quantized_weight_configs: Option<HashMap<String, WeightQuantization>>,
    /// Exact bounded GPT-OSS expert activation equation.
    pub gated_product_policy: GatedProductPolicy,
    /// Published shared upper/absolute bound for gate and up branches.
    pub swiglu_limit: f32,
}

impl ModelArgs {
    /// Validates normalized geometry and execution policy.
    pub fn validate(&self) -> Result<(), ConfigError> {
        validate_model_args(self)
    }

    /// Returns the canonical routed observation point for one decoder layer.
    pub fn routed_observation_points(
        &self,
        unit_path: &str,
        _layer: usize,
    ) -> eredu_runtime::RoutedObservationPoints {
        eredu_runtime::RoutedObservationPoints::new(
            eredu_runtime::RoutedBankId::new(0),
            format!("{unit_path}.mlp"),
            self.num_local_experts,
        )
    }

    /// Returns the physical encoding for one ordinary canonical parameter.
    pub fn weight_quantization_for(&self, name: &str) -> Option<WeightQuantization> {
        self.quantized_weight_configs
            .as_ref()
            .and_then(|configs| configs.get(name).copied())
            .or(self.quantization)
    }

    /// Returns only a checkpoint-native per-weight encoding, excluding any
    /// model-wide load-time quantization request.
    pub(crate) fn checkpoint_weight_quantization_for(
        &self,
        name: &str,
    ) -> Option<WeightQuantization> {
        self.quantized_weight_configs
            .as_ref()
            .and_then(|configs| configs.get(name).copied())
    }
}

impl Config for ModelArgs {
    fn model_family(&self) -> &'static str {
        "gpt_oss"
    }
    fn model_identity(&self) -> &str {
        &self.model_type
    }

    fn architecture_fingerprint(&self) -> String {
        prompt_cache_architecture_fingerprint(self)
    }

    fn parameter_root(&self) -> &str {
        &self.parameter_root
    }

    fn routed_observation_points(
        &self,
        unit_path: &str,
        layer: usize,
    ) -> Option<eredu_runtime::RoutedObservationPoints> {
        Some(ModelArgs::routed_observation_points(self, unit_path, layer))
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

    fn attention_bias(&self, _projection: AttentionProjection) -> bool {
        true
    }

    fn learned_attention_sinks(&self) -> bool {
        true
    }

    fn mlp_bias(&self) -> bool {
        true
    }

    fn gated_product_policy(&self) -> Option<GatedProductPolicy> {
        Some(self.gated_product_policy)
    }

    fn tie_word_embeddings(&self) -> bool {
        false
    }

    fn attention_schedule(&self) -> &LayerSchedule<AttentionPolicy> {
        &self.attention_schedule
    }

    fn weight_quantization(&self, name: &str) -> Option<WeightQuantization> {
        self.weight_quantization_for(name)
    }

    fn rotary_spec(&self, dimensions: i32) -> RotarySpec {
        RotarySpec {
            arithmetic: eredu_nn::RotaryArithmetic::Native,
            dimensions,
            base: self.rope_theta,
            traditional: false,
            algorithm: crate::rotary::normalize_algorithm(self.rope_scaling.as_ref())
                .expect("validated GPT-OSS RoPE algorithm"),
        }
    }
}

/// Reads and validates a Hugging Face GPT-OSS configuration.
pub fn model_args_from_config_reader(reader: impl Read) -> Result<ModelArgs, ConfigError> {
    model_args_from_config_value(&serde_json::from_reader(reader)?)
}

/// Parses and normalizes a Hugging Face GPT-OSS configuration.
pub fn model_args_from_config_value(config: &Value) -> Result<ModelArgs, ConfigError> {
    if config.get("layer_types").is_some_and(Value::is_null) {
        return Err(invalid(
            "GPT-OSS layer_types must be an array when supplied",
        ));
    }
    let source: ModelArgsSource = serde_json::from_value(config.clone())?;
    let layers = positive_usize(source.num_hidden_layers, "num_hidden_layers")?;
    let attention_schedule =
        normalize_attention_schedule(layers, source.sliding_window, source.layer_types.as_deref())?;
    let rope_scaling = normalize_rope_scaling(source.rope_scaling, source.max_position_embeddings)?;
    let gated_product_policy = gpt_oss_gated_product_policy(source.swiglu_limit)?;
    let args = ModelArgs {
        model_type: source.model_type,
        parameter_root: "model".into(),
        hidden_size: source.hidden_size,
        intermediate_size: source.intermediate_size,
        num_hidden_layers: source.num_hidden_layers,
        num_attention_heads: source.num_attention_heads,
        num_key_value_heads: source.num_key_value_heads,
        head_dim: source.head_dim,
        vocab_size: source.vocab_size,
        num_local_experts: source.num_local_experts,
        num_experts_per_tok: source.num_experts_per_tok,
        rms_norm_eps: source.rms_norm_eps,
        max_position_embeddings: source.max_position_embeddings,
        rope_theta: source.rope_theta,
        rope_scaling,
        attention_schedule,
        quantization_config: source.quantization_config,
        quantization: source.quantization,
        quantized_weight_configs: None,
        gated_product_policy,
        swiglu_limit: source.swiglu_limit,
    };
    args.validate()?;
    Ok(args)
}

/// Parses GPT-OSS arguments from backend-independent GGUF metadata.
pub fn model_args_from_gguf_catalog(
    metadata: &HashMap<String, MetadataValue>,
) -> Result<ModelArgs, ConfigError> {
    let architecture = gguf_string(metadata, "general.architecture")?;
    if architecture != "gpt-oss" {
        return Err(invalid(format!(
            "GGUF architecture {architecture:?}; expected gpt-oss"
        )));
    }
    let key = |suffix: &str| format!("gpt-oss.{suffix}");
    let hidden_size = gguf_i32(metadata, &key("embedding_length"))?;
    let num_hidden_layers = gguf_i32(metadata, &key("block_count"))?;
    let layers = positive_usize(num_hidden_layers, "GGUF block_count")?;
    let num_attention_heads = gguf_i32(metadata, &key("attention.head_count"))?;
    if num_attention_heads <= 0 {
        return Err(invalid(format!(
            "GGUF attention head count must be positive, got {num_attention_heads}"
        )));
    }
    let head_dim = match gguf_optional_i64(metadata, &key("attention.key_length"))? {
        Some(value) => {
            i32::try_from(value).map_err(|_| invalid("GGUF attention key length exceeds i32"))?
        }
        None if hidden_size > 0 && hidden_size % num_attention_heads == 0 => {
            hidden_size / num_attention_heads
        }
        None => return Err(invalid(
            "GGUF without attention.key_length requires embedding_length divisible by head_count",
        )),
    };
    for suffix in ["attention.value_length", "rope.dimension_count"] {
        if let Some(value) = gguf_optional_i64(metadata, &key(suffix))? {
            if value != i64::from(head_dim) {
                return Err(invalid(format!(
                    "GGUF {suffix} {value} does not match head dimension {head_dim}"
                )));
            }
        }
    }
    let sliding_window = gguf_i32(metadata, &key("attention.sliding_window"))?;
    let max_position_embeddings = gguf_i32(metadata, &key("context_length"))?;
    let bound =
        gguf_optional_f32(metadata, &key("swiglu_clamp_exp"))?.unwrap_or(DEFAULT_GATE_BOUND);
    let args = ModelArgs {
        model_type: "gpt_oss".into(),
        parameter_root: "model".into(),
        hidden_size,
        intermediate_size: gguf_i32(metadata, &key("expert_feed_forward_length"))?,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads: gguf_i32(metadata, &key("attention.head_count_kv"))?,
        head_dim,
        vocab_size: gguf_vocab_size(metadata, &key("vocab_size"))?,
        num_local_experts: gguf_i32(metadata, &key("expert_count"))?,
        num_experts_per_tok: gguf_i32(metadata, &key("expert_used_count"))?,
        rms_norm_eps: gguf_f32(metadata, &key("attention.layer_norm_rms_epsilon"))?,
        max_position_embeddings,
        rope_theta: gguf_optional_f32(metadata, &key("rope.freq_base"))?
            .unwrap_or(DEFAULT_ROPE_THETA),
        rope_scaling: gguf_rope_scaling(metadata, "gpt-oss", max_position_embeddings)?,
        attention_schedule: normalize_attention_schedule(layers, i64::from(sliding_window), None)?,
        quantization_config: MxFp4Config {
            quant_method: "mxfp4".into(),
        },
        quantization: None,
        quantized_weight_configs: None,
        gated_product_policy: gpt_oss_gated_product_policy(bound)?,
        swiglu_limit: bound,
    };
    args.validate()?;
    Ok(args)
}

/// Declares the exact full/sliding key/value state layout.
pub fn state_layout(args: &ModelArgs) -> Result<StateLayout, ConfigError> {
    decoder::state_layout(args).map_err(|error| invalid(error.to_string()))
}

/// Declares rank-local prompt-cache identity without constructing backend state.
pub fn state_identity(
    args: &ModelArgs,
    layout: &StateLayout,
    global_layer_start: usize,
    topology: PromptCacheTopology,
) -> Result<ModelStateIdentity, ConfigError> {
    args.validate()?;
    topology
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    let layer_count = usize::try_from(args.num_hidden_layers)
        .map_err(|_| invalid("GPT-OSS layer count exceeds usize"))?;
    let global_layer_end = global_layer_start
        .checked_add(layout.len())
        .ok_or_else(|| invalid("GPT-OSS owned layer range overflowed"))?;
    if global_layer_end > layer_count {
        return Err(invalid(format!(
            "GPT-OSS owns layers {global_layer_start}..{global_layer_end}, outside {layer_count} layers"
        )));
    }
    eredu_runtime::ModelStateIdentity::new(
        "gpt_oss",
        args.model_type.clone(),
        prompt_cache_architecture_fingerprint(args),
        layer_count,
        global_layer_start,
        0,
        topology,
    )
    .map_err(|error| invalid(error.to_string()))
}

/// Returns the stable cache-relevant architecture fingerprint.
pub fn prompt_cache_architecture_fingerprint(args: &ModelArgs) -> String {
    let rope_scaling = args.rope_scaling.as_ref().map_or_else(
        || "none".to_string(),
        |config| {
            let mut entries = config.iter().collect::<Vec<_>>();
            entries.sort_unstable_by_key(|(key, _)| key.as_str());
            entries
                .into_iter()
                .map(|(key, value)| format!("{key}={}", rope_value_fingerprint(value)))
                .collect::<Vec<_>>()
                .join(";")
        },
    );
    let policy = args.gated_product_policy;
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
        "gpt_oss",
        [
            ("model_type", args.model_type.clone()),
            ("parameter_root", args.parameter_root.clone()),
            ("hidden_size", args.hidden_size.to_string()),
            ("intermediate_size", args.intermediate_size.to_string()),
            ("layers", args.num_hidden_layers.to_string()),
            ("query_heads", args.num_attention_heads.to_string()),
            ("kv_heads", args.num_key_value_heads.to_string()),
            ("head_dim", args.head_dim.to_string()),
            ("vocab_size", args.vocab_size.to_string()),
            ("experts", args.num_local_experts.to_string()),
            ("top_k", args.num_experts_per_tok.to_string()),
            ("norm_eps", f32_fingerprint(args.rms_norm_eps)),
            (
                "attention_schedule",
                args.attention_schedule.fingerprint_component(),
            ),
            ("max_positions", args.max_position_embeddings.to_string()),
            ("rope_theta", f32_fingerprint(args.rope_theta)),
            ("rope_scaling", rope_scaling),
            (
                "expert_quantization",
                args.quantization_config.quant_method.clone(),
            ),
            ("dense_quantization", format!("{:?}", args.quantization)),
            (
                "quantized_weight_configs",
                quantized_weight_configs.join(";"),
            ),
            (
                "gate_bound",
                policy
                    .gate_upper_bound()
                    .map_or_else(|| "none".into(), f32_fingerprint),
            ),
            (
                "up_bound",
                policy
                    .up_absolute_bound()
                    .map_or_else(|| "none".into(), f32_fingerprint),
            ),
            (
                "sigmoid_multiplier",
                f32_fingerprint(policy.sigmoid_multiplier()),
            ),
            ("up_offset", f32_fingerprint(policy.up_offset())),
            ("swiglu_limit", f32_fingerprint(args.swiglu_limit)),
        ],
    )
}

fn validate_model_args(args: &ModelArgs) -> Result<(), ConfigError> {
    if args.model_type != "gpt_oss" {
        return Err(invalid(format!(
            "GPT-OSS requires model_type \"gpt_oss\", got {:?}",
            args.model_type
        )));
    }
    if args.parameter_root != "model" {
        return Err(invalid(format!(
            "GPT-OSS parameter root must be \"model\", got {:?}",
            args.parameter_root
        )));
    }
    if args.quantization_config.quant_method != "mxfp4" {
        return Err(invalid(format!(
            "GPT-OSS requires quantization_config.quant_method=\"mxfp4\", got {:?}",
            args.quantization_config.quant_method
        )));
    }
    for (name, value) in [
        ("hidden_size", args.hidden_size),
        ("intermediate_size", args.intermediate_size),
        ("num_hidden_layers", args.num_hidden_layers),
        ("num_attention_heads", args.num_attention_heads),
        ("num_key_value_heads", args.num_key_value_heads),
        ("head_dim", args.head_dim),
        ("vocab_size", args.vocab_size),
        ("num_local_experts", args.num_local_experts),
        ("num_experts_per_tok", args.num_experts_per_tok),
        ("max_position_embeddings", args.max_position_embeddings),
    ] {
        if value <= 0 {
            return Err(invalid(format!("{name} must be positive, got {value}")));
        }
    }
    if args.hidden_size % MXFP4_GROUP_SIZE != 0 || args.intermediate_size % MXFP4_GROUP_SIZE != 0 {
        return Err(invalid(format!(
            "GPT-OSS MXFP4 hidden and intermediate dimensions must be divisible by {MXFP4_GROUP_SIZE}"
        )));
    }
    args.num_attention_heads
        .checked_mul(args.head_dim)
        .ok_or_else(|| invalid("GPT-OSS query projection width overflows i32"))?;
    args.num_key_value_heads
        .checked_mul(args.head_dim)
        .ok_or_else(|| invalid("GPT-OSS key/value projection width overflows i32"))?;
    args.intermediate_size
        .checked_mul(2)
        .ok_or_else(|| invalid("GPT-OSS fused gate/up width overflows i32"))?;
    if args.num_attention_heads % args.num_key_value_heads != 0 {
        return Err(invalid(format!(
            "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
            args.num_attention_heads, args.num_key_value_heads
        )));
    }
    if args.num_experts_per_tok > args.num_local_experts {
        return Err(invalid(format!(
            "num_experts_per_tok ({}) exceeds num_local_experts ({})",
            args.num_experts_per_tok, args.num_local_experts
        )));
    }
    if !args.rms_norm_eps.is_finite() || args.rms_norm_eps <= 0.0 {
        return Err(invalid(format!(
            "rms_norm_eps must be finite and positive, got {}",
            args.rms_norm_eps
        )));
    }
    if !args.rope_theta.is_finite() || args.rope_theta <= 0.0 {
        return Err(invalid(format!(
            "rope_theta must be finite and positive, got {}",
            args.rope_theta
        )));
    }
    let layers = positive_usize(args.num_hidden_layers, "num_hidden_layers")?;
    if args.attention_schedule.len() != layers {
        return Err(invalid(format!(
            "attention schedule has {} entries for {layers} layers",
            args.attention_schedule.len()
        )));
    }
    for policy in args.attention_schedule.iter() {
        if let Some(window) = policy.window() {
            if window.get() > i32::MAX as u32
                || i64::from(window.get()) > i64::from(args.max_position_embeddings)
            {
                return Err(invalid(format!(
                    "sliding window {} exceeds maximum positions {}",
                    window.get(),
                    args.max_position_embeddings
                )));
            }
        }
    }
    validate_normalized_rope_scaling(args.rope_scaling.as_ref(), args.max_position_embeddings)?;
    args.gated_product_policy
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    if args.gated_product_policy.activation() != GatedProductActivation::Silu
        || args.gated_product_policy.gate_upper_bound()
            != args.gated_product_policy.up_absolute_bound()
        || args.gated_product_policy.gate_upper_bound().is_none()
        || args.gated_product_policy.gate_upper_bound() != Some(args.swiglu_limit)
        || args.gated_product_policy.sigmoid_multiplier() != GPT_OSS_SIGMOID_MULTIPLIER
        || args.gated_product_policy.up_offset() != GPT_OSS_UP_OFFSET
    {
        return Err(invalid(format!(
            "GPT-OSS requires its exact bounded gated-product policy, got {:?}",
            args.gated_product_policy
        )));
    }
    Ok(())
}

fn normalize_attention_schedule(
    layers: usize,
    window: i64,
    layer_types: Option<&[String]>,
) -> Result<LayerSchedule<AttentionPolicy>, ConfigError> {
    if layers == 0 {
        return Err(invalid("num_hidden_layers must be positive"));
    }
    let window = u32::try_from(window)
        .ok()
        .filter(|window| *window > 0 && *window <= i32::MAX as u32)
        .ok_or_else(|| invalid(format!("invalid GPT-OSS sliding_window {window}")))?;
    let sliding = AttentionPolicy::sliding(window).map_err(|error| invalid(error.to_string()))?;
    let policies = match layer_types {
        None => (0..layers)
            .map(|layer| {
                if layer % 2 == 0 {
                    sliding
                } else {
                    AttentionPolicy::Full
                }
            })
            .collect(),
        Some(layer_types) => {
            if layer_types.len() != layers {
                return Err(invalid(format!(
                    "layer_types has {} entries for {layers} layers",
                    layer_types.len()
                )));
            }
            layer_types
                .iter()
                .enumerate()
                .map(|(layer, kind)| match kind.as_str() {
                    "sliding_attention" => Ok(sliding),
                    "full_attention" => Ok(AttentionPolicy::Full),
                    _ => Err(invalid(format!(
                        "layer_types[{layer}] must be sliding_attention or full_attention, got {kind:?}"
                    ))),
                })
                .collect::<Result<Vec<_>, _>>()?
        }
    };
    LayerSchedule::new(layers, policies).map_err(|error| invalid(error.to_string()))
}

fn gpt_oss_gated_product_policy(bound: f32) -> Result<GatedProductPolicy, ConfigError> {
    GatedProductPolicy::new(
        GatedProductActivation::Silu,
        Some(bound),
        Some(bound),
        GPT_OSS_SIGMOID_MULTIPLIER,
        GPT_OSS_UP_OFFSET,
    )
    .map_err(|error| invalid(error.to_string()))
}

fn normalize_rope_scaling(
    scaling: Option<HashMap<String, RopeValue>>,
    max_positions: i32,
) -> Result<Option<HashMap<String, RopeValue>>, ConfigError> {
    let Some(mut scaling) = scaling else {
        return Ok(None);
    };
    let kind = rope_kind(&scaling)?;
    if matches!(kind.as_str(), "none" | "default") {
        return Ok(None);
    }
    if kind != "yarn" {
        return Err(invalid(format!(
            "GPT-OSS RoPE scaling type {kind:?} is unsupported"
        )));
    }
    for key in scaling.keys() {
        if !matches!(
            key.as_str(),
            "type"
                | "rope_type"
                | "factor"
                | "original_max_position_embeddings"
                | "beta_fast"
                | "beta_slow"
                | "mscale"
                | "mscale_all_dim"
                | "truncate"
        ) {
            return Err(invalid(format!("unsupported GPT-OSS YaRN field {key:?}")));
        }
    }
    let factor = required_rope_number(&scaling, "factor")?;
    let original = required_rope_number(&scaling, "original_max_position_embeddings")?;
    let beta_fast = optional_rope_number(&scaling, "beta_fast")?.unwrap_or(32.0);
    let beta_slow = optional_rope_number(&scaling, "beta_slow")?.unwrap_or(1.0);
    let mscale = optional_rope_number(&scaling, "mscale")?.unwrap_or(1.0);
    let mscale_all_dim = optional_rope_number(&scaling, "mscale_all_dim")?.unwrap_or(0.0);
    let truncate = match scaling.remove("truncate") {
        None => true,
        Some(RopeValue::Bool(value)) => value,
        Some(_) => return Err(invalid("GPT-OSS YaRN truncate must be boolean")),
    };
    validate_yarn_scalars(
        factor,
        original,
        beta_fast,
        beta_slow,
        mscale,
        mscale_all_dim,
        max_positions,
    )?;
    Ok(Some(HashMap::from([
        ("rope_type".into(), RopeValue::String("yarn".into())),
        ("factor".into(), RopeValue::Float(factor)),
        (
            "original_max_position_embeddings".into(),
            RopeValue::Float(original),
        ),
        ("beta_fast".into(), RopeValue::Float(beta_fast)),
        ("beta_slow".into(), RopeValue::Float(beta_slow)),
        ("mscale".into(), RopeValue::Float(mscale)),
        ("mscale_all_dim".into(), RopeValue::Float(mscale_all_dim)),
        ("truncate".into(), RopeValue::Bool(truncate)),
    ])))
}

fn validate_normalized_rope_scaling(
    scaling: Option<&HashMap<String, RopeValue>>,
    max_positions: i32,
) -> Result<(), ConfigError> {
    let Some(scaling) = scaling else {
        return Ok(());
    };
    if rope_kind(scaling)? != "yarn" {
        return Err(invalid("normalized GPT-OSS RoPE scaling must be YaRN"));
    }
    validate_yarn_scalars(
        required_rope_number(scaling, "factor")?,
        required_rope_number(scaling, "original_max_position_embeddings")?,
        required_rope_number(scaling, "beta_fast")?,
        required_rope_number(scaling, "beta_slow")?,
        required_rope_number(scaling, "mscale")?,
        required_rope_number(scaling, "mscale_all_dim")?,
        max_positions,
    )?;
    if !matches!(scaling.get("truncate"), Some(RopeValue::Bool(_))) {
        return Err(invalid("normalized GPT-OSS YaRN truncate must be boolean"));
    }
    match crate::rotary::normalize_algorithm(Some(scaling)).map_err(invalid)? {
        eredu_nn::RotaryAlgorithm::Yarn { .. } => Ok(()),
        _ => Err(invalid("normalized GPT-OSS RoPE scaling must be YaRN")),
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_yarn_scalars(
    factor: f32,
    original: f32,
    beta_fast: f32,
    beta_slow: f32,
    mscale: f32,
    mscale_all_dim: f32,
    max_positions: i32,
) -> Result<(), ConfigError> {
    if factor <= 0.0
        || original <= 0.0
        || beta_fast <= 0.0
        || beta_slow <= 0.0
        || beta_fast <= beta_slow
        || mscale <= 0.0
        || mscale_all_dim < 0.0
    {
        return Err(invalid(
            "GPT-OSS YaRN scalars are outside their valid ranges",
        ));
    }
    let expanded = original * factor;
    if !expanded.is_finite() || original > max_positions as f32 || max_positions as f32 > expanded {
        return Err(invalid(format!(
            "GPT-OSS max positions {max_positions} are incompatible with YaRN original={original} factor={factor}"
        )));
    }
    Ok(())
}

fn rope_kind(scaling: &HashMap<String, RopeValue>) -> Result<String, ConfigError> {
    let from_type = scaling.get("type");
    let from_rope_type = scaling.get("rope_type");
    let parse = |value: &RopeValue| match value {
        RopeValue::String(value) => Ok(value.clone()),
        _ => Err(invalid("RoPE type or rope_type must be a string")),
    };
    let kind = match (from_type, from_rope_type) {
        (Some(left), Some(right)) => {
            let left = parse(left)?;
            let right = parse(right)?;
            if left != right {
                return Err(invalid(format!(
                    "conflicting RoPE type {left:?} and rope_type {right:?}"
                )));
            }
            left
        }
        (Some(value), None) | (None, Some(value)) => parse(value)?,
        (None, None) => return Err(invalid("RoPE scaling requires type or rope_type")),
    };
    Ok(kind)
}

fn required_rope_number(
    values: &HashMap<String, RopeValue>,
    key: &str,
) -> Result<f32, ConfigError> {
    optional_rope_number(values, key)?
        .ok_or_else(|| invalid(format!("GPT-OSS YaRN requires numeric {key}")))
}

fn optional_rope_number(
    values: &HashMap<String, RopeValue>,
    key: &str,
) -> Result<Option<f32>, ConfigError> {
    match values.get(key) {
        None => Ok(None),
        Some(RopeValue::Float(value)) if value.is_finite() => Ok(Some(*value)),
        Some(RopeValue::String(value)) => value
            .parse::<f32>()
            .ok()
            .filter(|value| value.is_finite())
            .map(Some)
            .ok_or_else(|| invalid(format!("GPT-OSS YaRN {key} must be finite numeric"))),
        Some(_) => Err(invalid(format!(
            "GPT-OSS YaRN {key} must be finite numeric"
        ))),
    }
}

fn gguf_rope_scaling(
    metadata: &HashMap<String, MetadataValue>,
    architecture: &str,
    max_positions: i32,
) -> Result<Option<HashMap<String, RopeValue>>, ConfigError> {
    let key = |suffix: &str| format!("{architecture}.rope.scaling.{suffix}");
    let Some(kind) = gguf_optional_string(metadata, &key("type"))? else {
        return Ok(None);
    };
    if matches!(kind.as_str(), "none" | "default") {
        return Ok(None);
    }
    if kind != "yarn" {
        return Err(invalid(format!(
            "GPT-OSS GGUF RoPE scaling type {kind:?} is unsupported"
        )));
    }
    normalize_rope_scaling(
        Some(HashMap::from([
            ("rope_type".into(), RopeValue::String("yarn".into())),
            (
                "factor".into(),
                RopeValue::Float(gguf_f32(metadata, &key("factor"))?),
            ),
            (
                "original_max_position_embeddings".into(),
                RopeValue::Float(gguf_i32(metadata, &key("original_context_length"))? as f32),
            ),
            (
                "beta_fast".into(),
                RopeValue::Float(
                    gguf_optional_f32(metadata, &key("yarn_beta_fast"))?.unwrap_or(32.0),
                ),
            ),
            (
                "beta_slow".into(),
                RopeValue::Float(
                    gguf_optional_f32(metadata, &key("yarn_beta_slow"))?.unwrap_or(1.0),
                ),
            ),
            ("truncate".into(), RopeValue::Bool(false)),
        ])),
        max_positions,
    )
}

fn positive_usize(value: i32, name: &str) -> Result<usize, ConfigError> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| invalid(format!("{name} must be positive, got {value}")))
}

fn gguf_vocab_size(
    metadata: &HashMap<String, MetadataValue>,
    fallback: &str,
) -> Result<i32, ConfigError> {
    gguf_i32(metadata, fallback)
}

fn gguf_string(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<String, ConfigError> {
    gguf_optional_string(metadata, key)?
        .ok_or_else(|| invalid(format!("GGUF metadata is missing {key:?}")))
}

fn gguf_optional_string(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<String>, ConfigError> {
    match metadata.get(key) {
        Some(MetadataValue::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(invalid(format!("GGUF metadata {key:?} has wrong type"))),
        None => Ok(None),
    }
}

fn gguf_i32(metadata: &HashMap<String, MetadataValue>, key: &str) -> Result<i32, ConfigError> {
    i32::try_from(
        gguf_optional_i64(metadata, key)?
            .ok_or_else(|| invalid(format!("GGUF metadata is missing {key:?}")))?,
    )
    .map_err(|_| invalid(format!("GGUF metadata {key:?} exceeds i32")))
}

fn gguf_optional_i64(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<i64>, ConfigError> {
    match metadata.get(key) {
        Some(value) => value
            .as_i64()
            .map(Some)
            .ok_or_else(|| invalid(format!("GGUF metadata {key:?} has wrong type"))),
        None => Ok(None),
    }
}

fn gguf_f32(metadata: &HashMap<String, MetadataValue>, key: &str) -> Result<f32, ConfigError> {
    gguf_optional_f32(metadata, key)?
        .ok_or_else(|| invalid(format!("GGUF metadata is missing {key:?}")))
}

fn gguf_optional_f32(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<f32>, ConfigError> {
    match metadata.get(key) {
        Some(value) => value
            .as_f32()
            .filter(|value| value.is_finite())
            .map(Some)
            .ok_or_else(|| invalid(format!("GGUF metadata {key:?} must be finite f32"))),
        None => Ok(None),
    }
}

fn f32_fingerprint(value: f32) -> String {
    format!("{:08x}", value.to_bits())
}

fn rope_value_fingerprint(value: &RopeValue) -> String {
    match value {
        RopeValue::Float(value) => format!("f32:{}", f32_fingerprint(*value)),
        RopeValue::String(value) => format!("string:{value}"),
        RopeValue::Bool(value) => format!("bool:{value}"),
    }
}

fn invalid(message: impl Into<String>) -> ConfigError {
    ConfigError::Invalid(message.into())
}
