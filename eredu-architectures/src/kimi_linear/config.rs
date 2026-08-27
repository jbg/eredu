//! Strict backend-independent Kimi Linear configuration and state geometry.

use std::{collections::HashMap, io::Read};

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

use crate::GgufTensorCatalog;

fn default_model_type() -> String {
    "kimi_linear".into()
}
fn default_norm_eps() -> f32 {
    1e-5
}
fn default_rope_theta() -> f32 {
    10_000.0
}
fn default_conv_kernel() -> i32 {
    4
}
fn default_one() -> i32 {
    1
}
fn default_true() -> bool {
    true
}
fn default_moe_layer_freq() -> i32 {
    1
}
fn default_router_activation() -> String {
    "sigmoid".into()
}

/// Invalid or currently unsupported Kimi Linear configuration.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// JSON decoding failure.
    #[error("invalid Kimi Linear configuration: {0}")]
    Json(#[from] serde_json::Error),
    /// Invalid geometry or unsupported semantic policy.
    #[error("{0}")]
    Invalid(String),
}

fn invalid(message: impl Into<String>) -> ConfigError {
    ConfigError::Invalid(message.into())
}

/// KDA recurrent geometry shared by KDA layers.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct KdaConfig {
    /// KDA head count.
    pub num_heads: i32,
    /// Per-head key/value width.
    pub head_dim: i32,
    /// Causal convolution kernel width.
    pub short_conv_kernel_size: i32,
}

/// Stateful token mixer selected for one layer.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum AttentionKind {
    /// Kimi Delta Attention.
    Kda,
    /// No-positional multi-head latent attention.
    Mla,
}

/// Feed-forward policy selected for one layer.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum FeedForwardPolicy {
    /// Ordinary dense SwiGLU feed-forward network.
    Dense,
    /// Routed experts plus the shared expert.
    SparseMoe,
}

/// Complete physical layer policy.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub struct LayerPolicy {
    /// Token-mixing operator selected for this physical layer.
    pub attention: AttentionKind,
    /// Feed-forward operator selected for this physical layer.
    pub feed_forward: FeedForwardPolicy,
}

#[derive(Debug, Clone, Deserialize)]
struct LinearAttentionSource {
    kda_layers: Vec<i32>,
    full_attn_layers: Vec<i32>,
    num_heads: i32,
    head_dim: i32,
    #[serde(default = "default_conv_kernel")]
    short_conv_kernel_size: i32,
}

#[derive(Debug, Clone, Deserialize)]
struct Source {
    #[serde(default = "default_model_type")]
    model_type: String,
    vocab_size: i32,
    hidden_size: i32,
    num_hidden_layers: i32,
    num_attention_heads: i32,
    num_key_value_heads: i32,
    intermediate_size: i32,
    head_dim: i32,
    #[serde(alias = "max_position_embeddings")]
    model_max_length: i32,
    #[serde(default = "default_norm_eps")]
    rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    linear_attn_config: LinearAttentionSource,
    #[serde(alias = "n_routed_experts")]
    num_experts: i32,
    moe_intermediate_size: i32,
    kv_lora_rank: i32,
    #[serde(default)]
    q_lora_rank: Option<i32>,
    qk_nope_head_dim: i32,
    qk_rope_head_dim: i32,
    v_head_dim: i32,
    #[serde(default)]
    mla_use_nope: bool,
    #[serde(alias = "num_experts_per_tok")]
    num_experts_per_token: i32,
    #[serde(default = "default_one", alias = "n_shared_experts")]
    num_shared_experts: i32,
    #[serde(default = "default_router_activation")]
    moe_router_activation_func: String,
    #[serde(default = "default_true")]
    moe_renormalize: bool,
    routed_scaling_factor: f32,
    first_k_dense_replace: i32,
    #[serde(default = "default_moe_layer_freq")]
    moe_layer_freq: i32,
    #[serde(default = "default_true")]
    use_grouped_topk: bool,
    #[serde(alias = "n_group")]
    num_expert_group: i32,
    topk_group: i32,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default)]
    num_nextn_predict_layers: i32,
    #[serde(default)]
    quantization: Option<WeightQuantization>,
}

/// Validated Kimi Linear text configuration.
#[derive(Debug, Clone)]
pub struct ModelArgs {
    /// Canonical Hugging Face model type.
    pub model_type: String,
    /// Vocabulary row count.
    pub vocab_size: i32,
    /// Residual-stream width.
    pub hidden_size: i32,
    /// Number of physical decoder layers.
    pub num_hidden_layers: i32,
    /// MLA query-head count.
    pub num_attention_heads: i32,
    /// Declared MLA key/value-head count.
    pub num_key_value_heads: i32,
    /// Dense feed-forward intermediate width.
    pub intermediate_size: i32,
    /// Declared ordinary attention head width.
    pub head_dim: i32,
    /// Maximum admitted context length.
    pub model_max_length: i32,
    /// RMS normalization epsilon.
    pub rms_norm_eps: f32,
    /// Nominal rotary base retained for checkpoint identity.
    pub rope_theta: f32,
    /// Exact KDA/MLA and dense/MoE schedule.
    pub layer_schedule: LayerSchedule<LayerPolicy>,
    /// Shared KDA geometry.
    pub kda_config: KdaConfig,
    /// Routed expert count.
    pub num_experts: i32,
    /// Routed and shared expert intermediate width.
    pub moe_intermediate_size: i32,
    /// Compressed MLA key/value rank.
    pub kv_lora_rank: i32,
    /// Optional low-rank query projection width.
    pub q_lora_rank: Option<i32>,
    /// Non-positional MLA key/query width per head.
    pub qk_nope_head_dim: i32,
    /// Nominal positional MLA width per head.
    pub qk_rope_head_dim: i32,
    /// MLA value width per head.
    pub v_head_dim: i32,
    /// Whether MLA leaves the nominal positional subspace unrotated.
    pub mla_use_nope: bool,
    /// Routed experts selected per token.
    pub num_experts_per_token: i32,
    /// Shared expert count; currently required to be one.
    pub num_shared_experts: i32,
    /// Router score activation; currently required to be sigmoid.
    pub moe_router_activation_func: String,
    /// Whether selected route weights are renormalized.
    pub moe_renormalize: bool,
    /// Multiplier applied to routed expert output.
    pub routed_scaling_factor: f32,
    /// Whether routing selects expert groups before individual experts.
    pub use_grouped_topk: bool,
    /// Number of equal expert groups.
    pub num_expert_group: i32,
    /// Number of groups retained by grouped top-k selection.
    pub topk_group: i32,
    /// Whether the vocabulary head aliases the token embedding.
    pub tie_word_embeddings: bool,
    /// Embedded prediction depth, currently required to be zero.
    pub num_nextn_predict_layers: i32,
    /// Uniform checkpoint weight quantization.
    pub weight_quantization: Option<WeightQuantization>,
    /// Optional per-parameter quantization overrides.
    pub quantized_weight_configs: Option<HashMap<String, WeightQuantization>>,
    /// Whether MLA KV-B is stored as separate per-head projections.
    pub split_kv_b: bool,
}

impl Source {
    fn normalize(self) -> Result<ModelArgs, ConfigError> {
        let layers = usize::try_from(self.num_hidden_layers)
            .map_err(|_| invalid("Kimi Linear num_hidden_layers must be positive"))?;
        if layers == 0
            || self.first_k_dense_replace < 0
            || self.first_k_dense_replace > self.num_hidden_layers
            || self.moe_layer_freq <= 0
        {
            return Err(invalid("invalid Kimi Linear layer/MoE schedule geometry"));
        }
        let mut attention = vec![None; layers];
        for (name, indices, kind) in [
            (
                "KDA",
                &self.linear_attn_config.kda_layers,
                AttentionKind::Kda,
            ),
            (
                "MLA",
                &self.linear_attn_config.full_attn_layers,
                AttentionKind::Mla,
            ),
        ] {
            for &one_based in indices {
                if one_based <= 0 || one_based > self.num_hidden_layers {
                    return Err(invalid(format!(
                        "Kimi Linear {name} layer index {one_based} is out of range"
                    )));
                }
                if attention[one_based as usize - 1].replace(kind).is_some() {
                    return Err(invalid(format!(
                        "Kimi Linear layer {one_based} occurs more than once"
                    )));
                }
            }
        }
        if let Some(layer) = attention.iter().position(Option::is_none) {
            return Err(invalid(format!(
                "Kimi Linear KDA/MLA lists omit layer {}",
                layer + 1
            )));
        }
        let policies = attention
            .into_iter()
            .enumerate()
            .map(|(layer, attention)| LayerPolicy {
                attention: attention.expect("validated schedule"),
                feed_forward: if self.num_experts > 0
                    && layer as i32 >= self.first_k_dense_replace
                    && layer as i32 % self.moe_layer_freq == 0
                {
                    FeedForwardPolicy::SparseMoe
                } else {
                    FeedForwardPolicy::Dense
                },
            })
            .collect::<Vec<_>>();
        let args = ModelArgs {
            model_type: self.model_type,
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            intermediate_size: self.intermediate_size,
            head_dim: self.head_dim,
            model_max_length: self.model_max_length,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            layer_schedule: LayerSchedule::new(layers, policies)
                .map_err(|e| invalid(e.to_string()))?,
            kda_config: KdaConfig {
                num_heads: self.linear_attn_config.num_heads,
                head_dim: self.linear_attn_config.head_dim,
                short_conv_kernel_size: self.linear_attn_config.short_conv_kernel_size,
            },
            num_experts: self.num_experts,
            moe_intermediate_size: self.moe_intermediate_size,
            kv_lora_rank: self.kv_lora_rank,
            q_lora_rank: self.q_lora_rank,
            qk_nope_head_dim: self.qk_nope_head_dim,
            qk_rope_head_dim: self.qk_rope_head_dim,
            v_head_dim: self.v_head_dim,
            mla_use_nope: self.mla_use_nope,
            num_experts_per_token: self.num_experts_per_token,
            num_shared_experts: self.num_shared_experts,
            moe_router_activation_func: self.moe_router_activation_func,
            moe_renormalize: self.moe_renormalize,
            routed_scaling_factor: self.routed_scaling_factor,
            use_grouped_topk: self.use_grouped_topk,
            num_expert_group: self.num_expert_group,
            topk_group: self.topk_group,
            tie_word_embeddings: self.tie_word_embeddings,
            num_nextn_predict_layers: self.num_nextn_predict_layers,
            weight_quantization: self.quantization,
            quantized_weight_configs: None,
            split_kv_b: false,
        };
        args.validate()?;
        Ok(args)
    }
}

impl ModelArgs {
    /// Returns the exact policy for one physical layer.
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
            eredu_runtime::RoutedObservationPoint::new(format!("{unit_path}.mlp"), self.num_experts)
        })
    }
    /// Returns whether any layer uses routed experts.
    pub fn has_sparse_moe_layers(&self) -> bool {
        self.layer_schedule
            .iter()
            .any(|p| p.feed_forward == FeedForwardPolicy::SparseMoe)
    }
    /// Resolves a parameter's per-weight or uniform quantization policy.
    pub fn weight_quantization_for(&self, name: &str) -> Option<WeightQuantization> {
        self.quantized_weight_configs
            .as_ref()
            .and_then(|m| m.get(name).copied())
            .or(self.weight_quantization)
    }
    /// Validates all normalized geometry and admitted semantic policies.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.model_type != "kimi_linear" {
            return Err(invalid(format!(
                "unsupported model type {:?}",
                self.model_type
            )));
        }
        for (name, value) in [
            ("vocab_size", self.vocab_size),
            ("hidden_size", self.hidden_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("intermediate_size", self.intermediate_size),
            ("model_max_length", self.model_max_length),
            ("kv_lora_rank", self.kv_lora_rank),
            ("qk_nope_head_dim", self.qk_nope_head_dim),
            ("qk_rope_head_dim", self.qk_rope_head_dim),
            ("v_head_dim", self.v_head_dim),
            ("num_experts", self.num_experts),
            ("moe_intermediate_size", self.moe_intermediate_size),
            ("num_experts_per_token", self.num_experts_per_token),
            ("num_expert_group", self.num_expert_group),
            ("topk_group", self.topk_group),
            ("kda.num_heads", self.kda_config.num_heads),
            ("kda.head_dim", self.kda_config.head_dim),
            (
                "kda.short_conv_kernel_size",
                self.kda_config.short_conv_kernel_size,
            ),
        ] {
            if value <= 0 {
                return Err(invalid(format!(
                    "Kimi Linear {name} must be positive, got {value}"
                )));
            }
        }
        if !self.mla_use_nope {
            return Err(invalid("Kimi Linear currently requires mla_use_nope=true"));
        }
        if self.q_lora_rank.is_some_and(|v| v <= 0)
            || self.rms_norm_eps <= 0.0
            || self.rope_theta <= 0.0
            || self.routed_scaling_factor <= 0.0
            || self.num_experts_per_token > self.num_experts
            || self.num_experts % self.num_expert_group != 0
            || self.topk_group > self.num_expert_group
            || self.num_experts_per_token
                > self.topk_group * (self.num_experts / self.num_expert_group)
        {
            return Err(invalid("invalid Kimi Linear MLA/MoE dimensions"));
        }
        if self.num_shared_experts != 1 {
            return Err(invalid("Kimi Linear currently requires one shared expert"));
        }
        if self.moe_router_activation_func != "sigmoid" {
            return Err(invalid("Kimi Linear requires sigmoid routing"));
        }
        if self.num_nextn_predict_layers != 0 {
            return Err(invalid("Kimi Linear MTP layers are not implemented"));
        }
        if self.layer_schedule.len() != self.num_hidden_layers as usize {
            return Err(invalid("Kimi Linear schedule length mismatch"));
        }
        if let Some(q) = self.weight_quantization {
            q.validate().map_err(|e| invalid(e.to_string()))?;
        }
        Ok(())
    }
}

/// Parses and validates one Hugging Face configuration value.
pub fn model_args_from_config_value(value: &Value) -> Result<ModelArgs, ConfigError> {
    serde_json::from_value::<Source>(value.clone())?.normalize()
}

/// Reads, parses, and validates one Hugging Face configuration stream.
pub fn model_args_from_config_reader(mut reader: impl Read) -> Result<ModelArgs, ConfigError> {
    let mut data = String::new();
    reader
        .read_to_string(&mut data)
        .map_err(|e| invalid(e.to_string()))?;
    model_args_from_config_value(&serde_json::from_str(&data)?)
}

/// Parses and validates Kimi geometry from pure GGUF metadata and names.
pub fn model_args_from_gguf_catalog(
    catalog: &impl GgufTensorCatalog,
    metadata: &HashMap<String, MetadataValue>,
) -> Result<ModelArgs, ConfigError> {
    let architecture = gguf_string(metadata, "general.architecture")?;
    if architecture != "kimi-linear" {
        return Err(invalid(format!(
            "GGUF architecture {architecture:?}; expected kimi-linear"
        )));
    }
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let num_hidden_layers = gguf_i32(metadata, &key("block_count"))?;
    let num_attention_heads = gguf_i32(metadata, &key("attention.head_count"))?;
    let full_attn_layers = gguf_attention_layers(catalog, metadata, num_hidden_layers, &key)?;
    let kda_layers = (1..=num_hidden_layers)
        .filter(|layer| !full_attn_layers.contains(layer))
        .collect::<Vec<_>>();
    let qk_rope_head_dim = gguf_i32(metadata, &key("rope.dimension_count"))?;
    let qk_head_dim = gguf_i32(metadata, &key("attention.key_length_mla"))?;
    let qk_nope_head_dim = qk_head_dim.checked_sub(qk_rope_head_dim).ok_or_else(|| {
        invalid(format!(
            "Kimi MLA key length {qk_head_dim} is smaller than positional length {qk_rope_head_dim}"
        ))
    })?;
    let q_lora_rank = gguf_optional_i64(metadata, &key("attention.q_lora_rank"))?
        .map(i32::try_from)
        .transpose()
        .map_err(|_| invalid("GGUF query LoRA rank exceeds i32"))?
        .filter(|rank| *rank > 0);
    let vocab_size = gguf_i32(metadata, &key("vocab_size"))?;
    let gating = gguf_optional_i64(metadata, &key("expert_gating_func"))?.unwrap_or(2);
    if gating != 2 {
        return Err(invalid(format!(
            "Kimi expert_gating_func {gating} is unsupported; expected sigmoid (2)"
        )));
    }
    let hidden_size = gguf_i32(metadata, &key("embedding_length"))?;
    Source {
        model_type: "kimi_linear".into(),
        vocab_size,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads: 1,
        intermediate_size: gguf_i32(metadata, &key("feed_forward_length"))?,
        head_dim: hidden_size / num_attention_heads,
        model_max_length: gguf_i32(metadata, &key("context_length"))?,
        rms_norm_eps: gguf_f32(metadata, &key("attention.layer_norm_rms_epsilon"))?,
        rope_theta: gguf_optional_f32(metadata, &key("rope.freq_base"))?.unwrap_or(10_000.0),
        linear_attn_config: LinearAttentionSource {
            full_attn_layers,
            kda_layers,
            num_heads: num_attention_heads,
            head_dim: gguf_i32(metadata, &key("kda.head_dim"))?,
            short_conv_kernel_size: gguf_i32(metadata, &key("ssm.conv_kernel"))?,
        },
        num_experts: gguf_i32(metadata, &key("expert_count"))?,
        moe_intermediate_size: gguf_i32(metadata, &key("expert_feed_forward_length"))?,
        kv_lora_rank: gguf_i32(metadata, &key("attention.kv_lora_rank"))?,
        q_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim: gguf_i32(metadata, &key("attention.value_length_mla"))?,
        mla_use_nope: true,
        first_k_dense_replace: gguf_optional_i64(metadata, &key("leading_dense_block_count"))?
            .unwrap_or(1)
            .try_into()
            .map_err(|_| invalid("leading dense layer count exceeds i32"))?,
        moe_layer_freq: 1,
        num_experts_per_token: gguf_i32(metadata, &key("expert_used_count"))?,
        num_shared_experts: gguf_optional_i64(metadata, &key("expert_shared_count"))?
            .unwrap_or(1)
            .try_into()
            .map_err(|_| invalid("shared expert count exceeds i32"))?,
        num_expert_group: gguf_optional_i64(metadata, &key("expert_group_count"))?
            .unwrap_or(1)
            .try_into()
            .map_err(|_| invalid("expert group count exceeds i32"))?,
        topk_group: gguf_optional_i64(metadata, &key("expert_group_used_count"))?
            .unwrap_or(1)
            .try_into()
            .map_err(|_| invalid("expert group selection exceeds i32"))?,
        routed_scaling_factor: gguf_optional_f32(metadata, &key("expert_weights_scale"))?
            .unwrap_or(1.0),
        moe_renormalize: gguf_optional_bool(metadata, &key("expert_weights_norm"))?.unwrap_or(true),
        moe_router_activation_func: "sigmoid".into(),
        use_grouped_topk: true,
        tie_word_embeddings: !catalog.contains("output.weight"),
        num_nextn_predict_layers: 0,
        quantization: None,
    }
    .normalize()
    .map(|mut args| {
        args.split_kv_b = catalog.any(|name| name.contains(".attn_k_b."));
        args
    })
}

fn gguf_attention_layers(
    catalog: &impl GgufTensorCatalog,
    metadata: &HashMap<String, MetadataValue>,
    layers: i32,
    key: &impl Fn(&str) -> String,
) -> Result<Vec<i32>, ConfigError> {
    let metadata_key = key("attention.head_count_kv");
    if let Some(values) = metadata
        .get(&metadata_key)
        .and_then(MetadataValue::to_i64_vec)
    {
        if values.len() != layers as usize {
            return Err(invalid(format!(
                "GGUF per-layer attention head array has {} values for {layers} layers",
                values.len()
            )));
        }
        return Ok(values
            .into_iter()
            .enumerate()
            .filter_map(|(layer, heads)| (heads > 0).then_some(layer as i32 + 1))
            .collect());
    }
    Ok((0..layers)
        .filter(|layer| !catalog.any(|name| name.starts_with(&format!("blk.{layer}.ssm_"))))
        .map(|layer| layer + 1)
        .collect())
}

fn gguf_string(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<String, ConfigError> {
    match metadata.get(key) {
        Some(MetadataValue::String(value)) => Ok(value.clone()),
        Some(_) => Err(invalid(format!(
            "GGUF metadata key {key:?} has the wrong type"
        ))),
        None => Err(invalid(format!(
            "GGUF metadata is missing required key {key:?}"
        ))),
    }
}

fn gguf_i32(metadata: &HashMap<String, MetadataValue>, key: &str) -> Result<i32, ConfigError> {
    let value = gguf_optional_i64(metadata, key)?
        .ok_or_else(|| invalid(format!("GGUF metadata is missing required key {key:?}")))?;
    i32::try_from(value).map_err(|_| invalid(format!("GGUF metadata value {key:?} exceeds i32")))
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

fn gguf_optional_bool(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<bool>, ConfigError> {
    match metadata.get(key) {
        Some(MetadataValue::Bool(value)) => Ok(Some(*value)),
        Some(value) => value
            .as_i64()
            .map(|value| Some(value != 0))
            .ok_or_else(|| invalid(format!("GGUF metadata key {key:?} has the wrong type"))),
        None => Ok(None),
    }
}

/// Rank-local recurrent geometry resolved by semantic placement.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct LayerCacheGeometry {
    /// Rank-local KDA head count, or `None` for MLA layers.
    pub kda_heads: Option<i32>,
}

/// Declares global heterogeneous state geometry for all physical layers.
pub fn state_layout(args: &ModelArgs) -> Result<StateLayout, ConfigError> {
    let geometry = args
        .layer_schedule
        .iter()
        .map(|p| LayerCacheGeometry {
            kda_heads: (p.attention == AttentionKind::Kda).then_some(args.kda_config.num_heads),
        })
        .collect::<Vec<_>>();
    state_layout_with_geometry(args, &geometry)
}

/// Declares heterogeneous state using placement-resolved KDA head counts.
pub fn state_layout_with_geometry(
    args: &ModelArgs,
    geometry: &[LayerCacheGeometry],
) -> Result<StateLayout, ConfigError> {
    if geometry.len() != args.layer_schedule.len() {
        return Err(invalid("Kimi Linear state geometry length mismatch"));
    }
    let history = args.kda_config.short_conv_kernel_size - 1;
    let fixed = |v| StateTensorDimension::fixed(v).map_err(|e| invalid(e.to_string()));
    let policies = args
        .layer_schedule
        .iter()
        .zip(geometry)
        .enumerate()
        .map(
            |(layer, (policy, local))| match (policy.attention, local.kda_heads) {
                (AttentionKind::Kda, Some(heads)) => {
                    let width = heads
                        .checked_mul(args.kda_config.head_dim)
                        .ok_or_else(|| invalid("KDA width overflow"))?;
                    let mut tensors = (0..3)
                        .map(|slot| {
                            StateTensorPolicy::new(
                                StateTensorRole::Convolution { slot },
                                vec![StateTensorDimension::Batch, fixed(history)?, fixed(width)?],
                                StateTensorDtype::Floating,
                                MutableStateResidency::AlwaysDeviceMutable,
                            )
                            .map_err(|e| invalid(e.to_string()))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    tensors.push(
                        StateTensorPolicy::new(
                            StateTensorRole::Recurrent,
                            vec![
                                StateTensorDimension::Batch,
                                fixed(heads)?,
                                fixed(args.kda_config.head_dim)?,
                                fixed(args.kda_config.head_dim)?,
                            ],
                            StateTensorDtype::Float32,
                            MutableStateResidency::LayerScopedOffloadable,
                        )
                        .map_err(|e| invalid(e.to_string()))?,
                    );
                    LayerCachePolicy::fixed_only(tensors).map_err(|e| invalid(e.to_string()))
                }
                (AttentionKind::Mla, None) => LayerCachePolicy::compressed_latent_rotary(
                    AttentionPolicy::Full,
                    args.kv_lora_rank,
                    args.qk_rope_head_dim,
                )
                .map_err(|e| invalid(e.to_string())),
                _ => Err(invalid(format!(
                    "Kimi Linear state geometry mismatch at layer {layer}"
                ))),
            },
        )
        .collect::<Result<Vec<_>, _>>()?;
    let schedule =
        LayerSchedule::new(policies.len(), policies).map_err(|e| invalid(e.to_string()))?;
    StateLayout::new(schedule).map_err(|e| invalid(e.to_string()))
}

/// Derives stable prompt-cache architecture identity from normalized policy.
pub fn prompt_cache_architecture_fingerprint(args: &ModelArgs) -> String {
    derive_prompt_cache_architecture_fingerprint(
        "kimi-linear",
        [
            ("model_type", args.model_type.clone()),
            ("hidden_size", args.hidden_size.to_string()),
            ("num_hidden_layers", args.num_hidden_layers.to_string()),
            ("num_attention_heads", args.num_attention_heads.to_string()),
            ("kv_lora_rank", args.kv_lora_rank.to_string()),
            ("qk_nope_head_dim", args.qk_nope_head_dim.to_string()),
            ("qk_rope_head_dim", args.qk_rope_head_dim.to_string()),
            ("v_head_dim", args.v_head_dim.to_string()),
            ("kda_num_heads", args.kda_config.num_heads.to_string()),
            ("kda_head_dim", args.kda_config.head_dim.to_string()),
            (
                "kda_conv_kernel",
                args.kda_config.short_conv_kernel_size.to_string(),
            ),
            (
                "layer_schedule",
                args.layer_schedule
                    .iter()
                    .map(|p| {
                        format!(
                            "{}{}",
                            if p.attention == AttentionKind::Kda {
                                "k"
                            } else {
                                "m"
                            },
                            if p.feed_forward == FeedForwardPolicy::Dense {
                                "d"
                            } else {
                                "e"
                            }
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(","),
            ),
            ("quantization", format!("{:?}", args.weight_quantization)),
            (
                "quantized_weight_configs",
                crate::cache_identity::debug_map(args.quantized_weight_configs.as_ref()),
            ),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Value {
        serde_json::json!({
            "model_type":"kimi_linear","vocab_size":16,"hidden_size":12,"num_hidden_layers":2,
            "num_attention_heads":3,"num_key_value_heads":3,"intermediate_size":17,"head_dim":4,
            "model_max_length":64,"linear_attn_config":{"kda_layers":[1],"full_attn_layers":[2],"num_heads":3,"head_dim":4,"short_conv_kernel_size":3},
            "num_experts":2,"moe_intermediate_size":9,"kv_lora_rank":6,"qk_nope_head_dim":4,"qk_rope_head_dim":2,"v_head_dim":4,
            "mla_use_nope":true,"num_experts_per_token":1,"num_shared_experts":1,"routed_scaling_factor":1.0,
            "first_k_dense_replace":1,"num_expert_group":1,"topk_group":1
        })
    }

    #[test]
    fn normalizes_exact_kda_mla_and_dense_moe_schedules() {
        let args = model_args_from_config_value(&fixture()).unwrap();
        assert_eq!(
            *args.layer_schedule.get(0).unwrap(),
            LayerPolicy {
                attention: AttentionKind::Kda,
                feed_forward: FeedForwardPolicy::Dense
            }
        );
        assert_eq!(
            *args.layer_schedule.get(1).unwrap(),
            LayerPolicy {
                attention: AttentionKind::Mla,
                feed_forward: FeedForwardPolicy::SparseMoe
            }
        );
        let layout = state_layout(&args).unwrap();
        assert_eq!(layout.components(0).unwrap().len(), 4);
        assert!(matches!(
            layout.layers().get(1).unwrap(),
            LayerCachePolicy::CompressedLatentRotary { .. }
        ));
        assert!(args.routed_observation_point("model.layers.0", 0).is_none());
        let point = args.routed_observation_point("model.layers.1", 1).unwrap();
        assert_eq!(point.path(), "model.layers.1.mlp");
        assert_eq!(point.expert_count(), 2);
    }

    #[test]
    fn prompt_cache_fingerprint_includes_effective_quantization() {
        let mut args = model_args_from_config_value(&fixture()).unwrap();
        let dense = prompt_cache_architecture_fingerprint(&args);
        args.weight_quantization = Some(
            eredu_checkpoint::AffineQuantization::new(16, 4)
                .unwrap()
                .into(),
        );

        assert_ne!(dense, prompt_cache_architecture_fingerprint(&args));
    }

    #[test]
    fn rejects_nonzero_mtp_and_incomplete_layer_lists() {
        let mut value = fixture();
        value["num_nextn_predict_layers"] = 1.into();
        assert!(model_args_from_config_value(&value).is_err());
        let mut value = fixture();
        value["linear_attn_config"]["full_attn_layers"] = serde_json::json!([]);
        assert!(model_args_from_config_value(&value).is_err());
    }
}
