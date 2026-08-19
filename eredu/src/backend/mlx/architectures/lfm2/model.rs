//! Liquid AI LFM2/LFM2.5 dense and mixture-of-experts text models.

use eredu_checkpoint::WeightQuantization;
use eredu_nn::RopeValue;
use eredu_runtime::CausalModel;

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    path::Path,
};

use safemlx::{
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::{Module, Param},
    nn,
    ops::{concatenate_axis, indexing::TryIndexOp, GgufCheckpoint, GgufMetadataValue},
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::core::cache::{
    derive_prompt_cache_architecture_fingerprint, PromptCacheDescriptor, PromptCacheManifest,
    PromptCacheModelIdentity, PromptCacheOptions,
};

use crate::{
    backend::mlx::architectures::qwen::dense::gguf_string,
    backend::mlx::error::Error,
    backend::mlx::nn::{
        self as common,
        attention::{
            apply_rope_and_update_cache, batch_seq, finish_attention, reshape_attention_projection,
        },
        convolution::{causal_depthwise_conv1d, CausalConv1dCache, DepthwiseConv1d},
        linear::project_logits_maybe_quantized,
        moe::{PackedSwiGluExperts, TopKRouterScoreFunction},
    },
    backend::mlx::nn::{
        parallel::forward_row_parallel,
        tensor::{
            create_attention_mask,
            rope::{initialize_rope, RopeVariant},
            AttentionMask,
        },
    },
    backend::mlx::runtime::cache::{
        residency::{
            open_prompt_cache_snapshot, save_prompt_cache_snapshot, CacheBlockArrays,
            CacheResidencyManager, CacheResidencyPolicy, CacheResidencyReport, PagedCacheOptions,
            PromptCacheSnapshotBlock, PromptCacheStateArray,
        },
        ConcatKeyValueCache, KeyValueCache, LiveKeyValueCache,
    },
    backend::mlx::runtime::checkpoint::load::{gguf_quantization_configs, GgufTensorNames},
    backend::mlx::runtime::media::input,
    core::attention::{AttentionPolicy, LayerSchedule},
    core::cache::{
        CacheRankIdentity, CacheResidencyPool, LayerCachePolicy, StateTensorDimension,
        StateTensorDtype, StateTensorOwner, StateTensorPolicy, StateTensorRole,
    },
};

fn default_true() -> bool {
    true
}

fn default_rope_theta() -> f32 {
    1_000_000.0
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

#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
/// Stateful operator used by an LFM2 decoder layer.
pub enum OperatorPolicy {
    /// Gated causal depthwise convolution.
    CausalConvolution,
    /// Grouped-query self-attention with its exact cache-retention policy.
    SelfAttention(AttentionPolicy),
}

impl OperatorPolicy {
    fn parse(value: &str) -> Result<Self, Error> {
        match value {
            "conv" => Ok(Self::CausalConvolution),
            "full_attention" => Ok(Self::SelfAttention(AttentionPolicy::Full)),
            other => Err(Error::UnsupportedArchitecture(format!(
                "LFM2 layer type {other:?} is unsupported"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
/// Feed-forward operator used by an LFM2 decoder layer.
pub enum FeedForwardPolicy {
    /// Dense SwiGLU feed-forward block.
    Dense,
    /// Routed sparse mixture-of-experts block.
    SparseMoe,
}

#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
/// Complete execution policy for one LFM2 decoder layer.
pub struct LayerPolicy {
    /// Stateful token-mixing operator.
    pub operator: OperatorPolicy,
    /// Feed-forward operator.
    pub feed_forward: FeedForwardPolicy,
}

#[derive(Debug, Clone, Copy)]
/// Canonical rotary-position configuration used by LFM2 attention layers.
pub struct RopeConfig {
    /// RoPE frequency base.
    pub theta: f32,
}

#[derive(Debug, Clone)]
/// Validated LFM2/LFM2.5 decoder configuration.
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
    fn into_args(self) -> Result<ModelArgs, Error> {
        let layer_count = usize::try_from(self.num_hidden_layers).map_err(|_| {
            Error::UnsupportedArchitecture(format!(
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
            return Err(Error::UnsupportedArchitecture(format!(
                "LFM2 dense config conflicts with num_dense_layers={}",
                self.num_dense_layers
            )));
        }
        if self.model_type == "lfm2_moe"
            && (self.num_dense_layers < 0 || self.num_dense_layers > self.num_hidden_layers)
        {
            return Err(Error::UnsupportedArchitecture(format!(
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
            .map_err(|error| Error::UnsupportedArchitecture(format!("LFM2 {error}")))?;
        if self
            .block_dim
            .is_some_and(|value| value != self.hidden_size)
        {
            return Err(Error::UnsupportedArchitecture(format!(
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
                return Err(Error::UnsupportedArchitecture(format!(
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
                    return Err(Error::UnsupportedArchitecture(
                        "LFM2 dense FFN adjustment requires a positive rounding multiple and finite positive multiplier"
                            .into(),
                    ));
                }
                size = 2 * size / 3;
                size = (self.block_ffn_dim_multiplier * size as f32) as i64;
                let multiple = i64::from(self.block_multiple_of);
                size = multiple * ((size + multiple - 1) / multiple);
            }
            i32::try_from(size).map_err(|_| {
                Error::UnsupportedArchitecture(format!(
                    "LFM2 adjusted dense intermediate size {size} exceeds i32"
                ))
            })?
        };
        let top_level_rope_theta = self.rope_theta;
        let rope_theta = match self.rope_parameters {
            Some(parameters) => {
                for key in parameters.keys() {
                    if !matches!(key.as_str(), "rope_theta" | "rope_type") {
                        return Err(Error::UnsupportedArchitecture(format!(
                            "LFM2 rope_parameters key {key:?} is unsupported"
                        )));
                    }
                }
                if let Some(rope_type) = parameters.get("rope_type") {
                    let RopeValue::String(rope_type) = rope_type else {
                        return Err(Error::UnsupportedArchitecture(
                            "LFM2 rope_parameters.rope_type must be \"default\"".into(),
                        ));
                    };
                    if rope_type != "default" {
                        return Err(Error::UnsupportedArchitecture(format!(
                            "LFM2 rope type {rope_type:?} is unsupported"
                        )));
                    }
                }
                let nested = match parameters.get("rope_theta") {
                    Some(RopeValue::Float(value)) => Some(*value),
                    Some(RopeValue::String(value)) => Some(value.parse::<f32>().map_err(|_| {
                        Error::UnsupportedArchitecture(format!(
                            "LFM2 rope_parameters.rope_theta {value:?} is not a float"
                        ))
                    })?),
                    Some(RopeValue::Bool(_)) => {
                        return Err(Error::UnsupportedArchitecture(
                            "LFM2 rope_parameters.rope_theta must be a float".into(),
                        ));
                    }
                    None => None,
                };
                if let (Some(top), Some(nested)) = (top_level_rope_theta, nested) {
                    if top != nested {
                        return Err(Error::UnsupportedArchitecture(format!(
                            "LFM2 rope_theta {top} conflicts with rope_parameters.rope_theta {nested}"
                        )));
                    }
                }
                nested
                    .or(top_level_rope_theta)
                    .unwrap_or_else(default_rope_theta)
            }
            None => top_level_rope_theta.unwrap_or_else(default_rope_theta),
        };
        let weight_quantization = match (self.quantization, self.quantization_config) {
            (Some(first), Some(second)) if first != second => {
                return Err(Error::Quantization(
                    "LFM2 quantization and quantization_config disagree".into(),
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

impl ModelArgs {
    /// Returns one validated layer policy without an out-of-range fallback.
    pub fn layer_policy(&self, layer: usize) -> Option<&LayerPolicy> {
        self.layer_schedule.get(layer)
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

    pub(crate) fn weight_quantization_for(&self, weight_name: &str) -> Option<WeightQuantization> {
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

/// Validates a parsed LFM2 configuration.
pub(crate) fn validate_model_config_value(config: &Value) -> Result<(), Error> {
    model_args_from_config_value(config).map(|_| ())
}

/// Normalizes and validates Hugging Face configuration into executable model arguments.
pub fn model_args_from_config_value(config: &Value) -> Result<ModelArgs, Error> {
    let source: ModelArgsSource = serde_json::from_value(config.clone())
        .map_err(|error| Error::UnsupportedArchitecture(format!("invalid LFM2 config: {error}")))?;
    let args = source.into_args()?;
    validate_args(&args)?;
    Ok(args)
}

pub(crate) fn prompt_cache_architecture_fingerprint(args: &ModelArgs) -> String {
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
        ],
    )
}

#[cfg(test)]
pub(crate) fn prompt_cache_layer_layout(
    args: &ModelArgs,
) -> Result<LayerSchedule<LayerCachePolicy>, Error> {
    prompt_cache_layer_layout_with_geometry(
        args,
        &args
            .layer_schedule
            .iter()
            .map(|policy| match policy.operator {
                OperatorPolicy::CausalConvolution => Lfm2LayerCacheGeometry {
                    kv_heads: None,
                    convolution_channels: Some(args.hidden_size),
                },
                OperatorPolicy::SelfAttention(_) => Lfm2LayerCacheGeometry {
                    kv_heads: Some(args.num_key_value_heads),
                    convolution_channels: None,
                },
            })
            .collect::<Vec<_>>(),
    )
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) struct Lfm2LayerCacheGeometry {
    pub kv_heads: Option<i32>,
    pub convolution_channels: Option<i32>,
}

pub(crate) fn prompt_cache_layer_layout_with_geometry(
    args: &ModelArgs,
    geometry: &[Lfm2LayerCacheGeometry],
) -> Result<LayerSchedule<LayerCachePolicy>, Error> {
    if geometry.len() != args.layer_schedule.len() {
        return Err(Error::UnsupportedArchitecture(format!(
            "LFM2 cache geometry has {} layers, expected {}",
            geometry.len(),
            args.layer_schedule.len()
        )));
    }
    let cache_error = |error: crate::core::cache::CachePolicyError| {
        Error::UnsupportedArchitecture(error.to_string())
    };
    let history = args
        .conv_l_cache
        .checked_sub(1)
        .ok_or_else(|| Error::UnsupportedArchitecture("invalid LFM2 convolution width".into()))?;
    let fixed = |value| StateTensorDimension::fixed(value).map_err(cache_error);
    args.layer_schedule
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
                        crate::MutableStateResidency::AlwaysDeviceMutable,
                    )
                    .map_err(cache_error)?])
                    .map_err(cache_error)
                }
                (OperatorPolicy::SelfAttention(attention), Some(kv_heads), None) => {
                    LayerCachePolicy::key_value(
                        attention,
                        kv_heads,
                        args.hidden_size / args.num_attention_heads,
                    )
                    .map_err(cache_error)
                }
                (OperatorPolicy::CausalConvolution, _, _) => Err(Error::UnsupportedArchitecture(
                    "LFM2 convolution layer requires only convolution-channel cache geometry"
                        .into(),
                )),
                (OperatorPolicy::SelfAttention(_), _, _) => Err(Error::UnsupportedArchitecture(
                    "LFM2 attention layer requires only KV-head cache geometry".into(),
                )),
            }
        })
        .collect::<Result<Vec<_>, _>>()
        .and_then(|policies| {
            LayerSchedule::new(args.layer_schedule.len(), policies)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
        })
}

fn validate_args(args: &ModelArgs) -> Result<(), Error> {
    if !matches!(args.model_type.as_str(), "lfm2" | "lfm2_moe") {
        return Err(Error::UnsupportedArchitecture(format!(
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
            return Err(Error::UnsupportedArchitecture(format!(
                "LFM2 {name} must be positive, got {value}"
            )));
        }
    }
    if args.layer_schedule.len() != args.num_hidden_layers as usize {
        return Err(Error::UnsupportedArchitecture(format!(
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
        return Err(Error::UnsupportedArchitecture(
            "LFM2 supports only full self-attention policies".into(),
        ));
    }
    if args.hidden_size % args.num_attention_heads != 0
        || args.num_attention_heads % args.num_key_value_heads != 0
    {
        return Err(Error::UnsupportedArchitecture(
            "LFM2 attention head counts do not divide hidden/query dimensions".into(),
        ));
    }
    if args.dense_intermediate_size <= 0 {
        return Err(Error::UnsupportedArchitecture(
            "LFM2 adjusted dense intermediate size must be positive".into(),
        ));
    }
    if !args.rope.theta.is_finite() || args.rope.theta <= 0.0 {
        return Err(Error::UnsupportedArchitecture(format!(
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
        return Err(Error::UnsupportedArchitecture(
            "LFM2 MoE expert configuration is invalid".into(),
        ));
    }
    if args.model_type == "lfm2" && args.has_sparse_moe_layers() {
        return Err(Error::UnsupportedArchitecture(
            "LFM2 dense config contains a sparse-MoE layer policy".into(),
        ));
    }
    Ok(())
}

/// LFM2 attention input.
pub struct AttentionInput<'a> {
    /// Hidden states.
    pub x: &'a Array,
    /// Optional causal mask.
    pub mask: Option<&'a Array>,
    /// Optional KV cache.
    pub cache: Option<&'a mut dyn KeyValueCache>,
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// QK-normalized grouped-query attention used by LFM2 full-attention layers.
pub struct Attention {
    /// Query head count.
    pub n_heads: i32,
    /// Key/value head count.
    pub n_kv_heads: i32,
    /// Head dimension.
    pub head_dim: i32,
    /// Attention scale.
    pub scale: f32,
    #[quantizable]
    #[param]
    /// Query projection.
    pub q_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    /// Key projection.
    pub k_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    /// Value projection.
    pub v_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    /// Output projection.
    pub out_proj: MaybeQuantized<nn::Linear>,
    #[param]
    /// Query head RMSNorm.
    pub q_layernorm: nn::RmsNorm,
    #[param]
    /// Key head RMSNorm.
    pub k_layernorm: nn::RmsNorm,
    #[param]
    /// Rotary embedding.
    pub rope: RopeVariant,
}

impl Attention {
    fn new(args: &ModelArgs, layer: i32, stream: &Stream) -> Result<Self, Exception> {
        Self::new_with_geometry(
            args,
            layer,
            args.num_attention_heads,
            args.num_key_value_heads,
            args.hidden_size / args.num_attention_heads,
            stream,
        )
    }

    fn new_with_geometry(
        args: &ModelArgs,
        layer: i32,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let prefix = format!("model.layers.{layer}.self_attn");
        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            scale: (head_dim as f32).sqrt().recip(),
            q_proj: common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                n_heads * head_dim,
                false,
                args.weight_quantization_for(&format!("{prefix}.q_proj.weight")),
                stream,
            )?,
            k_proj: common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                n_kv_heads * head_dim,
                false,
                args.weight_quantization_for(&format!("{prefix}.k_proj.weight")),
                stream,
            )?,
            v_proj: common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                n_kv_heads * head_dim,
                false,
                args.weight_quantization_for(&format!("{prefix}.v_proj.weight")),
                stream,
            )?,
            out_proj: common::linear::unloaded_maybe_quantized_linear(
                n_heads * head_dim,
                args.hidden_size,
                false,
                args.weight_quantization_for(&format!("{prefix}.out_proj.weight")),
                stream,
            )?,
            q_layernorm: nn::RmsNorm::unloaded(head_dim, args.norm_eps, Dtype::Float32, stream)?,
            k_layernorm: nn::RmsNorm::unloaded(head_dim, args.norm_eps, Dtype::Float32, stream)?,
            rope: initialize_rope(
                head_dim,
                args.rope.theta,
                false,
                &None,
                args.max_position_embeddings,
                stream,
            )?,
        })
    }
}

impl Module<AttentionInput<'_>> for Attention {
    type Output = Array;
    type Error = Exception;

    fn forward(
        &mut self,
        input: AttentionInput<'_>,
        stream: &Stream,
    ) -> Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, mut cache } = input;
        let (batch, seq_len) = batch_seq(x);
        let queries = self.q_layernorm.forward(
            &reshape_attention_projection(
                self.q_proj.forward(x, stream)?,
                batch,
                seq_len,
                self.n_heads,
                stream,
            )?,
            stream,
        )?;
        let keys = self.k_layernorm.forward(
            &reshape_attention_projection(
                self.k_proj.forward(x, stream)?,
                batch,
                seq_len,
                self.n_kv_heads,
                stream,
            )?,
            stream,
        )?;
        let values = reshape_attention_projection(
            self.v_proj.forward(x, stream)?,
            batch,
            seq_len,
            self.n_kv_heads,
            stream,
        )?;
        let (queries, keys, values) =
            apply_rope_and_update_cache(&mut self.rope, queries, keys, values, &mut cache, stream)?;
        let output = finish_attention(
            queries, keys, values, cache, self.scale, mask, batch, seq_len, stream,
        )?;
        self.out_proj.forward(&output, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        self.k_proj.training_mode(mode);
        self.v_proj.training_mode(mode);
        self.out_proj.training_mode(mode);
        self.q_layernorm.training_mode(mode);
        self.k_layernorm.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// LFM2 gated causal short convolution.
pub struct ShortConv {
    #[param]
    /// Depthwise convolution kernel.
    pub conv: DepthwiseConv1d,
    #[quantizable]
    #[param]
    /// Joint B/C/x projection.
    pub in_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    /// Output projection.
    pub out_proj: MaybeQuantized<nn::Linear>,
}

impl ShortConv {
    fn new(args: &ModelArgs, layer: i32, stream: &Stream) -> Result<Self, Exception> {
        Self::new_with_channels(args, layer, args.hidden_size, stream)
    }

    fn new_with_channels(
        args: &ModelArgs,
        layer: i32,
        channels: i32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let prefix = format!("model.layers.{layer}.conv");
        Ok(Self {
            conv: DepthwiseConv1d::new(channels, args.conv_l_cache, args.conv_bias, stream)?,
            in_proj: common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                3 * channels,
                args.conv_bias,
                args.weight_quantization_for(&format!("{prefix}.in_proj.weight")),
                stream,
            )?,
            out_proj: common::linear::unloaded_maybe_quantized_linear(
                channels,
                args.hidden_size,
                args.conv_bias,
                args.weight_quantization_for(&format!("{prefix}.out_proj.weight")),
                stream,
            )?,
        })
    }

    fn forward(
        &mut self,
        x: &Array,
        cache: Option<&mut CausalConv1dCache>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let projected = self.in_proj.forward(x, stream)?;
        let channels = projected.dim(-1) / 3;
        let b = projected.try_index_device((.., .., ..channels), stream)?;
        let c = projected.try_index_device((.., .., channels..2 * channels), stream)?;
        let x = projected.try_index_device((.., .., 2 * channels..), stream)?;
        let bx = b.multiply(x, stream)?;
        let convolution = causal_depthwise_conv1d(&self.conv, &bx, cache, stream)?;
        self.out_proj
            .forward(&c.multiply(convolution, stream)?, stream)
    }

    fn forward_tensor_parallel(
        &mut self,
        x: &Array,
        cache: Option<&mut CausalConv1dCache>,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let projected = self.in_proj.forward(x, stream)?;
        if projected.dim(-1) % 3 != 0 {
            return Err(Exception::custom(format!(
                "LFM2 fused short-convolution projection width {} is not divisible by three",
                projected.dim(-1)
            )));
        }
        let channels = projected.dim(-1) / 3;
        let b = projected.try_index_device((.., .., ..channels), stream)?;
        let c = projected.try_index_device((.., .., channels..2 * channels), stream)?;
        let x = projected.try_index_device((.., .., 2 * channels..), stream)?;
        let bx = b.multiply(x, stream)?;
        let convolution = causal_depthwise_conv1d(&self.conv, &bx, cache, stream)?;
        forward_row_parallel(
            &mut self.out_proj,
            &c.multiply(convolution, stream)?,
            group,
            stream,
        )
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// Dense or sparse LFM2 feed-forward block with checkpoint-compatible names.
pub struct FeedForward {
    /// Whether this block is sparse MoE.
    pub is_moe: bool,
    #[quantizable]
    #[param]
    /// Dense gate projection.
    pub w1: Option<MaybeQuantized<nn::Linear>>,
    #[quantizable]
    #[param]
    /// Dense down projection.
    pub w2: Option<MaybeQuantized<nn::Linear>>,
    #[quantizable]
    #[param]
    /// Dense up projection.
    pub w3: Option<MaybeQuantized<nn::Linear>>,
    #[param]
    /// Sparse router.
    pub gate: Option<common::moe::TopKRouter>,
    #[param]
    /// Packed routed experts.
    pub experts: Option<PackedSwiGluExperts>,
    #[param]
    /// Optional selection-only expert bias.
    pub expert_bias: Param<Option<Array>>,
}

impl FeedForward {
    fn dense(
        args: &ModelArgs,
        layer: i32,
        intermediate_size: i32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let prefix = format!("model.layers.{layer}.feed_forward");
        Ok(Self {
            is_moe: false,
            w1: Some(common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                intermediate_size,
                false,
                args.weight_quantization_for(&format!("{prefix}.w1.weight")),
                stream,
            )?),
            w2: Some(common::linear::unloaded_maybe_quantized_linear(
                intermediate_size,
                args.hidden_size,
                false,
                args.weight_quantization_for(&format!("{prefix}.w2.weight")),
                stream,
            )?),
            w3: Some(common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                intermediate_size,
                false,
                args.weight_quantization_for(&format!("{prefix}.w3.weight")),
                stream,
            )?),
            gate: None,
            experts: None,
            expert_bias: Param::new(None),
        })
    }

    fn moe(args: &ModelArgs, layer: i32, stream: &Stream) -> Result<Self, Exception> {
        let prefix = format!("model.layers.{layer}.feed_forward.experts");
        Ok(Self {
            is_moe: true,
            w1: None,
            w2: None,
            w3: None,
            gate: Some(common::moe::TopKRouter::new(
                common::moe::TopKRouterConfig {
                    top_k: args.num_experts_per_tok,
                    num_experts: args.num_experts,
                    hidden_size: args.hidden_size,
                    score_function: TopKRouterScoreFunction::Sigmoid,
                    norm_topk_prob: args.norm_topk_prob,
                    normalization_epsilon: 1e-6,
                    routed_scaling_factor: args.routed_scaling_factor,
                    n_group: 1,
                    topk_group: 1,
                    score_correction_bias: false,
                },
                stream,
            )?),
            experts: Some(PackedSwiGluExperts::new(
                args.num_experts,
                args.hidden_size,
                args.moe_intermediate_size,
                args.weight_quantization_for(&format!("{prefix}.gate_up_proj")),
                args.weight_quantization_for(&format!("{prefix}.down_proj")),
                stream,
            )?),
            expert_bias: if args.use_expert_bias {
                Param::<Option<Array>>::unloaded_some(&[args.num_experts], Dtype::Float32, stream)?
            } else {
                Param::new(None)
            },
        })
    }

    pub(crate) fn forward_with_expert_executor<F>(
        &mut self,
        x: &Array,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        if !self.is_moe {
            return self.forward(x, stream);
        }
        let shape = x.shape();
        let flat = x.reshape(&[-1, shape[2]], stream)?;
        let (indices, weights) = self
            .gate
            .as_mut()
            .expect("MoE gate")
            .forward_with_selection_bias(&flat, self.expert_bias.as_ref().as_ref(), stream)?;
        execute(&flat, &indices, &weights, stream)?.reshape(shape, stream)
    }

    pub(crate) fn forward_expert_parallel(
        &mut self,
        x: &Array,
        assignment: &crate::backend::mlx::architectures::distributed::expert::ExpertAssignment,
        group: &safemlx::distributed::Group,
        statistics: &mut crate::backend::mlx::architectures::distributed::expert::RoutingStatistics,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if !self.is_moe {
            return self.forward(x, stream);
        }
        let shape = x.shape();
        let flat = x.reshape(&[-1, shape[2]], stream)?;
        let router_started = std::time::Instant::now();
        let (indices, weights) = self
            .gate
            .as_mut()
            .expect("MoE gate")
            .forward_with_selection_bias(&flat, self.expert_bias.as_ref().as_ref(), stream)?;
        statistics.router_time += router_started.elapsed();
        let returned =
            crate::backend::mlx::architectures::distributed::expert::dispatch_replicated(
                &flat,
                &indices,
                &weights,
                assignment,
                self.experts.as_mut().expect("MoE experts"),
                group,
                stream,
            )
            .map_err(|error| Exception::custom(error.to_string()))?;
        statistics.accumulate(&returned.statistics);
        returned.reduced_output.reshape(shape, stream)
    }
}

impl Module<&Array> for FeedForward {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array, stream: &Stream) -> Result<Self::Output, Self::Error> {
        if !self.is_moe {
            let gate = self.w1.as_mut().expect("dense w1").forward(x, stream)?;
            let up = self.w3.as_mut().expect("dense w3").forward(x, stream)?;
            let hidden = common::layers::silu(gate, stream)?.multiply(up, stream)?;
            return self.w2.as_mut().expect("dense w2").forward(&hidden, stream);
        }
        let shape = x.shape();
        let flat = x.reshape(&[-1, shape[2]], stream)?;
        let (indices, weights) = self
            .gate
            .as_mut()
            .expect("MoE gate")
            .forward_with_selection_bias(&flat, self.expert_bias.as_ref().as_ref(), stream)?;
        self.experts
            .as_mut()
            .expect("MoE experts")
            .forward(&flat, &indices, &weights, stream)?
            .reshape(shape, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        for projection in [&mut self.w1, &mut self.w2, &mut self.w3]
            .into_iter()
            .flatten()
        {
            projection.training_mode(mode);
        }
    }
}

#[derive(Debug, Clone)]
/// Cache for one LFM2 operator layer.
pub enum LayerCache {
    /// Full-attention KV cache.
    Attention(LiveKeyValueCache),
    /// Short-convolution state.
    Conv(CausalConv1dCache),
}

/// Borrowed execution state for one LFM2 operator.
///
/// This separates decoder execution from the resident cache container so
/// generalized runtimes can supply any exact KV implementation alongside
/// descriptor-backed convolution state.
pub(crate) enum OperatorCache<'a> {
    /// Full-attention state implementing the shared KV contract.
    Attention(&'a mut dyn KeyValueCache),
    /// Short-convolution history.
    Convolution(&'a mut CausalConv1dCache),
}

impl LayerCache {
    pub(crate) fn new(policy: LayerPolicy) -> Self {
        match policy.operator {
            OperatorPolicy::CausalConvolution => Self::Conv(CausalConv1dCache::default()),
            // Match mlx-lm's KVCache growth policy. Chunked backing arrays
            // avoid concatenating the complete cache for every decode token.
            OperatorPolicy::SelfAttention(AttentionPolicy::Full) => Self::Attention(
                LiveKeyValueCache::resident(ConcatKeyValueCache::new_with_step(256)),
            ),
            OperatorPolicy::SelfAttention(AttentionPolicy::Sliding { .. }) => {
                unreachable!("validated LFM2 schedules contain only full self-attention")
            }
        }
    }

    pub(crate) fn offset(&self) -> i32 {
        match self {
            Self::Attention(cache) => cache.offset(),
            Self::Conv(cache) => cache.offset,
        }
    }

    pub(crate) fn retained_arrays(&self) -> Vec<&Array> {
        match self {
            Self::Attention(cache) => cache.retained_arrays(),
            Self::Conv(cache) => cache.state.iter().collect(),
        }
    }
}

#[derive(Debug, Clone)]
/// Heterogeneous LFM2 generation cache.
pub struct Cache {
    /// Per-layer operator caches.
    pub layers: Vec<LayerCache>,
}

impl Cache {
    pub(crate) fn new(args: &ModelArgs) -> Result<Self, Error> {
        Ok(Self {
            layers: args
                .layer_schedule
                .iter()
                .copied()
                .map(LayerCache::new)
                .collect(),
        })
    }

    pub(crate) fn new_paged(
        args: &ModelArgs,
        options: PagedCacheOptions,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, Exception> {
        let manager = CacheResidencyManager::new(options)
            .map_err(|error| Exception::custom(error.to_string()))?;
        Ok(Self {
            layers: args
                .layer_schedule
                .iter()
                .copied()
                .enumerate()
                .map(|(layer, policy)| match policy.operator {
                    OperatorPolicy::CausalConvolution => {
                        Ok(LayerCache::Conv(CausalConv1dCache::default()))
                    }
                    OperatorPolicy::SelfAttention(AttentionPolicy::Full) => {
                        LiveKeyValueCache::paged(manager.clone(), layer, None, 0, rank)
                            .map(LayerCache::Attention)
                    }
                    OperatorPolicy::SelfAttention(AttentionPolicy::Sliding { .. }) => {
                        unreachable!("validated LFM2 schedules contain only full self-attention")
                    }
                })
                .collect::<Result<Vec<_>, Exception>>()?,
        })
    }

    /// Returns aggregate live KV paging observations, if enabled.
    pub fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        self.layers
            .iter()
            .find_map(|layer| match layer {
                LayerCache::Attention(cache) => cache.residency_report().transpose(),
                LayerCache::Conv(_) => None,
            })
            .transpose()
    }

    /// Returns the aggregate process pool for paged attention state.
    pub fn residency_pool(&self) -> Option<&CacheResidencyPool> {
        self.layers.iter().find_map(|layer| match layer {
            LayerCache::Attention(cache) => cache.manager().map(CacheResidencyManager::pool),
            LayerCache::Conv(_) => None,
        })
    }

    /// Returns the consumed-token offset.
    pub fn offset(&self) -> i32 {
        self.layers.first().map(LayerCache::offset).unwrap_or(0)
    }

    pub(crate) fn reset(&mut self) -> Result<(), Exception> {
        if let Some(manager) = self.layers.iter().find_map(|layer| match layer {
            LayerCache::Attention(cache) => cache.manager().cloned(),
            LayerCache::Conv(_) => None,
        }) {
            manager
                .clear()
                .map_err(|error| Exception::custom(error.to_string()))?;
        }
        for layer in &mut self.layers {
            match layer {
                LayerCache::Attention(cache) => cache.reset_local_after_manager_clear(),
                LayerCache::Conv(cache) => *cache = CausalConv1dCache::default(),
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// One LFM2 decoder layer.
pub struct DecoderLayer {
    /// Operator kind.
    pub layer_policy: LayerPolicy,
    #[quantizable]
    #[param]
    /// Full attention operator.
    pub self_attn: Option<Attention>,
    #[quantizable]
    #[param]
    /// Short convolution operator.
    pub conv: Option<ShortConv>,
    #[quantizable]
    #[param]
    /// Dense or sparse feed-forward block.
    pub feed_forward: FeedForward,
    #[param]
    /// Operator pre-norm.
    pub operator_norm: nn::RmsNorm,
    #[param]
    /// Feed-forward pre-norm.
    pub ffn_norm: nn::RmsNorm,
}

impl DecoderLayer {
    pub(crate) fn new(args: &ModelArgs, index: i32, stream: &Stream) -> Result<Self, Error> {
        Self::new_with_widths(
            args,
            index,
            args.dense_intermediate_size,
            args.moe_intermediate_size,
            None,
            None,
            stream,
        )
    }

    pub(crate) fn new_with_widths(
        args: &ModelArgs,
        index: i32,
        dense_intermediate_size: i32,
        moe_intermediate_size: i32,
        attention_head_dim: Option<i32>,
        convolution_channels: Option<i32>,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let layer_policy = args
            .layer_schedule
            .get(index as usize)
            .copied()
            .ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "LFM2 layer schedule has no policy for layer {index}"
                ))
            })?;
        Ok(Self {
            layer_policy,
            self_attn: if matches!(layer_policy.operator, OperatorPolicy::SelfAttention(_)) {
                Some(match attention_head_dim {
                    Some(head_dim) => Attention::new_with_geometry(
                        args,
                        index,
                        args.num_attention_heads,
                        args.num_key_value_heads,
                        head_dim,
                        stream,
                    )?,
                    None => Attention::new(args, index, stream)?,
                })
            } else {
                None
            },
            conv: if layer_policy.operator == OperatorPolicy::CausalConvolution {
                Some(match convolution_channels {
                    Some(channels) => ShortConv::new_with_channels(args, index, channels, stream)?,
                    None => ShortConv::new(args, index, stream)?,
                })
            } else {
                None
            },
            feed_forward: match layer_policy.feed_forward {
                FeedForwardPolicy::SparseMoe => {
                    let mut local = args.clone();
                    local.moe_intermediate_size = moe_intermediate_size;
                    FeedForward::moe(&local, index, stream)?
                }
                FeedForwardPolicy::Dense => {
                    FeedForward::dense(args, index, dense_intermediate_size, stream)?
                }
            },
            operator_norm: nn::RmsNorm::unloaded(
                args.hidden_size,
                args.norm_eps,
                Dtype::Float32,
                stream,
            )?,
            ffn_norm: nn::RmsNorm::unloaded(
                args.hidden_size,
                args.norm_eps,
                Dtype::Float32,
                stream,
            )?,
        })
    }

    pub(crate) fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut LayerCache>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let cache = cache.map(|cache| match cache {
            LayerCache::Attention(cache) => OperatorCache::Attention(cache),
            LayerCache::Conv(cache) => OperatorCache::Convolution(cache),
        });
        self.forward_with_operator_cache(x, mask, cache, stream)
    }

    pub(crate) fn forward_with_operator_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<OperatorCache<'_>>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let normalized = self.operator_norm.forward(x, stream)?;
        let operator = match (self.layer_policy.operator, cache) {
            (OperatorPolicy::SelfAttention(_), Some(OperatorCache::Attention(cache))) => {
                self.self_attn.as_mut().expect("attention layer").forward(
                    AttentionInput {
                        x: &normalized,
                        mask,
                        cache: Some(cache),
                    },
                    stream,
                )?
            }
            (OperatorPolicy::SelfAttention(_), None) => {
                self.self_attn.as_mut().expect("attention layer").forward(
                    AttentionInput {
                        x: &normalized,
                        mask,
                        cache: None,
                    },
                    stream,
                )?
            }
            (OperatorPolicy::CausalConvolution, Some(OperatorCache::Convolution(cache))) => self
                .conv
                .as_mut()
                .expect("conv layer")
                .forward(&normalized, Some(cache), stream)?,
            (OperatorPolicy::CausalConvolution, None) => self
                .conv
                .as_mut()
                .expect("conv layer")
                .forward(&normalized, None, stream)?,
            (policy, Some(_)) => {
                return Err(Exception::custom(format!(
                    "LFM2 cache kind does not match layer policy {policy:?}"
                )))
            }
        };
        let h = x.add(operator, stream)?;
        let feed_forward = self
            .feed_forward
            .forward(&self.ffn_norm.forward(&h, stream)?, stream)?;
        h.add(feed_forward, stream)
    }

    /// Executes one hybrid LFM2 layer with local attention heads, convolution
    /// channels, and feed-forward intermediates.
    pub(crate) fn forward_tensor_parallel(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut LayerCache>,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let cache = cache.map(|cache| match cache {
            LayerCache::Attention(cache) => OperatorCache::Attention(cache),
            LayerCache::Conv(cache) => OperatorCache::Convolution(cache),
        });
        self.forward_tensor_parallel_with_operator_cache(x, mask, cache, group, stream)
    }

    pub(crate) fn forward_tensor_parallel_with_operator_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<OperatorCache<'_>>,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let normalized = self.operator_norm.forward(x, stream)?;
        let operator = match (self.layer_policy.operator, cache) {
            (OperatorPolicy::SelfAttention(_), Some(OperatorCache::Attention(cache))) => {
                let partial = self.self_attn.as_mut().expect("attention layer").forward(
                    AttentionInput {
                        x: &normalized,
                        mask,
                        cache: Some(cache),
                    },
                    stream,
                )?;
                safemlx::distributed::all_sum(&partial, group, stream)?
            }
            (OperatorPolicy::SelfAttention(_), None) => {
                let partial = self.self_attn.as_mut().expect("attention layer").forward(
                    AttentionInput {
                        x: &normalized,
                        mask,
                        cache: None,
                    },
                    stream,
                )?;
                safemlx::distributed::all_sum(&partial, group, stream)?
            }
            (OperatorPolicy::CausalConvolution, Some(OperatorCache::Convolution(cache))) => self
                .conv
                .as_mut()
                .expect("conv layer")
                .forward_tensor_parallel(&normalized, Some(cache), group, stream)?,
            (OperatorPolicy::CausalConvolution, None) => self
                .conv
                .as_mut()
                .expect("conv layer")
                .forward_tensor_parallel(&normalized, None, group, stream)?,
            (policy, Some(_)) => {
                return Err(Exception::custom(format!(
                    "LFM2 tensor-parallel cache kind does not match layer policy {policy:?}"
                )))
            }
        };
        let hidden = x.add(operator, stream)?;
        let normalized = self.ffn_norm.forward(&hidden, stream)?;
        let partial = self.feed_forward.forward(&normalized, stream)?;
        let feed_forward = safemlx::distributed::all_sum(&partial, group, stream)?;
        hidden.add(feed_forward, stream)
    }

    pub(crate) fn forward_with_expert_executor<F>(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut LayerCache>,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let cache = cache.map(|cache| match cache {
            LayerCache::Attention(cache) => OperatorCache::Attention(cache),
            LayerCache::Conv(cache) => OperatorCache::Convolution(cache),
        });
        self.forward_with_operator_cache_and_expert_executor(x, mask, cache, stream, execute)
    }

    pub(crate) fn forward_with_operator_cache_and_expert_executor<F>(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<OperatorCache<'_>>,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let normalized = self.operator_norm.forward(x, stream)?;
        let operator = match (self.layer_policy.operator, cache) {
            (OperatorPolicy::SelfAttention(_), Some(OperatorCache::Attention(cache))) => {
                self.self_attn.as_mut().expect("attention layer").forward(
                    AttentionInput {
                        x: &normalized,
                        mask,
                        cache: Some(cache),
                    },
                    stream,
                )?
            }
            (OperatorPolicy::SelfAttention(_), None) => {
                self.self_attn.as_mut().expect("attention layer").forward(
                    AttentionInput {
                        x: &normalized,
                        mask,
                        cache: None,
                    },
                    stream,
                )?
            }
            (OperatorPolicy::CausalConvolution, Some(OperatorCache::Convolution(cache))) => self
                .conv
                .as_mut()
                .expect("conv layer")
                .forward(&normalized, Some(cache), stream)?,
            (OperatorPolicy::CausalConvolution, None) => self
                .conv
                .as_mut()
                .expect("conv layer")
                .forward(&normalized, None, stream)?,
            (policy, Some(_)) => {
                return Err(Exception::custom(format!(
                    "LFM2 cache kind does not match layer policy {policy:?}"
                )))
            }
        };
        let h = x.add(operator, stream)?;
        let normalized = self.ffn_norm.forward(&h, stream)?;
        let feed_forward =
            self.feed_forward
                .forward_with_expert_executor(&normalized, stream, execute)?;
        h.add(feed_forward, stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_expert_parallel_with_operator_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<OperatorCache<'_>>,
        assignment: &crate::backend::mlx::architectures::distributed::expert::ExpertAssignment,
        group: &safemlx::distributed::Group,
        statistics: &mut crate::backend::mlx::architectures::distributed::expert::RoutingStatistics,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let normalized = self.operator_norm.forward(x, stream)?;
        let operator = match (self.layer_policy.operator, cache) {
            (OperatorPolicy::SelfAttention(_), Some(OperatorCache::Attention(cache))) => {
                self.self_attn.as_mut().expect("attention layer").forward(
                    AttentionInput {
                        x: &normalized,
                        mask,
                        cache: Some(cache),
                    },
                    stream,
                )?
            }
            (OperatorPolicy::SelfAttention(_), None) => {
                self.self_attn.as_mut().expect("attention layer").forward(
                    AttentionInput {
                        x: &normalized,
                        mask,
                        cache: None,
                    },
                    stream,
                )?
            }
            (OperatorPolicy::CausalConvolution, Some(OperatorCache::Convolution(cache))) => self
                .conv
                .as_mut()
                .expect("conv layer")
                .forward(&normalized, Some(cache), stream)?,
            (OperatorPolicy::CausalConvolution, None) => self
                .conv
                .as_mut()
                .expect("conv layer")
                .forward(&normalized, None, stream)?,
            (policy, Some(_)) => {
                return Err(Exception::custom(format!(
                    "LFM2 expert-parallel cache kind does not match layer policy {policy:?}"
                )))
            }
        };
        let hidden = x.add(operator, stream)?;
        let normalized = self.ffn_norm.forward(&hidden, stream)?;
        let feed_forward = self.feed_forward.forward_expert_parallel(
            &normalized,
            assignment,
            group,
            statistics,
            stream,
        )?;
        hidden.add(feed_forward, stream)
    }

    /// Executes TP-sharded hybrid operators and dense feed-forward blocks while
    /// delegating sparse routed experts to an EP-scoped executor.
    pub(crate) fn forward_tensor_with_expert_executor<F>(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut LayerCache>,
        group: &safemlx::distributed::Group,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let cache = cache.map(|cache| match cache {
            LayerCache::Attention(cache) => OperatorCache::Attention(cache),
            LayerCache::Conv(cache) => OperatorCache::Convolution(cache),
        });
        self.forward_tensor_with_operator_cache_and_expert_executor(
            x, mask, cache, group, stream, execute,
        )
    }

    /// Executes TP-sharded hybrid operators while delegating full-width routed
    /// experts to a caller. The expert result is already complete for this TP
    /// lane and therefore does not participate in the TP reduction.
    pub(crate) fn forward_tensor_with_operator_cache_and_expert_executor<F>(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<OperatorCache<'_>>,
        group: &safemlx::distributed::Group,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let normalized = self.operator_norm.forward(x, stream)?;
        let operator = match (self.layer_policy.operator, cache) {
            (OperatorPolicy::SelfAttention(_), Some(OperatorCache::Attention(cache))) => {
                let partial = self.self_attn.as_mut().expect("attention layer").forward(
                    AttentionInput {
                        x: &normalized,
                        mask,
                        cache: Some(cache),
                    },
                    stream,
                )?;
                safemlx::distributed::all_sum(&partial, group, stream)?
            }
            (OperatorPolicy::SelfAttention(_), None) => {
                let partial = self.self_attn.as_mut().expect("attention layer").forward(
                    AttentionInput {
                        x: &normalized,
                        mask,
                        cache: None,
                    },
                    stream,
                )?;
                safemlx::distributed::all_sum(&partial, group, stream)?
            }
            (OperatorPolicy::CausalConvolution, Some(OperatorCache::Convolution(cache))) => self
                .conv
                .as_mut()
                .expect("conv layer")
                .forward_tensor_parallel(&normalized, Some(cache), group, stream)?,
            (OperatorPolicy::CausalConvolution, None) => self
                .conv
                .as_mut()
                .expect("conv layer")
                .forward_tensor_parallel(&normalized, None, group, stream)?,
            (policy, Some(_)) => {
                return Err(Exception::custom(format!(
                    "LFM2 tensor/expert cache kind does not match layer policy {policy:?}"
                )))
            }
        };
        let hidden = x.add(operator, stream)?;
        let normalized = self.ffn_norm.forward(&hidden, stream)?;
        let feed_forward = if self.feed_forward.is_moe {
            self.feed_forward
                .forward_with_expert_executor(&normalized, stream, execute)?
        } else {
            let partial = self.feed_forward.forward(&normalized, stream)?;
            safemlx::distributed::all_sum(&partial, group, stream)?
        };
        hidden.add(feed_forward, stream)
    }

    /// Executes one layer whose operators and expert intermediates are TP
    /// sharded while routed expert ownership is partitioned by EP.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_tensor_expert_parallel_with_operator_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<OperatorCache<'_>>,
        tensor_group: &safemlx::distributed::Group,
        assignment: &crate::backend::mlx::architectures::distributed::expert::ExpertAssignment,
        expert_group: &safemlx::distributed::Group,
        statistics: &mut crate::backend::mlx::architectures::distributed::expert::RoutingStatistics,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let normalized = self.operator_norm.forward(x, stream)?;
        let operator = match (self.layer_policy.operator, cache) {
            (OperatorPolicy::SelfAttention(_), Some(OperatorCache::Attention(cache))) => {
                let partial = self.self_attn.as_mut().expect("attention layer").forward(
                    AttentionInput {
                        x: &normalized,
                        mask,
                        cache: Some(cache),
                    },
                    stream,
                )?;
                safemlx::distributed::all_sum(&partial, tensor_group, stream)?
            }
            (OperatorPolicy::SelfAttention(_), None) => {
                let partial = self.self_attn.as_mut().expect("attention layer").forward(
                    AttentionInput {
                        x: &normalized,
                        mask,
                        cache: None,
                    },
                    stream,
                )?;
                safemlx::distributed::all_sum(&partial, tensor_group, stream)?
            }
            (OperatorPolicy::CausalConvolution, Some(OperatorCache::Convolution(cache))) => self
                .conv
                .as_mut()
                .expect("conv layer")
                .forward_tensor_parallel(&normalized, Some(cache), tensor_group, stream)?,
            (OperatorPolicy::CausalConvolution, None) => self
                .conv
                .as_mut()
                .expect("conv layer")
                .forward_tensor_parallel(&normalized, None, tensor_group, stream)?,
            (policy, Some(_)) => {
                return Err(Exception::custom(format!(
                    "LFM2 tensor/expert cache kind does not match layer policy {policy:?}"
                )))
            }
        };
        let hidden = x.add(operator, stream)?;
        let normalized = self.ffn_norm.forward(&hidden, stream)?;
        let partial = if self.feed_forward.is_moe {
            self.feed_forward.forward_expert_parallel(
                &normalized,
                assignment,
                expert_group,
                statistics,
                stream,
            )?
        } else {
            self.feed_forward.forward(&normalized, stream)?
        };
        let feed_forward = safemlx::distributed::all_sum(&partial, tensor_group, stream)?;
        hidden.add(feed_forward, stream)
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// LFM2 transformer body.
pub struct Lfm2Model {
    #[quantizable]
    #[param]
    /// Token embeddings.
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[quantizable]
    #[param]
    /// Decoder layers.
    pub layers: Vec<DecoderLayer>,
    #[param]
    /// Final embedding normalization.
    pub embedding_norm: nn::RmsNorm,
}

impl Lfm2Model {
    fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Error> {
        Ok(Self {
            embed_tokens: common::linear::unloaded_maybe_quantized_embedding(
                args.vocab_size,
                args.hidden_size,
                args.weight_quantization_for("model.embed_tokens.weight"),
                stream,
            )?,
            layers: (0..args.num_hidden_layers)
                .map(|index| DecoderLayer::new(args, index, stream))
                .collect::<Result<Vec<_>, _>>()?,
            embedding_norm: nn::RmsNorm::unloaded(
                args.hidden_size,
                args.norm_eps,
                Dtype::Float32,
                stream,
            )?,
        })
    }

    fn forward(
        &mut self,
        inputs: &Array,
        mut cache: Option<&mut Cache>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let mut h = self.embed_tokens.forward(inputs, stream)?;
        let offset = cache.as_ref().map(|cache| cache.offset()).unwrap_or(0);
        let mask = if h.dim(1) > 1 {
            match create_attention_mask(&h, &offset_cache(offset), Some(true), stream)? {
                Some(AttentionMask::Array(mask)) => Some(mask),
                Some(AttentionMask::Causal) => {
                    return Err(Exception::custom("LFM2 requires an array causal mask"));
                }
                None => None,
            }
        } else {
            None
        };
        if let Some(cache) = cache.as_mut() {
            for (layer, layer_cache) in self.layers.iter_mut().zip(cache.layers.iter_mut()) {
                h = layer.forward(&h, mask.as_ref(), Some(layer_cache), stream)?;
            }
        } else {
            for layer in &mut self.layers {
                h = layer.forward(&h, mask.as_ref(), None, stream)?;
            }
        }
        self.embedding_norm.forward(&h, stream)
    }
}

fn offset_cache(offset: i32) -> Vec<Option<OffsetOnlyCache>> {
    vec![Some(OffsetOnlyCache { offset })]
}

struct OffsetOnlyCache {
    offset: i32,
}

impl KeyValueCache for OffsetOnlyCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn max_size(&self) -> Option<i32> {
        None
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
        _stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        Ok((keys, values))
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// LFM2 causal language model.
pub struct Model {
    /// Model configuration.
    pub args: ModelArgs,
    #[quantizable]
    #[param]
    /// Transformer body.
    pub model: Lfm2Model,
    #[quantizable]
    #[param]
    /// Optional untied language-model head.
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    /// Creates an unloaded LFM2 model.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        validate_args(&args)?;
        let model = Lfm2Model::new(&args, stream)?;
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                args.vocab_size,
                false,
                args.weight_quantization_for("lm_head.weight"),
                stream,
            )?)
        };
        Ok(Self {
            args,
            model,
            lm_head,
        })
    }

    /// Returns the configured model type.
    pub fn model_type(&self) -> &str {
        &self.args.model_type
    }

    /// Creates an empty heterogeneous cache.
    pub fn new_cache(&self) -> Cache {
        Cache::new(&self.args).expect("validated LFM2 layer schedule")
    }

    /// Creates resident heterogeneous state or pages growing attention state
    /// under the same policy used by layerwise and tensor-parallel execution.
    pub fn new_cache_with_options(&self, policy: CacheResidencyPolicy) -> Result<Cache, Exception> {
        match policy {
            CacheResidencyPolicy::Device => Ok(self.new_cache()),
            CacheResidencyPolicy::Paged(options) => Cache::new_paged(&self.args, options, None),
        }
    }

    pub(crate) fn save_prompt_cache_with_rank(
        cache: &mut Cache,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        rank: Option<CacheRankIdentity>,
        stream: &Stream,
    ) -> Result<PromptCacheManifest, Exception> {
        let end = i64::try_from(prefix_token_ids.len())
            .map_err(|_| Exception::custom("LFM2 prompt length exceeds i64"))?;
        let mut blocks = Vec::new();
        let mut state = Vec::new();
        let paged_manager = cache.layers.iter().find_map(|layer| match layer {
            LayerCache::Attention(cache) => cache.manager().cloned(),
            LayerCache::Conv(_) => None,
        });
        if paged_manager.is_some() {
            for layer in &mut cache.layers {
                if let LayerCache::Attention(cache) = layer {
                    cache.finalize()?;
                }
            }
        }
        for (layer, cache) in cache.layers.iter().enumerate() {
            if i64::from(cache.offset()) != end {
                return Err(Exception::custom(format!(
                    "LFM2 layer {layer} cache offset does not match the persisted prefix"
                )));
            }
            match cache {
                LayerCache::Attention(cache) => {
                    if paged_manager.is_some() {
                        continue;
                    }
                    let (keys, values) = cache.snapshot_arrays(stream)?.ok_or_else(|| {
                        Exception::custom(format!("LFM2 layer {layer} attention state is missing"))
                    })?;
                    blocks.push(PromptCacheSnapshotBlock {
                        global_layer: layer,
                        start: 0,
                        end,
                        rank,
                        arrays: CacheBlockArrays::KeyValue { keys, values },
                    });
                }
                LayerCache::Conv(cache) => {
                    let convolution = cache.state.as_ref().ok_or_else(|| {
                        Exception::custom(format!(
                            "LFM2 layer {layer} convolution state is missing"
                        ))
                    })?;
                    if convolution.dim(1) > 0 {
                        state.push(PromptCacheStateArray {
                            owner: StateTensorOwner::Layer(layer),
                            role: StateTensorRole::Convolution { slot: 0 },
                            array: convolution,
                        });
                    }
                }
            }
        }
        if let Some(manager) = paged_manager {
            return manager
                .save_prompt_cache(destination, descriptor, prefix_token_ids, &state, options)
                .map_err(|error| Exception::custom(error.to_string()));
        }
        save_prompt_cache_snapshot(
            destination,
            descriptor,
            prefix_token_ids,
            blocks,
            &state,
            options,
        )
        .map_err(|error| Exception::custom(error.to_string()))
    }

    #[cfg(test)]
    pub(crate) fn save_prompt_cache(
        cache: &mut Cache,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        stream: &Stream,
    ) -> Result<PromptCacheManifest, Exception> {
        Self::save_prompt_cache_with_rank(
            cache,
            destination,
            descriptor,
            prefix_token_ids,
            options,
            None,
            stream,
        )
    }

    #[cfg(test)]
    pub(crate) fn load_prompt_cache(
        args: &ModelArgs,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        stream: &Stream,
    ) -> Result<(Cache, PromptCacheManifest), Exception> {
        let layer_count = usize::try_from(args.num_hidden_layers)
            .map_err(|_| Exception::custom("invalid LFM2 cache layer count"))?;
        let identity = PromptCacheModelIdentity {
            model_family: "lfm2".into(),
            effective_model_type: args.model_type.clone(),
            architecture_fingerprint: prompt_cache_architecture_fingerprint(args),
            layer_count,
            global_layer_start: 0,
            global_layer_end: layer_count,
            sink_tokens: 0,
            layer_prefix_offsets: vec![0; layer_count],
            topology: Default::default(),
            layer_layout: prompt_cache_layer_layout(args)
                .map_err(|error| Exception::custom(error.to_string()))?,
        };
        Self::load_prompt_cache_with_identity(
            args,
            directory,
            expected,
            prefix_token_ids,
            identity,
            stream,
        )
    }

    pub(crate) fn load_prompt_cache_with_identity(
        args: &ModelArgs,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        identity: PromptCacheModelIdentity,
        stream: &Stream,
    ) -> Result<(Cache, PromptCacheManifest), Exception> {
        let (blocks, state, manifest) =
            open_prompt_cache_snapshot(directory, expected, &identity, prefix_token_ids, stream)
                .map_err(|error| Exception::custom(error.to_string()))?;
        let mut grouped_blocks: BTreeMap<usize, Vec<_>> = BTreeMap::new();
        for block in blocks {
            grouped_blocks
                .entry(block.global_layer)
                .or_default()
                .push(block);
        }
        let mut blocks = grouped_blocks;
        let mut state = state
            .into_iter()
            .map(|tensor| ((tensor.owner, tensor.role), tensor.array))
            .collect::<BTreeMap<_, _>>();
        let end = i32::try_from(prefix_token_ids.len())
            .map_err(|_| Exception::custom("LFM2 prompt length exceeds i32"))?;
        let mut cache = Cache::new(args).map_err(|error| Exception::custom(error.to_string()))?;
        for (layer, cache) in cache.layers.iter_mut().enumerate() {
            match cache {
                LayerCache::Attention(cache) => {
                    let mut layer_blocks = blocks.remove(&layer).ok_or_else(|| {
                        Exception::custom(format!(
                            "LFM2 layer {layer} attention prompt-cache block is missing"
                        ))
                    })?;
                    layer_blocks.sort_by_key(|block| block.start);
                    let mut expected_start = 0;
                    let mut keys = Vec::with_capacity(layer_blocks.len());
                    let mut values = Vec::with_capacity(layer_blocks.len());
                    for block in layer_blocks {
                        if block.start != expected_start {
                            return Err(Exception::custom(format!(
                                "LFM2 layer {layer} prompt-cache blocks are not contiguous"
                            )));
                        }
                        expected_start = block.end;
                        match block.arrays {
                            CacheBlockArrays::KeyValue {
                                keys: block_keys,
                                values: block_values,
                            } => {
                                keys.push(block_keys);
                                values.push(block_values);
                            }
                            CacheBlockArrays::CompressedLatentRotary { .. } => {
                                return Err(Exception::custom(format!(
                                    "LFM2 layer {layer} prompt-cache kind mismatch"
                                )));
                            }
                        }
                    }
                    if expected_start != i64::from(end) {
                        return Err(Exception::custom(format!(
                            "LFM2 layer {layer} prompt-cache blocks end at {expected_start}, expected {end}"
                        )));
                    }
                    let keys = concatenate_axis(&keys, -2, stream)?;
                    let values = concatenate_axis(&values, -2, stream)?;
                    cache.restore_resident(keys, values, end)?;
                }
                LayerCache::Conv(cache) => {
                    if args.conv_l_cache > 1 {
                        cache.state = Some(
                            state
                                .remove(&(
                                    StateTensorOwner::Layer(layer),
                                    StateTensorRole::Convolution { slot: 0 },
                                ))
                                .ok_or_else(|| {
                                    Exception::custom(format!(
                                        "LFM2 layer {layer} convolution state is missing"
                                    ))
                                })?,
                        );
                    }
                    cache.offset = end;
                }
            }
        }
        if !blocks.is_empty() || !state.is_empty() {
            return Err(Exception::custom("LFM2 prompt cache has unexpected state"));
        }
        Ok((cache, manifest))
    }

    pub(crate) fn forward_logits(
        &mut self,
        inputs: &Array,
        cache: Option<&mut Cache>,
        last_token_only: bool,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let hidden = self.model.forward(inputs, cache, stream)?;
        let hidden = if last_token_only {
            hidden.try_index_device((.., -1, ..), stream)?
        } else {
            hidden
        };
        project_logits_maybe_quantized(
            &mut self.lm_head,
            &mut self.model.embed_tokens,
            &hidden,
            stream,
        )
    }
}

impl CausalModel<Cache> for Model {
    type Tensor = Array;
    type Input<'a> = input::ModelInput<'a>;
    type Error = Exception;

    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let tokens = input::text_token_ids(input, stream)?;
        self.forward_logits(&tokens, Some(cache), true, stream)
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.forward_logits(input_tokens, Some(cache), true, stream)
    }
}

/// LFM2 token generation iterator.
pub type Generate<'a, S = crate::backend::mlx::runtime::generation::sampler::DefaultSampler> =
    common::generation::Generate<'a, Model, Cache, S>;

/// Reads and validates LFM2 model arguments.
pub fn get_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let value: Value =
        serde_json::from_reader(std::fs::File::open(model_dir.as_ref().join("config.json"))?)?;
    model_args_from_config_value(&value)
}

/// Loads the tokenizer stored next to an LFM2 checkpoint.
pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    Ok(Tokenizer::from_file(
        model_dir.as_ref().join("tokenizer.json"),
    )?)
}

pub(crate) struct PreparedLfm2Gguf {
    pub(crate) args: ModelArgs,
    pub(crate) eos_token_ids: Vec<u32>,
}

pub(crate) fn prepare_gguf_checkpoint(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    _weights_stream: &Stream,
) -> Result<PreparedLfm2Gguf, Error> {
    let architecture = gguf_string(metadata, "general.architecture")?;
    if !matches!(architecture.as_str(), "lfm2" | "lfm2moe") {
        return Err(Error::UnsupportedArchitecture(format!(
            "GGUF architecture {architecture:?}; this loader supports lfm2 and lfm2moe"
        )));
    }
    let is_moe = architecture == "lfm2moe";
    let gguf_architecture = crate::core::GgufArchitecture::resolve(&architecture)?;
    crate::backend::mlx::structural::validate_gguf(
        gguf_architecture,
        checkpoint,
        metadata,
        crate::backend::mlx::ModelLoadOptions::default(),
    )
    .into_loader_result()?;
    let mut args = args_from_gguf_catalog(checkpoint, metadata, &architecture, is_moe)?;
    let translate = |name: &str| translate_gguf_weight_name(name, is_moe);
    checkpoint
        .catalog()
        .translated_outputs(translate)
        .map_err(safemlx::error::IoError::from)?;
    let mut configs = gguf_quantization_configs(checkpoint, translate)?;
    if is_moe {
        for (layer, _) in args
            .layer_schedule
            .iter()
            .enumerate()
            .filter(|(_, policy)| policy.feed_forward == FeedForwardPolicy::SparseMoe)
        {
            let prefix = format!("model.layers.{layer}.feed_forward.experts");
            if let Some(config) = configs.remove(&format!("{prefix}.gate_proj")) {
                configs.remove(&format!("{prefix}.up_proj"));
                configs.insert(format!("{prefix}.gate_up_proj"), config);
            }
        }
    }
    args.quantized_weights = Some(configs.keys().cloned().collect());
    args.quantized_weight_configs = Some(configs);
    validate_args(&args)?;
    let eos_token_ids = crate::backend::mlx::gguf_eos_token_ids(metadata)?;
    Ok(PreparedLfm2Gguf {
        args,
        eos_token_ids,
    })
}

/// Parses GGUF arguments without creating an MLX stream.
pub(crate) fn args_from_gguf_catalog(
    arrays: &impl GgufTensorNames,
    metadata: &HashMap<String, GgufMetadataValue>,
    architecture: &str,
    is_moe: bool,
) -> Result<ModelArgs, Error> {
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let num_hidden_layers = gguf_i32_catalog(metadata, &key("block_count"))?;
    if num_hidden_layers <= 0 {
        return Err(Error::UnsupportedArchitecture(format!(
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
        gguf_i32_catalog(metadata, &key("leading_dense_block_count"))?
    } else {
        0
    };
    if num_dense_layers < 0 || num_dense_layers > num_hidden_layers {
        return Err(Error::UnsupportedArchitecture(format!(
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
    .map_err(|error| Error::UnsupportedArchitecture(format!("LFM2 GGUF {error}")))?;
    let vocab_size = match metadata
        .get("tokenizer.ggml.tokens")
        .and_then(GgufMetadataValue::as_strings)
    {
        Some(tokens) => i32::try_from(tokens.len()).map_err(|_| {
            Error::UnsupportedArchitecture("GGUF tokenizer vocabulary exceeds i32".into())
        })?,
        None if metadata.contains_key("tokenizer.ggml.tokens") => {
            return Err(Error::UnsupportedArchitecture(
                "GGUF tokenizer.ggml.tokens metadata has the wrong type".into(),
            ));
        }
        None => gguf_i32_catalog(metadata, &key("vocab_size"))?,
    };
    let expert_bias_name =
        |name: &str| name.contains("ffn_exp_probs_b") || name.contains("exp_probs_b");
    let args = ModelArgs {
        model_type: if is_moe { "lfm2_moe" } else { "lfm2" }.into(),
        vocab_size,
        hidden_size: gguf_i32_catalog(metadata, &key("embedding_length"))?,
        dense_intermediate_size: gguf_i32_catalog(metadata, &key("feed_forward_length"))?,
        num_hidden_layers,
        num_attention_heads: gguf_i32_catalog(metadata, &key("attention.head_count"))?,
        num_key_value_heads,
        max_position_embeddings: gguf_i32_catalog(metadata, &key("context_length"))?,
        norm_eps: gguf_f32(metadata, &key("attention.layer_norm_rms_epsilon"))?,
        layer_schedule,
        conv_l_cache: gguf_i32_catalog(metadata, &key("shortconv.l_cache"))?,
        conv_bias: arrays
            .any_gguf_tensor(|name| name.contains("shortconv") && name.ends_with(".bias")),
        tie_word_embeddings: !arrays.contains_gguf_tensor("output.weight"),
        rope: RopeConfig {
            theta: gguf_optional_f32(metadata, &key("rope.freq_base"))?
                .unwrap_or_else(default_rope_theta),
        },
        moe_intermediate_size: if is_moe {
            gguf_i32_catalog(metadata, &key("expert_feed_forward_length"))?
        } else {
            0
        },
        num_experts: if is_moe {
            gguf_i32_catalog(metadata, &key("expert_count"))?
        } else {
            0
        },
        num_experts_per_tok: if is_moe {
            gguf_i32_catalog(metadata, &key("expert_used_count"))?
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
        use_expert_bias: arrays.any_gguf_tensor(expert_bias_name),
        weight_quantization: None,
        quantized_weights: None,
        quantized_weight_configs: None,
    };
    validate_args(&args)?;
    Ok(args)
}

pub(crate) fn translate_gguf_weight_name(name: &str, is_moe: bool) -> String {
    for (source, target) in [
        ("token_embd", "model.embed_tokens"),
        ("token_embd_norm", "model.embedding_norm"),
        ("output", "lm_head"),
    ] {
        if name == source || name.starts_with(&format!("{source}.")) {
            return name.replacen(source, target, 1);
        }
    }
    let Some(rest) = name.strip_prefix("blk.") else {
        return name.to_string();
    };
    let Some((layer, parameter)) = rest.split_once('.') else {
        return name.to_string();
    };
    if is_moe {
        for (source, target) in [
            ("ffn_gate_inp", "feed_forward.gate"),
            ("ffn_gate_exps", "feed_forward.experts.gate_proj"),
            ("ffn_up_exps", "feed_forward.experts.up_proj"),
            ("ffn_down_exps", "feed_forward.experts.down_proj"),
            ("ffn_exp_probs_b", "feed_forward.expert_bias"),
            ("exp_probs_b", "feed_forward.expert_bias"),
        ] {
            if parameter == source || parameter.starts_with(&format!("{source}.")) {
                let suffix = parameter.strip_prefix(source).unwrap_or_default();
                let suffix = if target.ends_with("expert_bias") && suffix == ".bias" {
                    ""
                } else if target.contains("experts.") {
                    match suffix {
                        ".weight" => "",
                        ".scales" => "_scales",
                        ".biases" => "_biases",
                        other => other,
                    }
                } else {
                    suffix
                };
                return format!("model.layers.{layer}.{target}{suffix}");
            }
        }
    }
    for (source, target) in [
        ("shortconv.conv", "conv.conv"),
        ("shortconv.in_proj", "conv.in_proj"),
        ("shortconv.out_proj", "conv.out_proj"),
        ("attn_q_norm", "self_attn.q_layernorm"),
        ("attn_k_norm", "self_attn.k_layernorm"),
        ("attn_q", "self_attn.q_proj"),
        ("attn_k", "self_attn.k_proj"),
        ("attn_v", "self_attn.v_proj"),
        ("attn_output", "self_attn.out_proj"),
        ("attn_norm", "operator_norm"),
        ("ffn_norm", "ffn_norm"),
        ("ffn_gate", "feed_forward.w1"),
        ("ffn_down", "feed_forward.w2"),
        ("ffn_up", "feed_forward.w3"),
    ] {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            return format!(
                "model.layers.{layer}.{}",
                parameter.replacen(source, target, 1)
            );
        }
    }
    name.to_string()
}

fn gguf_i64_values(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<Vec<i64>, Error> {
    metadata
        .get(key)
        .and_then(GgufMetadataValue::to_i64_vec)
        .ok_or_else(|| {
            Error::UnsupportedArchitecture(format!("GGUF metadata is missing numeric key {key:?}"))
        })
}

fn gguf_i32_catalog(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<i32, Error> {
    let value = gguf_optional_i64(metadata, key)?.ok_or_else(|| {
        Error::UnsupportedArchitecture(format!("GGUF metadata is missing required key {key:?}"))
    })?;
    i32::try_from(value).map_err(|_| {
        Error::UnsupportedArchitecture(format!("GGUF metadata value {key:?} exceeds i32"))
    })
}

fn gguf_optional_i64(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<Option<i64>, Error> {
    match metadata.get(key) {
        Some(value) => value.as_i64().map(Some).ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "GGUF metadata key {key:?} must be a numeric scalar"
            ))
        }),
        None => Ok(None),
    }
}

fn gguf_f32(metadata: &HashMap<String, GgufMetadataValue>, key: &str) -> Result<f32, Error> {
    gguf_optional_f32(metadata, key)?.ok_or_else(|| {
        Error::UnsupportedArchitecture(format!("GGUF metadata is missing required key {key:?}"))
    })
}

fn gguf_optional_f32(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<Option<f32>, Error> {
    match metadata.get(key) {
        Some(value) => value.as_f32().map(Some).ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "GGUF metadata key {key:?} must be a numeric scalar"
            ))
        }),
        None => Ok(None),
    }
}

fn expand_layer_values(key: &str, values: Vec<i64>, layers: i32) -> Result<Vec<i32>, Error> {
    let values = if values.len() == 1 {
        vec![values[0]; layers as usize]
    } else if values.len() == layers as usize {
        values
    } else {
        return Err(Error::UnsupportedArchitecture(format!(
            "GGUF metadata key {key:?} has {} values for {layers} layers",
            values.len()
        )));
    };
    values
        .into_iter()
        .map(|value| {
            i32::try_from(value).map_err(|_| {
                Error::UnsupportedArchitecture(format!("GGUF metadata value {key:?} exceeds i32"))
            })
        })
        .collect()
}

fn unique_nonzero(key: &str, values: &[i32]) -> Result<i32, Error> {
    let Some(value) = values.iter().copied().find(|value| *value > 0) else {
        return Err(Error::UnsupportedArchitecture(format!(
            "GGUF metadata key {key:?} has no attention-layer value"
        )));
    };
    if values.iter().any(|other| *other > 0 && *other != value) {
        return Err(Error::UnsupportedArchitecture(format!(
            "GGUF metadata key {key:?} has non-uniform attention-layer values"
        )));
    }
    Ok(value)
}
