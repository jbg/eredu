//! Liquid AI LFM2/LFM2.5 dense and mixture-of-experts text models.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    path::Path,
};

use safemlx::{
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::{Module, ModuleParametersExt, Param},
    nn,
    ops::{concatenate_axis, indexing::TryIndexOp, GgufCheckpoint, GgufMetadataValue},
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::{
    api::{
        common::{
            self,
            attention::{
                apply_rope_and_update_cache, batch_seq, finish_attention,
                reshape_attention_projection,
            },
            convolution::{causal_depthwise_conv1d, CausalConv1dCache, DepthwiseConv1d},
            generation::CausalLm,
            linear::project_logits_maybe_quantized,
            moe::{PackedSwiGluExperts, TopKRouterScoreFunction},
        },
        input,
    },
    architectures::qwen::dense::gguf_string,
    error::Error,
    nn::{
        parallel::forward_row_parallel,
        tensor::{
            create_attention_mask,
            rope::{initialize_rope, FloatOrString, RopeVariant},
            AttentionMask,
        },
    },
    runtime::attention::{AttentionPolicy, LayerSchedule},
    runtime::cache::{
        residency::{
            derive_prompt_cache_architecture_fingerprint, open_prompt_cache_snapshot,
            save_prompt_cache_snapshot, CacheBlockArrays, CacheRankIdentity, LayerCachePolicy,
            PromptCacheDescriptor, PromptCacheManifest, PromptCacheModelIdentity,
            PromptCacheOptions, PromptCacheSnapshotBlock, PromptCacheStateArray,
            StateTensorDimension, StateTensorDtype, StateTensorOwner, StateTensorPolicy,
            StateTensorRole,
        },
        ConcatKeyValueCache, KeyValueCache,
    },
    runtime::checkpoint::load::{
        gguf_metadata, gguf_quantization_configs, load_named_array_strict,
        load_safetensors_dir_quantized_strict, load_safetensors_dir_strict,
        load_safetensors_dir_strict_with_split_swiglu_experts, GgufTensorNames, StrictLoadConfig,
        StrictLoadReport,
    },
    runtime::checkpoint::quantization::WeightQuantization,
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
    rope_parameters: Option<HashMap<String, FloatOrString>>,
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
                    let FloatOrString::String(rope_type) = rope_type else {
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
                    Some(FloatOrString::Float(value)) => Some(*value),
                    Some(FloatOrString::String(value)) => {
                        Some(value.parse::<f32>().map_err(|_| {
                            Error::UnsupportedArchitecture(format!(
                                "LFM2 rope_parameters.rope_theta {value:?} is not a float"
                            ))
                        })?)
                    }
                    Some(FloatOrString::Bool(_)) => {
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
    let cache_error = |error: crate::runtime::cache::residency::CacheResidencyError| {
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
    Attention(ConcatKeyValueCache),
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
            OperatorPolicy::SelfAttention(AttentionPolicy::Full) => {
                Self::Attention(ConcatKeyValueCache::new_with_step(256))
            }
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

    /// Returns the consumed-token offset.
    pub fn offset(&self) -> i32 {
        self.layers.first().map(LayerCache::offset).unwrap_or(0)
    }

    pub(crate) fn reset(&mut self) {
        for layer in &mut self.layers {
            match layer {
                LayerCache::Attention(cache) => cache.clear(),
                LayerCache::Conv(cache) => *cache = CausalConv1dCache::default(),
            }
        }
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
        let normalized = self.operator_norm.forward(x, stream)?;
        let operator = match (self.layer_policy.operator, cache) {
            (OperatorPolicy::SelfAttention(_), Some(LayerCache::Attention(cache))) => {
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
            (OperatorPolicy::CausalConvolution, Some(LayerCache::Conv(cache))) => self
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
        let normalized = self.operator_norm.forward(x, stream)?;
        let operator = match (self.layer_policy.operator, cache) {
            (OperatorPolicy::SelfAttention(_), Some(LayerCache::Attention(cache))) => {
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
            (OperatorPolicy::CausalConvolution, Some(LayerCache::Conv(cache))) => self
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

    fn forward_with_expert_executor<F>(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Exception>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let mut h = self.embed_tokens.forward(inputs, stream)?;
        let offset = cache.offset();
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
        for (index, (layer, layer_cache)) in self
            .layers
            .iter_mut()
            .zip(cache.layers.iter_mut())
            .enumerate()
        {
            h = if layer.layer_policy.feed_forward == FeedForwardPolicy::SparseMoe {
                layer.forward_with_expert_executor(
                    &h,
                    mask.as_ref(),
                    Some(layer_cache),
                    stream,
                    |flat, ids, weights, stream| execute(index, flat, ids, weights, stream),
                )?
            } else {
                layer.forward(&h, mask.as_ref(), Some(layer_cache), stream)?
            };
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

    pub(crate) fn save_prompt_cache_with_rank(
        cache: &Cache,
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
        for (layer, cache) in cache.layers.iter().enumerate() {
            if i64::from(cache.offset()) != end {
                return Err(Exception::custom(format!(
                    "LFM2 layer {layer} cache offset does not match the persisted prefix"
                )));
            }
            match cache {
                LayerCache::Attention(cache) => {
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
        cache: &Cache,
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
        let mut blocks = blocks
            .into_iter()
            .map(|block| (block.global_layer, block))
            .collect::<BTreeMap<_, _>>();
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
                    let block = blocks.remove(&layer).ok_or_else(|| {
                        Exception::custom(format!(
                            "LFM2 layer {layer} attention prompt-cache block is missing"
                        ))
                    })?;
                    match block.arrays {
                        CacheBlockArrays::KeyValue { keys, values } => {
                            cache.restore_resident(keys, values, end)?;
                        }
                        CacheBlockArrays::CompressedLatentRotary { .. } => {
                            return Err(Exception::custom(format!(
                                "LFM2 layer {layer} prompt-cache kind mismatch"
                            )))
                        }
                    }
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

    pub(crate) fn forward_cached_expert_parallel<F>(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        execute: F,
        stream: &Stream,
    ) -> Result<Array, Exception>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let hidden = self
            .model
            .forward_with_expert_executor(inputs, cache, execute, stream)?;
        project_logits_maybe_quantized(
            &mut self.lm_head,
            &mut self.model.embed_tokens,
            &hidden,
            stream,
        )
    }
}

impl CausalLm<Cache> for Model {
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
pub type Generate<'a, S = crate::runtime::generation::sampler::DefaultSampler> =
    common::generation::Generate<'a, Model, Cache, S>;

/// Reads and validates LFM2 model arguments.
pub fn get_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let value: Value =
        serde_json::from_reader(std::fs::File::open(model_dir.as_ref().join("config.json"))?)?;
    model_args_from_config_value(&value)
}

/// Loads an LFM2 safetensors checkpoint.
pub fn load_model(
    model_dir: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    crate::api::structural::validate_safetensors_load_path(
        crate::api::ModelKind::Lfm2,
        model_dir,
        crate::api::ModelLoadOptions::default(),
    )?;
    let args = get_model_args(model_dir)?;
    let mut model = Model::new(args.clone(), stream)?;
    let config = StrictLoadConfig::default();
    let mut report = StrictLoadReport::default();
    if args.has_sparse_moe_layers() {
        load_safetensors_dir_strict_with_split_swiglu_experts(
            &mut model,
            model_dir,
            weights_stream,
            stream,
            None,
            &config,
            &mut report,
            args.num_experts,
        )?;
    } else {
        load_safetensors_dir_strict(&mut model, model_dir, weights_stream, &config, &mut report)?;
    }
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;
    Ok(model)
}

/// Loads an LFM2 checkpoint while quantizing eligible projections.
pub fn load_model_quantized(
    model_dir: impl AsRef<Path>,
    quantization: WeightQuantization,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    crate::api::structural::validate_safetensors_load_path(
        crate::api::ModelKind::Lfm2,
        model_dir,
        crate::api::ModelLoadOptions::with_quantization(quantization),
    )?;
    let mut args = get_model_args(model_dir)?;
    if !crate::runtime::checkpoint::quantization::should_quantize_on_load(
        "LFM2",
        args.weight_quantization,
        quantization,
    )? {
        return load_model(model_dir, stream, weights_stream);
    }
    args.weight_quantization = Some(quantization);
    let mut model = Model::new(args.clone(), stream)?;
    let config = StrictLoadConfig::default();
    let mut report = StrictLoadReport::default();
    if args.has_sparse_moe_layers() {
        load_safetensors_dir_strict_with_split_swiglu_experts(
            &mut model,
            model_dir,
            weights_stream,
            stream,
            Some(quantization),
            &config,
            &mut report,
            args.num_experts,
        )?;
    } else {
        load_safetensors_dir_quantized_strict(
            &mut model,
            model_dir,
            weights_stream,
            stream,
            quantization,
            &config,
            &mut report,
        )?;
    }
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;
    Ok(model)
}

/// Loads the tokenizer stored next to an LFM2 checkpoint.
pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    Ok(Tokenizer::from_file(
        model_dir.as_ref().join("tokenizer.json"),
    )?)
}

pub(crate) struct LoadedLfm2Gguf {
    pub(crate) model: Model,
    pub(crate) eos_token_ids: Vec<u32>,
}

pub(crate) struct PreparedLfm2Gguf {
    pub(crate) args: ModelArgs,
    pub(crate) eos_token_ids: Vec<u32>,
}

/// Loads an LFM2 or LFM2-MoE GGUF checkpoint.
pub fn load_gguf(
    gguf_file: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let checkpoint = GgufCheckpoint::open(gguf_file)?;
    let metadata = gguf_metadata(&checkpoint);
    Ok(load_gguf_checkpoint(&checkpoint, metadata, None, stream, weights_stream)?.model)
}

pub(crate) fn load_gguf_checkpoint(
    checkpoint: &GgufCheckpoint,
    metadata: HashMap<String, GgufMetadataValue>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LoadedLfm2Gguf, Error> {
    let architecture = gguf_string(&metadata, "general.architecture")?;
    if !matches!(architecture.as_str(), "lfm2" | "lfm2moe") {
        return Err(Error::UnsupportedArchitecture(format!(
            "GGUF architecture {architecture:?}; this loader supports lfm2 and lfm2moe"
        )));
    }
    let is_moe = architecture == "lfm2moe";
    let gguf_architecture = crate::api::GgufArchitecture::resolve(&architecture)?;
    crate::api::structural::validate_gguf(
        gguf_architecture,
        checkpoint,
        &metadata,
        crate::api::ModelLoadOptions::default(),
    )
    .into_loader_result()?;
    let mut args = args_from_gguf_catalog(checkpoint, &metadata, &architecture, is_moe)?;
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
            for suffix in ["", "_scales", "_biases"] {
                let gate = format!("{prefix}.gate_proj{suffix}");
                let up = format!("{prefix}.up_proj{suffix}");
                let combined = format!("{prefix}.gate_up_proj{suffix}");
                if let Some(config) = configs.remove(&gate) {
                    configs.remove(&up);
                    configs.insert(combined, config);
                }
            }
        }
    }
    args.quantized_weights = Some(configs.keys().cloned().collect());
    args.quantized_weight_configs = Some(configs);
    if let Some(quantization) = quantization {
        args.weight_quantization = Some(quantization);
        args.quantized_weights = None;
        args.quantized_weight_configs = None;
    }
    validate_args(&args)?;

    let mut model = Model::new(args, stream)?;
    let config = StrictLoadConfig::default().allow_unused_prefix("rope_freqs.");
    let mut report = StrictLoadReport::default();
    let mut materializer = checkpoint.materializer();
    for tensor in checkpoint.catalog().tensors() {
        let physical_name = &tensor.descriptor().name;
        if is_moe
            && (physical_name.contains("ffn_gate_exps") || physical_name.contains("ffn_up_exps"))
        {
            continue;
        }
        for (name, mut value) in materializer.converted_tensor(physical_name)?.into_arrays() {
            if name.ends_with(".shortconv.conv.weight") && value.ndim() == 2 {
                value = value.reshape(&[value.dim(0), 1, value.dim(1)], weights_stream)?;
            }
            load_named_array_strict(
                &mut model,
                translate_gguf_weight_name(&name, is_moe),
                value,
                quantization.map(|value| (value, stream)),
                &config,
                &mut report,
            )?;
        }
    }
    if is_moe {
        let sparse_layers = model
            .args
            .layer_schedule
            .iter()
            .enumerate()
            .filter_map(|(layer, policy)| {
                (policy.feed_forward == FeedForwardPolicy::SparseMoe).then_some(layer)
            })
            .collect::<Vec<_>>();
        for layer in sparse_layers {
            let source_prefix = format!("blk.{layer}");
            let target_prefix = format!("model.layers.{layer}.feed_forward.experts");
            let gate =
                materializer.converted_tensor(&format!("{source_prefix}.ffn_gate_exps.weight"))?;
            let up =
                materializer.converted_tensor(&format!("{source_prefix}.ffn_up_exps.weight"))?;
            let gate = gate.into_arrays().into_iter().collect::<HashMap<_, _>>();
            let up = up.into_arrays().into_iter().collect::<HashMap<_, _>>();
            for (source_suffix, target_suffix) in
                [("weight", ""), ("scales", "_scales"), ("biases", "_biases")]
            {
                let gate_name = format!("{source_prefix}.ffn_gate_exps.{source_suffix}");
                let up_name = format!("{source_prefix}.ffn_up_exps.{source_suffix}");
                match (gate.get(&gate_name), up.get(&up_name)) {
                    (Some(gate), Some(up)) => {
                        let value =
                            concatenate_axis(&[gate.clone(), up.clone()], 1, weights_stream)?;
                        load_named_array_strict(
                            &mut model,
                            format!("{target_prefix}.gate_up_proj{target_suffix}"),
                            value,
                            quantization.map(|value| (value, stream)),
                            &config,
                            &mut report,
                        )?;
                    }
                    (None, None) if source_suffix == "biases" => {}
                    _ => {
                        return Err(Error::UnsupportedArchitecture(format!(
                        "LFM2 MoE GGUF has incomplete gate/up expert tensors under {source_prefix}"
                    )))
                    }
                }
            }
        }
    }
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;
    let eos_token_ids = crate::api::gguf_eos_token_ids(&metadata)?;
    Ok(LoadedLfm2Gguf {
        model,
        eos_token_ids,
    })
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
    let gguf_architecture = crate::api::GgufArchitecture::resolve(&architecture)?;
    crate::api::structural::validate_gguf(
        gguf_architecture,
        checkpoint,
        metadata,
        crate::api::ModelLoadOptions::default(),
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
    let eos_token_ids = crate::api::gguf_eos_token_ids(metadata)?;
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use safemlx::{
        module::ModuleParameters,
        ops::{GgufMetadataArray, GgufMetadataValue},
        Array, Device, DeviceType, ExecutionContext,
    };

    use super::{
        model_args_from_config_value, translate_gguf_weight_name, validate_model_config_value,
        FeedForwardPolicy, LayerPolicy, OperatorPolicy,
    };
    use crate::LayerSchedule;
    use serde_json::json;

    fn dense_config() -> serde_json::Value {
        json!({
            "model_type": "lfm2",
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "norm_eps": 0.00001,
            "conv_L_cache": 3,
            "block_multiple_of": 4,
            "block_ffn_dim_multiplier": 1.0,
            "block_auto_adjust_ff_dim": true,
            "layer_types": ["conv", "full_attention", "conv"],
            "tie_embedding": true
        })
    }

    #[test]
    fn parses_dense_schedule_and_adjusts_ffn() {
        let config = dense_config();
        let args = model_args_from_config_value(&config).unwrap();
        assert_eq!(
            args.layer_schedule.get(0),
            Some(&LayerPolicy {
                operator: OperatorPolicy::CausalConvolution,
                feed_forward: FeedForwardPolicy::Dense,
            })
        );
        assert_eq!(
            args.layer_schedule.get(1),
            Some(&LayerPolicy {
                operator: OperatorPolicy::SelfAttention(crate::AttentionPolicy::Full),
                feed_forward: FeedForwardPolicy::Dense,
            })
        );
        assert_eq!(args.dense_intermediate_size, 16);
        let cache = super::Cache::new(&args).unwrap();
        assert!(matches!(cache.layers[0], super::LayerCache::Conv(_)));
        assert!(matches!(cache.layers[1], super::LayerCache::Attention(_)));
        assert!(matches!(cache.layers[2], super::LayerCache::Conv(_)));
        validate_model_config_value(&config).unwrap();
    }

    #[test]
    fn prompt_cache_layout_records_convolution_and_attention_in_order() {
        use crate::runtime::cache::residency::{LayerCachePolicy, StateTensorRole};

        let args = model_args_from_config_value(&dense_config()).unwrap();
        let layout = super::prompt_cache_layer_layout(&args).unwrap();
        assert_eq!(layout.len(), 3);
        for layer in [0, 2] {
            match layout.get(layer).unwrap() {
                LayerCachePolicy::FixedState { tensors } => {
                    assert_eq!(tensors.len(), 1);
                    assert_eq!(tensors[0].role, StateTensorRole::Convolution { slot: 0 });
                }
                policy => panic!("unexpected LFM2 convolution policy {policy:?}"),
            }
        }
        assert!(matches!(
            layout.get(1).unwrap(),
            LayerCachePolicy::KeyValue { .. }
        ));
        let mut reordered = args.clone();
        reordered.layer_schedule = LayerSchedule::new(
            3,
            vec![
                LayerPolicy {
                    operator: OperatorPolicy::SelfAttention(crate::AttentionPolicy::Full),
                    feed_forward: FeedForwardPolicy::Dense,
                },
                LayerPolicy {
                    operator: OperatorPolicy::CausalConvolution,
                    feed_forward: FeedForwardPolicy::Dense,
                },
                LayerPolicy {
                    operator: OperatorPolicy::CausalConvolution,
                    feed_forward: FeedForwardPolicy::Dense,
                },
            ],
        )
        .unwrap();
        assert_ne!(
            super::prompt_cache_architecture_fingerprint(&args),
            super::prompt_cache_architecture_fingerprint(&reordered)
        );
    }

    #[test]
    fn rejects_bad_schedule_length() {
        let mut config = dense_config();
        config["layer_types"] = json!(["conv"]);
        assert!(validate_model_config_value(&config).is_err());
    }

    #[test]
    fn rejects_unknown_layer_policy_during_normalization() {
        let mut config = dense_config();
        config["layer_types"][1] = json!("linear_attention");
        let error = model_args_from_config_value(&config)
            .unwrap_err()
            .to_string();
        assert!(error.contains("layer type \"linear_attention\" is unsupported"));
    }

    #[test]
    fn source_aliases_normalize_into_canonical_geometry() {
        let mut config = dense_config();
        config["block_dim"] = json!(16);
        config["block_ff_dim"] = json!(24);
        config["block_norm_eps"] = json!(0.00001);
        config["rope_parameters"] = json!({
            "rope_theta": 1_000_000.0,
            "rope_type": "default"
        });
        validate_model_config_value(&config).unwrap();
        let args = model_args_from_config_value(&config).unwrap();
        assert!(args.tie_word_embeddings);
        assert_eq!(args.dense_intermediate_size, 16);
        assert_eq!(args.rope.theta, 1_000_000.0);
    }

    #[test]
    fn source_aliases_fail_closed_on_conflicting_geometry_or_rope() {
        let mut wrong_hidden = dense_config();
        wrong_hidden["block_dim"] = json!(32);
        assert!(model_args_from_config_value(&wrong_hidden)
            .unwrap_err()
            .to_string()
            .contains("block_dim must equal hidden_size"));

        let mut invalid_rope = dense_config();
        invalid_rope["rope_parameters"] = json!({
            "rope_theta": "not-a-number",
            "rope_type": "default"
        });
        assert!(model_args_from_config_value(&invalid_rope)
            .unwrap_err()
            .to_string()
            .contains("rope_parameters.rope_theta"));

        let mut unsupported_rope = dense_config();
        unsupported_rope["rope_parameters"] = json!({
            "rope_theta": 1_000_000.0,
            "rope_type": "yarn"
        });
        assert!(model_args_from_config_value(&unsupported_rope)
            .unwrap_err()
            .to_string()
            .contains("rope type \"yarn\" is unsupported"));

        let mut conflicting_rope = dense_config();
        conflicting_rope["rope_theta"] = json!(10_000.0);
        conflicting_rope["rope_parameters"] = json!({
            "rope_theta": 1_000_000.0,
            "rope_type": "default"
        });
        assert!(model_args_from_config_value(&conflicting_rope)
            .unwrap_err()
            .to_string()
            .contains("conflicts with rope_parameters.rope_theta"));

        let mut conflicting_quantization = dense_config();
        conflicting_quantization["quantization"] =
            json!({"group_size": 32, "bits": 4, "mode": "affine"});
        conflicting_quantization["quantization_config"] =
            json!({"group_size": 64, "bits": 4, "mode": "affine"});
        assert!(model_args_from_config_value(&conflicting_quantization)
            .unwrap_err()
            .to_string()
            .contains("quantization and quantization_config disagree"));
    }

    #[test]
    fn accepts_published_moe_shape() {
        let config = json!({
            "model_type": "lfm2_moe",
            "vocab_size": 65536,
            "hidden_size": 2048,
            "intermediate_size": 7168,
            "num_hidden_layers": 4,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "max_position_embeddings": 128000,
            "norm_eps": 0.00001,
            "conv_L_cache": 3,
            "layer_types": ["conv", "conv", "full_attention", "conv"],
            "moe_intermediate_size": 1792,
            "num_dense_layers": 2,
            "num_experts": 32,
            "num_experts_per_tok": 4,
            "norm_topk_prob": true,
            "use_expert_bias": true
        });
        let args = model_args_from_config_value(&config).unwrap();
        assert_eq!(
            args.layer_schedule
                .iter()
                .map(|policy| policy.feed_forward)
                .collect::<Vec<_>>(),
            vec![
                FeedForwardPolicy::Dense,
                FeedForwardPolicy::Dense,
                FeedForwardPolicy::SparseMoe,
                FeedForwardPolicy::SparseMoe,
            ]
        );
        assert_eq!(args.layer_schedule_fingerprint(), "cd,cd,afe,ce");
        validate_model_config_value(&config).unwrap();
    }

    #[test]
    fn rejects_invalid_or_conflicting_dense_prefix_metadata() {
        let mut moe = dense_config();
        moe["model_type"] = json!("lfm2_moe");
        moe["moe_intermediate_size"] = json!(8);
        moe["num_experts"] = json!(4);
        moe["num_experts_per_tok"] = json!(2);
        moe["num_dense_layers"] = json!(4);
        let error = model_args_from_config_value(&moe).unwrap_err().to_string();
        assert!(error.contains("num_dense_layers must be between 0 and 3"));

        let mut dense = dense_config();
        dense["num_dense_layers"] = json!(1);
        let error = model_args_from_config_value(&dense)
            .unwrap_err()
            .to_string();
        assert!(error.contains("dense config conflicts with num_dense_layers=1"));
    }

    #[test]
    fn inspection_and_loading_reject_invalid_prefix_with_same_diagnostic() {
        let mut config = dense_config();
        config["model_type"] = json!("lfm2_moe");
        config["moe_intermediate_size"] = json!(8);
        config["num_experts"] = json!(4);
        config["num_experts_per_tok"] = json!(2);
        config["num_dense_layers"] = json!(-1);
        let loading = model_args_from_config_value(&config)
            .unwrap_err()
            .to_string();
        let inspection = crate::api::resolve_model_config(&config)
            .unwrap_err()
            .to_string();
        assert_eq!(inspection, loading);
        assert!(loading.contains("num_dense_layers must be between 0 and 3"));
    }

    #[test]
    fn arbitrary_feed_forward_order_is_authoritative_and_fingerprinted() {
        let mut config = dense_config();
        config["model_type"] = json!("lfm2_moe");
        config["moe_intermediate_size"] = json!(8);
        config["num_experts"] = json!(4);
        config["num_experts_per_tok"] = json!(2);
        config["num_dense_layers"] = json!(1);
        let mut args = model_args_from_config_value(&config).unwrap();
        let baseline = args.layer_schedule_fingerprint();
        let baseline_cache_identity = super::prompt_cache_architecture_fingerprint(&args);
        args.layer_schedule = LayerSchedule::new(
            3,
            vec![
                LayerPolicy {
                    operator: OperatorPolicy::CausalConvolution,
                    feed_forward: FeedForwardPolicy::SparseMoe,
                },
                LayerPolicy {
                    operator: OperatorPolicy::SelfAttention(crate::AttentionPolicy::Full),
                    feed_forward: FeedForwardPolicy::Dense,
                },
                LayerPolicy {
                    operator: OperatorPolicy::CausalConvolution,
                    feed_forward: FeedForwardPolicy::SparseMoe,
                },
            ],
        )
        .unwrap();
        super::validate_args(&args).unwrap();
        assert_eq!(args.layer_schedule_fingerprint(), "ce,afd,ce");
        assert_ne!(args.layer_schedule_fingerprint(), baseline);
        assert_ne!(
            super::prompt_cache_architecture_fingerprint(&args),
            baseline_cache_identity
        );
        assert_eq!(
            args.layer_policy(1).unwrap().feed_forward,
            FeedForwardPolicy::Dense
        );
        assert!(args.layer_policy(3).is_none());
    }

    #[test]
    fn gguf_leading_dense_count_normalizes_into_combined_schedule() {
        let arrays = HashMap::<String, Array>::new();
        let mut metadata = HashMap::from([
            ("lfm2moe.block_count".into(), GgufMetadataValue::Uint32(4)),
            (
                "lfm2moe.attention.head_count_kv".into(),
                GgufMetadataValue::Array(GgufMetadataArray::Uint32(vec![0, 2, 0, 2])),
            ),
            (
                "lfm2moe.embedding_length".into(),
                GgufMetadataValue::Uint32(16),
            ),
            (
                "lfm2moe.feed_forward_length".into(),
                GgufMetadataValue::Uint32(24),
            ),
            (
                "lfm2moe.attention.head_count".into(),
                GgufMetadataValue::Uint32(4),
            ),
            (
                "lfm2moe.context_length".into(),
                GgufMetadataValue::Uint32(128),
            ),
            (
                "lfm2moe.attention.layer_norm_rms_epsilon".into(),
                GgufMetadataValue::Float32(1e-5),
            ),
            (
                "lfm2moe.shortconv.l_cache".into(),
                GgufMetadataValue::Uint32(3),
            ),
            ("lfm2moe.vocab_size".into(), GgufMetadataValue::Uint32(32)),
            (
                "lfm2moe.expert_feed_forward_length".into(),
                GgufMetadataValue::Uint32(8),
            ),
            (
                "lfm2moe.leading_dense_block_count".into(),
                GgufMetadataValue::Uint32(2),
            ),
            ("lfm2moe.expert_count".into(), GgufMetadataValue::Uint32(4)),
            (
                "lfm2moe.expert_used_count".into(),
                GgufMetadataValue::Uint32(2),
            ),
        ]);
        let args = super::args_from_gguf_catalog(&arrays, &metadata, "lfm2moe", true).unwrap();
        assert_eq!(args.layer_schedule_fingerprint(), "cd,afd,ce,afe");

        metadata.insert(
            "lfm2moe.leading_dense_block_count".into(),
            GgufMetadataValue::Uint32(5),
        );
        let error = super::args_from_gguf_catalog(&arrays, &metadata, "lfm2moe", true)
            .unwrap_err()
            .to_string();
        assert!(error.contains("leading_dense_block_count must be between 0 and 4"));

        metadata.insert(
            "lfm2moe.leading_dense_block_count".into(),
            GgufMetadataValue::String("2".into()),
        );
        let error = super::args_from_gguf_catalog(&arrays, &metadata, "lfm2moe", true)
            .unwrap_err()
            .to_string();
        assert!(error.contains("must be a numeric scalar"));
    }

    #[test]
    fn translates_dense_and_moe_gguf_tensor_names() {
        assert_eq!(
            translate_gguf_weight_name("token_embd.weight", false),
            "model.embed_tokens.weight"
        );
        assert_eq!(
            translate_gguf_weight_name("blk.2.shortconv.conv.weight", false),
            "model.layers.2.conv.conv.weight"
        );
        assert_eq!(
            translate_gguf_weight_name("blk.2.attn_q_norm.weight", false),
            "model.layers.2.self_attn.q_layernorm.weight"
        );
        assert_eq!(
            translate_gguf_weight_name("blk.3.ffn_gate_exps.scales", true),
            "model.layers.3.feed_forward.experts.gate_proj_scales"
        );
        assert_eq!(
            translate_gguf_weight_name("blk.3.ffn_exp_probs_b.bias", true),
            "model.layers.3.feed_forward.expert_bias"
        );
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn dense_parameter_tree_and_cache_match_public_checkpoint_layout() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let args = model_args_from_config_value(&dense_config()).unwrap();
        let model = super::Model::new(args, context.stream()).unwrap();
        let params = model.parameters().flatten();
        assert_eq!(
            params["model.layers.0.conv.conv.weight"].shape(),
            &[16, 1, 3]
        );
        assert!(params.contains_key("model.layers.0.conv.in_proj.weight"));
        assert!(params.contains_key("model.layers.0.feed_forward.w1.weight"));
        assert!(params.contains_key("model.layers.1.self_attn.q_proj.weight"));
        assert!(params.contains_key("model.layers.1.self_attn.q_layernorm.weight"));
        assert!(params.contains_key("model.embedding_norm.weight"));
        assert!(!params.contains_key("lm_head.weight"));
        let cache = model.new_cache();
        assert!(matches!(cache.layers[0], super::LayerCache::Conv(_)));
        assert!(matches!(cache.layers[1], super::LayerCache::Attention(_)));
        assert_eq!(cache.offset(), 0);
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn moe_parameter_tree_packs_experts_after_dense_prefix() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let config = json!({
            "model_type": "lfm2_moe",
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "norm_eps": 0.00001,
            "conv_L_cache": 3,
            "layer_types": ["conv", "full_attention"],
            "moe_intermediate_size": 8,
            "num_dense_layers": 1,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "norm_topk_prob": true,
            "use_expert_bias": true
        });
        let args = model_args_from_config_value(&config).unwrap();
        let model = super::Model::new(args, context.stream()).unwrap();
        let params = model.parameters().flatten();
        assert!(params.contains_key("model.layers.0.feed_forward.w1.weight"));
        assert!(params.contains_key("model.layers.1.feed_forward.gate.weight"));
        assert_eq!(
            params["model.layers.1.feed_forward.experts.gate_up_proj"].shape(),
            &[4, 16, 16]
        );
        assert_eq!(
            params["model.layers.1.feed_forward.experts.down_proj"].shape(),
            &[4, 16, 8]
        );
        assert_eq!(
            params["model.layers.1.feed_forward.expert_bias"].shape(),
            &[4]
        );
        assert!(!params.contains_key("model.layers.1.feed_forward.w1.weight"));
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn dense_and_moe_prefill_decode_smoke() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        let dense = model_args_from_config_value(&dense_config()).unwrap();
        let moe_config = json!({
            "model_type": "lfm2_moe",
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "norm_eps": 0.00001,
            "conv_L_cache": 3,
            "layer_types": ["conv", "full_attention", "conv"],
            "moe_intermediate_size": 8,
            "num_dense_layers": 1,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "norm_topk_prob": true,
            "use_expert_bias": true
        });
        let moe = model_args_from_config_value(&moe_config).unwrap();

        for args in [dense, moe] {
            let mut model = super::Model::new(args, stream).unwrap();
            for (_, parameter) in model.parameters_mut().flatten() {
                *parameter =
                    safemlx::ops::zeros_dtype(parameter.shape(), parameter.dtype(), stream)
                        .unwrap();
            }
            let mut cache = model.new_cache();
            let prompt = safemlx::Array::from_slice(&[1_u32, 2, 3], &[1, 3]);
            let parts = [crate::runtime::media::input::InputPart::text_token_ids(
                &prompt,
            )];
            let logits = crate::nn::generation::CausalLm::prefill_input_logits(
                &mut model,
                crate::runtime::media::input::ModelInput::new(&parts),
                &mut cache,
                stream,
            )
            .unwrap();
            assert_eq!(logits.shape(), &[1, 32]);
            assert_eq!(cache.offset(), 3);
            assert_eq!(logits.max(None, stream).unwrap().item::<f32>(stream), 0.0);

            let next = safemlx::Array::from_slice(&[4_u32], &[1, 1]);
            let logits = crate::nn::generation::CausalLm::decode_logits(
                &mut model, &next, &mut cache, stream,
            )
            .unwrap();
            assert_eq!(logits.shape(), &[1, 32]);
            assert_eq!(cache.offset(), 4);
        }
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn schema_v4_lfm2_save_drop_reload_continue_matches_uninterrupted() {
        use crate::{
            nn::generation::CausalLm,
            runtime::{
                cache::residency::{
                    PromptCacheDescriptor, PromptCacheOptions, PromptCacheTopology,
                },
                media::input::{InputPart, ModelInput},
            },
        };

        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        let args = model_args_from_config_value(&dense_config()).unwrap();
        let mut model = super::Model::new(args.clone(), stream).unwrap();
        for (index, parameter) in model.parameters_mut().flatten().values_mut().enumerate() {
            **parameter = safemlx::Array::full::<f32>(
                parameter.shape(),
                safemlx::Array::from_f32((index + 1) as f32 * 0.001),
                stream,
            )
            .unwrap();
        }
        let prefix_ids = [1_u32, 2, 3, 4];
        let prefix = safemlx::Array::from_slice(&prefix_ids, &[1, 4]);
        let parts = [InputPart::text_token_ids(&prefix)];
        let mut cache = model.new_cache();
        CausalLm::prefill_input_logits(&mut model, ModelInput::new(&parts), &mut cache, stream)
            .unwrap();
        assert_eq!(cache.offset(), 4);
        match &cache.layers[0] {
            super::LayerCache::Conv(cache) => {
                assert_eq!(cache.state.as_ref().unwrap().shape(), &[1, 2, 16]);
            }
            _ => panic!("expected LFM2 convolution cache"),
        }
        match &cache.layers[1] {
            super::LayerCache::Attention(cache) => {
                assert_eq!(cache.snapshot_arrays(stream).unwrap().unwrap().0.dim(-2), 4);
            }
            _ => panic!("expected LFM2 attention cache"),
        }
        let mut uninterrupted_cache = cache.clone();
        let suffix = safemlx::Array::from_slice(&[5_u32], &[1, 1]);
        let uninterrupted =
            CausalLm::decode_logits(&mut model, &suffix, &mut uninterrupted_cache, stream).unwrap();
        let layout = super::prompt_cache_layer_layout(&args).unwrap();
        let descriptor = PromptCacheDescriptor {
            model_family: "lfm2".into(),
            effective_model_type: args.model_type.clone(),
            checkpoint_fingerprint: "deterministic-fixture".into(),
            prefix_content_fingerprint: "tokens:1,2,3,4".into(),
            architecture_fingerprint: super::prompt_cache_architecture_fingerprint(&args),
            layer_count: layout.len(),
            global_layer_start: 0,
            global_layer_end: layout.len(),
            batch_size: 1,
            layer_layout: layout,
            sink_tokens: 0,
            topology: PromptCacheTopology::default(),
        };
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("prompt-cache");
        super::Model::save_prompt_cache(
            &cache,
            &destination,
            descriptor.clone(),
            &prefix_ids,
            &PromptCacheOptions::default(),
            stream,
        )
        .unwrap();
        drop(cache);
        let (mut restored, _) =
            super::Model::load_prompt_cache(&args, &destination, &descriptor, &prefix_ids, stream)
                .unwrap();
        assert_eq!(restored.offset(), 4);
        match &restored.layers[2] {
            super::LayerCache::Conv(cache) => {
                assert_eq!(cache.state.as_ref().unwrap().shape(), &[1, 2, 16]);
            }
            _ => panic!("expected restored LFM2 convolution cache"),
        }
        match &restored.layers[1] {
            super::LayerCache::Attention(cache) => {
                assert_eq!(cache.snapshot_arrays(stream).unwrap().unwrap().0.dim(-2), 4);
            }
            _ => panic!("expected restored LFM2 attention cache"),
        }
        let continued =
            CausalLm::decode_logits(&mut model, &suffix, &mut restored, stream).unwrap();
        assert!(uninterrupted
            .all_close(&continued, 1e-5, 1e-5, None, stream)
            .unwrap()
            .item::<bool>(stream));
    }
}
