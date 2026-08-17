//! Nemotron-H configuration parsing, runtime blocks, and strict checkpoint loading.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    num::NonZeroU32,
    path::Path,
};

use safemlx::{
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::{Module, ModuleParametersExt, Param},
    native_quantization::{native_grouped_linear, NativeQuantizedTensor},
    nn,
    ops::{
        broadcast_to, concatenate_axis, exp, gather_grouped_rows, grouped_matmul,
        indexing::{NewAxis, TryIndexOp},
        quantized_packed_dimension, sigmoid, sum_axis, topk_route_plan, zeros, GgufCheckpoint,
        GgufMetadataValue,
    },
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
    api::{
        common::{
            self,
            attention::{batch_seq, finish_attention, reshape_attention_projection},
            convolution::DepthwiseConv1d,
            generation::CausalLm,
            layers::relu2,
            linear::project_logits_maybe_quantized,
            moe::{packed_grouped_linear, weighted_route_sum, TopKRouterScoreFunction},
        },
        input,
    },
    core::cache::{
        CacheRankIdentity, LayerCachePolicy, StateTensorDimension, StateTensorDtype,
        StateTensorOwner, StateTensorPolicy, StateTensorRole,
    },
    error::Error,
    nn::tensor::{create_attention_mask, AttentionMask},
    runtime::attention::{AttentionPolicy, LayerSchedule},
    runtime::cache::{
        residency::{
            load_prompt_cache_state_tensors, open_prompt_cache, save_prompt_cache_snapshot,
            CacheBlockArrays, CacheResidencyManager, CacheResidencyPolicy, CacheResidencyReport,
            PagedCacheOptions, PromptCacheSnapshotBlock, PromptCacheStateArray,
        },
        ConcatKeyValueCache, KeyValueCache, PagedKeyValueCache,
    },
    runtime::checkpoint::load::{
        gguf_metadata, gguf_quantization_configs, load_gguf_strict,
        load_safetensors_dir_strict_with_split_relu2_experts, transform_split_relu2_experts,
        GgufTensorNames, StrictLoadConfig, StrictLoadReport,
    },
    runtime::checkpoint::quantization::{AffineQuantization, WeightQuantization},
};

/// Executable operator and state policy for one Nemotron-H decoder layer.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum LayerPolicy {
    /// Mamba2 state-space layer.
    Mamba,
    /// Grouped-query self-attention layer.
    SelfAttention(AttentionPolicy),
    /// Dense feed-forward MLP layer.
    DenseMlp,
    /// Sparse mixture-of-experts feed-forward layer.
    SparseMoe,
}

impl LayerPolicy {
    fn from_pattern_char(ch: char, attention: AttentionPolicy) -> Result<Self, Error> {
        match ch {
            'M' => Ok(Self::Mamba),
            '*' => Ok(Self::SelfAttention(attention)),
            '-' => Ok(Self::DenseMlp),
            'E' => Ok(Self::SparseMoe),
            other => Err(Error::UnsupportedArchitecture(format!(
                "Nemotron-H hybrid_override_pattern contains unsupported layer marker '{other}'"
            ))),
        }
    }
}

/// Deserialized Nemotron-H configuration fields used by this loader.
#[derive(Debug, Clone, Deserialize)]
struct ModelArgsSource {
    /// Model type from the configuration.
    pub model_type: String,
    /// Token vocabulary size.
    #[serde(default = "default_vocab_size")]
    pub vocab_size: i32,
    /// Whether logits use tied input embeddings.
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Transformer hidden size.
    #[serde(default = "default_hidden_size")]
    pub hidden_size: i32,
    /// Dense MLP intermediate size.
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: i32,
    /// Number of hybrid layers.
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: i32,
    #[serde(default)]
    pub num_nextn_predict_layers: i32,
    #[serde(default)]
    pub mtp_hybrid_override_pattern: Option<String>,
    #[serde(default)]
    pub mtp_layers_block_type: Option<Vec<String>>,
    /// Per-layer block pattern: `M` Mamba2, `*` attention, `-` MLP, `E` MoE.
    #[serde(default = "default_hybrid_override_pattern")]
    pub hybrid_override_pattern: String,
    /// Number of query attention heads for GQA layers.
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: i32,
    /// Per-head attention dimension.
    #[serde(default = "default_head_dim")]
    pub head_dim: i32,
    /// Number of key/value heads for GQA layers.
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: i32,
    /// RoPE base frequency for attention layers.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// Maximum configured sequence length.
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,
    /// Whether attention projections include bias terms.
    #[serde(default)]
    pub attention_bias: bool,
    /// Whether MLP projections include bias terms.
    #[serde(default)]
    pub mlp_bias: bool,
    /// Global bias flag from Nemotron-H configs.
    #[serde(default)]
    pub use_bias: bool,
    /// LayerNorm epsilon used by the reference implementation.
    #[serde(default = "default_layer_norm_epsilon")]
    pub layer_norm_epsilon: f32,
    /// RMSNorm epsilon alias used by released Nemotron-H checkpoints.
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f32,
    /// Whether residual paths are accumulated in float32.
    #[serde(default)]
    pub residual_in_fp32: bool,
    /// Number of prompt logits kept during generation.
    #[serde(default = "default_num_logits_to_keep")]
    pub num_logits_to_keep: i32,
    /// Optional sliding-window attention size.
    #[serde(default)]
    pub sliding_window: Option<i64>,
    /// Mamba2 state dimension.
    #[serde(default = "default_ssm_state_size")]
    pub ssm_state_size: i32,
    /// Number of Mamba heads.
    #[serde(default = "default_mamba_num_heads")]
    pub mamba_num_heads: i32,
    /// Number of Mamba groups.
    #[serde(default = "default_n_groups")]
    pub n_groups: i32,
    /// Mamba head dimension.
    #[serde(default = "default_mamba_head_dim")]
    pub mamba_head_dim: i32,
    /// Mamba causal convolution kernel width.
    #[serde(default = "default_conv_kernel")]
    pub conv_kernel: i32,
    /// Mamba expansion factor.
    #[serde(default = "default_expand")]
    pub expand: i32,
    /// Mamba activation name.
    #[serde(default = "default_mamba_hidden_act")]
    pub mamba_hidden_act: String,
    /// Minimum Mamba time step.
    #[serde(default = "default_time_step_min")]
    pub time_step_min: f32,
    /// Maximum Mamba time step.
    #[serde(default = "default_time_step_max")]
    pub time_step_max: f32,
    /// Floor for Mamba time-step initialization.
    #[serde(default = "default_time_step_floor")]
    pub time_step_floor: f32,
    /// Whether Mamba convolution includes a bias.
    #[serde(default = "default_true")]
    pub use_conv_bias: bool,
    /// Whether Mamba projections include bias terms.
    #[serde(default)]
    pub mamba_proj_bias: bool,
    /// Mamba scan chunk size.
    #[serde(default = "default_chunk_size")]
    pub chunk_size: i32,
    /// Whether prenorm residuals are rescaled.
    #[serde(default = "default_true")]
    pub rescale_prenorm_residual: bool,
    /// Dense MLP activation name.
    #[serde(default = "default_mlp_hidden_act")]
    pub mlp_hidden_act: String,
    /// Number of routed MoE experts.
    #[serde(default = "default_n_routed_experts")]
    pub n_routed_experts: i32,
    /// Number of shared MoE experts.
    #[serde(default = "default_n_shared_experts")]
    pub n_shared_experts: i32,
    /// Routed-expert intermediate size.
    #[serde(default = "default_moe_intermediate_size")]
    pub moe_intermediate_size: i32,
    /// Shared-expert intermediate size.
    #[serde(default = "default_moe_shared_expert_intermediate_size")]
    pub moe_shared_expert_intermediate_size: i32,
    /// Number of experts selected per token.
    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: i32,
    /// Routed expert scaling factor.
    #[serde(default = "default_routed_scaling_factor")]
    pub routed_scaling_factor: f32,
    /// Router group count.
    #[serde(default = "default_n_group")]
    pub n_group: i32,
    /// Number of router groups considered by top-k routing.
    #[serde(default = "default_topk_group")]
    pub topk_group: i32,
    /// Whether selected top-k probabilities are normalized.
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,
    /// Torch dtype string from the Hugging Face config.
    #[serde(default)]
    pub torch_dtype: Option<String>,
    /// Optional MLX affine quantization metadata.
    #[serde(default)]
    pub quantization: Option<AffineQuantization>,
    /// Optional exact weight names that use affine quantization.
    #[serde(skip)]
    pub quantized_weights: Option<HashSet<String>>,
    /// Per-weight affine settings for GGUF files with mixed Q2/Q3/Q4/Q5/Q6/Q8 tensors.
    #[serde(skip)]
    pub quantized_weight_configs: Option<HashMap<String, WeightQuantization>>,
}

/// Validated Nemotron-H decoder configuration.
#[derive(Debug, Clone)]
pub struct ModelArgs {
    /// Hugging Face model type.
    pub model_type: String,
    /// Token vocabulary size.
    pub vocab_size: i32,
    /// Whether logits reuse input embeddings.
    pub tie_word_embeddings: bool,
    /// Decoder hidden width.
    pub hidden_size: i32,
    /// Dense MLP intermediate width.
    pub intermediate_size: i32,
    /// Decoder layer count.
    pub num_hidden_layers: i32,
    /// Number of checkpoint-embedded prediction steps.
    pub num_nextn_predict_layers: i32,
    /// Physical operator pattern executed by each prediction step.
    pub mtp_hybrid_override_pattern: Option<String>,
    /// Authoritative operator and attention policy in decoder-layer order.
    pub layer_schedule: LayerSchedule<LayerPolicy>,
    /// Query-head count.
    pub num_attention_heads: i32,
    /// Attention head width.
    pub head_dim: i32,
    /// Key/value-head count.
    pub num_key_value_heads: i32,
    /// RoPE base frequency.
    pub rope_theta: f32,
    /// Configured context limit.
    pub max_position_embeddings: i32,
    /// Whether attention projections use biases.
    pub attention_bias: bool,
    /// Whether dense MLP projections use biases.
    pub mlp_bias: bool,
    /// Global Mamba projection-bias setting.
    pub use_bias: bool,
    /// Layer normalization epsilon.
    pub layer_norm_epsilon: f32,
    /// Released checkpoint RMSNorm epsilon alias.
    pub norm_eps: f32,
    /// Whether residual accumulation uses float32.
    pub residual_in_fp32: bool,
    /// Prompt logits retained during generation.
    pub num_logits_to_keep: i32,
    /// Mamba SSM state width.
    pub ssm_state_size: i32,
    /// Mamba head count.
    pub mamba_num_heads: i32,
    /// Mamba B/C group count.
    pub n_groups: i32,
    /// Mamba head width.
    pub mamba_head_dim: i32,
    /// Causal convolution kernel width.
    pub conv_kernel: i32,
    /// Mamba expansion factor.
    pub expand: i32,
    /// Mamba activation name.
    pub mamba_hidden_act: String,
    /// Minimum Mamba time step.
    pub time_step_min: f32,
    /// Maximum Mamba time step.
    pub time_step_max: f32,
    /// Mamba time-step floor.
    pub time_step_floor: f32,
    /// Whether Mamba convolution uses bias.
    pub use_conv_bias: bool,
    /// Whether Mamba projections use bias.
    pub mamba_proj_bias: bool,
    /// Mamba scan chunk size.
    pub chunk_size: i32,
    /// Whether prenorm residuals are rescaled.
    pub rescale_prenorm_residual: bool,
    /// Dense and expert MLP activation name.
    pub mlp_hidden_act: String,
    /// Routed expert count.
    pub n_routed_experts: i32,
    /// Shared expert count.
    pub n_shared_experts: i32,
    /// Routed expert intermediate width.
    pub moe_intermediate_size: i32,
    /// Shared expert intermediate width.
    pub moe_shared_expert_intermediate_size: i32,
    /// Experts selected per token.
    pub num_experts_per_tok: i32,
    /// Routed expert output scale.
    pub routed_scaling_factor: f32,
    /// Router group count.
    pub n_group: i32,
    /// Router groups considered by top-k selection.
    pub topk_group: i32,
    /// Whether selected routing weights are normalized.
    pub norm_topk_prob: bool,
    /// Source checkpoint dtype label.
    pub torch_dtype: Option<String>,
    /// Optional uniform MLX affine or MXFP4 quantization settings.
    pub quantization: Option<WeightQuantization>,
    /// Exact quantized weight names.
    pub quantized_weights: Option<HashSet<String>>,
    /// Per-weight GGUF quantization layouts.
    pub quantized_weight_configs: Option<HashMap<String, WeightQuantization>>,
}

/// Rank-local geometry for one Nemotron-H operator.
///
/// The tensor-parallel planner produces this descriptor from actual checkpoint
/// placements. Keeping it per layer avoids reconstructing heterogeneous model
/// geometry from a global TP divisor.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum ParallelLayerGeometry {
    Mamba {
        heads: i32,
        groups: i32,
    },
    Attention {
        query_heads: i32,
        kv_heads: i32,
    },
    DenseMlp {
        intermediate: i32,
    },
    SparseMoe {
        routed_intermediate: i32,
        shared_intermediate: i32,
    },
}

impl ModelArgsSource {
    fn into_args(self) -> Result<ModelArgs, Error> {
        let layer_count = usize::try_from(self.num_hidden_layers).map_err(|_| {
            Error::UnsupportedArchitecture(format!(
                "Nemotron-H num_hidden_layers must be positive, got {}",
                self.num_hidden_layers
            ))
        })?;
        let attention = match self.sliding_window {
            None => AttentionPolicy::Full,
            Some(window) => {
                let window = u32::try_from(window)
                    .ok()
                    .filter(|window| *window <= i32::MAX as u32)
                    .and_then(NonZeroU32::new)
                    .ok_or_else(|| {
                        Error::UnsupportedArchitecture(format!(
                            "Nemotron-H sliding_window must be a positive integer in the executable i32 range, got {window}"
                        ))
                    })?;
                AttentionPolicy::Sliding { window }
            }
        };
        let policies = self
            .hybrid_override_pattern
            .chars()
            .map(|ch| LayerPolicy::from_pattern_char(ch, attention))
            .collect::<Result<Vec<_>, _>>()?;
        let layer_schedule = LayerSchedule::new(layer_count, policies).map_err(|error| {
            Error::UnsupportedArchitecture(format!("Nemotron-H hybrid_override_pattern {error}"))
        })?;
        let mtp_hybrid_override_pattern =
            match (self.mtp_hybrid_override_pattern, self.mtp_layers_block_type) {
                (Some(pattern), _) => Some(pattern),
                (None, Some(layers)) => Some(
                    layers
                        .iter()
                        .map(|layer| match layer.as_str() {
                            "attention" | "full_attention" => Ok('*'),
                            "moe" => Ok('E'),
                            other => Err(Error::UnsupportedArchitecture(format!(
                                "Nemotron-H MTP block type {other:?} is unsupported"
                            ))),
                        })
                        .collect::<Result<String, Error>>()?,
                ),
                (None, None) => None,
            };
        Ok(ModelArgs {
            model_type: self.model_type,
            vocab_size: self.vocab_size,
            tie_word_embeddings: self.tie_word_embeddings,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_nextn_predict_layers: self.num_nextn_predict_layers,
            mtp_hybrid_override_pattern,
            layer_schedule,
            num_attention_heads: self.num_attention_heads,
            head_dim: self.head_dim,
            num_key_value_heads: self.num_key_value_heads,
            rope_theta: self.rope_theta,
            max_position_embeddings: self.max_position_embeddings,
            attention_bias: self.attention_bias,
            mlp_bias: self.mlp_bias,
            use_bias: self.use_bias,
            layer_norm_epsilon: self.layer_norm_epsilon,
            norm_eps: self.norm_eps,
            residual_in_fp32: self.residual_in_fp32,
            num_logits_to_keep: self.num_logits_to_keep,
            ssm_state_size: self.ssm_state_size,
            mamba_num_heads: self.mamba_num_heads,
            n_groups: self.n_groups,
            mamba_head_dim: self.mamba_head_dim,
            conv_kernel: self.conv_kernel,
            expand: self.expand,
            mamba_hidden_act: self.mamba_hidden_act,
            time_step_min: self.time_step_min,
            time_step_max: self.time_step_max,
            time_step_floor: self.time_step_floor,
            use_conv_bias: self.use_conv_bias,
            mamba_proj_bias: self.mamba_proj_bias,
            chunk_size: self.chunk_size,
            rescale_prenorm_residual: self.rescale_prenorm_residual,
            mlp_hidden_act: self.mlp_hidden_act,
            n_routed_experts: self.n_routed_experts,
            n_shared_experts: self.n_shared_experts,
            moe_intermediate_size: self.moe_intermediate_size,
            moe_shared_expert_intermediate_size: self.moe_shared_expert_intermediate_size,
            num_experts_per_tok: self.num_experts_per_tok,
            routed_scaling_factor: self.routed_scaling_factor,
            n_group: self.n_group,
            topk_group: self.topk_group,
            norm_topk_prob: self.norm_topk_prob,
            torch_dtype: self.torch_dtype,
            quantization: self.quantization.map(Into::into),
            quantized_weights: self.quantized_weights,
            quantized_weight_configs: self.quantized_weight_configs,
        })
    }
}

impl ModelArgs {
    pub(crate) fn mtp_policies(&self) -> Result<Vec<LayerPolicy>, Error> {
        if self.num_nextn_predict_layers == 0 {
            return Ok(Vec::new());
        }
        if self.num_nextn_predict_layers < 0 {
            return Err(Error::UnsupportedArchitecture(
                "Nemotron-H MTP layer count cannot be negative".into(),
            ));
        }
        let pattern = self.mtp_hybrid_override_pattern.as_deref().ok_or_else(|| {
            Error::UnsupportedArchitecture(
                "Nemotron-H MTP weights require mtp_hybrid_override_pattern".into(),
            )
        })?;
        if pattern.is_empty() {
            return Err(Error::UnsupportedArchitecture(
                "Nemotron-H MTP operator pattern cannot be empty".into(),
            ));
        }
        let attention = self
            .layer_schedule
            .iter()
            .find_map(|policy| match policy {
                LayerPolicy::SelfAttention(policy) => Some(*policy),
                _ => None,
            })
            .unwrap_or(AttentionPolicy::Full);
        let mut policies = Vec::new();
        for _ in 0..self.num_nextn_predict_layers {
            for marker in pattern.chars() {
                let policy = LayerPolicy::from_pattern_char(marker, attention)?;
                if !matches!(
                    policy,
                    LayerPolicy::SelfAttention(_) | LayerPolicy::SparseMoe
                ) {
                    return Err(Error::UnsupportedArchitecture(format!(
                        "Nemotron-H MTP pattern supports only '*' and 'E', got {marker:?}"
                    )));
                }
                policies.push(policy);
            }
        }
        Ok(policies)
    }

    /// Returns a stable ordered representation of all layer operators and attention windows.
    pub fn layer_schedule_fingerprint(&self) -> String {
        self.layer_schedule
            .iter()
            .map(|policy| match policy {
                LayerPolicy::Mamba => "m".to_string(),
                LayerPolicy::SelfAttention(AttentionPolicy::Full) => "af".to_string(),
                LayerPolicy::SelfAttention(AttentionPolicy::Sliding { window }) => {
                    format!("as{}", window.get())
                }
                LayerPolicy::DenseMlp => "d".to_string(),
                LayerPolicy::SparseMoe => "e".to_string(),
            })
            .collect::<Vec<_>>()
            .join(",")
    }

    pub(crate) fn weight_quantization_for(&self, weight_name: &str) -> Option<WeightQuantization> {
        if let Some(configs) = &self.quantized_weight_configs {
            return configs.get(weight_name).copied();
        }
        let quantization = self.quantization?;
        match &self.quantized_weights {
            Some(names) if !names.contains(weight_name) => None,
            _ => Some(quantization),
        }
    }

    fn layer_policy(&self, index: usize) -> Result<LayerPolicy, Error> {
        self.layer_schedule.get(index).copied().ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "Nemotron-H layer schedule has no policy for layer {index}"
            ))
        })
    }
}

pub(crate) fn prompt_cache_architecture_fingerprint(args: &ModelArgs) -> String {
    derive_prompt_cache_architecture_fingerprint(
        "nemotron_h",
        [
            ("model_type", args.model_type.clone()),
            ("hidden_size", args.hidden_size.to_string()),
            ("layers", args.num_hidden_layers.to_string()),
            ("layer_schedule", args.layer_schedule_fingerprint()),
            ("query_heads", args.num_attention_heads.to_string()),
            ("kv_heads", args.num_key_value_heads.to_string()),
            ("head_dim", args.head_dim.to_string()),
            ("max_positions", args.max_position_embeddings.to_string()),
            ("rope_theta", format!("{:08x}", args.rope_theta.to_bits())),
            ("attention_bias", args.attention_bias.to_string()),
            (
                "layer_norm_eps",
                format!("{:08x}", args.layer_norm_epsilon.to_bits()),
            ),
            ("norm_eps", format!("{:08x}", args.norm_eps.to_bits())),
            ("residual_f32", args.residual_in_fp32.to_string()),
            ("mamba_state", args.ssm_state_size.to_string()),
            ("mamba_heads", args.mamba_num_heads.to_string()),
            ("mamba_groups", args.n_groups.to_string()),
            ("mamba_head_dim", args.mamba_head_dim.to_string()),
            ("mamba_conv", args.conv_kernel.to_string()),
            ("mamba_chunk", args.chunk_size.to_string()),
            ("mamba_activation", args.mamba_hidden_act.clone()),
            ("mamba_bias", args.use_bias.to_string()),
            ("mamba_conv_bias", args.use_conv_bias.to_string()),
        ],
    )
}

pub(crate) fn prompt_cache_layer_layout(
    args: &ModelArgs,
) -> Result<LayerSchedule<LayerCachePolicy>, Error> {
    let geometry = args
        .layer_schedule
        .iter()
        .map(|policy| match policy {
            LayerPolicy::Mamba => ParallelLayerGeometry::Mamba {
                heads: args.mamba_num_heads,
                groups: args.n_groups,
            },
            LayerPolicy::SelfAttention(_) => ParallelLayerGeometry::Attention {
                query_heads: args.num_attention_heads,
                kv_heads: args.num_key_value_heads,
            },
            LayerPolicy::DenseMlp => ParallelLayerGeometry::DenseMlp {
                intermediate: args.intermediate_size,
            },
            LayerPolicy::SparseMoe => ParallelLayerGeometry::SparseMoe {
                routed_intermediate: args.moe_intermediate_size,
                shared_intermediate: args.moe_shared_expert_intermediate_size,
            },
        })
        .collect::<Vec<_>>();
    prompt_cache_layer_layout_with_geometry(args, &geometry)
}

pub(crate) fn prompt_cache_layer_layout_with_geometry(
    args: &ModelArgs,
    geometry: &[ParallelLayerGeometry],
) -> Result<LayerSchedule<LayerCachePolicy>, Error> {
    let cache_error = |error: crate::core::cache::CachePolicyError| {
        Error::UnsupportedArchitecture(error.to_string())
    };
    let fixed = |value| StateTensorDimension::fixed(value).map_err(cache_error);
    let history = args
        .conv_kernel
        .checked_sub(1)
        .ok_or_else(|| Error::UnsupportedArchitecture("invalid Nemotron-H kernel width".into()))?;
    if geometry.len() != args.layer_schedule.len() {
        return Err(Error::UnsupportedArchitecture(format!(
            "Nemotron-H cache geometry has {} layers, expected {}",
            geometry.len(),
            args.layer_schedule.len()
        )));
    }
    let mamba_tensors = |heads: i32, groups: i32| -> Result<Vec<StateTensorPolicy>, Error> {
        let intermediate = heads.checked_mul(args.mamba_head_dim).ok_or_else(|| {
            Error::UnsupportedArchitecture("Nemotron-H Mamba width overflow".into())
        })?;
        let conv_dim = groups
            .checked_mul(args.ssm_state_size)
            .and_then(|grouped| grouped.checked_mul(2))
            .and_then(|grouped| intermediate.checked_add(grouped))
            .ok_or_else(|| {
                Error::UnsupportedArchitecture("Nemotron-H convolution width overflow".into())
            })?;
        let mut tensors = Vec::with_capacity(2);
        if history > 0 {
            tensors.push(
                StateTensorPolicy::new(
                    StateTensorRole::Convolution { slot: 0 },
                    vec![
                        StateTensorDimension::Batch,
                        fixed(history)?,
                        fixed(conv_dim)?,
                    ],
                    StateTensorDtype::Floating,
                    crate::MutableStateResidency::AlwaysDeviceMutable,
                )
                .map_err(cache_error)?,
            );
        }
        tensors.push(
            StateTensorPolicy::new(
                StateTensorRole::Recurrent,
                vec![
                    StateTensorDimension::Batch,
                    fixed(heads)?,
                    fixed(args.mamba_head_dim)?,
                    fixed(args.ssm_state_size)?,
                ],
                StateTensorDtype::Float32,
                crate::MutableStateResidency::LayerScopedOffloadable,
            )
            .map_err(cache_error)?,
        );
        Ok(tensors)
    };
    args.layer_schedule
        .iter()
        .zip(geometry)
        .map(|(layer, geometry)| match (*layer, *geometry) {
            (LayerPolicy::Mamba, ParallelLayerGeometry::Mamba { heads, groups }) => {
                LayerCachePolicy::fixed_only(mamba_tensors(heads, groups)?).map_err(cache_error)
            }
            (
                LayerPolicy::SelfAttention(attention),
                ParallelLayerGeometry::Attention { kv_heads, .. },
            ) => {
                LayerCachePolicy::key_value(attention, kv_heads, args.head_dim).map_err(cache_error)
            }
            (LayerPolicy::DenseMlp, ParallelLayerGeometry::DenseMlp { .. })
            | (LayerPolicy::SparseMoe, ParallelLayerGeometry::SparseMoe { .. }) => {
                Ok(LayerCachePolicy::NoState)
            }
            (policy, geometry) => Err(Error::UnsupportedArchitecture(format!(
                "Nemotron-H cache geometry {geometry:?} does not match layer policy {policy:?}"
            ))),
        })
        .collect::<Result<Vec<_>, _>>()
        .and_then(|policies| {
            LayerSchedule::new(args.layer_schedule.len(), policies)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
        })
}

fn default_true() -> bool {
    true
}

fn default_vocab_size() -> i32 {
    131_072
}

fn default_hidden_size() -> i32 {
    4096
}

fn default_intermediate_size() -> i32 {
    21_504
}

fn default_num_hidden_layers() -> i32 {
    52
}

fn default_hybrid_override_pattern() -> String {
    "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-".to_string()
}

fn default_num_attention_heads() -> i32 {
    32
}

fn default_head_dim() -> i32 {
    128
}

fn default_num_key_value_heads() -> i32 {
    8
}

fn default_rope_theta() -> f32 {
    10_000.0
}

fn default_max_position_embeddings() -> i32 {
    4096
}

fn default_layer_norm_epsilon() -> f32 {
    1e-5
}

fn default_norm_eps() -> f32 {
    1e-5
}

fn default_num_logits_to_keep() -> i32 {
    1
}

fn default_ssm_state_size() -> i32 {
    128
}

fn default_mamba_num_heads() -> i32 {
    128
}

fn default_n_groups() -> i32 {
    8
}

fn default_mamba_head_dim() -> i32 {
    64
}

fn default_conv_kernel() -> i32 {
    4
}

fn default_expand() -> i32 {
    2
}

fn default_mamba_hidden_act() -> String {
    "silu".to_string()
}

fn default_time_step_min() -> f32 {
    0.001
}

fn default_time_step_max() -> f32 {
    0.1
}

fn default_time_step_floor() -> f32 {
    0.0001
}

fn default_chunk_size() -> i32 {
    128
}

fn default_mlp_hidden_act() -> String {
    "relu2".to_string()
}

fn default_n_routed_experts() -> i32 {
    8
}

fn default_n_shared_experts() -> i32 {
    1
}

fn default_moe_intermediate_size() -> i32 {
    7688
}

fn default_moe_shared_expert_intermediate_size() -> i32 {
    7688
}

fn default_num_experts_per_tok() -> i32 {
    2
}

fn default_routed_scaling_factor() -> f32 {
    1.0
}

fn default_n_group() -> i32 {
    1
}

fn default_topk_group() -> i32 {
    1
}

fn ensure_positive(name: &str, value: i32) -> Result<(), Error> {
    if value > 0 {
        Ok(())
    } else {
        Err(Error::UnsupportedArchitecture(format!(
            "Nemotron-H {name} must be positive, got {value}"
        )))
    }
}

fn silu(x: Array, stream: &Stream) -> Result<Array, Exception> {
    x.multiply(sigmoid(&x, stream)?, stream)
}

#[derive(Debug, Clone, ModuleParameters)]
/// Gated RMSNorm used by Nemotron-H Mamba2 layers.
pub struct MambaRmsNormGated {
    #[param]
    /// Learned RMSNorm scale.
    pub weight: Param<Array>,
    /// Numerical epsilon.
    pub eps: f32,
    /// Number of feature groups.
    pub n_groups: i32,
    /// Features per group.
    pub group_size: i32,
}

impl MambaRmsNormGated {
    /// Creates an unloaded gated RMSNorm.
    pub fn new(
        intermediate_size: i32,
        n_groups: i32,
        eps: f32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Ok(Self {
            weight: Param::<Array>::unloaded(&[intermediate_size], Dtype::Float32, stream)?,
            eps,
            n_groups,
            group_size: intermediate_size / n_groups,
        })
    }

    /// Applies SiLU gate modulation followed by grouped RMS normalization.
    pub fn forward(&self, x: &Array, gate: &Array, stream: &Stream) -> Result<Array, Exception> {
        let original_shape = x.shape().to_vec();
        let gated = x.multiply(silu(gate.clone(), stream)?, stream)?;
        let grouped = gated.reshape(&[-1, self.n_groups, self.group_size], stream)?;
        let variance = safemlx::ops::mean_axis(&grouped.square(stream)?, -1, true, stream)?;
        let normalized = grouped
            .multiply(
                safemlx::ops::rsqrt(variance.add(Array::from_f32(self.eps), stream)?, stream)?,
                stream,
            )?
            .reshape(&original_shape, stream)?;
        normalized.multiply(&*self.weight, stream)
    }

    /// Sets training mode.
    pub fn training_mode(&mut self, _mode: bool) {}
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// Dense Nemotron-H feed-forward block using `relu2(up_proj(x))`.
pub struct Mlp {
    #[quantizable]
    #[param]
    /// Up projection.
    pub up_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    /// Down projection.
    pub down_proj: MaybeQuantized<nn::Linear>,
}

impl Mlp {
    /// Creates an unloaded MLP.
    pub fn new(
        hidden_size: i32,
        intermediate_size: i32,
        bias: bool,
        quantization: [Option<WeightQuantization>; 2],
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Ok(Self {
            up_proj: common::linear::unloaded_maybe_quantized_linear(
                hidden_size,
                intermediate_size,
                bias,
                quantization[0],
                stream,
            )?,
            down_proj: common::linear::unloaded_maybe_quantized_linear(
                intermediate_size,
                hidden_size,
                bias,
                quantization[1],
                stream,
            )?,
        })
    }
}

impl Module<&Array> for Mlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array, stream: &Stream) -> Result<Self::Output, Self::Error> {
        let hidden = relu2(self.up_proj.forward(x, stream)?, stream)?;
        self.down_proj.forward(&hidden, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.up_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
    }
}

fn ordinary_linear_bias(linear: &MaybeQuantized<nn::Linear>) -> Option<&Array> {
    match linear {
        MaybeQuantized::Original(linear) => linear.bias.as_ref().as_ref(),
        MaybeQuantized::Quantized(linear) => linear.inner.bias.as_ref().as_ref(),
    }
}

fn reduce_row_parallel_output(
    mut partial: Array,
    projection: &MaybeQuantized<nn::Linear>,
    group: &safemlx::distributed::Group,
    stream: &Stream,
) -> Result<Array, Exception> {
    let bias = ordinary_linear_bias(projection);
    if let Some(bias) = bias {
        partial = partial.subtract(bias, stream)?;
    }
    let mut output = safemlx::distributed::all_sum(&partial, group, stream)?;
    if let Some(bias) = bias {
        output = output.add(bias, stream)?;
    }
    Ok(output)
}

/// Sigmoid top-k router used by Nemotron-H MoE layers.
pub type TopKRouter = common::moe::TopKRouter;

#[derive(Debug, Clone, ModuleParameters)]
/// Packed routed ReLU2 expert bank for Nemotron-H MoE layers.
pub struct Experts {
    /// Number of routed experts.
    pub num_experts: i32,
    /// Model hidden dimension.
    pub hidden_size: i32,
    /// Per-expert intermediate dimension.
    pub intermediate_size: i32,
    /// Optional affine or MXFP4 settings for the up-projection bank.
    pub up_quantization: Option<WeightQuantization>,
    /// Optional affine or MXFP4 settings for the down-projection bank.
    pub down_quantization: Option<WeightQuantization>,
    /// Optional checkpoint-native IQ settings for the up-projection bank.
    pub up_iquant: Option<WeightQuantization>,
    /// Optional checkpoint-native IQ settings for the down-projection bank.
    pub down_iquant: Option<WeightQuantization>,
    #[param]
    /// Expert up-projection weights.
    pub up_proj: Param<Array>,
    #[param]
    /// Expert up-projection packed scales.
    pub up_proj_scales: Param<Option<Array>>,
    #[param]
    /// Expert up-projection affine biases, absent for MXFP4.
    pub up_proj_biases: Param<Option<Array>>,
    #[param]
    /// Expert down-projection weights.
    pub down_proj: Param<Array>,
    #[param]
    /// Expert down-projection packed scales.
    pub down_proj_scales: Param<Option<Array>>,
    #[param]
    /// Expert down-projection affine biases, absent for MXFP4.
    pub down_proj_biases: Param<Option<Array>>,
}

type ExpertProjectionParams = (Param<Array>, Param<Option<Array>>, Param<Option<Array>>);

impl Experts {
    /// Creates an unloaded dense, packed, or checkpoint-native IQ expert bank.
    pub fn new(
        num_experts: i32,
        hidden_size: i32,
        intermediate_size: i32,
        quantization: [Option<WeightQuantization>; 2],
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let split = |quantization| {
            Ok::<_, Exception>(match quantization {
                Some(iq @ WeightQuantization::GgufIQuant { .. }) => (None, Some(iq)),
                packed => (packed, None),
            })
        };
        let (up_quantization, up_iquant) = split(quantization[0])?;
        let (down_quantization, down_iquant) = split(quantization[1])?;
        let projection = |out_features: i32,
                          in_features: i32,
                          quantization: Option<WeightQuantization>,
                          iquant: Option<WeightQuantization>|
         -> Result<ExpertProjectionParams, Exception> {
            if let Some(iquant) = iquant {
                let (ggml_type, _) = iquant.gguf_iquant().expect("IQ expert format");
                let (block_values, block_bytes) = ggml_type
                    .block_and_bytes()
                    .expect("canonical IQ block geometry");
                return Ok((
                    Param::<Array>::unloaded(
                        &[
                            num_experts,
                            out_features,
                            in_features / block_values as i32 * block_bytes as i32,
                        ],
                        Dtype::Uint8,
                        stream,
                    )?,
                    Param::new(None),
                    Param::new(None),
                ));
            }
            match quantization {
                Some(quantization) => Ok((
                    Param::<Array>::unloaded(
                        &[
                            num_experts,
                            out_features,
                            quantized_packed_dimension(in_features, quantization.bits()),
                        ],
                        Dtype::Uint32,
                        stream,
                    )?,
                    Param::<Option<Array>>::unloaded_some(
                        &[
                            num_experts,
                            out_features,
                            in_features / quantization.group_size(),
                        ],
                        if quantization == WeightQuantization::MxFp4 {
                            Dtype::Uint8
                        } else {
                            Dtype::Float16
                        },
                        stream,
                    )?,
                    if quantization.has_biases() {
                        Param::<Option<Array>>::unloaded_some(
                            &[
                                num_experts,
                                out_features,
                                in_features / quantization.group_size(),
                            ],
                            Dtype::Float16,
                            stream,
                        )?
                    } else {
                        Param::new(None)
                    },
                )),
                None => Ok((
                    Param::<Array>::unloaded(
                        &[num_experts, out_features, in_features],
                        Dtype::Float32,
                        stream,
                    )?,
                    Param::new(None),
                    Param::new(None),
                )),
            }
        };
        let (up_proj, up_proj_scales, up_proj_biases) =
            projection(intermediate_size, hidden_size, up_quantization, up_iquant)?;
        let (down_proj, down_proj_scales, down_proj_biases) = projection(
            hidden_size,
            intermediate_size,
            down_quantization,
            down_iquant,
        )?;
        Ok(Self {
            num_experts,
            hidden_size,
            intermediate_size,
            up_quantization,
            down_quantization,
            up_iquant,
            down_iquant,
            up_proj,
            up_proj_scales,
            up_proj_biases,
            down_proj,
            down_proj_scales,
            down_proj_biases,
        })
    }

    /// Evaluates routed experts and reduces route outputs back to tokens.
    pub fn forward(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let num_tokens = hidden_states.dim(0);
        let plan = topk_route_plan(top_k_index, self.num_experts, stream)?;
        let hidden = gather_grouped_rows(hidden_states, &plan, stream)?;
        let hidden = if let Some(iquant) = self.up_iquant {
            let (ggml_type, endian) = iquant.gguf_iquant().expect("IQ expert format");
            let native = NativeQuantizedTensor::from_iq_array(
                self.up_proj.value.clone(),
                &[self.num_experts, self.intermediate_size, self.hidden_size],
                ggml_type,
                endian,
            )?;
            native_grouped_linear(&hidden, &native, &plan.sorted_group_ids, stream)?
        } else {
            match self.up_quantization {
                Some(quantization) => packed_grouped_linear(
                    &hidden,
                    &self.up_proj,
                    self.up_proj_scales
                        .as_ref()
                        .as_ref()
                        .expect("quantized expert scales"),
                    self.up_proj_biases.as_ref().as_ref(),
                    &plan.sorted_group_ids,
                    quantization,
                    stream,
                )?,
                None => grouped_matmul(
                    &hidden,
                    &self.up_proj.as_ref().swap_axes(-1, -2, stream)?,
                    &plan.sorted_group_ids,
                    true,
                    stream,
                )?,
            }
        };
        let hidden = relu2(hidden, stream)?;
        let current = if let Some(iquant) = self.down_iquant {
            let (ggml_type, endian) = iquant.gguf_iquant().expect("IQ expert format");
            let native = NativeQuantizedTensor::from_iq_array(
                self.down_proj.value.clone(),
                &[self.num_experts, self.hidden_size, self.intermediate_size],
                ggml_type,
                endian,
            )?;
            native_grouped_linear(&hidden, &native, &plan.sorted_group_ids, stream)?
        } else {
            match self.down_quantization {
                Some(quantization) => packed_grouped_linear(
                    &hidden,
                    &self.down_proj,
                    self.down_proj_scales
                        .as_ref()
                        .as_ref()
                        .expect("quantized expert scales"),
                    self.down_proj_biases.as_ref().as_ref(),
                    &plan.sorted_group_ids,
                    quantization,
                    stream,
                )?,
                None => grouped_matmul(
                    &hidden,
                    &self.down_proj.as_ref().swap_axes(-1, -2, stream)?,
                    &plan.sorted_group_ids,
                    true,
                    stream,
                )?,
            }
        };
        weighted_route_sum(current, top_k_weights, &plan, num_tokens, stream)
    }

    /// Sets training mode.
    pub fn training_mode(&mut self, _mode: bool) {}
}

#[derive(Debug, Clone, ModuleParameters)]
/// Sparse MoE block with routed experts plus one shared dense expert.
pub struct SparseMoeBlock {
    #[param]
    /// Top-k router.
    pub gate: TopKRouter,
    #[param]
    /// Routed expert bank.
    pub experts: Experts,
    #[param]
    /// Shared dense expert.
    pub shared_experts: Mlp,
}

impl SparseMoeBlock {
    /// Creates an unloaded sparse MoE block.
    pub fn new(args: &ModelArgs, layer_idx: usize, stream: &Stream) -> Result<Self, Exception> {
        Self::new_with_widths(
            args,
            layer_idx,
            args.moe_intermediate_size,
            args.moe_shared_expert_intermediate_size,
            stream,
        )
    }

    fn new_with_widths(
        args: &ModelArgs,
        layer_idx: usize,
        routed_intermediate: i32,
        shared_intermediate: i32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let prefix = format!("model.layers.{layer_idx}.moe");
        Ok(Self {
            gate: TopKRouter::new(
                common::moe::TopKRouterConfig {
                    top_k: args.num_experts_per_tok,
                    num_experts: args.n_routed_experts,
                    hidden_size: args.hidden_size,
                    score_function: TopKRouterScoreFunction::Sigmoid,
                    norm_topk_prob: args.norm_topk_prob,
                    normalization_epsilon: 1e-20,
                    routed_scaling_factor: args.routed_scaling_factor,
                    n_group: args.n_group,
                    topk_group: args.topk_group,
                    score_correction_bias: true,
                },
                stream,
            )?,
            experts: Experts::new(
                args.n_routed_experts,
                args.hidden_size,
                routed_intermediate,
                [
                    args.weight_quantization_for(&format!("{prefix}.experts.up_proj")),
                    args.weight_quantization_for(&format!("{prefix}.experts.down_proj")),
                ],
                stream,
            )?,
            shared_experts: Mlp::new(
                args.hidden_size,
                shared_intermediate,
                args.mlp_bias,
                [
                    args.weight_quantization_for(&format!(
                        "{prefix}.shared_experts.up_proj.weight"
                    )),
                    args.weight_quantization_for(&format!(
                        "{prefix}.shared_experts.down_proj.weight"
                    )),
                ],
                stream,
            )?,
        })
    }

    pub(crate) fn forward_with_expert_executor<F>(
        &mut self,
        hidden_states: &Array,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let shape = hidden_states.shape();
        let flat = hidden_states.reshape(&[-1, shape[2]], stream)?;
        let (indices, weights) = self.gate.forward(&flat, stream)?;
        let routed = execute(&flat, &indices, &weights, stream)?;
        let shared = self
            .shared_experts
            .forward(hidden_states, stream)?
            .reshape(&[-1, shape[2]], stream)?;
        routed.add(shared, stream)?.reshape(shape, stream)
    }

    pub(crate) fn forward_tensor_with_expert_executor<F>(
        &mut self,
        hidden_states: &Array,
        group: &safemlx::distributed::Group,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let shape = hidden_states.shape();
        let flat = hidden_states.reshape(&[-1, shape[2]], stream)?;
        let (indices, weights) = self.gate.forward(&flat, stream)?;
        let routed = execute(&flat, &indices, &weights, stream)?.reshape(shape, stream)?;
        let shared_partial = self.shared_experts.forward(hidden_states, stream)?;
        let shared = reduce_row_parallel_output(
            shared_partial,
            &self.shared_experts.down_proj,
            group,
            stream,
        )?;
        routed.add(shared, stream)
    }

    pub(crate) fn forward_expert_parallel(
        &mut self,
        hidden_states: &Array,
        assignment: &crate::architectures::distributed::expert::ExpertAssignment,
        group: &safemlx::distributed::Group,
        statistics: &mut crate::architectures::distributed::expert::RoutingStatistics,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let shape = hidden_states.shape();
        let flat = hidden_states.reshape(&[-1, shape[2]], stream)?;
        let router_started = std::time::Instant::now();
        let (indices, weights) = self.gate.forward(&flat, stream)?;
        statistics.router_time += router_started.elapsed();
        let returned = crate::architectures::distributed::expert::dispatch_replicated(
            &flat,
            &indices,
            &weights,
            assignment,
            &mut self.experts,
            group,
            stream,
        )
        .map_err(|error| Exception::custom(error.to_string()))?;
        statistics.accumulate(&returned.statistics);
        let routed = returned.reduced_output.reshape(shape, stream)?;
        let shared = self.shared_experts.forward(hidden_states, stream)?;
        routed.add(shared, stream)
    }

    pub(crate) fn forward_tensor_expert_parallel(
        &mut self,
        hidden_states: &Array,
        tensor_group: &safemlx::distributed::Group,
        assignment: &crate::architectures::distributed::expert::ExpertAssignment,
        expert_group: &safemlx::distributed::Group,
        statistics: &mut crate::architectures::distributed::expert::RoutingStatistics,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let shape = hidden_states.shape();
        let flat = hidden_states.reshape(&[-1, shape[2]], stream)?;
        let router_started = std::time::Instant::now();
        let (indices, weights) = self.gate.forward(&flat, stream)?;
        statistics.router_time += router_started.elapsed();
        let returned = crate::architectures::distributed::expert::dispatch_replicated(
            &flat,
            &indices,
            &weights,
            assignment,
            &mut self.experts,
            expert_group,
            stream,
        )
        .map_err(|error| Exception::custom(error.to_string()))?;
        statistics.accumulate(&returned.statistics);
        let routed_partial = returned.reduced_output.reshape(shape, stream)?;
        let routed = safemlx::distributed::all_sum(&routed_partial, tensor_group, stream)?;
        let shared_partial = self.shared_experts.forward(hidden_states, stream)?;
        let shared = reduce_row_parallel_output(
            shared_partial,
            &self.shared_experts.down_proj,
            tensor_group,
            stream,
        )?;
        routed.add(shared, stream)
    }
}

impl Module<&Array> for SparseMoeBlock {
    type Output = Array;
    type Error = Exception;

    fn forward(
        &mut self,
        hidden_states: &Array,
        stream: &Stream,
    ) -> Result<Self::Output, Self::Error> {
        let shape = hidden_states.shape();
        let flat = hidden_states.reshape(&[-1, shape[2]], stream)?;
        let (top_k_index, top_k_weights) = self.gate.forward(&flat, stream)?;
        let routed = self
            .experts
            .forward(&flat, &top_k_index, &top_k_weights, stream)?;
        let shared = self
            .shared_experts
            .forward(hidden_states, stream)?
            .reshape(&[-1, shape[2]], stream)?;
        routed.add(shared, stream)?.reshape(shape, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate.training_mode(mode);
        self.experts.training_mode(mode);
        self.shared_experts.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// Grouped-query self-attention layer used by `*` Nemotron-H blocks.
pub struct Attention {
    /// Number of query heads.
    pub n_heads: i32,
    /// Number of key/value heads.
    pub n_kv_heads: i32,
    /// Attention scale.
    pub scale: f32,
    /// Layer-local attention policy.
    pub policy: AttentionPolicy,
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
    pub o_proj: MaybeQuantized<nn::Linear>,
}

impl Attention {
    /// Creates an unloaded attention layer.
    pub fn new(args: &ModelArgs, layer_idx: usize, stream: &Stream) -> Result<Self, Exception> {
        Self::new_with_heads(
            args,
            layer_idx,
            args.num_attention_heads,
            args.num_key_value_heads,
            stream,
        )
    }

    fn new_with_heads(
        args: &ModelArgs,
        layer_idx: usize,
        query_heads: i32,
        kv_heads: i32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let policy = match args.layer_schedule.get(layer_idx) {
            Some(LayerPolicy::SelfAttention(policy)) => *policy,
            Some(other) => {
                return Err(Exception::custom(format!(
                    "Nemotron-H layer {layer_idx} has non-attention policy {other:?}"
                )))
            }
            None => {
                return Err(Exception::custom(format!(
                    "Nemotron-H layer schedule has no policy for layer {layer_idx}"
                )))
            }
        };
        let prefix = format!("model.layers.{layer_idx}.attention");
        Ok(Self {
            n_heads: query_heads,
            n_kv_heads: kv_heads,
            scale: (args.head_dim as f32).sqrt().recip(),
            policy,
            q_proj: common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                query_heads * args.head_dim,
                args.attention_bias,
                args.weight_quantization_for(&format!("{prefix}.q_proj.weight")),
                stream,
            )?,
            k_proj: common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                kv_heads * args.head_dim,
                args.attention_bias,
                args.weight_quantization_for(&format!("{prefix}.k_proj.weight")),
                stream,
            )?,
            v_proj: common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                kv_heads * args.head_dim,
                args.attention_bias,
                args.weight_quantization_for(&format!("{prefix}.v_proj.weight")),
                stream,
            )?,
            o_proj: common::linear::unloaded_maybe_quantized_linear(
                query_heads * args.head_dim,
                args.hidden_size,
                args.attention_bias,
                args.weight_quantization_for(&format!("{prefix}.o_proj.weight")),
                stream,
            )?,
        })
    }
}

/// Input for a Nemotron-H attention layer.
pub struct AttentionInput<'a> {
    /// Hidden states.
    pub x: &'a Array,
    /// Optional attention mask.
    pub mask: Option<&'a Array>,
    /// Optional key/value cache.
    pub cache: Option<&'a mut dyn KeyValueCache>,
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
        if let Some(cache) = cache.as_ref() {
            let cache_window = cache.max_size();
            let expected_window = self
                .policy
                .window()
                .map(|window| i32::try_from(window.get()))
                .transpose()
                .map_err(|_| Exception::custom("Nemotron-H attention window exceeds i32"))?;
            if cache_window != expected_window {
                return Err(Exception::custom(format!(
                    "Nemotron-H attention cache window {cache_window:?} does not match layer policy {:?}", self.policy
                )));
            }
        }
        let (batch, seq_len) = batch_seq(x);
        let queries = reshape_attention_projection(
            self.q_proj.forward(x, stream)?,
            batch,
            seq_len,
            self.n_heads,
            stream,
        )?;
        let mut keys = reshape_attention_projection(
            self.k_proj.forward(x, stream)?,
            batch,
            seq_len,
            self.n_kv_heads,
            stream,
        )?;
        let mut values = reshape_attention_projection(
            self.v_proj.forward(x, stream)?,
            batch,
            seq_len,
            self.n_kv_heads,
            stream,
        )?;
        let position_offset = cache.as_ref().map_or(0, |cache| cache.offset());
        if let Some(cache) = cache.as_mut() {
            (keys, values) = cache.update_for_attention(keys, values, stream)?;
        }
        let paged = match cache.as_mut() {
            Some(cache) => cache.paged_attention(&queries, self.scale, mask, None, stream)?,
            None => None,
        };
        let out = if let Some(attended) = paged {
            attended
                .transpose_axes(&[0, 2, 1, 3], stream)?
                .reshape(&[batch, seq_len, -1], stream)?
        } else {
            match self.policy {
                AttentionPolicy::Sliding { window } if seq_len > 1 => {
                    common::attention::sliding_window_prefill_attention(
                        queries,
                        keys,
                        values,
                        self.scale,
                        i32::try_from(window.get()).map_err(|_| {
                            Exception::custom("Nemotron-H sliding attention window exceeds i32")
                        })?,
                        position_offset,
                        batch,
                        seq_len,
                        stream,
                    )?
                }
                _ => finish_attention(
                    queries,
                    keys,
                    values,
                    None::<&mut ConcatKeyValueCache>,
                    self.scale,
                    mask,
                    batch,
                    seq_len,
                    stream,
                )?,
            }
        };
        self.o_proj.forward(&out, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        self.k_proj.training_mode(mode);
        self.v_proj.training_mode(mode);
        self.o_proj.training_mode(mode);
    }
}

#[derive(Debug, Clone, Default)]
/// Cache state for a Nemotron-H Mamba2 block.
pub struct Mamba2Cache {
    /// Cached causal-convolution state shaped `[batch, kernel - 1, conv_dim]`.
    pub conv_state: Option<Array>,
    /// Cached SSM state shaped `[batch, heads, head_dim, state]`.
    pub ssm_state: Option<Array>,
    /// Number of consumed tokens.
    pub offset: i32,
}

#[allow(non_snake_case)]
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// Mamba2 mixer used by `M` Nemotron-H blocks.
pub struct Mamba2Mixer {
    /// Number of Mamba heads.
    pub num_heads: i32,
    /// Mamba head dimension.
    pub head_dim: i32,
    /// SSM state size.
    pub ssm_state_size: i32,
    /// Number of B/C groups.
    pub n_groups: i32,
    /// Intermediate Mamba width.
    pub intermediate_size: i32,
    /// Convolution input dimension.
    pub conv_dim: i32,
    /// Causal convolution kernel size.
    pub conv_kernel_size: i32,
    /// Unused MLP branch size from the reference split.
    pub d_mlp: i32,
    /// Number of tokens per prefill scan chunk.
    pub chunk_size: i32,
    #[quantizable]
    #[param]
    /// Joint input projection.
    pub in_proj: MaybeQuantized<nn::Linear>,
    #[param]
    /// Depthwise causal convolution.
    pub conv1d: DepthwiseConv1d,
    #[param]
    /// Timestep bias.
    pub dt_bias: Param<Array>,
    #[param]
    /// Log transition parameter.
    pub A_log: Param<Array>,
    #[param]
    /// Skip parameter.
    pub D: Param<Array>,
    #[param]
    /// Gated RMSNorm.
    pub norm: MambaRmsNormGated,
    #[quantizable]
    #[param]
    /// Output projection.
    pub out_proj: MaybeQuantized<nn::Linear>,
}

impl Mamba2Mixer {
    /// Creates an unloaded Mamba2 mixer.
    pub fn new(args: &ModelArgs, layer_idx: usize, stream: &Stream) -> Result<Self, Exception> {
        Self::new_with_geometry(args, layer_idx, args.mamba_num_heads, args.n_groups, stream)
    }

    fn new_with_geometry(
        args: &ModelArgs,
        layer_idx: usize,
        heads: i32,
        groups: i32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let intermediate_size = heads * args.mamba_head_dim;
        let conv_dim = intermediate_size + 2 * groups * args.ssm_state_size;
        let projection_size = intermediate_size + conv_dim + heads;
        let d_mlp =
            (projection_size - 2 * intermediate_size - 2 * groups * args.ssm_state_size - heads)
                / 2;
        Ok(Self {
            num_heads: heads,
            head_dim: args.mamba_head_dim,
            ssm_state_size: args.ssm_state_size,
            n_groups: groups,
            intermediate_size,
            conv_dim,
            conv_kernel_size: args.conv_kernel,
            d_mlp,
            chunk_size: args.chunk_size,
            in_proj: common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                projection_size,
                args.use_bias,
                args.weight_quantization_for(&format!(
                    "model.layers.{layer_idx}.mamba.in_proj.weight"
                )),
                stream,
            )?,
            conv1d: DepthwiseConv1d::new(conv_dim, args.conv_kernel, args.use_conv_bias, stream)?,
            dt_bias: Param::<Array>::unloaded(&[heads], Dtype::Float32, stream)?,
            A_log: Param::<Array>::unloaded(&[heads], Dtype::Float32, stream)?,
            D: Param::<Array>::unloaded(&[heads], Dtype::Float32, stream)?,
            norm: MambaRmsNormGated::new(
                intermediate_size,
                groups,
                args.layer_norm_epsilon,
                stream,
            )?,
            out_proj: common::linear::unloaded_maybe_quantized_linear(
                intermediate_size,
                args.hidden_size,
                args.use_bias,
                args.weight_quantization_for(&format!(
                    "model.layers.{layer_idx}.mamba.out_proj.weight"
                )),
                stream,
            )?,
        })
    }

    fn depthwise_causal_conv(
        &self,
        hidden_states_b_c: &Array,
        cache: Option<&mut Mamba2Cache>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let shape = hidden_states_b_c.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let channels = shape[2];
        let state_len = self.conv_kernel_size - 1;
        let state = cache
            .as_ref()
            .and_then(|cache| cache.conv_state.clone())
            .unwrap_or(zeros::<f32>(&[batch, state_len, channels], stream)?);
        let padded = concatenate_axis(&[state, hidden_states_b_c.clone()], 1, stream)?;
        if let Some(cache) = cache {
            cache.conv_state = Some(padded.try_index_device((.., seq_len.., ..), stream)?);
            cache.offset += seq_len;
        }

        silu(self.conv1d.forward_padded(&padded, stream)?, stream)
    }

    fn expand_group_states(
        &self,
        x: Array,
        batch: i32,
        seq_len: i32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let repeats = self.num_heads / self.n_groups;
        let grouped = x.reshape(
            &[batch, seq_len, self.n_groups, self.ssm_state_size],
            stream,
        )?;
        if repeats == 1 {
            return Ok(grouped);
        }
        broadcast_to(
            &grouped.try_index_device((.., .., .., NewAxis, ..), stream)?,
            &[batch, seq_len, self.n_groups, repeats, self.ssm_state_size],
            stream,
        )?
        .reshape(
            &[batch, seq_len, self.num_heads, self.ssm_state_size],
            stream,
        )
    }

    fn initial_ssm_state(
        &self,
        batch: i32,
        cache: Option<&Mamba2Cache>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        cache
            .and_then(|cache| cache.ssm_state.clone())
            .map(Ok)
            .unwrap_or_else(|| {
                zeros::<f32>(
                    &[batch, self.num_heads, self.head_dim, self.ssm_state_size],
                    stream,
                )
            })
    }

    #[allow(clippy::too_many_arguments)]
    fn scan_token(
        &self,
        state: Array,
        x_t: Array,
        b_t: Array,
        c_t: Array,
        dt_t: Array,
        a: &Array,
        d: &Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        let batch = x_t.dim(0);
        let d_a = exp(
            dt_t.reshape(&[batch, self.num_heads, 1, 1], stream)?
                .multiply(a, stream)?,
            stream,
        )?;
        let d_b = dt_t
            .reshape(&[batch, self.num_heads, 1], stream)?
            .multiply(&b_t, stream)?;
        let d_b_x = x_t
            .try_index_device((.., .., .., NewAxis), stream)?
            .multiply(d_b.try_index_device((.., .., NewAxis, ..), stream)?, stream)?;
        let state = state.multiply(d_a, stream)?.add(d_b_x, stream)?;
        let y = sum_axis(
            state.multiply(c_t.try_index_device((.., .., NewAxis, ..), stream)?, stream)?,
            -1,
            false,
            stream,
        )?
        .add(x_t.multiply(d, stream)?, stream)?;
        Ok((state, y))
    }

    #[allow(clippy::too_many_arguments)]
    fn scan_chunk(
        &self,
        hidden_states: &Array,
        b_states: &Array,
        c_states: &Array,
        dt: &Array,
        start: i32,
        end: i32,
        mut state: Array,
        a: &Array,
        d: &Array,
        dt_bias: &Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        let batch = hidden_states.dim(0);
        let mut outputs = Vec::with_capacity((end - start) as usize);
        for index in start..end {
            let x_t = hidden_states.try_index_device((.., index, .., ..), stream)?;
            let b_t = b_states.try_index_device((.., index, .., ..), stream)?;
            let c_t = c_states.try_index_device((.., index, .., ..), stream)?;
            let dt_t = nn::softplus(
                dt.try_index_device((.., index..index + 1, ..), stream)?
                    .add(dt_bias, stream)?,
                stream,
            )?
            .reshape(&[batch, self.num_heads], stream)?;
            let (next_state, y) = self.scan_token(state, x_t, b_t, c_t, dt_t, a, d, stream)?;
            state = next_state;
            outputs.push(y.try_index_device((.., NewAxis, .., ..), stream)?);
        }
        Ok((state, concatenate_axis(&outputs, 1, stream)?))
    }

    fn selective_scan_prefill_chunked(
        &self,
        hidden_states: Array,
        b_states: Array,
        c_states: Array,
        dt: Array,
        cache: Option<&mut Mamba2Cache>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let shape = hidden_states.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let mut state = self.initial_ssm_state(batch, cache.as_deref(), stream)?;
        let a = exp(self.A_log.as_ref(), stream)?
            .multiply(Array::from_f32(-1.0), stream)?
            .reshape(&[1, self.num_heads, 1, 1], stream)?;
        let d = self.D.as_ref().reshape(&[1, self.num_heads, 1], stream)?;
        let dt_bias = self
            .dt_bias
            .as_ref()
            .reshape(&[1, 1, self.num_heads], stream)?;
        let chunk_size = self.chunk_size.max(1);
        let mut outputs = Vec::with_capacity(((seq_len + chunk_size - 1) / chunk_size) as usize);
        let mut start = 0;
        while start < seq_len {
            let end = (start + chunk_size).min(seq_len);
            let (next_state, chunk) = self.scan_chunk(
                &hidden_states,
                &b_states,
                &c_states,
                &dt,
                start,
                end,
                state,
                &a,
                &d,
                &dt_bias,
                stream,
            )?;
            state = next_state;
            outputs.push(chunk);
            start = end;
        }
        if let Some(cache) = cache {
            cache.ssm_state = Some(state);
        }
        concatenate_axis(&outputs, 1, stream)
    }

    fn selective_scan_decode(
        &self,
        hidden_states: Array,
        b_states: Array,
        c_states: Array,
        dt: Array,
        cache: Option<&mut Mamba2Cache>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if hidden_states.dim(1) != 1 {
            return Err(Exception::custom(
                "Nemotron-H Mamba2 decode scan expects exactly one token",
            ));
        }
        let batch = hidden_states.dim(0);
        let state = self.initial_ssm_state(batch, cache.as_deref(), stream)?;
        let a = exp(self.A_log.as_ref(), stream)?
            .multiply(Array::from_f32(-1.0), stream)?
            .reshape(&[1, self.num_heads, 1, 1], stream)?;
        let d = self.D.as_ref().reshape(&[1, self.num_heads, 1], stream)?;
        let dt_bias = self
            .dt_bias
            .as_ref()
            .reshape(&[1, 1, self.num_heads], stream)?;
        let x_t = hidden_states.try_index_device((.., 0, .., ..), stream)?;
        let b_t = b_states.try_index_device((.., 0, .., ..), stream)?;
        let c_t = c_states.try_index_device((.., 0, .., ..), stream)?;
        let dt_t = nn::softplus(
            dt.try_index_device((.., 0..1, ..), stream)?
                .add(&dt_bias, stream)?,
            stream,
        )?
        .reshape(&[batch, self.num_heads], stream)?;
        let (state, y) = self.scan_token(state, x_t, b_t, c_t, dt_t, &a, &d, stream)?;
        if let Some(cache) = cache {
            cache.ssm_state = Some(state);
        }
        y.try_index_device((.., NewAxis, .., ..), stream)
    }
}

/// Input for a Mamba2 mixer.
pub struct Mamba2Input<'a> {
    /// Hidden states.
    pub x: &'a Array,
    /// Optional Mamba2 cache.
    pub cache: Option<&'a mut Mamba2Cache>,
}

impl Module<Mamba2Input<'_>> for Mamba2Mixer {
    type Output = Array;
    type Error = Exception;

    fn forward(
        &mut self,
        input: Mamba2Input<'_>,
        stream: &Stream,
    ) -> Result<Self::Output, Self::Error> {
        let Mamba2Input { x, mut cache } = input;
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let is_decode = seq_len == 1 && cache.as_ref().is_some_and(|cache| cache.offset > 0);
        let projected = self.in_proj.forward(x, stream)?;
        let gate_start = 2 * self.d_mlp;
        let conv_start = gate_start + self.intermediate_size;
        let dt_start = conv_start + self.conv_dim;
        let gate = projected.try_index_device((.., .., gate_start..conv_start), stream)?;
        let hidden_states_b_c =
            projected.try_index_device((.., .., conv_start..dt_start), stream)?;
        let dt = projected.try_index_device((.., .., dt_start..), stream)?;
        let hidden_states_b_c =
            self.depthwise_causal_conv(&hidden_states_b_c, cache.as_deref_mut(), stream)?;
        let hidden_states = hidden_states_b_c
            .try_index_device((.., .., ..self.intermediate_size), stream)?
            .reshape(&[batch, seq_len, self.num_heads, self.head_dim], stream)?;
        let b_start = self.intermediate_size;
        let c_start = b_start + self.n_groups * self.ssm_state_size;
        let b_states = self.expand_group_states(
            hidden_states_b_c.try_index_device((.., .., b_start..c_start), stream)?,
            batch,
            seq_len,
            stream,
        )?;
        let c_states = self.expand_group_states(
            hidden_states_b_c.try_index_device((.., .., c_start..), stream)?,
            batch,
            seq_len,
            stream,
        )?;
        let scan = if is_decode {
            self.selective_scan_decode(hidden_states, b_states, c_states, dt, cache, stream)?
        } else {
            self.selective_scan_prefill_chunked(
                hidden_states,
                b_states,
                c_states,
                dt,
                cache,
                stream,
            )?
        }
        .reshape(&[batch, seq_len, self.intermediate_size], stream)?;
        let scan = self.norm.forward(&scan, &gate, stream)?;
        self.out_proj.forward(&scan, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.in_proj.training_mode(mode);
        self.norm.training_mode(mode);
        self.out_proj.training_mode(mode);
    }
}

#[derive(Debug, Clone)]
/// Key/value state matching one attention policy.
pub enum AttentionCache {
    /// Context-growing full-attention state.
    Full(ConcatKeyValueCache),
    /// Window-bounded sliding-attention state.
    Sliding {
        /// Exact visible-position count, including the current token.
        window: NonZeroU32,
        /// Cache retaining at most `window - 1` past states between calls.
        cache: ConcatKeyValueCache,
    },
    /// Block-addressable full or sliding attention state.
    Paged(PagedKeyValueCache),
}

impl AttentionCache {
    fn new(policy: AttentionPolicy) -> Self {
        match policy {
            AttentionPolicy::Full => Self::Full(ConcatKeyValueCache::new()),
            AttentionPolicy::Sliding { window } => Self::Sliding {
                window,
                cache: ConcatKeyValueCache::new_for_sliding_attention(
                    i32::try_from(window.get()).expect("validated executable attention window"),
                ),
            },
        }
    }

    fn new_paged(
        policy: AttentionPolicy,
        manager: CacheResidencyManager,
        layer: usize,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, Exception> {
        PagedKeyValueCache::new_with_layout(
            manager,
            layer,
            policy.window().map(|window| {
                i32::try_from(window.get()).expect("validated Nemotron-H attention window fits i32")
            }),
            0,
            rank,
        )
        .map(Self::Paged)
    }

    fn clear(&mut self) {
        match self {
            Self::Full(cache) => cache.clear(),
            Self::Sliding { cache, .. } => cache.clear(),
            Self::Paged(cache) => cache.reset_local_after_manager_clear(),
        }
    }

    /// Returns the attention behavior encoded by this cache.
    pub fn policy(&self) -> AttentionPolicy {
        match self {
            Self::Full(_) => AttentionPolicy::Full,
            Self::Sliding { window, .. } => AttentionPolicy::Sliding { window: *window },
            Self::Paged(cache) => AttentionPolicy::from_sliding_window(cache.attention_window())
                .expect("paged cache construction validates its attention window"),
        }
    }

    fn snapshot_arrays(&self, stream: &Stream) -> Result<Option<(Array, Array)>, Exception> {
        match self {
            Self::Full(cache) | Self::Sliding { cache, .. } => cache.snapshot_arrays(stream),
            Self::Paged(_) => Ok(None),
        }
    }

    fn restore_checkpoint(&mut self, checkpoint: &Self, stream: &Stream) -> Result<(), Exception> {
        match (self, checkpoint) {
            (Self::Full(cache), Self::Full(previous)) => {
                cache.clone_from(previous);
                Ok(())
            }
            (
                Self::Sliding { cache, window },
                Self::Sliding {
                    cache: previous,
                    window: previous_window,
                },
            ) if window == previous_window => {
                cache.clone_from(previous);
                Ok(())
            }
            (Self::Paged(cache), Self::Paged(previous)) => {
                cache.restore_checkpoint(previous, stream)
            }
            _ => Err(Exception::custom(
                "Nemotron-H cache checkpoint changed attention policy or residency",
            )),
        }
    }
}

impl KeyValueCache for AttentionCache {
    fn offset(&self) -> i32 {
        match self {
            Self::Full(cache) => cache.offset(),
            Self::Sliding { cache, .. } => cache.offset(),
            Self::Paged(cache) => cache.offset(),
        }
    }

    fn max_size(&self) -> Option<i32> {
        match self {
            Self::Full(cache) => cache.max_size(),
            Self::Sliding { cache, .. } => cache.max_size(),
            Self::Paged(cache) => cache.max_size(),
        }
    }

    fn retained_arrays(&self) -> Vec<&Array> {
        match self {
            Self::Full(cache) => cache.retained_arrays(),
            Self::Sliding { cache, .. } => cache.retained_arrays(),
            Self::Paged(cache) => cache.retained_arrays(),
        }
    }

    fn is_paged(&self) -> bool {
        matches!(self, Self::Paged(_))
    }

    fn paged_attention(
        &mut self,
        queries: &Array,
        scale: f32,
        mask: Option<&Array>,
        sinks: Option<&Array>,
        stream: &Stream,
    ) -> Result<Option<Array>, Exception> {
        match self {
            Self::Paged(cache) => cache.paged_attention(queries, scale, mask, sinks, stream),
            Self::Full(_) | Self::Sliding { .. } => Ok(None),
        }
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        match self {
            Self::Full(cache) => cache.update_and_fetch(keys, values, stream),
            Self::Sliding { cache, .. } => cache.update_and_fetch(keys, values, stream),
            Self::Paged(cache) => cache.update_and_fetch(keys, values, stream),
        }
    }

    fn update_for_attention(
        &mut self,
        keys: Array,
        values: Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        match self {
            Self::Full(cache) => cache.update_for_attention(keys, values, stream),
            Self::Sliding { cache, .. } => cache.update_for_attention(keys, values, stream),
            Self::Paged(cache) => cache.update_for_attention(keys, values, stream),
        }
    }
}

#[derive(Debug, Clone)]
/// Per-layer cache for a Nemotron-H block.
pub enum LayerCache {
    /// Mamba2 convolution and SSM cache.
    Mamba(Mamba2Cache),
    /// Attention key/value cache.
    Attention(AttentionCache),
    /// MLP block cache placeholder.
    Mlp,
    /// MoE block cache placeholder.
    Moe,
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// Pattern-selected Nemotron-H block.
pub struct TransformerBlock {
    /// Executable layer policy.
    pub policy: LayerPolicy,
    #[param]
    /// Pre-mixer RMSNorm.
    pub norm: nn::RmsNorm,
    #[quantizable]
    #[param]
    /// Mamba2 mixer for `M` layers.
    pub mamba: Option<Mamba2Mixer>,
    #[quantizable]
    #[param]
    /// GQA attention mixer for `*` layers.
    pub attention: Option<Attention>,
    #[quantizable]
    #[param]
    /// Dense MLP mixer for `-` layers.
    pub mlp: Option<Mlp>,
    #[param]
    /// Sparse MoE mixer for `E` layers.
    pub moe: Option<SparseMoeBlock>,
}

impl TransformerBlock {
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
        let residual = x;
        let normalized = self.norm.forward(x, stream)?;
        let output = match (self.policy, cache) {
            (LayerPolicy::Mamba, Some(OperatorCache::Mamba(cache))) => {
                self.mamba.as_mut().expect("mamba block").forward(
                    Mamba2Input {
                        x: &normalized,
                        cache: Some(cache),
                    },
                    stream,
                )?
            }
            (LayerPolicy::Mamba, None) => self.mamba.as_mut().expect("mamba block").forward(
                Mamba2Input {
                    x: &normalized,
                    cache: None,
                },
                stream,
            )?,
            (LayerPolicy::SelfAttention(_), Some(OperatorCache::Attention(cache))) => {
                self.attention.as_mut().expect("attention block").forward(
                    AttentionInput {
                        x: &normalized,
                        mask,
                        cache: Some(cache),
                    },
                    stream,
                )?
            }
            (LayerPolicy::SelfAttention(_), None) => {
                self.attention.as_mut().expect("attention block").forward(
                    AttentionInput {
                        x: &normalized,
                        mask,
                        cache: None,
                    },
                    stream,
                )?
            }
            (LayerPolicy::DenseMlp, None) => self
                .mlp
                .as_mut()
                .expect("mlp block")
                .forward(&normalized, stream)?,
            (LayerPolicy::SparseMoe, None) => self
                .moe
                .as_mut()
                .expect("moe block")
                .forward_with_expert_executor(&normalized, stream, execute)?,
            (policy, Some(_)) => {
                return Err(Exception::custom(format!(
                "Nemotron-H external-expert operator cache does not match layer policy {policy:?}"
            )))
            }
        };
        residual.add(output, stream)
    }

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
        let normalized = self.norm.forward(x, stream)?;
        let output = match (self.policy, cache) {
            (LayerPolicy::Mamba, Some(OperatorCache::Mamba(cache))) => {
                let mamba = self.mamba.as_mut().expect("mamba block");
                let partial = mamba.forward(
                    Mamba2Input {
                        x: &normalized,
                        cache: Some(cache),
                    },
                    stream,
                )?;
                reduce_row_parallel_output(partial, &mamba.out_proj, group, stream)?
            }
            (LayerPolicy::Mamba, None) => {
                let mamba = self.mamba.as_mut().expect("mamba block");
                let partial = mamba.forward(
                    Mamba2Input {
                        x: &normalized,
                        cache: None,
                    },
                    stream,
                )?;
                reduce_row_parallel_output(partial, &mamba.out_proj, group, stream)?
            }
            (LayerPolicy::SelfAttention(_), Some(OperatorCache::Attention(cache))) => {
                let attention = self.attention.as_mut().expect("attention block");
                let partial = attention.forward(
                    AttentionInput {
                        x: &normalized,
                        mask,
                        cache: Some(cache),
                    },
                    stream,
                )?;
                reduce_row_parallel_output(partial, &attention.o_proj, group, stream)?
            }
            (LayerPolicy::SelfAttention(_), None) => {
                let attention = self.attention.as_mut().expect("attention block");
                let partial = attention.forward(
                    AttentionInput {
                        x: &normalized,
                        mask,
                        cache: None,
                    },
                    stream,
                )?;
                reduce_row_parallel_output(partial, &attention.o_proj, group, stream)?
            }
            (LayerPolicy::DenseMlp, None) => {
                let mlp = self.mlp.as_mut().expect("mlp block");
                let partial = mlp.forward(&normalized, stream)?;
                reduce_row_parallel_output(partial, &mlp.down_proj, group, stream)?
            }
            (LayerPolicy::SparseMoe, None) => self
                .moe
                .as_mut()
                .expect("moe block")
                .forward_tensor_with_expert_executor(&normalized, group, stream, execute)?,
            (policy, Some(_)) => {
                return Err(Exception::custom(format!(
                    "Nemotron-H tensor/external-expert operator cache does not match layer policy {policy:?}"
                )))
            }
        };
        x.add(output, stream)
    }

    /// Creates an unloaded block for `layer_idx`.
    pub fn new(args: &ModelArgs, layer_idx: usize, stream: &Stream) -> Result<Self, Error> {
        let geometry = match args.layer_policy(layer_idx)? {
            LayerPolicy::Mamba => ParallelLayerGeometry::Mamba {
                heads: args.mamba_num_heads,
                groups: args.n_groups,
            },
            LayerPolicy::SelfAttention(_) => ParallelLayerGeometry::Attention {
                query_heads: args.num_attention_heads,
                kv_heads: args.num_key_value_heads,
            },
            LayerPolicy::DenseMlp => ParallelLayerGeometry::DenseMlp {
                intermediate: args.intermediate_size,
            },
            LayerPolicy::SparseMoe => ParallelLayerGeometry::SparseMoe {
                routed_intermediate: args.moe_intermediate_size,
                shared_intermediate: args.moe_shared_expert_intermediate_size,
            },
        };
        Self::new_with_geometry(args, layer_idx, geometry, stream)
    }

    pub(crate) fn new_parallel_layerwise(
        args: &ModelArgs,
        layer_idx: usize,
        geometry: ParallelLayerGeometry,
        stream: &Stream,
    ) -> Result<Self, Error> {
        Self::new_with_geometry(args, layer_idx, geometry, stream)
    }

    fn new_with_geometry(
        args: &ModelArgs,
        layer_idx: usize,
        geometry: ParallelLayerGeometry,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let policy = args.layer_policy(layer_idx)?;
        let geometry_matches = matches!(
            (policy, geometry),
            (LayerPolicy::Mamba, ParallelLayerGeometry::Mamba { .. })
                | (
                    LayerPolicy::SelfAttention(_),
                    ParallelLayerGeometry::Attention { .. }
                )
                | (
                    LayerPolicy::DenseMlp,
                    ParallelLayerGeometry::DenseMlp { .. }
                )
                | (
                    LayerPolicy::SparseMoe,
                    ParallelLayerGeometry::SparseMoe { .. }
                )
        );
        if !geometry_matches {
            return Err(Error::Parallel(format!(
                "Nemotron-H local geometry {geometry:?} does not match layer {layer_idx} policy {policy:?}"
            )));
        }
        Ok(Self {
            policy,
            norm: nn::RmsNorm::unloaded(
                args.hidden_size,
                args.layer_norm_epsilon,
                Dtype::Float32,
                stream,
            )?,
            mamba: if policy == LayerPolicy::Mamba {
                let ParallelLayerGeometry::Mamba { heads, groups } = geometry else {
                    unreachable!()
                };
                Some(Mamba2Mixer::new_with_geometry(
                    args, layer_idx, heads, groups, stream,
                )?)
            } else {
                None
            },
            attention: if matches!(policy, LayerPolicy::SelfAttention(_)) {
                let ParallelLayerGeometry::Attention {
                    query_heads,
                    kv_heads,
                } = geometry
                else {
                    unreachable!()
                };
                Some(Attention::new_with_heads(
                    args,
                    layer_idx,
                    query_heads,
                    kv_heads,
                    stream,
                )?)
            } else {
                None
            },
            mlp: if policy == LayerPolicy::DenseMlp {
                let ParallelLayerGeometry::DenseMlp { intermediate } = geometry else {
                    unreachable!()
                };
                let prefix = format!("model.layers.{layer_idx}.mlp");
                Some(Mlp::new(
                    args.hidden_size,
                    intermediate,
                    args.mlp_bias,
                    [
                        args.weight_quantization_for(&format!("{prefix}.up_proj.weight")),
                        args.weight_quantization_for(&format!("{prefix}.down_proj.weight")),
                    ],
                    stream,
                )?)
            } else {
                None
            },
            moe: if policy == LayerPolicy::SparseMoe {
                let ParallelLayerGeometry::SparseMoe {
                    routed_intermediate,
                    shared_intermediate,
                } = geometry
                else {
                    unreachable!()
                };
                Some(SparseMoeBlock::new_with_widths(
                    args,
                    layer_idx,
                    routed_intermediate,
                    shared_intermediate,
                    stream,
                )?)
            } else {
                None
            },
        })
    }

    pub(crate) fn forward_sparse_experts<F>(
        &mut self,
        input: BlockInput<'_>,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let BlockInput { x, mask, cache } = input;
        let residual = x;
        let h = self.norm.forward(x, stream)?;
        let h = match (self.policy, cache) {
            (LayerPolicy::Mamba, Some(LayerCache::Mamba(cache))) => {
                self.mamba.as_mut().expect("mamba block").forward(
                    Mamba2Input {
                        x: &h,
                        cache: Some(cache),
                    },
                    stream,
                )?
            }
            (LayerPolicy::Mamba, None) => self
                .mamba
                .as_mut()
                .expect("mamba block")
                .forward(Mamba2Input { x: &h, cache: None }, stream)?,
            (LayerPolicy::SelfAttention(_), Some(LayerCache::Attention(cache))) => {
                self.attention.as_mut().expect("attention block").forward(
                    AttentionInput {
                        x: &h,
                        mask,
                        cache: Some(cache),
                    },
                    stream,
                )?
            }
            (LayerPolicy::SelfAttention(_), None) => {
                self.attention.as_mut().expect("attention block").forward(
                    AttentionInput {
                        x: &h,
                        mask,
                        cache: None,
                    },
                    stream,
                )?
            }
            (LayerPolicy::DenseMlp, None | Some(LayerCache::Mlp)) => {
                self.mlp.as_mut().expect("mlp block").forward(&h, stream)?
            }
            (LayerPolicy::SparseMoe, None | Some(LayerCache::Moe)) => self
                .moe
                .as_mut()
                .expect("moe block")
                .forward_with_expert_executor(&h, stream, execute)?,
            (policy, Some(cache)) => {
                return Err(Exception::custom(format!(
                    "Nemotron-H cache {cache:?} does not match layer policy {policy:?}"
                )))
            }
        };
        residual.add(h, stream)
    }

    /// Executes one Nemotron-H layer with rank-local attention, Mamba, dense,
    /// or expert channels and one row-parallel reduction.
    pub(crate) fn forward_tensor_parallel(
        &mut self,
        input: BlockInput<'_>,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let BlockInput { x, mask, cache } = input;
        let normalized = self.norm.forward(x, stream)?;
        let output = match (self.policy, cache) {
            (LayerPolicy::Mamba, Some(LayerCache::Mamba(cache))) => {
                let mamba = self.mamba.as_mut().expect("mamba block");
                let partial = mamba.forward(
                    Mamba2Input {
                        x: &normalized,
                        cache: Some(cache),
                    },
                    stream,
                )?;
                reduce_row_parallel_output(partial, &mamba.out_proj, group, stream)?
            }
            (LayerPolicy::Mamba, None) => {
                let mamba = self.mamba.as_mut().expect("mamba block");
                let partial = mamba.forward(
                    Mamba2Input {
                        x: &normalized,
                        cache: None,
                    },
                    stream,
                )?;
                reduce_row_parallel_output(partial, &mamba.out_proj, group, stream)?
            }
            (LayerPolicy::SelfAttention(_), Some(LayerCache::Attention(cache))) => {
                let attention = self.attention.as_mut().expect("attention block");
                let partial = attention.forward(
                    AttentionInput {
                        x: &normalized,
                        mask,
                        cache: Some(cache),
                    },
                    stream,
                )?;
                reduce_row_parallel_output(partial, &attention.o_proj, group, stream)?
            }
            (LayerPolicy::SelfAttention(_), None) => {
                let attention = self.attention.as_mut().expect("attention block");
                let partial = attention.forward(
                    AttentionInput {
                        x: &normalized,
                        mask,
                        cache: None,
                    },
                    stream,
                )?;
                reduce_row_parallel_output(partial, &attention.o_proj, group, stream)?
            }
            (LayerPolicy::DenseMlp, None | Some(LayerCache::Mlp)) => {
                let mlp = self.mlp.as_mut().expect("mlp block");
                let partial = mlp.forward(&normalized, stream)?;
                reduce_row_parallel_output(partial, &mlp.down_proj, group, stream)?
            }
            (LayerPolicy::SparseMoe, None | Some(LayerCache::Moe)) => {
                let moe = self.moe.as_mut().expect("moe block");
                let mut partial = moe.forward(&normalized, stream)?;
                let bias = ordinary_linear_bias(&moe.shared_experts.down_proj);
                if let Some(bias) = bias {
                    partial = partial.subtract(bias, stream)?;
                }
                let mut reduced = safemlx::distributed::all_sum(&partial, group, stream)?;
                if let Some(bias) = bias {
                    reduced = reduced.add(bias, stream)?;
                }
                reduced
            }
            (policy, Some(cache)) => {
                return Err(Exception::custom(format!(
                "Nemotron-H tensor-parallel cache {cache:?} does not match layer policy {policy:?}"
            )))
            }
        };
        x.add(output, stream)
    }

    /// Executes TP-sharded nonexpert and shared-expert projections while an
    /// EP-scoped caller executes complete routed experts.
    pub(crate) fn forward_tensor_with_expert_executor<F>(
        &mut self,
        input: BlockInput<'_>,
        group: &safemlx::distributed::Group,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let BlockInput { x, mask, cache } = input;
        let normalized = self.norm.forward(x, stream)?;
        let output = match (self.policy, cache) {
            (LayerPolicy::Mamba, Some(LayerCache::Mamba(cache))) => {
                let mamba = self.mamba.as_mut().expect("mamba block");
                let partial = mamba.forward(
                    Mamba2Input {
                        x: &normalized,
                        cache: Some(cache),
                    },
                    stream,
                )?;
                reduce_row_parallel_output(partial, &mamba.out_proj, group, stream)?
            }
            (LayerPolicy::Mamba, None) => {
                let mamba = self.mamba.as_mut().expect("mamba block");
                let partial = mamba.forward(
                    Mamba2Input {
                        x: &normalized,
                        cache: None,
                    },
                    stream,
                )?;
                reduce_row_parallel_output(partial, &mamba.out_proj, group, stream)?
            }
            (LayerPolicy::SelfAttention(_), Some(LayerCache::Attention(cache))) => {
                let attention = self.attention.as_mut().expect("attention block");
                let partial = attention.forward(
                    AttentionInput {
                        x: &normalized,
                        mask,
                        cache: Some(cache),
                    },
                    stream,
                )?;
                reduce_row_parallel_output(partial, &attention.o_proj, group, stream)?
            }
            (LayerPolicy::SelfAttention(_), None) => {
                let attention = self.attention.as_mut().expect("attention block");
                let partial = attention.forward(
                    AttentionInput {
                        x: &normalized,
                        mask,
                        cache: None,
                    },
                    stream,
                )?;
                reduce_row_parallel_output(partial, &attention.o_proj, group, stream)?
            }
            (LayerPolicy::DenseMlp, None | Some(LayerCache::Mlp)) => {
                let mlp = self.mlp.as_mut().expect("mlp block");
                let partial = mlp.forward(&normalized, stream)?;
                reduce_row_parallel_output(partial, &mlp.down_proj, group, stream)?
            }
            (LayerPolicy::SparseMoe, None | Some(LayerCache::Moe)) => self
                .moe
                .as_mut()
                .expect("moe block")
                .forward_tensor_with_expert_executor(&normalized, group, stream, execute)?,
            (policy, Some(cache)) => {
                return Err(Exception::custom(format!(
                "Nemotron-H tensor/expert cache {cache:?} does not match layer policy {policy:?}"
            )))
            }
        };
        x.add(output, stream)
    }
}

/// Input for a Nemotron-H block.
pub struct BlockInput<'a> {
    /// Hidden states.
    pub x: &'a Array,
    /// Optional attention mask for attention layers.
    pub mask: Option<&'a Array>,
    /// Optional layer cache.
    pub cache: Option<&'a mut LayerCache>,
}

/// Borrowed operator state used by architecture-independent execution runtimes.
pub(crate) enum OperatorCache<'a> {
    /// Mamba convolution and recurrent state.
    Mamba(&'a mut Mamba2Cache),
    /// Full or sliding key/value state.
    Attention(&'a mut dyn KeyValueCache),
}

impl TransformerBlock {
    /// Executes this layer from semantic operator state rather than the
    /// resident model's family-specific cache container.
    pub(crate) fn forward_with_operator_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<OperatorCache<'_>>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let residual = x;
        let h = self.norm.forward(x, stream)?;
        let h = match (self.policy, cache) {
            (LayerPolicy::Mamba, Some(OperatorCache::Mamba(cache))) => {
                self.mamba.as_mut().expect("mamba block").forward(
                    Mamba2Input {
                        x: &h,
                        cache: Some(cache),
                    },
                    stream,
                )?
            }
            (LayerPolicy::Mamba, None) => self
                .mamba
                .as_mut()
                .expect("mamba block")
                .forward(Mamba2Input { x: &h, cache: None }, stream)?,
            (LayerPolicy::SelfAttention(_), Some(OperatorCache::Attention(cache))) => {
                self.attention.as_mut().expect("attention block").forward(
                    AttentionInput {
                        x: &h,
                        mask,
                        cache: Some(cache),
                    },
                    stream,
                )?
            }
            (LayerPolicy::SelfAttention(_), None) => {
                self.attention.as_mut().expect("attention block").forward(
                    AttentionInput {
                        x: &h,
                        mask,
                        cache: None,
                    },
                    stream,
                )?
            }
            (LayerPolicy::DenseMlp, None) => {
                self.mlp.as_mut().expect("mlp block").forward(&h, stream)?
            }
            (LayerPolicy::SparseMoe, None) => {
                self.moe.as_mut().expect("moe block").forward(&h, stream)?
            }
            (policy, Some(_)) => {
                return Err(Exception::custom(format!(
                    "Nemotron-H operator cache does not match layer policy {policy:?}"
                )))
            }
        };
        residual.add(h, stream)
    }

    pub(crate) fn forward_tensor_parallel_with_operator_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<OperatorCache<'_>>,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let normalized = self.norm.forward(x, stream)?;
        let output = match (self.policy, cache) {
            (LayerPolicy::Mamba, Some(OperatorCache::Mamba(cache))) => {
                let mamba = self.mamba.as_mut().expect("mamba block");
                let partial = mamba.forward(
                    Mamba2Input {
                        x: &normalized,
                        cache: Some(cache),
                    },
                    stream,
                )?;
                reduce_row_parallel_output(partial, &mamba.out_proj, group, stream)?
            }
            (LayerPolicy::Mamba, None) => {
                let mamba = self.mamba.as_mut().expect("mamba block");
                let partial = mamba.forward(
                    Mamba2Input {
                        x: &normalized,
                        cache: None,
                    },
                    stream,
                )?;
                reduce_row_parallel_output(partial, &mamba.out_proj, group, stream)?
            }
            (LayerPolicy::SelfAttention(_), Some(OperatorCache::Attention(cache))) => {
                let attention = self.attention.as_mut().expect("attention block");
                let partial = attention.forward(
                    AttentionInput {
                        x: &normalized,
                        mask,
                        cache: Some(cache),
                    },
                    stream,
                )?;
                reduce_row_parallel_output(partial, &attention.o_proj, group, stream)?
            }
            (LayerPolicy::SelfAttention(_), None) => {
                let attention = self.attention.as_mut().expect("attention block");
                let partial = attention.forward(
                    AttentionInput {
                        x: &normalized,
                        mask,
                        cache: None,
                    },
                    stream,
                )?;
                reduce_row_parallel_output(partial, &attention.o_proj, group, stream)?
            }
            (LayerPolicy::DenseMlp, None) => {
                let mlp = self.mlp.as_mut().expect("mlp block");
                let partial = mlp.forward(&normalized, stream)?;
                reduce_row_parallel_output(partial, &mlp.down_proj, group, stream)?
            }
            (LayerPolicy::SparseMoe, None) => {
                let moe = self.moe.as_mut().expect("moe block");
                let mut partial = moe.forward(&normalized, stream)?;
                let bias = ordinary_linear_bias(&moe.shared_experts.down_proj);
                if let Some(bias) = bias {
                    partial = partial.subtract(bias, stream)?;
                }
                let mut reduced = safemlx::distributed::all_sum(&partial, group, stream)?;
                if let Some(bias) = bias {
                    reduced = reduced.add(bias, stream)?;
                }
                reduced
            }
            (policy, Some(_)) => {
                return Err(Exception::custom(format!(
                "Nemotron-H tensor-parallel operator cache does not match layer policy {policy:?}"
            )))
            }
        };
        x.add(output, stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_expert_parallel_with_operator_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<OperatorCache<'_>>,
        assignment: &crate::architectures::distributed::expert::ExpertAssignment,
        group: &safemlx::distributed::Group,
        statistics: &mut crate::architectures::distributed::expert::RoutingStatistics,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let normalized = self.norm.forward(x, stream)?;
        let output = match (self.policy, cache) {
            (LayerPolicy::Mamba, Some(OperatorCache::Mamba(cache))) => {
                self.mamba.as_mut().expect("mamba block").forward(
                    Mamba2Input {
                        x: &normalized,
                        cache: Some(cache),
                    },
                    stream,
                )?
            }
            (LayerPolicy::Mamba, None) => self.mamba.as_mut().expect("mamba block").forward(
                Mamba2Input {
                    x: &normalized,
                    cache: None,
                },
                stream,
            )?,
            (LayerPolicy::SelfAttention(_), Some(OperatorCache::Attention(cache))) => {
                self.attention.as_mut().expect("attention block").forward(
                    AttentionInput {
                        x: &normalized,
                        mask,
                        cache: Some(cache),
                    },
                    stream,
                )?
            }
            (LayerPolicy::SelfAttention(_), None) => {
                self.attention.as_mut().expect("attention block").forward(
                    AttentionInput {
                        x: &normalized,
                        mask,
                        cache: None,
                    },
                    stream,
                )?
            }
            (LayerPolicy::DenseMlp, None) => self
                .mlp
                .as_mut()
                .expect("mlp block")
                .forward(&normalized, stream)?,
            (LayerPolicy::SparseMoe, None) => self
                .moe
                .as_mut()
                .expect("moe block")
                .forward_expert_parallel(&normalized, assignment, group, statistics, stream)?,
            (policy, Some(_)) => {
                return Err(Exception::custom(format!(
                "Nemotron-H expert-parallel operator cache does not match layer policy {policy:?}"
            )))
            }
        };
        x.add(output, stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_tensor_expert_parallel_with_operator_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<OperatorCache<'_>>,
        tensor_group: &safemlx::distributed::Group,
        assignment: &crate::architectures::distributed::expert::ExpertAssignment,
        expert_group: &safemlx::distributed::Group,
        statistics: &mut crate::architectures::distributed::expert::RoutingStatistics,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let normalized = self.norm.forward(x, stream)?;
        let output = match (self.policy, cache) {
            (LayerPolicy::Mamba, Some(OperatorCache::Mamba(cache))) => {
                let mamba = self.mamba.as_mut().expect("mamba block");
                let partial = mamba.forward(
                    Mamba2Input {
                        x: &normalized,
                        cache: Some(cache),
                    },
                    stream,
                )?;
                reduce_row_parallel_output(partial, &mamba.out_proj, tensor_group, stream)?
            }
            (LayerPolicy::Mamba, None) => {
                let mamba = self.mamba.as_mut().expect("mamba block");
                let partial = mamba.forward(
                    Mamba2Input {
                        x: &normalized,
                        cache: None,
                    },
                    stream,
                )?;
                reduce_row_parallel_output(partial, &mamba.out_proj, tensor_group, stream)?
            }
            (LayerPolicy::SelfAttention(_), Some(OperatorCache::Attention(cache))) => {
                let attention = self.attention.as_mut().expect("attention block");
                let partial = attention.forward(
                    AttentionInput {
                        x: &normalized,
                        mask,
                        cache: Some(cache),
                    },
                    stream,
                )?;
                reduce_row_parallel_output(partial, &attention.o_proj, tensor_group, stream)?
            }
            (LayerPolicy::SelfAttention(_), None) => {
                let attention = self.attention.as_mut().expect("attention block");
                let partial = attention.forward(
                    AttentionInput {
                        x: &normalized,
                        mask,
                        cache: None,
                    },
                    stream,
                )?;
                reduce_row_parallel_output(partial, &attention.o_proj, tensor_group, stream)?
            }
            (LayerPolicy::DenseMlp, None) => {
                let mlp = self.mlp.as_mut().expect("mlp block");
                let partial = mlp.forward(&normalized, stream)?;
                reduce_row_parallel_output(partial, &mlp.down_proj, tensor_group, stream)?
            }
            (LayerPolicy::SparseMoe, None) => self
                .moe
                .as_mut()
                .expect("moe block")
                .forward_tensor_expert_parallel(
                    &normalized,
                    tensor_group,
                    assignment,
                    expert_group,
                    statistics,
                    stream,
                )?,
            (policy, Some(_)) => {
                return Err(Exception::custom(format!(
                    "Nemotron-H tensor/expert operator cache does not match layer policy {policy:?}"
                )))
            }
        };
        x.add(output, stream)
    }
}

impl Module<BlockInput<'_>> for TransformerBlock {
    type Output = Array;
    type Error = Exception;

    fn forward(
        &mut self,
        input: BlockInput<'_>,
        stream: &Stream,
    ) -> Result<Self::Output, Self::Error> {
        let BlockInput { x, mask, cache } = input;
        let residual = x;
        let h = self.norm.forward(x, stream)?;
        let h = match (self.policy, cache) {
            (LayerPolicy::Mamba, Some(LayerCache::Mamba(cache))) => {
                self.mamba.as_mut().expect("mamba block").forward(
                    Mamba2Input {
                        x: &h,
                        cache: Some(cache),
                    },
                    stream,
                )?
            }
            (LayerPolicy::Mamba, None) => self
                .mamba
                .as_mut()
                .expect("mamba block")
                .forward(Mamba2Input { x: &h, cache: None }, stream)?,
            (LayerPolicy::SelfAttention(_), Some(LayerCache::Attention(cache))) => {
                self.attention.as_mut().expect("attention block").forward(
                    AttentionInput {
                        x: &h,
                        mask,
                        cache: Some(cache),
                    },
                    stream,
                )?
            }
            (LayerPolicy::SelfAttention(_), None) => {
                self.attention.as_mut().expect("attention block").forward(
                    AttentionInput {
                        x: &h,
                        mask,
                        cache: None,
                    },
                    stream,
                )?
            }
            (LayerPolicy::DenseMlp, None | Some(LayerCache::Mlp)) => {
                self.mlp.as_mut().expect("mlp block").forward(&h, stream)?
            }
            (LayerPolicy::SparseMoe, None | Some(LayerCache::Moe)) => {
                self.moe.as_mut().expect("moe block").forward(&h, stream)?
            }
            (policy, Some(cache)) => {
                return Err(Exception::custom(format!(
                    "Nemotron-H cache {cache:?} does not match layer policy {policy:?}"
                )))
            }
        };
        residual.add(h, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.norm.training_mode(mode);
        if let Some(mamba) = &mut self.mamba {
            mamba.training_mode(mode);
        }
        if let Some(attention) = &mut self.attention {
            attention.training_mode(mode);
        }
        if let Some(mlp) = &mut self.mlp {
            mlp.training_mode(mode);
        }
        if let Some(moe) = &mut self.moe {
            moe.training_mode(mode);
        }
    }
}

impl LayerCache {
    pub(crate) fn new(policy: LayerPolicy) -> Self {
        match policy {
            LayerPolicy::Mamba => Self::Mamba(Mamba2Cache::default()),
            LayerPolicy::SelfAttention(policy) => Self::Attention(AttentionCache::new(policy)),
            LayerPolicy::DenseMlp => Self::Mlp,
            LayerPolicy::SparseMoe => Self::Moe,
        }
    }

    fn new_paged(
        policy: LayerPolicy,
        manager: CacheResidencyManager,
        layer: usize,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, Exception> {
        match policy {
            LayerPolicy::Mamba => Ok(Self::Mamba(Mamba2Cache::default())),
            LayerPolicy::SelfAttention(policy) => {
                AttentionCache::new_paged(policy, manager, layer, rank).map(Self::Attention)
            }
            LayerPolicy::DenseMlp => Ok(Self::Mlp),
            LayerPolicy::SparseMoe => Ok(Self::Moe),
        }
    }

    pub(crate) fn offset(&self) -> Option<i32> {
        match self {
            Self::Mamba(cache) => Some(cache.offset),
            Self::Attention(cache) => Some(cache.offset()),
            Self::Mlp | Self::Moe => None,
        }
    }

    pub(crate) fn retained_arrays(&self) -> Vec<&Array> {
        match self {
            Self::Mamba(cache) => cache
                .conv_state
                .iter()
                .chain(cache.ssm_state.iter())
                .collect(),
            Self::Attention(cache) => cache.retained_arrays(),
            Self::Mlp | Self::Moe => Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
/// Heterogeneous cache for Nemotron-H hybrid layers.
pub struct Cache {
    /// Per-layer cache state.
    pub layers: Vec<LayerCache>,
    /// State owned by checkpoint-embedded prediction operators.
    pub(crate) mtp_layers: Vec<LayerCache>,
}

impl Cache {
    /// Creates an empty cache matching the authoritative layer schedule.
    pub fn new(args: &ModelArgs) -> Self {
        Self {
            layers: args
                .layer_schedule
                .iter()
                .copied()
                .map(LayerCache::new)
                .collect(),
            mtp_layers: Vec::new(),
        }
    }

    pub(crate) fn with_mtp_policies(mut self, policies: &[LayerPolicy]) -> Self {
        self.mtp_layers = policies.iter().copied().map(LayerCache::new).collect();
        self
    }

    pub(crate) fn with_paged_mtp_policies(
        mut self,
        policies: &[LayerPolicy],
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, Exception> {
        if policies.is_empty() {
            return Ok(self);
        }
        let manager = self.residency_manager()?.cloned().ok_or_else(|| {
            Exception::custom("Nemotron-H paged MTP requires a shared cache manager")
        })?;
        let start = self.layers.len();
        self.mtp_layers = policies
            .iter()
            .copied()
            .enumerate()
            .map(|(index, policy)| {
                LayerCache::new_paged(policy, manager.clone(), start + index, rank)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self)
    }

    pub(crate) fn restore_target_checkpoint(
        &mut self,
        checkpoint: &Self,
        stream: &Stream,
    ) -> Result<(), Exception> {
        if self.layers.len() != checkpoint.layers.len() {
            return Err(Exception::custom(
                "Nemotron-H target cache checkpoint has a different layer count",
            ));
        }
        for (cache, previous) in self.layers.iter_mut().zip(&checkpoint.layers) {
            match (cache, previous) {
                (LayerCache::Attention(cache), LayerCache::Attention(previous)) => {
                    cache.restore_checkpoint(previous, stream)?;
                }
                (LayerCache::Mamba(cache), LayerCache::Mamba(previous)) => {
                    cache.clone_from(previous);
                }
                (LayerCache::Mlp, LayerCache::Mlp) | (LayerCache::Moe, LayerCache::Moe) => {}
                _ => {
                    return Err(Exception::custom(
                        "Nemotron-H target cache checkpoint changed layer policy",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Creates device-resident state or pages only context-growing attention
    /// blocks while bounded Mamba state remains device resident.
    pub fn new_with_options(
        args: &ModelArgs,
        policy: CacheResidencyPolicy,
    ) -> Result<Self, Exception> {
        Self::new_with_options_and_rank(args, policy, None)
    }

    pub(crate) fn new_with_options_and_rank(
        args: &ModelArgs,
        policy: CacheResidencyPolicy,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, Exception> {
        match policy {
            CacheResidencyPolicy::Device => Ok(Self::new(args)),
            CacheResidencyPolicy::Paged(options) => {
                let manager = CacheResidencyManager::new(options)
                    .map_err(|error| Exception::custom(error.to_string()))?;
                Self::new_paged_with_manager(args, manager, rank)
            }
        }
    }

    fn new_paged_with_manager(
        args: &ModelArgs,
        manager: CacheResidencyManager,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, Exception> {
        if !args
            .layer_schedule
            .iter()
            .any(|policy| matches!(policy, LayerPolicy::SelfAttention(_)))
        {
            return Err(Exception::custom(
                "paged Nemotron-H cache residency requires at least one attention layer",
            ));
        }
        let layers = args
            .layer_schedule
            .iter()
            .copied()
            .enumerate()
            .map(|(layer, policy)| LayerCache::new_paged(policy, manager.clone(), layer, rank))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            layers,
            mtp_layers: Vec::new(),
        })
    }

    fn residency_manager(&self) -> Result<Option<&CacheResidencyManager>, Exception> {
        let mut managers =
            self.layers
                .iter()
                .chain(&self.mtp_layers)
                .filter_map(|layer| match layer {
                    LayerCache::Attention(AttentionCache::Paged(cache)) => Some(cache.manager()),
                    LayerCache::Mamba(_)
                    | LayerCache::Attention(AttentionCache::Full(_))
                    | LayerCache::Attention(AttentionCache::Sliding { .. })
                    | LayerCache::Mlp
                    | LayerCache::Moe => None,
                });
        let Some(first) = managers.next() else {
            return Ok(None);
        };
        if managers.any(|manager| manager.session_id() != first.session_id()) {
            return Err(Exception::custom(
                "Nemotron-H paged attention layers must share one residency manager",
            ));
        }
        Ok(Some(first))
    }

    /// Returns aggregate live attention paging observations, if enabled.
    pub fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        self.residency_manager()?
            .map(|manager| {
                manager
                    .report()
                    .map_err(|error| Exception::custom(error.to_string()))
            })
            .transpose()
    }

    /// Returns the current sequence offset represented by the first stateful layer.
    pub fn offset(&self) -> i32 {
        self.layers.iter().find_map(LayerCache::offset).unwrap_or(0)
    }

    pub(crate) fn reset(&mut self) -> Result<(), Exception> {
        if let Some(manager) = self.residency_manager()?.cloned() {
            manager
                .clear()
                .map_err(|error| Exception::custom(error.to_string()))?;
        }
        for layer in self.layers.iter_mut().chain(&mut self.mtp_layers) {
            match layer {
                LayerCache::Mamba(cache) => *cache = Mamba2Cache::default(),
                LayerCache::Attention(cache) => cache.clear(),
                LayerCache::Mlp | LayerCache::Moe => {}
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// Nemotron-H transformer body without the language-model head.
pub struct NemotronHModel {
    /// Token vocabulary size.
    pub vocab_size: i32,
    /// Number of hybrid layers.
    pub num_hidden_layers: i32,
    #[quantizable]
    #[param]
    /// Token embedding table.
    pub embeddings: MaybeQuantized<nn::Embedding>,
    #[quantizable]
    #[param]
    /// Hybrid transformer blocks.
    pub layers: Vec<TransformerBlock>,
    #[param]
    /// Final normalization.
    pub norm_f: nn::RmsNorm,
}

impl NemotronHModel {
    /// Creates an unloaded Nemotron-H transformer body.
    pub fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        let embeddings = common::linear::unloaded_maybe_quantized_embedding(
            args.vocab_size,
            args.hidden_size,
            args.weight_quantization_for("model.embeddings.weight"),
            stream,
        )?;
        let layers = (0..args.num_hidden_layers)
            .map(|idx| TransformerBlock::new(args, idx as usize, stream))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| Exception::custom(error.to_string()))?;
        let norm_f =
            nn::RmsNorm::unloaded(args.hidden_size, args.norm_eps, Dtype::Float32, stream)?;
        Ok(Self {
            vocab_size: args.vocab_size,
            num_hidden_layers: args.num_hidden_layers,
            embeddings,
            layers,
            norm_f,
        })
    }
}

/// Input for a Nemotron-H forward pass.
pub struct ModelInput<'a> {
    /// Token ids with shape `[batch, sequence]`.
    pub inputs: &'a Array,
    /// Optional attention mask.
    pub mask: Option<&'a Array>,
    /// Optional heterogeneous cache.
    pub cache: Option<&'a mut Cache>,
}

impl Module<ModelInput<'_>> for NemotronHModel {
    type Output = Array;
    type Error = Exception;

    fn forward(
        &mut self,
        input: ModelInput<'_>,
        stream: &Stream,
    ) -> Result<Self::Output, Self::Error> {
        let ModelInput {
            inputs,
            mask,
            mut cache,
        } = input;
        let mut h = self.embeddings.forward(inputs, stream)?;
        let mask = match mask {
            Some(mask) => Some(mask.clone()),
            None => {
                let offset = cache.as_ref().map(|cache| cache.offset()).unwrap_or(0);
                if h.shape()[1] > 1 {
                    match create_attention_mask(&h, &offset_cache(offset), Some(true), stream)? {
                        Some(AttentionMask::Array(a)) => Some(a),
                        Some(AttentionMask::Causal) => {
                            return Err(Exception::custom("Only `Array` mask is supported"));
                        }
                        None => None,
                    }
                } else {
                    None
                }
            }
        };

        if let Some(cache) = cache.as_mut() {
            if cache.layers.len() != self.layers.len() {
                return Err(Exception::custom(format!(
                    "Nemotron-H cache has {} layers, expected {}",
                    cache.layers.len(),
                    self.layers.len()
                )));
            }
            for (layer, layer_cache) in self.layers.iter_mut().zip(cache.layers.iter_mut()) {
                h = layer.forward(
                    BlockInput {
                        x: &h,
                        mask: mask.as_ref(),
                        cache: Some(layer_cache),
                    },
                    stream,
                )?;
            }
        } else {
            for layer in &mut self.layers {
                h = layer.forward(
                    BlockInput {
                        x: &h,
                        mask: mask.as_ref(),
                        cache: None,
                    },
                    stream,
                )?;
            }
        }
        self.norm_f.forward(&h, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.embeddings.training_mode(mode);
        for layer in &mut self.layers {
            layer.training_mode(mode);
        }
        self.norm_f.training_mode(mode);
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
/// Nemotron-H causal language model.
pub struct Model {
    /// Model configuration.
    pub args: ModelArgs,
    #[quantizable]
    #[param]
    /// Transformer body.
    pub model: NemotronHModel,
    #[quantizable]
    #[param]
    /// Optional untied language-model head.
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    /// Creates an unloaded Nemotron-H causal language model.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        validate_model_args(&args).map_err(|error| Exception::custom(error.to_string()))?;
        let model = NemotronHModel::new(&args, stream)?;
        let lm_head = if !args.tie_word_embeddings {
            Some(common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                args.vocab_size,
                false,
                args.weight_quantization_for("lm_head.weight"),
                stream,
            )?)
        } else {
            None
        };
        Ok(Self {
            args,
            model,
            lm_head,
        })
    }

    /// Creates an empty heterogeneous cache for this model.
    pub fn new_cache(&self) -> Cache {
        Cache::new(&self.args)
    }

    /// Creates resident state or blockwise-paged attention state while Mamba
    /// convolution and recurrent tensors remain device resident.
    pub fn new_cache_with_options(&self, policy: CacheResidencyPolicy) -> Result<Cache, Exception> {
        Cache::new_with_options(&self.args, policy)
    }

    /// Returns aggregate live attention paging observations, if enabled.
    pub fn cache_residency_report(
        &self,
        cache: &Cache,
    ) -> Result<Option<CacheResidencyReport>, Exception> {
        cache.residency_report()
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
            .map_err(|_| Exception::custom("Nemotron-H prompt length exceeds i64"))?;
        let paged_attention = cache
            .layers
            .iter()
            .filter(|layer| matches!(layer, LayerCache::Attention(AttentionCache::Paged(_))))
            .count();
        let resident_attention = cache
            .layers
            .iter()
            .filter(|layer| {
                matches!(
                    layer,
                    LayerCache::Attention(AttentionCache::Full(_) | AttentionCache::Sliding { .. })
                )
            })
            .count();
        if paged_attention > 0 && resident_attention > 0 {
            return Err(Exception::custom(
                "Nemotron-H prompt caches cannot mix paged and resident attention layers",
            ));
        }
        let paged = paged_attention > 0;
        let residency_manager = cache.residency_manager()?.cloned();
        if paged != residency_manager.is_some() {
            return Err(Exception::custom(
                "Nemotron-H paged attention cache has inconsistent residency state",
            ));
        }
        if paged {
            for layer in &mut cache.layers {
                if let LayerCache::Attention(AttentionCache::Paged(cache)) = layer {
                    cache.finalize()?;
                }
            }
        }
        let mut blocks = Vec::new();
        let mut state = Vec::new();
        for (layer, cache) in cache.layers.iter().enumerate() {
            if cache
                .offset()
                .is_some_and(|offset| i64::from(offset) != end)
            {
                return Err(Exception::custom(format!(
                    "Nemotron-H layer {layer} cache offset does not match the persisted prefix"
                )));
            }
            match cache {
                LayerCache::Mamba(cache) => {
                    let convolution = cache.conv_state.as_ref().ok_or_else(|| {
                        Exception::custom(format!(
                            "Nemotron-H layer {layer} convolution state is missing"
                        ))
                    })?;
                    if convolution.dim(1) > 0 {
                        state.push(PromptCacheStateArray {
                            owner: StateTensorOwner::Layer(layer),
                            role: StateTensorRole::Convolution { slot: 0 },
                            array: convolution,
                        });
                    }
                    state.push(PromptCacheStateArray {
                        owner: StateTensorOwner::Layer(layer),
                        role: StateTensorRole::Recurrent,
                        array: cache.ssm_state.as_ref().ok_or_else(|| {
                            Exception::custom(format!(
                                "Nemotron-H layer {layer} recurrent state is missing"
                            ))
                        })?,
                    });
                }
                LayerCache::Attention(cache) => {
                    if matches!(cache, AttentionCache::Paged(_)) {
                        continue;
                    }
                    let (keys, values) = cache.snapshot_arrays(stream)?.ok_or_else(|| {
                        Exception::custom(format!(
                            "Nemotron-H layer {layer} attention state is missing"
                        ))
                    })?;
                    let start = end.checked_sub(i64::from(keys.dim(-2))).ok_or_else(|| {
                        Exception::custom(format!(
                            "Nemotron-H layer {layer} retained range exceeds its cache offset"
                        ))
                    })?;
                    blocks.push(PromptCacheSnapshotBlock {
                        global_layer: layer,
                        start,
                        end,
                        rank,
                        arrays: CacheBlockArrays::KeyValue { keys, values },
                    });
                }
                LayerCache::Mlp | LayerCache::Moe => {}
            }
        }
        if let Some(manager) = residency_manager {
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
    pub(crate) fn load_paged_prompt_cache(
        args: &ModelArgs,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(Cache, PromptCacheManifest), Exception> {
        let layer_count = usize::try_from(args.num_hidden_layers)
            .map_err(|_| Exception::custom("invalid Nemotron-H cache layer count"))?;
        let identity = PromptCacheModelIdentity {
            model_family: "nemotron_h".into(),
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
        Self::load_paged_prompt_cache_with_identity(
            args,
            directory,
            expected,
            prefix_token_ids,
            &identity,
            options,
            stream,
        )
    }

    pub(crate) fn load_paged_prompt_cache_with_identity(
        args: &ModelArgs,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        identity: &PromptCacheModelIdentity,
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(Cache, PromptCacheManifest), Exception> {
        let directory = directory.as_ref();
        let (manager, manifest) =
            open_prompt_cache(directory, expected, identity, prefix_token_ids, options)
                .map_err(|error| Exception::custom(error.to_string()))?;
        let mut state = load_prompt_cache_state_tensors(directory, &manifest, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .into_iter()
            .map(|tensor| ((tensor.owner, tensor.role), tensor.array))
            .collect::<BTreeMap<_, _>>();
        let end = i32::try_from(prefix_token_ids.len())
            .map_err(|_| Exception::custom("Nemotron-H prompt length exceeds i32"))?;
        let mut cache =
            Cache::new_paged_with_manager(args, manager, identity.topology.cache_rank_identity())?;
        for (layer, layer_cache) in cache.layers.iter_mut().enumerate() {
            if let LayerCache::Mamba(cache) = layer_cache {
                if args.conv_kernel > 1 {
                    cache.conv_state = Some(
                        state
                            .remove(&(
                                StateTensorOwner::Layer(layer),
                                StateTensorRole::Convolution { slot: 0 },
                            ))
                            .ok_or_else(|| {
                                Exception::custom(format!(
                                    "Nemotron-H layer {layer} convolution state is missing"
                                ))
                            })?,
                    );
                }
                cache.ssm_state = Some(
                    state
                        .remove(&(StateTensorOwner::Layer(layer), StateTensorRole::Recurrent))
                        .ok_or_else(|| {
                            Exception::custom(format!(
                                "Nemotron-H layer {layer} recurrent state is missing"
                            ))
                        })?,
                );
                cache.offset = end;
            }
        }
        if !state.is_empty() {
            return Err(Exception::custom(
                "Nemotron-H paged prompt cache has unexpected fixed state",
            ));
        }
        Ok((cache, manifest))
    }

    /// Returns the configured model type.
    pub fn model_type(&self) -> &str {
        &self.args.model_type
    }

    fn project_logits(
        &mut self,
        hidden_states: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        project_logits_maybe_quantized(
            &mut self.lm_head,
            &mut self.model.embeddings,
            hidden_states,
            stream,
        )
    }

    pub(crate) fn forward_logits(
        &mut self,
        input: ModelInput<'_>,
        last_token_only: bool,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let hidden_states = self.model.forward(input, stream)?;
        let hidden_states = if last_token_only {
            hidden_states.try_index_device((.., -1, ..), stream)?
        } else {
            hidden_states
        };
        self.project_logits(&hidden_states, stream)
    }
}

impl Module<ModelInput<'_>> for Model {
    type Output = Array;
    type Error = Exception;

    fn forward(
        &mut self,
        input: ModelInput<'_>,
        stream: &Stream,
    ) -> Result<Self::Output, Self::Error> {
        self.forward_logits(input, false, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.model.training_mode(mode);
        if let Some(lm_head) = &mut self.lm_head {
            lm_head.training_mode(mode);
        }
    }
}

impl CausalLm<Cache> for Model {
    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let prompt_tokens = input::text_token_ids(input, stream)?;
        self.forward_logits(
            ModelInput {
                inputs: &prompt_tokens,
                mask: None,
                cache: Some(cache),
            },
            true,
            stream,
        )
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let logits = self.forward(
            ModelInput {
                inputs: input_tokens,
                mask: None,
                cache: Some(cache),
            },
            stream,
        )?;
        logits.try_index_device((.., -1, ..), stream)
    }
}

/// Nemotron-H token generation iterator.
pub type Generate<'a, S = crate::runtime::generation::sampler::DefaultSampler> =
    common::generation::Generate<'a, Model, Cache, S>;

pub(crate) struct LoadedNemotronHGguf {
    pub(crate) model: Model,
}

pub(crate) struct PreparedNemotronHGguf {
    pub(crate) args: ModelArgs,
    pub(crate) eos_token_ids: Vec<u32>,
}

/// Loads a dense or sparse-MoE Nemotron-H text model from a GGUF checkpoint.
pub fn load_nemotron_h_gguf(
    gguf_file: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    Ok(load_nemotron_h_gguf_with_metadata(gguf_file, stream, weights_stream)?.model)
}

pub(crate) fn load_nemotron_h_gguf_with_metadata(
    gguf_file: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LoadedNemotronHGguf, Error> {
    let checkpoint = GgufCheckpoint::open(gguf_file)?;
    let metadata = gguf_metadata(&checkpoint);
    load_nemotron_h_gguf_checkpoint(&checkpoint, metadata, stream, weights_stream)
}

pub(crate) fn load_nemotron_h_gguf_checkpoint(
    checkpoint: &GgufCheckpoint,
    metadata: HashMap<String, GgufMetadataValue>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LoadedNemotronHGguf, Error> {
    let architecture = gguf_string(&metadata, "general.architecture")?;
    if !matches!(architecture.as_str(), "nemotron_h" | "nemotron_h_moe") {
        return Err(Error::UnsupportedArchitecture(format!(
            "GGUF architecture {architecture:?}; this loader supports nemotron_h and nemotron_h_moe"
        )));
    }
    checkpoint
        .catalog()
        .translated_outputs(translate_gguf_weight_name)
        .map_err(safemlx::error::IoError::from)?;

    let gguf_architecture = if architecture == "nemotron_h_moe" {
        crate::api::GgufArchitecture::NemotronHMoe
    } else {
        crate::api::GgufArchitecture::NemotronH
    };
    crate::backend::mlx::structural::validate_gguf(
        gguf_architecture,
        checkpoint,
        &metadata,
        crate::api::ModelLoadOptions::default(),
    )
    .into_loader_result()?;
    let mut args = model_args_from_gguf_catalog(checkpoint, &metadata, &architecture)?;
    let quantized_weight_configs =
        gguf_quantization_configs(checkpoint, translate_gguf_weight_name)?;
    args.quantized_weights = Some(quantized_weight_configs.keys().cloned().collect());
    args.quantization = None;
    args.quantized_weight_configs = Some(quantized_weight_configs);

    let mut model = Model::new(args, stream)?;
    let config = StrictLoadConfig::default().allow_unused_prefix("rope_freqs.");
    let mut report = StrictLoadReport::default();
    load_gguf_strict(
        &mut model,
        checkpoint,
        None,
        &config,
        &mut report,
        |name, value| translate_gguf_weight(name, value, weights_stream),
    )?;
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;
    Ok(LoadedNemotronHGguf { model })
}

pub(crate) fn prepare_nemotron_h_gguf_checkpoint(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    _weights_stream: &Stream,
) -> Result<PreparedNemotronHGguf, Error> {
    let architecture = gguf_string(metadata, "general.architecture")?;
    if !matches!(architecture.as_str(), "nemotron_h" | "nemotron_h_moe") {
        return Err(Error::UnsupportedArchitecture(format!(
            "GGUF architecture {architecture:?}; this loader supports nemotron_h and nemotron_h_moe"
        )));
    }
    checkpoint
        .catalog()
        .translated_outputs(translate_gguf_weight_name)
        .map_err(safemlx::error::IoError::from)?;
    let gguf_architecture = if architecture == "nemotron_h_moe" {
        crate::api::GgufArchitecture::NemotronHMoe
    } else {
        crate::api::GgufArchitecture::NemotronH
    };
    crate::backend::mlx::structural::validate_gguf(
        gguf_architecture,
        checkpoint,
        metadata,
        crate::api::ModelLoadOptions::default(),
    )
    .into_loader_result()?;
    let mut args = model_args_from_gguf_catalog(checkpoint, metadata, &architecture)?;
    let configs = gguf_quantization_configs(checkpoint, translate_gguf_weight_name)?;
    args.quantized_weights = Some(configs.keys().cloned().collect());
    args.quantized_weight_configs = Some(configs);
    args.quantization = None;
    let eos_token_ids = crate::api::gguf_eos_token_ids(metadata)?;
    Ok(PreparedNemotronHGguf {
        args,
        eos_token_ids,
    })
}

pub(crate) fn model_args_from_gguf_catalog(
    arrays: &impl GgufTensorNames,
    metadata: &HashMap<String, GgufMetadataValue>,
    architecture: &str,
) -> Result<ModelArgs, Error> {
    if !matches!(architecture, "nemotron_h" | "nemotron_h_moe") {
        return Err(Error::UnsupportedArchitecture(format!(
            "GGUF architecture {architecture:?}; this loader supports nemotron_h and nemotron_h_moe"
        )));
    }
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let is_moe = architecture == "nemotron_h_moe";
    let expert_count_key = key("expert_count");
    let has_experts = gguf_optional_i64(metadata, &expert_count_key)?.unwrap_or(0) > 0
        || arrays.any_gguf_tensor(|name| name.contains("_exps"));
    if is_moe != has_experts {
        return Err(Error::UnsupportedArchitecture(
            "Nemotron-H GGUF architecture and expert tensors disagree".into(),
        ));
    }
    let latent_size_key = key("moe_latent_size");
    if gguf_optional_i64(metadata, &latent_size_key)?.unwrap_or(0) > 0
        || arrays.any_gguf_tensor(|name| name.contains("ffn_latent_"))
    {
        return Err(Error::UnsupportedArchitecture(
            "Nemotron-H latent-space MoE GGUF checkpoints are not supported".into(),
        ));
    }

    let num_hidden_layers = gguf_i32(metadata, &key("block_count"))?;
    ensure_positive("block_count", num_hidden_layers)?;
    let feed_forward_lengths = expand_layer_values(
        &key("feed_forward_length"),
        gguf_i64_values(metadata, &key("feed_forward_length"))?,
        num_hidden_layers,
    )?;
    let kv_head_counts = expand_layer_values(
        &key("attention.head_count_kv"),
        gguf_i64_values(metadata, &key("attention.head_count_kv"))?,
        num_hidden_layers,
    )?;
    let hybrid_override_pattern =
        hybrid_pattern_from_gguf_layers(&feed_forward_lengths, &kv_head_counts, is_moe);
    let intermediate_size =
        unique_nonzero_layer_value(&key("feed_forward_length"), &feed_forward_lengths)?;
    let num_key_value_heads =
        unique_nonzero_layer_value(&key("attention.head_count_kv"), &kv_head_counts)?;

    let inner_size = gguf_i32(metadata, &key("ssm.inner_size"))?;
    let mamba_num_heads = gguf_i32(metadata, &key("ssm.time_step_rank"))?;
    ensure_positive("ssm.inner_size", inner_size)?;
    ensure_positive("ssm.time_step_rank", mamba_num_heads)?;
    if inner_size % mamba_num_heads != 0 {
        return Err(Error::UnsupportedArchitecture(format!(
            "Nemotron-H SSM inner size {inner_size} is not divisible by {mamba_num_heads} heads"
        )));
    }
    let hidden_size = gguf_i32(metadata, &key("embedding_length"))?;
    let num_attention_heads = gguf_i32(metadata, &key("attention.head_count"))?;
    ensure_positive("embedding_length", hidden_size)?;
    ensure_positive("attention.head_count", num_attention_heads)?;
    let head_dim = gguf_optional_i64(metadata, &key("attention.key_length"))?
        .map(i32::try_from)
        .transpose()
        .map_err(|_| Error::UnsupportedArchitecture("Nemotron-H head size exceeds i32".into()))?
        .unwrap_or(hidden_size / num_attention_heads);
    let norm_eps = gguf_f32(metadata, &key("attention.layer_norm_rms_epsilon"))?;
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
        None => gguf_i32(metadata, &key("vocab_size"))?,
    };

    let n_routed_experts = if is_moe {
        gguf_i32(metadata, &key("expert_count"))?
    } else {
        default_n_routed_experts()
    };
    let n_shared_experts = if is_moe {
        gguf_optional_i64(metadata, &key("expert_shared_count"))?
            .unwrap_or(1)
            .try_into()
            .map_err(|_| Error::UnsupportedArchitecture("expert_shared_count exceeds i32".into()))?
    } else {
        default_n_shared_experts()
    };
    let moe_intermediate_size = if is_moe {
        gguf_i32(metadata, &key("expert_feed_forward_length"))?
    } else {
        default_moe_intermediate_size()
    };
    let moe_shared_expert_intermediate_size = if is_moe {
        gguf_i32(metadata, &key("expert_shared_feed_forward_length"))?
    } else {
        default_moe_shared_expert_intermediate_size()
    };
    let num_experts_per_tok = if is_moe {
        gguf_i32(metadata, &key("expert_used_count"))?
    } else {
        default_num_experts_per_tok()
    };

    let args = ModelArgsSource {
        model_type: "nemotron_h".into(),
        vocab_size,
        tie_word_embeddings: !arrays.contains_gguf_tensor("output.weight"),
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_nextn_predict_layers: 0,
        mtp_hybrid_override_pattern: None,
        mtp_layers_block_type: None,
        hybrid_override_pattern,
        num_attention_heads,
        head_dim,
        num_key_value_heads,
        rope_theta: gguf_optional_f32(metadata, &key("rope.freq_base"))?
            .unwrap_or_else(default_rope_theta),
        max_position_embeddings: gguf_i32(metadata, &key("context_length"))?,
        attention_bias: arrays.any_gguf_tensor(|name| {
            name.ends_with("attn_q.bias")
                || name.ends_with("attn_k.bias")
                || name.ends_with("attn_v.bias")
                || name.ends_with("attn_output.bias")
        }),
        mlp_bias: arrays.any_gguf_tensor(|name| {
            name.ends_with("ffn_up.bias") || name.ends_with("ffn_down.bias")
        }),
        use_bias: arrays.any_gguf_tensor(|name| {
            name.ends_with("ssm_in.bias") || name.ends_with("ssm_out.bias")
        }),
        layer_norm_epsilon: norm_eps,
        norm_eps,
        residual_in_fp32: false,
        num_logits_to_keep: 1,
        sliding_window: gguf_optional_i64(metadata, &key("attention.sliding_window"))?,
        ssm_state_size: gguf_i32(metadata, &key("ssm.state_size"))?,
        mamba_num_heads,
        n_groups: gguf_i32(metadata, &key("ssm.group_count"))?,
        mamba_head_dim: inner_size / mamba_num_heads,
        conv_kernel: gguf_i32(metadata, &key("ssm.conv_kernel"))?,
        expand: 2,
        mamba_hidden_act: "silu".into(),
        time_step_min: default_time_step_min(),
        time_step_max: default_time_step_max(),
        time_step_floor: default_time_step_floor(),
        use_conv_bias: arrays.any_gguf_tensor(|name| name.ends_with("ssm_conv1d.bias")),
        mamba_proj_bias: arrays.any_gguf_tensor(|name| {
            name.ends_with("ssm_in.bias") || name.ends_with("ssm_out.bias")
        }),
        chunk_size: default_chunk_size(),
        rescale_prenorm_residual: true,
        mlp_hidden_act: "relu2".into(),
        n_routed_experts,
        n_shared_experts,
        moe_intermediate_size,
        moe_shared_expert_intermediate_size,
        num_experts_per_tok,
        routed_scaling_factor: if is_moe {
            gguf_optional_f32(metadata, &key("expert_weights_scale"))?.unwrap_or(1.0)
        } else {
            default_routed_scaling_factor()
        },
        n_group: if is_moe {
            gguf_optional_i64(metadata, &key("expert_group_count"))?
                .unwrap_or(1)
                .try_into()
                .map_err(|_| {
                    Error::UnsupportedArchitecture("expert_group_count exceeds i32".into())
                })?
        } else {
            default_n_group()
        },
        topk_group: if is_moe {
            gguf_optional_i64(metadata, &key("expert_group_used_count"))?
                .unwrap_or(1)
                .try_into()
                .map_err(|_| {
                    Error::UnsupportedArchitecture("expert_group_used_count exceeds i32".into())
                })?
        } else {
            default_topk_group()
        },
        norm_topk_prob: if is_moe {
            gguf_optional_i64(metadata, &key("expert_weights_norm"))?.unwrap_or(1) != 0
        } else {
            true
        },
        torch_dtype: None,
        quantization: None,
        quantized_weights: None,
        quantized_weight_configs: None,
    }
    .into_args()?;
    validate_model_args(&args)?;
    Ok(args)
}

#[cfg(test)]
fn gguf_affine_quantization(
    weight_shape: &[i32],
    scales_shape: &[i32],
    weight_name: &str,
) -> Result<AffineQuantization, Error> {
    crate::runtime::checkpoint::quantization::gguf_affine_quantization(
        weight_shape,
        scales_shape,
        weight_name,
    )
}

fn expand_layer_values(
    key: &str,
    values: Vec<i64>,
    num_hidden_layers: i32,
) -> Result<Vec<i32>, Error> {
    let values = if values.len() == 1 {
        vec![values[0]; num_hidden_layers as usize]
    } else if values.len() == num_hidden_layers as usize {
        values
    } else {
        return Err(Error::UnsupportedArchitecture(format!(
            "GGUF metadata key {key:?} has {} values for {num_hidden_layers} layers",
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

fn hybrid_pattern_from_gguf_layers(
    feed_forward_lengths: &[i32],
    kv_head_counts: &[i32],
    is_moe: bool,
) -> String {
    feed_forward_lengths
        .iter()
        .zip(kv_head_counts)
        .map(|(feed_forward, kv_heads)| {
            if *feed_forward > 0 {
                if is_moe {
                    'E'
                } else {
                    '-'
                }
            } else if *kv_heads > 0 {
                '*'
            } else {
                'M'
            }
        })
        .collect()
}

fn unique_nonzero_layer_value(key: &str, values: &[i32]) -> Result<i32, Error> {
    let Some(value) = values.iter().copied().find(|value| *value > 0) else {
        return Err(Error::UnsupportedArchitecture(format!(
            "GGUF metadata key {key:?} has no non-zero layer value"
        )));
    };
    if values.iter().any(|other| *other > 0 && *other != value) {
        return Err(Error::UnsupportedArchitecture(format!(
            "GGUF metadata key {key:?} has non-uniform non-zero layer values"
        )));
    }
    Ok(value)
}

fn translate_gguf_weight(
    name: String,
    value: Array,
    stream: &Stream,
) -> Result<(String, Array), Error> {
    let translated = translate_gguf_weight_name(&name);
    let value = if name.ends_with(".ssm_conv1d.weight") {
        value.reshape(&[value.shape()[0], 1, value.shape()[1]], stream)?
    } else if name.ends_with(".ssm_a") {
        value
            .multiply(Array::from_f32(-1.0), stream)?
            .log(stream)?
            .reshape(&[-1], stream)?
    } else if name.ends_with(".ssm_d") || name.ends_with(".ssm_norm.weight") {
        value.reshape(&[-1], stream)?
    } else {
        value
    };
    Ok((translated, value))
}

pub(crate) fn translate_gguf_weight_name(name: &str) -> String {
    const ROOTS: [(&str, &str); 3] = [
        ("token_embd", "model.embeddings"),
        ("output_norm", "model.norm_f"),
        ("output", "lm_head"),
    ];
    for (source, target) in ROOTS {
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

    const MOE_PARAMETERS: [(&str, &str); 7] = [
        ("ffn_gate_inp", "gate"),
        ("exp_probs_b", "gate.e_score_correction_bias"),
        ("ffn_up_exps", "experts.up_proj"),
        ("ffn_down_exps", "experts.down_proj"),
        ("ffn_up_shexp", "shared_experts.up_proj"),
        ("ffn_down_shexp", "shared_experts.down_proj"),
        ("ffn_exp_probs_b", "gate.e_score_correction_bias"),
    ];
    for (source, target) in MOE_PARAMETERS {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            let suffix = parameter.strip_prefix(source).unwrap_or_default();
            let suffix = if target == "gate.e_score_correction_bias" && suffix == ".bias" {
                ""
            } else if target.starts_with("experts.") {
                match suffix {
                    ".weight" => "",
                    ".scales" => "_scales",
                    ".biases" => "_biases",
                    other => other,
                }
            } else {
                suffix
            };
            return format!("model.layers.{layer}.moe.{target}{suffix}");
        }
    }

    const PARAMETERS: [(&str, &str); 16] = [
        ("attn_norm", "norm"),
        ("attn_q", "attention.q_proj"),
        ("attn_k", "attention.k_proj"),
        ("attn_v", "attention.v_proj"),
        ("attn_output", "attention.o_proj"),
        ("ffn_up", "mlp.up_proj"),
        ("ffn_down", "mlp.down_proj"),
        ("ssm_in", "mamba.in_proj"),
        ("ssm_conv1d", "mamba.conv1d"),
        ("ssm_dt.bias", "mamba.dt_bias"),
        ("ssm_a", "mamba.A_log"),
        ("ssm_d", "mamba.D"),
        ("ssm_norm", "mamba.norm"),
        ("ssm_out", "mamba.out_proj"),
        ("rope_freqs", "rope_freqs"),
        ("ffn_norm", "ffn_norm"),
    ];
    for (source, target) in PARAMETERS {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            return format!(
                "model.layers.{layer}.{}",
                parameter.replacen(source, target, 1)
            );
        }
    }
    name.to_string()
}

fn gguf_string(metadata: &HashMap<String, GgufMetadataValue>, key: &str) -> Result<String, Error> {
    match metadata.get(key) {
        Some(GgufMetadataValue::String(value)) => Ok(value.clone()),
        Some(_) => Err(Error::UnsupportedArchitecture(format!(
            "GGUF metadata key {key:?} has the wrong type"
        ))),
        None => Err(Error::UnsupportedArchitecture(format!(
            "GGUF metadata is missing required key {key:?}"
        ))),
    }
}

fn gguf_i32(metadata: &HashMap<String, GgufMetadataValue>, key: &str) -> Result<i32, Error> {
    i32::try_from(gguf_i64(metadata, key)?).map_err(|_| {
        Error::UnsupportedArchitecture(format!("GGUF metadata value {key:?} exceeds i32"))
    })
}

fn gguf_i64(metadata: &HashMap<String, GgufMetadataValue>, key: &str) -> Result<i64, Error> {
    gguf_optional_i64(metadata, key)?.ok_or_else(|| {
        Error::UnsupportedArchitecture(format!("GGUF metadata is missing required key {key:?}"))
    })
}

fn gguf_optional_i64(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<Option<i64>, Error> {
    let Some(values) = gguf_optional_i64_values(metadata, key)? else {
        return Ok(None);
    };
    if values.len() != 1 {
        return Err(Error::UnsupportedArchitecture(format!(
            "GGUF metadata key {key:?} must be scalar"
        )));
    }
    Ok(values.into_iter().next())
}

fn gguf_i64_values(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<Vec<i64>, Error> {
    gguf_optional_i64_values(metadata, key)?.ok_or_else(|| {
        Error::UnsupportedArchitecture(format!("GGUF metadata is missing required key {key:?}"))
    })
}

fn gguf_optional_i64_values(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<Option<Vec<i64>>, Error> {
    match metadata.get(key) {
        Some(value) => value.to_i64_vec().map(Some).ok_or_else(|| {
            Error::UnsupportedArchitecture(format!("GGUF metadata key {key:?} has the wrong type"))
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

/// Loads `tokenizer.json` from a Nemotron-H model directory.
pub fn load_nemotron_h_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let file = model_dir.as_ref().join("tokenizer.json");
    Tokenizer::from_file(file).map_err(Into::into)
}

/// Reads Nemotron-H model arguments from `config.json`.
pub fn get_nemotron_h_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let file = std::fs::File::open(model_dir.as_ref().join("config.json"))?;
    model_args_from_config_value(&serde_json::from_reader(file)?)
}

/// Loads a Nemotron-H model and safetensors weights from a model directory.
pub fn load_nemotron_h_model(
    model_dir: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    crate::backend::mlx::structural::validate_safetensors_load_path(
        crate::api::ModelKind::NemotronH,
        model_dir,
        crate::api::ModelLoadOptions::default(),
    )?;
    let args = get_nemotron_h_model_args(model_dir)?;
    let mut model = Model::new(args.clone(), stream)?;
    let config = nemotron_h_strict_load_config();
    let mut report = StrictLoadReport::default();
    load_nemotron_h_safetensors_strict(
        &mut model,
        model_dir,
        &args,
        weights_stream,
        stream,
        &config,
        &mut report,
    )?;
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;
    Ok(model)
}

/// Strict-loading key rules for Nemotron-H checkpoints.
pub fn nemotron_h_strict_load_config() -> StrictLoadConfig {
    StrictLoadConfig::default()
        .rewrite_prefix("backbone.", "model.")
        .rewrite_prefix("model.backbone.", "model.")
}

/// Remaps public Nemotron-H checkpoint tensors to the runtime parameter tree.
pub fn transform_nemotron_h_weights(
    loaded: std::collections::HashMap<String, Array>,
    args: &ModelArgs,
    stream: &Stream,
) -> Result<std::collections::HashMap<String, Array>, Error> {
    let mut rewritten = std::collections::HashMap::with_capacity(loaded.len());
    for (key, value) in loaded {
        rewritten.insert(rewrite_nemotron_h_weight_key(&key, args)?, value);
    }
    transform_split_relu2_experts(rewritten, args.n_routed_experts, stream)
}

pub(crate) fn rewrite_nemotron_h_weight_key(key: &str, args: &ModelArgs) -> Result<String, Error> {
    let Some(rest) = key.strip_prefix("backbone.layers.") else {
        return Ok(key.to_string());
    };
    let Some((layer_idx, suffix)) = rest.split_once(".mixer.") else {
        return Ok(key.to_string());
    };
    let layer_idx = layer_idx.parse::<usize>().map_err(|error| {
        Error::UnsupportedArchitecture(format!(
            "invalid Nemotron-H layer index in checkpoint key '{key}': {error}"
        ))
    })?;
    let field = match args.layer_policy(layer_idx)? {
        LayerPolicy::Mamba => "mamba",
        LayerPolicy::SelfAttention(_) => "attention",
        LayerPolicy::DenseMlp => "mlp",
        LayerPolicy::SparseMoe => "moe",
    };
    Ok(format!("backbone.layers.{layer_idx}.{field}.{suffix}"))
}

/// Strict-loads Nemotron-H safetensors into a module after applying checkpoint remaps.
pub fn load_nemotron_h_safetensors_strict<M: safemlx::module::ModuleParameters>(
    model: &mut M,
    model_dir: impl AsRef<Path>,
    args: &ModelArgs,
    weights_stream: &Stream,
    transform_stream: &Stream,
    config: &StrictLoadConfig,
    report: &mut StrictLoadReport,
) -> Result<(), Error> {
    load_safetensors_dir_strict_with_split_relu2_experts(
        model,
        model_dir,
        weights_stream,
        transform_stream,
        config,
        report,
        args.n_routed_experts,
        |key| rewrite_nemotron_h_weight_key(key, args),
    )
}

/// Validates a parsed Nemotron-H config value.
pub fn model_args_from_config_value(config: &Value) -> Result<ModelArgs, Error> {
    let source: ModelArgsSource = serde_json::from_value(config.clone()).map_err(|error| {
        Error::UnsupportedArchitecture(format!("invalid nemotron_h config: {error}"))
    })?;
    let args = source.into_args()?;
    validate_model_args(&args)?;
    Ok(args)
}

/// Validates a parsed Nemotron-H config value.
pub(crate) fn validate_model_config_value(config: &Value) -> Result<(), Error> {
    model_args_from_config_value(config).map(|_| ())
}

pub(crate) fn validate_model_args(args: &ModelArgs) -> Result<(), Error> {
    if args.model_type != "nemotron_h" {
        return Err(Error::UnsupportedModelType(args.model_type.clone()));
    }

    if args.layer_schedule.len() != args.num_hidden_layers as usize {
        return Err(Error::UnsupportedArchitecture(format!(
            "Nemotron-H layer schedule has {} entries, expected {}",
            args.layer_schedule.len(),
            args.num_hidden_layers
        )));
    }

    let has_attention = args
        .layer_schedule
        .iter()
        .any(|policy| matches!(policy, LayerPolicy::SelfAttention(_)));
    let has_mamba = args
        .layer_schedule
        .iter()
        .any(|policy| *policy == LayerPolicy::Mamba);
    let has_mlp = args
        .layer_schedule
        .iter()
        .any(|policy| *policy == LayerPolicy::DenseMlp);
    let has_moe = args
        .layer_schedule
        .iter()
        .any(|policy| *policy == LayerPolicy::SparseMoe);

    ensure_positive("vocab_size", args.vocab_size)?;
    ensure_positive("hidden_size", args.hidden_size)?;
    ensure_positive("num_hidden_layers", args.num_hidden_layers)?;
    ensure_positive("num_attention_heads", args.num_attention_heads)?;
    ensure_positive("num_key_value_heads", args.num_key_value_heads)?;
    ensure_positive("head_dim", args.head_dim)?;
    ensure_positive("max_position_embeddings", args.max_position_embeddings)?;
    ensure_positive("ssm_state_size", args.ssm_state_size)?;
    ensure_positive("mamba_num_heads", args.mamba_num_heads)?;
    ensure_positive("n_groups", args.n_groups)?;
    ensure_positive("mamba_head_dim", args.mamba_head_dim)?;
    ensure_positive("conv_kernel", args.conv_kernel)?;
    ensure_positive("chunk_size", args.chunk_size)?;
    ensure_positive("n_routed_experts", args.n_routed_experts)?;
    ensure_positive("n_shared_experts", args.n_shared_experts)?;
    ensure_positive("num_experts_per_tok", args.num_experts_per_tok)?;

    if has_attention && args.num_attention_heads % args.num_key_value_heads != 0 {
        return Err(Error::UnsupportedArchitecture(format!(
            "Nemotron-H num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
            args.num_attention_heads, args.num_key_value_heads
        )));
    }
    if has_mamba {
        if args.mamba_num_heads % args.n_groups != 0 {
            return Err(Error::UnsupportedArchitecture(format!(
                "Nemotron-H mamba_num_heads ({}) must be divisible by n_groups ({})",
                args.mamba_num_heads, args.n_groups
            )));
        }
        let intermediate = args
            .mamba_num_heads
            .checked_mul(args.mamba_head_dim)
            .ok_or_else(|| {
                Error::UnsupportedArchitecture(
                    "Nemotron-H Mamba intermediate size overflows i32".into(),
                )
            })?;
        let grouped_state = args
            .n_groups
            .checked_mul(args.ssm_state_size)
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                Error::UnsupportedArchitecture("Nemotron-H Mamba state width overflows i32".into())
            })?;
        intermediate
            .checked_add(grouped_state)
            .and_then(|conv| conv.checked_add(intermediate))
            .and_then(|projection| projection.checked_add(args.mamba_num_heads))
            .ok_or_else(|| {
                Error::UnsupportedArchitecture(
                    "Nemotron-H Mamba projection size overflows i32".into(),
                )
            })?;
    }
    if has_mlp {
        ensure_positive("intermediate_size", args.intermediate_size)?;
    }
    if has_moe {
        ensure_positive("moe_intermediate_size", args.moe_intermediate_size)?;
        ensure_positive(
            "moe_shared_expert_intermediate_size",
            args.moe_shared_expert_intermediate_size,
        )?;
        ensure_positive("n_group", args.n_group)?;
        ensure_positive("topk_group", args.topk_group)?;
        if args.n_shared_experts != 1 {
            return Err(Error::UnsupportedArchitecture(format!(
                "Nemotron-H loader requires exactly one shared expert, got {}",
                args.n_shared_experts
            )));
        }
        let grouped_capacity = args
            .topk_group
            .checked_mul(args.n_routed_experts / args.n_group);
        if args.n_routed_experts % args.n_group != 0
            || args.topk_group > args.n_group
            || grouped_capacity.is_none_or(|capacity| args.num_experts_per_tok > capacity)
        {
            return Err(Error::UnsupportedArchitecture(format!(
                "invalid Nemotron-H grouped expert routing: experts={}, groups={}, topk_group={}, experts_per_token={}",
                args.n_routed_experts, args.n_group, args.topk_group, args.num_experts_per_tok
            )));
        }
    }

    if args.num_experts_per_tok > args.n_routed_experts {
        return Err(Error::UnsupportedArchitecture(format!(
            "Nemotron-H num_experts_per_tok ({}) exceeds n_routed_experts ({})",
            args.num_experts_per_tok, args.n_routed_experts
        )));
    }
    if args.mlp_hidden_act != "relu2" {
        return Err(Error::UnsupportedArchitecture(format!(
            "unsupported Nemotron-H MLP activation '{}'",
            args.mlp_hidden_act
        )));
    }
    if args.mamba_hidden_act != "silu" {
        return Err(Error::UnsupportedArchitecture(format!(
            "unsupported Nemotron-H Mamba activation '{}'",
            args.mamba_hidden_act
        )));
    }
    if let Some(torch_dtype) = &args.torch_dtype {
        if !matches!(
            torch_dtype.as_str(),
            "bfloat16" | "bf16" | "float16" | "float32"
        ) {
            return Err(Error::UnsupportedArchitecture(format!(
                "unsupported Nemotron-H torch_dtype '{torch_dtype}'"
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        expand_layer_values, gguf_affine_quantization, hybrid_pattern_from_gguf_layers,
        load_nemotron_h_gguf, load_nemotron_h_model, load_nemotron_h_safetensors_strict,
        model_args_from_config_value, rewrite_nemotron_h_weight_key, translate_gguf_weight_name,
        unique_nonzero_layer_value, validate_model_config_value, AttentionCache, Experts,
        LayerCache, LayerPolicy, Model, ModelArgs, ModelInput, SparseMoeBlock,
    };
    use crate::runtime::checkpoint::load::{StrictLoadConfig, StrictLoadReport};
    use crate::{
        nn::{
            generation::CausalLm,
            moe::{quantize_expert_bank, TopKRouterScoreFunction},
        },
        runtime::checkpoint::quantization::{AffineQuantization, WeightQuantization},
        AttentionPolicy, LayerSchedule,
    };
    use safemlx::{
        module::{Module, ModuleParameters, Param},
        ops::{dequantize_with_mode, indexing::TryIndexOp},
        Array, Device, DeviceType, ExecutionContext,
    };
    use serde_json::json;
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicUsize, Ordering},
        time::{SystemTime, UNIX_EPOCH},
    };

    static TEMP_DIR_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn nemotron_nano_config() -> serde_json::Value {
        serde_json::from_str(
            r#"{
              "attention_bias": false,
              "chunk_size": 128,
              "conv_kernel": 4,
              "eos_token_id": 2,
              "expand": 2,
              "head_dim": 128,
              "hidden_size": 2688,
              "hybrid_override_pattern": "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
              "intermediate_size": 1856,
              "layer_norm_epsilon": 1e-05,
              "mamba_head_dim": 64,
              "mamba_hidden_act": "silu",
              "mamba_num_heads": 64,
              "mamba_proj_bias": false,
              "max_position_embeddings": 262144,
              "mlp_bias": false,
              "mlp_hidden_act": "relu2",
              "model_type": "nemotron_h",
              "moe_intermediate_size": 1856,
              "moe_shared_expert_intermediate_size": 3712,
              "n_group": 1,
              "n_groups": 8,
              "n_routed_experts": 128,
              "n_shared_experts": 1,
              "norm_eps": 1e-05,
              "norm_topk_prob": true,
              "num_attention_heads": 32,
              "num_experts_per_tok": 6,
              "num_hidden_layers": 52,
              "num_key_value_heads": 2,
              "rescale_prenorm_residual": true,
              "rope_theta": 10000,
              "routed_scaling_factor": 2.5,
              "ssm_state_size": 128,
              "tie_word_embeddings": false,
              "time_step_floor": 0.0001,
              "time_step_max": 0.1,
              "time_step_min": 0.001,
              "topk_group": 1,
              "torch_dtype": "bfloat16",
              "use_bias": false,
              "use_conv_bias": true,
              "vocab_size": 131072
            }"#,
        )
        .unwrap()
    }

    fn temp_model_dir() -> std::path::PathBuf {
        let id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let counter = TEMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "nemotron_h_test_{}_{}_{}",
            std::process::id(),
            id,
            counter
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn tiny_moe_args() -> ModelArgs {
        let mut config = nemotron_nano_config();
        config["hidden_size"] = json!(4);
        config["intermediate_size"] = json!(3);
        config["moe_intermediate_size"] = json!(3);
        config["moe_shared_expert_intermediate_size"] = json!(5);
        config["n_routed_experts"] = json!(2);
        config["num_experts_per_tok"] = json!(2);
        model_args_from_config_value(&config).unwrap()
    }

    fn tiny_full_args() -> ModelArgs {
        let mut config = nemotron_nano_config();
        config["vocab_size"] = json!(16);
        config["hidden_size"] = json!(8);
        config["intermediate_size"] = json!(12);
        config["num_hidden_layers"] = json!(4);
        config["hybrid_override_pattern"] = json!("M-E*");
        config["num_attention_heads"] = json!(2);
        config["num_key_value_heads"] = json!(1);
        config["head_dim"] = json!(4);
        config["mamba_num_heads"] = json!(2);
        config["mamba_head_dim"] = json!(4);
        config["n_groups"] = json!(1);
        config["ssm_state_size"] = json!(4);
        config["conv_kernel"] = json!(3);
        config["chunk_size"] = json!(2);
        config["moe_intermediate_size"] = json!(6);
        config["moe_shared_expert_intermediate_size"] = json!(10);
        config["n_routed_experts"] = json!(2);
        config["num_experts_per_tok"] = json!(2);
        model_args_from_config_value(&config).unwrap()
    }

    #[test]
    fn parses_nemotron_nano_config_fields() {
        let args = model_args_from_config_value(&nemotron_nano_config()).unwrap();
        assert_eq!(args.model_type, "nemotron_h");
        assert_eq!(args.hidden_size, 2688);
        assert_eq!(args.num_hidden_layers, 52);
        assert_eq!(args.n_routed_experts, 128);
        assert_eq!(args.n_shared_experts, 1);
        assert_eq!(args.num_experts_per_tok, 6);
        assert_eq!(args.mlp_hidden_act, "relu2");
        assert_eq!(args.mamba_hidden_act, "silu");
        assert_eq!(args.torch_dtype.as_deref(), Some("bfloat16"));

        let blocks = &args.layer_schedule;
        assert_eq!(blocks.len(), 52);
        assert_eq!(
            blocks
                .iter()
                .filter(|&&block| block == LayerPolicy::Mamba)
                .count(),
            23
        );
        assert_eq!(
            blocks
                .iter()
                .filter(|&&block| block == LayerPolicy::SparseMoe)
                .count(),
            23
        );
        assert_eq!(
            blocks
                .iter()
                .filter(|block| matches!(block, LayerPolicy::SelfAttention(_)))
                .count(),
            6
        );
    }

    #[test]
    fn prompt_cache_layout_records_mamba_attention_and_stateless_layers() {
        use crate::core::cache::{LayerCachePolicy, StateTensorRole};

        let args = tiny_full_args();
        let layout = super::prompt_cache_layer_layout(&args).unwrap();
        assert_eq!(layout.len(), 4);
        match layout.get(0).unwrap() {
            LayerCachePolicy::FixedState { tensors } => {
                assert_eq!(tensors.len(), 2);
                assert_eq!(tensors[0].role, StateTensorRole::Convolution { slot: 0 });
                assert_eq!(tensors[1].role, StateTensorRole::Recurrent);
            }
            policy => panic!("unexpected Nemotron-H Mamba policy {policy:?}"),
        }
        assert_eq!(layout.get(1), Some(&LayerCachePolicy::NoState));
        assert_eq!(layout.get(2), Some(&LayerCachePolicy::NoState));
        assert!(matches!(
            layout.get(3).unwrap(),
            LayerCachePolicy::KeyValue { .. }
        ));
        let mut reordered = args.clone();
        reordered.layer_schedule = LayerSchedule::new(
            4,
            vec![
                LayerPolicy::SelfAttention(AttentionPolicy::Full),
                LayerPolicy::SparseMoe,
                LayerPolicy::DenseMlp,
                LayerPolicy::Mamba,
            ],
        )
        .unwrap();
        assert_ne!(
            super::prompt_cache_architecture_fingerprint(&args),
            super::prompt_cache_architecture_fingerprint(&reordered)
        );
    }

    #[test]
    fn validates_nemotron_nano_config() {
        validate_model_config_value(&nemotron_nano_config()).unwrap();
    }

    #[test]
    fn rewrites_public_mixer_keys_by_layer_type() {
        let args = tiny_full_args();
        assert_eq!(
            rewrite_nemotron_h_weight_key("backbone.layers.0.mixer.in_proj.weight", &args).unwrap(),
            "backbone.layers.0.mamba.in_proj.weight"
        );
        assert_eq!(
            rewrite_nemotron_h_weight_key("backbone.layers.1.mixer.up_proj.weight", &args).unwrap(),
            "backbone.layers.1.mlp.up_proj.weight"
        );
        assert_eq!(
            rewrite_nemotron_h_weight_key("backbone.layers.2.mixer.gate.weight", &args).unwrap(),
            "backbone.layers.2.moe.gate.weight"
        );
        assert_eq!(
            rewrite_nemotron_h_weight_key("backbone.layers.3.mixer.q_proj.weight", &args).unwrap(),
            "backbone.layers.3.attention.q_proj.weight"
        );
    }

    #[test]
    fn translates_nemotron_h_gguf_tensor_names() {
        assert_eq!(
            translate_gguf_weight_name("token_embd.weight"),
            "model.embeddings.weight"
        );
        assert_eq!(
            translate_gguf_weight_name("blk.12.attn_q.scales"),
            "model.layers.12.attention.q_proj.scales"
        );
        assert_eq!(
            translate_gguf_weight_name("blk.0.ssm_conv1d.weight"),
            "model.layers.0.mamba.conv1d.weight"
        );
        assert_eq!(
            translate_gguf_weight_name("blk.0.ssm_dt.bias"),
            "model.layers.0.mamba.dt_bias"
        );
        assert_eq!(
            translate_gguf_weight_name("blk.1.ffn_down.weight"),
            "model.layers.1.mlp.down_proj.weight"
        );
        assert_eq!(
            translate_gguf_weight_name("output.weight"),
            "lm_head.weight"
        );
        assert_eq!(
            translate_gguf_weight_name("blk.1.ffn_gate_inp.weight"),
            "model.layers.1.moe.gate.weight"
        );
        assert_eq!(
            translate_gguf_weight_name("blk.1.exp_probs_b.bias"),
            "model.layers.1.moe.gate.e_score_correction_bias"
        );
        assert_eq!(
            translate_gguf_weight_name("blk.1.ffn_up_exps.weight"),
            "model.layers.1.moe.experts.up_proj"
        );
        assert_eq!(
            translate_gguf_weight_name("blk.1.ffn_up_exps.scales"),
            "model.layers.1.moe.experts.up_proj_scales"
        );
        assert_eq!(
            translate_gguf_weight_name("blk.1.ffn_down_exps.biases"),
            "model.layers.1.moe.experts.down_proj_biases"
        );
        assert_eq!(
            translate_gguf_weight_name("blk.1.ffn_up_shexp.weight"),
            "model.layers.1.moe.shared_experts.up_proj.weight"
        );
    }

    #[test]
    fn reconstructs_dense_hybrid_layer_metadata() {
        let feed_forward =
            expand_layer_values("nemotron_h.feed_forward_length", vec![0, 12544, 0, 0], 4).unwrap();
        let kv_heads =
            expand_layer_values("nemotron_h.attention.head_count_kv", vec![0, 0, 0, 8], 4).unwrap();
        let pattern = hybrid_pattern_from_gguf_layers(&feed_forward, &kv_heads, false);
        assert_eq!(pattern, "M-M*");
        assert_eq!(
            hybrid_pattern_from_gguf_layers(&feed_forward, &kv_heads, true),
            "MEM*"
        );
        assert_eq!(
            unique_nonzero_layer_value("feed_forward", &feed_forward).unwrap(),
            12544
        );
    }

    #[test]
    fn infers_mixed_q4_and_q8_affine_packing_per_tensor() {
        assert_eq!(
            gguf_affine_quantization(&[17504, 392], &[17504, 98], "ssm_in.weight").unwrap(),
            AffineQuantization::new(32, 4).unwrap()
        );
        assert_eq!(
            gguf_affine_quantization(&[131072, 784], &[131072, 98], "lm_head.weight").unwrap(),
            AffineQuantization::new(32, 8).unwrap()
        );
        assert!(gguf_affine_quantization(&[16, 196], &[16, 98], "bad.weight").is_err());
    }

    #[test]
    #[ignore = "requires an MLX Metal device"]
    fn packed_nemotron_experts_match_dequantized_reference_with_empty_routes() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        let bank_values = |offset: usize| {
            (0..3 * 32 * 32)
                .map(|index| {
                    let value = (index + offset) % 41;
                    (value as f32 - 20.0) / 32.0
                })
                .collect::<Vec<_>>()
        };
        for quantization in [
            WeightQuantization::Affine(AffineQuantization::new(32, 4).unwrap()),
            WeightQuantization::MxFp4,
        ] {
            let up = Array::from_slice(&bank_values(0), &[3, 32, 32]);
            let down = Array::from_slice(&bank_values(17), &[3, 32, 32]);
            let up = quantize_expert_bank(&up, quantization, stream).unwrap();
            let down = quantize_expert_bank(&down, quantization, stream).unwrap();
            let up_scales = up.scales.clone();
            let up_biases = up.biases.clone();
            let down_scales = down.scales.clone();
            let down_biases = down.biases.clone();

            let mut packed =
                Experts::new(3, 32, 32, [Some(quantization), Some(quantization)], stream).unwrap();
            packed.up_proj = Param::new(up.weight.clone());
            packed.up_proj_scales = Param::new(Some(up_scales.clone()));
            packed.up_proj_biases = Param::new(up_biases.clone());
            packed.down_proj = Param::new(down.weight.clone());
            packed.down_proj_scales = Param::new(Some(down_scales.clone()));
            packed.down_proj_biases = Param::new(down_biases.clone());

            let mut reference = Experts::new(3, 32, 32, [None, None], stream).unwrap();
            reference.up_proj = Param::new(
                dequantize_with_mode(
                    &up.weight,
                    &up_scales,
                    up_biases.as_ref(),
                    quantization.group_size(),
                    quantization.bits(),
                    quantization.mode(),
                    stream,
                )
                .unwrap(),
            );
            reference.down_proj = Param::new(
                dequantize_with_mode(
                    &down.weight,
                    &down_scales,
                    down_biases.as_ref(),
                    quantization.group_size(),
                    quantization.bits(),
                    quantization.mode(),
                    stream,
                )
                .unwrap(),
            );

            let input_values = (0..3 * 32)
                .map(|index| ((index * 7 % 29) as f32 - 14.0) / 16.0)
                .collect::<Vec<_>>();
            let hidden = Array::from_slice(&input_values, &[3, 32]);
            // Expert 1 deliberately receives no routes.
            let indices = Array::from_slice(&[2i32, 0, 2, 0, 2, 0], &[3, 2]);
            let weights = Array::from_slice(&[0.75f32, 0.25, 0.6, 0.4, 0.9, 0.1], &[3, 2]);
            let actual = packed.forward(&hidden, &indices, &weights, stream).unwrap();
            let actual = actual.evaluated().unwrap();
            let expected = reference
                .forward(&hidden, &indices, &weights, stream)
                .unwrap();
            let expected = expected.evaluated().unwrap();
            let maximum_error = actual
                .as_slice::<f32>()
                .iter()
                .zip(expected.as_slice::<f32>())
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            let tolerance = if quantization == WeightQuantization::MxFp4 {
                0.4
            } else {
                2e-2
            };
            assert!(
                maximum_error < tolerance,
                "{quantization:?} maximum error {maximum_error}"
            );
        }
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn strict_load_packs_public_split_moe_expert_weights() {
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let stream = ctx.stream();
        let args = tiny_moe_args();
        let dir = temp_model_dir();
        let weights_path = dir.join("model.safetensors");
        let arrays = [
            (
                "backbone.layers.1.mixer.gate.weight",
                Array::zeros::<f32>(&[2, 4], stream).unwrap(),
            ),
            (
                "backbone.layers.1.mixer.gate.e_score_correction_bias",
                Array::zeros::<f32>(&[2], stream).unwrap(),
            ),
            (
                "backbone.layers.1.mixer.experts.0.up_proj.weight",
                Array::zeros::<f32>(&[3, 4], stream).unwrap(),
            ),
            (
                "backbone.layers.1.mixer.experts.0.down_proj.weight",
                Array::zeros::<f32>(&[4, 3], stream).unwrap(),
            ),
            (
                "backbone.layers.1.mixer.experts.1.up_proj.weight",
                Array::zeros::<f32>(&[3, 4], stream).unwrap(),
            ),
            (
                "backbone.layers.1.mixer.experts.1.down_proj.weight",
                Array::zeros::<f32>(&[4, 3], stream).unwrap(),
            ),
            (
                "backbone.layers.1.mixer.shared_experts.up_proj.weight",
                Array::zeros::<f32>(&[5, 4], stream).unwrap(),
            ),
            (
                "backbone.layers.1.mixer.shared_experts.down_proj.weight",
                Array::zeros::<f32>(&[4, 5], stream).unwrap(),
            ),
        ];
        Array::save_safetensors(
            arrays.iter().map(|(key, value)| (*key, value)),
            None,
            &weights_path,
        )
        .unwrap();

        let mut moe = SparseMoeBlock::new(&args, 1, stream).unwrap();
        let config = StrictLoadConfig::default().rewrite_prefix("backbone.layers.1.moe.", "");
        let mut report = StrictLoadReport::default();
        load_nemotron_h_safetensors_strict(
            &mut moe,
            &dir,
            &args,
            stream,
            stream,
            &config,
            &mut report,
        )
        .unwrap();
        report.finish(&moe, &config).unwrap();
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn full_model_parameter_tree_matches_public_checkpoint_roots() {
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let model = Model::new(tiny_full_args(), stream).unwrap();
        let params = model.parameters().flatten();

        for key in [
            "model.embeddings.weight",
            "model.layers.0.norm.weight",
            "model.layers.0.mamba.in_proj.weight",
            "model.layers.0.mamba.conv1d.weight",
            "model.layers.1.mlp.up_proj.weight",
            "model.layers.2.moe.gate.weight",
            "model.layers.2.moe.gate.e_score_correction_bias",
            "model.layers.2.moe.experts.up_proj",
            "model.layers.2.moe.shared_experts.up_proj.weight",
            "model.layers.3.attention.q_proj.weight",
            "model.norm_f.weight",
            "lm_head.weight",
        ] {
            assert!(params.contains_key(key), "missing parameter key {key}");
        }
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn full_model_prefill_and_decode_shape_smoke() {
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let mut model = Model::new(tiny_full_args(), stream).unwrap();
        let mut cache = model.new_cache();
        let prompt = Array::from_slice(&[1_u32, 2, 3], &[1, 3]);
        let input_parts = [crate::runtime::media::input::InputPart::text_token_ids(
            &prompt,
        )];
        let input = crate::runtime::media::input::ModelInput::new(&input_parts);
        let logits = CausalLm::prefill_input_logits(&mut model, input, &mut cache, stream).unwrap();
        assert_eq!(logits.shape(), &[1, 16]);
        assert!(cache.offset() >= 3);

        let next = Array::from_slice(&[4_u32], &[1, 1]);
        let logits = CausalLm::decode_logits(&mut model, &next, &mut cache, stream).unwrap();
        assert_eq!(logits.shape(), &[1, 16]);
        assert!(cache.offset() >= 4);
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn heterogeneous_live_paging_matches_resident_mamba_and_attention() {
        use crate::{
            core::cache::{PromptCacheDescriptor, PromptCacheOptions, PromptCacheTopology},
            runtime::cache::residency::{CacheResidencyPolicy, PagedCacheOptions},
        };

        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let args = tiny_full_args();
        let mut resident = Model::new(args, stream).unwrap();
        for (index, parameter) in resident.parameters_mut().flatten().values_mut().enumerate() {
            **parameter = Array::full::<f32>(
                parameter.shape(),
                Array::from_f32((index + 1) as f32 * 0.001),
                stream,
            )
            .unwrap();
        }
        let mut paged = resident.clone();
        let mut resident_cache = resident.new_cache();
        let mut paged_cache = paged
            .new_cache_with_options(CacheResidencyPolicy::Paged(
                PagedCacheOptions::new(2, 1 << 20, 1 << 20, 1)
                    .unwrap()
                    .with_full_attention(true),
            ))
            .unwrap();
        for tokens in [
            Array::from_slice(&[1_u32, 2, 3], &[1, 3]),
            Array::from_slice(&[4_u32, 5], &[1, 2]),
            Array::from_slice(&[6_u32], &[1, 1]),
        ] {
            let expected = resident
                .forward(
                    ModelInput {
                        inputs: &tokens,
                        mask: None,
                        cache: Some(&mut resident_cache),
                    },
                    stream,
                )
                .unwrap();
            let actual = paged
                .forward(
                    ModelInput {
                        inputs: &tokens,
                        mask: None,
                        cache: Some(&mut paged_cache),
                    },
                    stream,
                )
                .unwrap();
            assert!(expected
                .all_close(&actual, 1e-5, 1e-5, None, stream)
                .unwrap()
                .item::<bool>(stream));
            assert_eq!(paged_cache.offset(), resident_cache.offset());
        }
        match (&resident_cache.layers[0], &paged_cache.layers[0]) {
            (LayerCache::Mamba(resident), LayerCache::Mamba(paged)) => {
                assert_eq!(
                    resident.conv_state.as_ref().unwrap().shape(),
                    paged.conv_state.as_ref().unwrap().shape()
                );
                assert_eq!(
                    resident.ssm_state.as_ref().unwrap().shape(),
                    paged.ssm_state.as_ref().unwrap().shape()
                );
            }
            _ => panic!("Nemotron-H layer zero must retain resident Mamba state"),
        }
        assert!(matches!(
            &paged_cache.layers[3],
            LayerCache::Attention(AttentionCache::Paged(_))
        ));
        let report = paged_cache.residency_report().unwrap().unwrap();
        assert_eq!(report.logical_cached_tokens, 6);
        assert!(report.key_value_blocks > 0);
        assert!(report.prefill_full_attention_blocks > 0);
        assert!(report.decode_full_attention_blocks > 0);

        let prefix_ids = [1_u32, 2, 3, 4, 5, 6];
        let layout = super::prompt_cache_layer_layout(&paged.args).unwrap();
        let descriptor = PromptCacheDescriptor {
            model_family: "nemotron_h".into(),
            effective_model_type: paged.args.model_type.clone(),
            checkpoint_fingerprint: "deterministic-paged-fixture".into(),
            prefix_content_fingerprint: "tokens:1,2,3,4,5,6".into(),
            architecture_fingerprint: super::prompt_cache_architecture_fingerprint(&paged.args),
            layer_count: layout.len(),
            global_layer_start: 0,
            global_layer_end: layout.len(),
            batch_size: 1,
            layer_prefix_offsets: vec![0; layout.len()],
            layer_layout: layout,
            sink_tokens: 0,
            topology: PromptCacheTopology::default(),
        };
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("paged-prompt-cache");
        Model::save_prompt_cache(
            &mut paged_cache,
            &destination,
            descriptor.clone(),
            &prefix_ids,
            &PromptCacheOptions::default(),
            stream,
        )
        .unwrap();
        drop(paged_cache);
        let (mut restored, _) = Model::load_paged_prompt_cache(
            &paged.args,
            &destination,
            &descriptor,
            &prefix_ids,
            PagedCacheOptions::new(2, 1 << 20, 1 << 20, 1)
                .unwrap()
                .with_full_attention(true),
            stream,
        )
        .unwrap();
        let next = Array::from_slice(&[7_u32], &[1, 1]);
        let expected = resident
            .forward(
                ModelInput {
                    inputs: &next,
                    mask: None,
                    cache: Some(&mut resident_cache),
                },
                stream,
            )
            .unwrap();
        let actual = paged
            .forward(
                ModelInput {
                    inputs: &next,
                    mask: None,
                    cache: Some(&mut restored),
                },
                stream,
            )
            .unwrap();
        assert!(expected
            .all_close(&actual, 1e-5, 1e-5, None, stream)
            .unwrap()
            .item::<bool>(stream));
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn heterogeneous_sliding_live_paging_matches_resident() {
        use crate::runtime::cache::residency::{CacheResidencyPolicy, PagedCacheOptions};

        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let mut args = tiny_full_args();
        args.layer_schedule = LayerSchedule::new(
            args.layer_schedule.len(),
            args.layer_schedule
                .iter()
                .copied()
                .map(|policy| match policy {
                    LayerPolicy::SelfAttention(_) => {
                        LayerPolicy::SelfAttention(AttentionPolicy::sliding(3).unwrap())
                    }
                    policy => policy,
                })
                .collect(),
        )
        .unwrap();
        let mut resident = Model::new(args, stream).unwrap();
        for (index, parameter) in resident.parameters_mut().flatten().values_mut().enumerate() {
            **parameter = Array::full::<f32>(
                parameter.shape(),
                Array::from_f32((index + 1) as f32 * 0.001),
                stream,
            )
            .unwrap();
        }
        let mut paged = resident.clone();
        let mut resident_cache = resident.new_cache();
        let mut paged_cache = paged
            .new_cache_with_options(CacheResidencyPolicy::Paged(
                PagedCacheOptions::new(2, 1 << 20, 1 << 20, 1).unwrap(),
            ))
            .unwrap();
        for tokens in [
            Array::from_slice(&[1_u32, 2, 3, 4, 5], &[1, 5]),
            Array::from_slice(&[6_u32], &[1, 1]),
            Array::from_slice(&[7_u32, 8], &[1, 2]),
        ] {
            let expected = resident
                .forward(
                    ModelInput {
                        inputs: &tokens,
                        mask: None,
                        cache: Some(&mut resident_cache),
                    },
                    stream,
                )
                .unwrap();
            let actual = paged
                .forward(
                    ModelInput {
                        inputs: &tokens,
                        mask: None,
                        cache: Some(&mut paged_cache),
                    },
                    stream,
                )
                .unwrap();
            assert!(expected
                .all_close(&actual, 1e-5, 1e-5, None, stream)
                .unwrap()
                .item::<bool>(stream));
        }
        let report = paged_cache.residency_report().unwrap().unwrap();
        assert_eq!(report.logical_cached_tokens, 8);
        assert!(report.discarded_sliding_blocks > 0);
        assert!(matches!(
            &paged_cache.layers[3],
            LayerCache::Attention(AttentionCache::Paged(cache))
                if cache.attention_window() == Some(3)
        ));
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn schema_v4_nemotron_h_save_drop_paged_reload_continue_matches_uninterrupted() {
        use crate::{
            core::cache::{PromptCacheDescriptor, PromptCacheOptions, PromptCacheTopology},
            runtime::{
                cache::residency::PagedCacheOptions,
                media::input::{InputPart, ModelInput as RuntimeModelInput},
            },
        };

        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let args = tiny_full_args();
        let mut model = Model::new(args.clone(), stream).unwrap();
        for (index, parameter) in model.parameters_mut().flatten().values_mut().enumerate() {
            **parameter = Array::full::<f32>(
                parameter.shape(),
                Array::from_f32((index + 1) as f32 * 0.001),
                stream,
            )
            .unwrap();
        }
        let prefix_ids = [1_u32, 2, 3, 4];
        let prefix = Array::from_slice(&prefix_ids, &[1, 4]);
        let parts = [InputPart::text_token_ids(&prefix)];
        let mut cache = model.new_cache();
        CausalLm::prefill_input_logits(
            &mut model,
            RuntimeModelInput::new(&parts),
            &mut cache,
            stream,
        )
        .unwrap();
        assert_eq!(cache.offset(), 4);
        match &cache.layers[0] {
            super::LayerCache::Mamba(cache) => {
                assert_eq!(cache.conv_state.as_ref().unwrap().shape(), &[1, 2, 16]);
                assert_eq!(cache.ssm_state.as_ref().unwrap().shape(), &[1, 2, 4, 4]);
            }
            _ => panic!("expected Nemotron-H Mamba cache"),
        }
        assert_eq!(cache.layers[3].retained_arrays()[0].dim(-2), 4);
        let mut uninterrupted_cache = cache.clone();
        let suffix = Array::from_slice(&[5_u32], &[1, 1]);
        let uninterrupted =
            CausalLm::decode_logits(&mut model, &suffix, &mut uninterrupted_cache, stream).unwrap();
        let layout = super::prompt_cache_layer_layout(&args).unwrap();
        let descriptor = PromptCacheDescriptor {
            model_family: "nemotron_h".into(),
            effective_model_type: args.model_type.clone(),
            checkpoint_fingerprint: "deterministic-fixture".into(),
            prefix_content_fingerprint: "tokens:1,2,3,4".into(),
            architecture_fingerprint: super::prompt_cache_architecture_fingerprint(&args),
            layer_count: layout.len(),
            global_layer_start: 0,
            global_layer_end: layout.len(),
            batch_size: 1,
            layer_prefix_offsets: vec![0; layout.len()],
            layer_layout: layout,
            sink_tokens: 0,
            topology: PromptCacheTopology::default(),
        };
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("prompt-cache");
        Model::save_prompt_cache(
            &mut cache,
            &destination,
            descriptor.clone(),
            &prefix_ids,
            &PromptCacheOptions::default(),
            stream,
        )
        .unwrap();
        drop(cache);
        let (mut restored, _) = Model::load_paged_prompt_cache(
            &args,
            &destination,
            &descriptor,
            &prefix_ids,
            PagedCacheOptions::new(4, 1 << 20, 1 << 20, 1)
                .unwrap()
                .with_full_attention(true),
            stream,
        )
        .unwrap();
        assert_eq!(restored.offset(), 4);
        match &restored.layers[0] {
            super::LayerCache::Mamba(cache) => {
                assert_eq!(cache.conv_state.as_ref().unwrap().shape(), &[1, 2, 16]);
                assert_eq!(cache.ssm_state.as_ref().unwrap().shape(), &[1, 2, 4, 4]);
            }
            _ => panic!("expected restored Nemotron-H Mamba cache"),
        }
        assert!(matches!(
            &restored.layers[3],
            LayerCache::Attention(AttentionCache::Paged(_))
        ));
        assert_eq!(
            restored
                .residency_report()
                .unwrap()
                .unwrap()
                .logical_cached_tokens,
            4
        );
        let continued =
            CausalLm::decode_logits(&mut model, &suffix, &mut restored, stream).unwrap();
        assert!(uninterrupted
            .all_close(&continued, 1e-5, 1e-5, None, stream)
            .unwrap()
            .item::<bool>(stream));
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn strict_load_full_model_from_public_checkpoint_key_shapes() {
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let weights_ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let weights_stream = weights_ctx.stream();
        let args = tiny_full_args();
        let source = Model::new(args.clone(), stream).unwrap();
        let dir = temp_model_dir();
        let weights_path = dir.join("model.safetensors");
        let mut arrays = Vec::new();
        for (key, value) in source.parameters().flatten() {
            if key.as_ref() == "model.layers.2.moe.experts.up_proj" {
                for expert in 0..args.n_routed_experts {
                    arrays.push((
                        format!("backbone.layers.2.mixer.experts.{expert}.up_proj.weight"),
                        value.try_index_device((expert, .., ..), stream).unwrap(),
                    ));
                }
                continue;
            }
            if key.as_ref() == "model.layers.2.moe.experts.down_proj" {
                for expert in 0..args.n_routed_experts {
                    arrays.push((
                        format!("backbone.layers.2.mixer.experts.{expert}.down_proj.weight"),
                        value.try_index_device((expert, .., ..), stream).unwrap(),
                    ));
                }
                continue;
            }

            let public_key = key
                .strip_prefix("model.embeddings.")
                .map(|rest| format!("backbone.embeddings.{rest}"))
                .or_else(|| {
                    key.strip_prefix("model.norm_f.")
                        .map(|rest| format!("backbone.norm_f.{rest}"))
                })
                .or_else(|| {
                    key.strip_prefix("model.layers.").map(|rest| {
                        let (layer, suffix) = rest.split_once('.').unwrap();
                        if suffix.starts_with("norm.") {
                            return format!("backbone.layers.{layer}.{suffix}");
                        }
                        let suffix = suffix
                            .strip_prefix("mamba.")
                            .or_else(|| suffix.strip_prefix("attention."))
                            .or_else(|| suffix.strip_prefix("mlp."))
                            .or_else(|| suffix.strip_prefix("moe."))
                            .unwrap_or(suffix);
                        format!("backbone.layers.{layer}.mixer.{suffix}")
                    })
                })
                .unwrap_or_else(|| key.to_string());
            arrays.push((public_key, value.clone()));
        }
        Array::save_safetensors(
            arrays.iter().map(|(key, value)| (key.as_str(), value)),
            None,
            &weights_path,
        )
        .unwrap();

        let mut target = Model::new(args.clone(), stream).unwrap();
        let config = super::nemotron_h_strict_load_config();
        let mut report = StrictLoadReport::default();
        load_nemotron_h_safetensors_strict(
            &mut target,
            &dir,
            &args,
            weights_stream,
            stream,
            &config,
            &mut report,
        )
        .unwrap();
        report.finish(&target, &config).unwrap();
    }

    #[test]
    #[ignore = "requires a local Nemotron-H checkpoint and MLX runtime execution"]
    fn strict_loads_real_public_checkpoint() {
        let model_dir = PathBuf::from(
            std::env::var("NEMOTRON_H_MODEL_DIR")
                .expect("set NEMOTRON_H_MODEL_DIR to a local Nemotron-H snapshot"),
        );
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let weights_ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let weights_stream = weights_ctx.stream();

        let model = load_nemotron_h_model(&model_dir, stream, weights_stream).unwrap();
        assert_eq!(model.model_type(), "nemotron_h");
        assert_eq!(model.args.num_hidden_layers, 52);
        assert_eq!(model.args.n_routed_experts, 128);
        assert_eq!(model.args.num_experts_per_tok, 6);
        assert_eq!(model.new_cache().layers.len(), 52);
    }

    #[test]
    #[ignore = "requires NEMOTRON_H_MOE_GGUF and Metal"]
    fn strict_loads_and_runs_real_nemotron_h_moe_gguf() {
        let gguf_file = PathBuf::from(
            std::env::var("NEMOTRON_H_MOE_GGUF")
                .expect("set NEMOTRON_H_MOE_GGUF to a local checkpoint"),
        );
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let weights_ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let mut model = load_nemotron_h_gguf(&gguf_file, stream, weights_ctx.stream()).unwrap();

        assert_eq!(model.args.num_hidden_layers, 52);
        assert_eq!(model.args.n_routed_experts, 128);
        assert_eq!(model.args.num_experts_per_tok, 6);
        assert!(model
            .args
            .layer_schedule
            .iter()
            .any(|policy| *policy == LayerPolicy::SparseMoe));

        let tokens = Array::from_slice(&[1_u32, 2], &[1, 2]);
        let parts = [crate::runtime::media::input::InputPart::text_token_ids(
            &tokens,
        )];
        let mut cache = model.new_cache();
        let logits = CausalLm::prefill_input_logits(
            &mut model,
            crate::runtime::media::input::ModelInput::new(&parts),
            &mut cache,
            stream,
        )
        .unwrap();
        assert_eq!(logits.shape(), &[1, 131072]);
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn moe_parameter_tree_uses_nemotron_weight_names_and_policy() {
        let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let args = model_args_from_config_value(&nemotron_nano_config()).unwrap();
        let moe = SparseMoeBlock::new(&args, 0, stream).unwrap();
        let params = moe.parameters().flatten();

        assert_eq!(moe.gate.top_k, 6);
        assert_eq!(moe.gate.num_experts, 128);
        assert_eq!(moe.gate.score_function, TopKRouterScoreFunction::Sigmoid);
        assert_eq!(moe.gate.routed_scaling_factor, 2.5);
        assert!(moe.gate.norm_topk_prob);
        assert_eq!(moe.experts.num_experts, 128);

        for key in [
            "gate.weight",
            "gate.e_score_correction_bias",
            "experts.up_proj",
            "experts.down_proj",
            "shared_experts.up_proj.weight",
            "shared_experts.down_proj.weight",
        ] {
            assert!(params.contains_key(key), "missing parameter key {key}");
        }
        assert_eq!(params["gate.weight"].shape(), &[128, args.hidden_size]);
        assert_eq!(params["gate.e_score_correction_bias"].shape(), &[128]);
        assert_eq!(
            params["experts.up_proj"].shape(),
            &[128, args.moe_intermediate_size, args.hidden_size]
        );
        assert_eq!(
            params["experts.down_proj"].shape(),
            &[128, args.hidden_size, args.moe_intermediate_size]
        );
    }

    #[test]
    fn rejects_mismatched_hybrid_pattern_length() {
        let mut config = nemotron_nano_config();
        config["hybrid_override_pattern"] = json!("ME");

        let error = validate_model_config_value(&config).unwrap_err();
        assert_eq!(
            error.to_string(),
            "unsupported model architecture: Nemotron-H hybrid_override_pattern layer schedule has 2 entries for 52 decoder layers"
        );
    }

    #[test]
    fn normalizes_four_operator_kinds_and_sliding_attention() {
        let mut config = nemotron_nano_config();
        config["num_hidden_layers"] = json!(4);
        config["hybrid_override_pattern"] = json!("M*-E");
        config["sliding_window"] = json!(7);
        let args = model_args_from_config_value(&config).unwrap();
        assert_eq!(
            args.layer_schedule.iter().copied().collect::<Vec<_>>(),
            vec![
                LayerPolicy::Mamba,
                LayerPolicy::SelfAttention(
                    crate::runtime::attention::AttentionPolicy::sliding(7).unwrap()
                ),
                LayerPolicy::DenseMlp,
                LayerPolicy::SparseMoe,
            ]
        );
        let cache = super::Cache::new(&args);
        assert!(matches!(cache.layers[0], super::LayerCache::Mamba(_)));
        assert!(matches!(
            &cache.layers[1],
            super::LayerCache::Attention(cache)
                if cache.policy() == crate::runtime::attention::AttentionPolicy::sliding(7).unwrap()
        ));
        assert!(matches!(cache.layers[2], super::LayerCache::Mlp));
        assert!(matches!(cache.layers[3], super::LayerCache::Moe));
    }

    #[test]
    fn rejects_invalid_sliding_windows_before_execution() {
        for value in [json!(0), json!(-1), json!(i64::from(i32::MAX) + 1)] {
            let mut config = nemotron_nano_config();
            config["sliding_window"] = value;
            let error = model_args_from_config_value(&config).unwrap_err();
            assert!(error
                .to_string()
                .contains("sliding_window must be a positive integer in the executable i32 range"));
        }
    }

    #[test]
    fn architecture_fingerprint_tracks_ordered_layer_schedule() {
        let mut first = nemotron_nano_config();
        first["num_hidden_layers"] = json!(4);
        first["hybrid_override_pattern"] = json!("M*-E");
        let mut second = first.clone();
        second["hybrid_override_pattern"] = json!("*M-E");
        let first = model_args_from_config_value(&first).unwrap();
        let second = model_args_from_config_value(&second).unwrap();
        assert_ne!(
            super::prompt_cache_architecture_fingerprint(&first),
            super::prompt_cache_architecture_fingerprint(&second)
        );
    }

    #[test]
    fn rejects_incompatible_mamba_and_grouped_expert_geometry() {
        let mut config = nemotron_nano_config();
        config["n_groups"] = json!(3);
        let error = validate_model_config_value(&config).unwrap_err();
        assert!(error
            .to_string()
            .contains("mamba_num_heads (64) must be divisible by n_groups (3)"));

        let mut config = nemotron_nano_config();
        config["n_group"] = json!(3);
        let error = validate_model_config_value(&config).unwrap_err();
        assert!(error
            .to_string()
            .contains("invalid Nemotron-H grouped expert routing"));
    }
}
