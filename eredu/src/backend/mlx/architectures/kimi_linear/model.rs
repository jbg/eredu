//! Kimi Linear hybrid KDA/MLA causal language model.

use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
};

use safemlx::{
    error::Exception,
    macros::ModuleParameters,
    module::{Module, ModuleParameters, Param},
    nn,
    ops::{
        exp, indexing::TryIndexOp, mean_axis, rsqrt, sigmoid, GgufCheckpoint, GgufMetadataValue,
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
    backend::mlx::architectures::deepseek_v3::model::{
        DeepSeekQuantizationConfig, LayerPolicy as DeepSeekLayerPolicy, ModelArgs as DeepSeekArgs,
        MultiHeadLatentAttention,
    },
    backend::mlx::architectures::qwen::hybrid::qwen3_5::{QwenLinear, QwenWeightFormat},
    backend::mlx::error::Error,
    backend::mlx::nn::{
        convolution::{causal_depthwise_conv1d, CausalConv1dCache, DepthwiseConv1d},
        gated_delta::gated_delta_scan,
        generation::CausalLm,
        layers::{silu, SwiGluMlp},
        linear::{
            project_logits_maybe_quantized, unloaded_maybe_quantized_embedding,
            unloaded_maybe_quantized_linear,
        },
        moe::{PackedSwiGluExperts, TopKRouter, TopKRouterConfig, TopKRouterScoreFunction},
        tensor::create_causal_mask,
    },
    backend::mlx::runtime::{
        cache::{
            residency::{
                load_prompt_cache_state_tensors, open_prompt_cache, save_prompt_cache_snapshot,
                CacheBlockArrays, CacheResidencyManager, CacheResidencyPolicy,
                CacheResidencyReport, PagedCacheOptions, PromptCacheSnapshotBlock,
                PromptCacheStateArray,
            },
            CompressedLatentCache,
        },
        checkpoint::{
            load::{gguf_quantization_configs, GgufTensorNames},
            quantization::WeightQuantization,
        },
        execution::inspection::{ActivationObserver, MoeRoutingObservation},
        media::input as runtime_input,
    },
    core::attention::{AttentionPolicy, LayerSchedule},
    core::cache::{
        CacheRankIdentity, LayerCachePolicy, StateTensorDimension, StateTensorDtype,
        StateTensorOwner, StateTensorPolicy, StateTensorRole,
    },
};

#[cfg(test)]
use crate::backend::mlx::runtime::cache::residency::open_prompt_cache_snapshot;

fn default_model_type() -> String {
    "kimi_linear".into()
}

fn default_rms_norm_eps() -> f32 {
    1e-5
}

fn default_rope_theta() -> f32 {
    10_000.0
}

fn default_moe_layer_freq() -> i32 {
    1
}

fn default_one() -> i32 {
    1
}

fn default_true() -> bool {
    true
}

fn default_router_activation() -> String {
    "sigmoid".into()
}

/// KDA recurrent geometry shared by every KDA layer.
#[derive(Debug, Clone)]
pub struct KdaConfig {
    /// KDA head count.
    pub num_heads: i32,
    /// KDA key/value dimension per head.
    pub head_dim: i32,
    /// Causal convolution kernel width.
    pub short_conv_kernel_size: i32,
}

/// Stateful attention operator used by one Kimi Linear decoder layer.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum AttentionKind {
    /// Kimi Delta Attention with bounded convolution and recurrent state.
    Kda,
    /// No-RoPE Multi-head Latent Attention with context-growing compressed state.
    Mla,
}

/// Feed-forward operator used by one Kimi Linear decoder layer.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum FeedForwardPolicy {
    /// Dense SwiGLU feed-forward block.
    Dense,
    /// Routed and shared sparse mixture-of-experts block.
    SparseMoe,
}

/// Complete execution policy for one Kimi Linear decoder layer.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub struct LayerPolicy {
    /// Stateful attention operator.
    pub attention: AttentionKind,
    /// Feed-forward operator.
    pub feed_forward: FeedForwardPolicy,
}

/// Raw Hugging Face KDA/MLA layout and recurrent dimensions.
#[derive(Debug, Clone, Deserialize)]
struct LinearAttentionConfigSource {
    /// One-based KDA layer indices.
    pub kda_layers: Vec<i32>,
    /// One-based MLA layer indices.
    pub full_attn_layers: Vec<i32>,
    /// KDA head count.
    pub num_heads: i32,
    /// KDA key/value dimension per head.
    pub head_dim: i32,
    /// Causal convolution kernel width.
    #[serde(default = "default_conv_kernel")]
    pub short_conv_kernel_size: i32,
}

fn default_conv_kernel() -> i32 {
    4
}

/// Raw Hugging Face Kimi Linear text configuration.
#[derive(Debug, Clone, Deserialize)]
struct ModelArgsSource {
    /// Hugging Face model type.
    #[serde(default = "default_model_type")]
    pub model_type: String,
    /// Vocabulary size.
    pub vocab_size: i32,
    /// Transformer hidden width.
    pub hidden_size: i32,
    /// Decoder layer count.
    pub num_hidden_layers: i32,
    /// MLA query head count.
    pub num_attention_heads: i32,
    /// MLA key/value head count from source metadata.
    pub num_key_value_heads: i32,
    /// Dense SwiGLU width.
    pub intermediate_size: i32,
    /// Conventional head width metadata.
    pub head_dim: i32,
    /// Maximum context length.
    #[serde(alias = "max_position_embeddings")]
    pub model_max_length: i32,
    /// RMSNorm epsilon.
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    /// RoPE base retained for compatible MLA construction.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// Hybrid KDA/MLA layout.
    pub linear_attn_config: LinearAttentionConfigSource,
    /// Routed expert count.
    #[serde(alias = "n_routed_experts")]
    pub num_experts: i32,
    /// Per-expert SwiGLU width.
    pub moe_intermediate_size: i32,
    /// MLA compressed latent width.
    pub kv_lora_rank: i32,
    /// Optional query LoRA width.
    #[serde(default)]
    pub q_lora_rank: Option<i32>,
    /// MLA non-positional width per head.
    pub qk_nope_head_dim: i32,
    /// MLA nominal positional width per head.
    pub qk_rope_head_dim: i32,
    /// MLA value width per head.
    pub v_head_dim: i32,
    /// Kimi MLA leaves its nominal positional subspace unrotated.
    #[serde(default)]
    pub mla_use_nope: bool,
    /// Experts selected per token.
    #[serde(alias = "num_experts_per_tok")]
    pub num_experts_per_token: i32,
    /// Shared expert count.
    #[serde(default = "default_one", alias = "n_shared_experts")]
    pub num_shared_experts: i32,
    /// Router score transform.
    #[serde(default = "default_router_activation")]
    pub moe_router_activation_func: String,
    /// Whether selected expert weights are renormalized.
    #[serde(default = "default_true")]
    pub moe_renormalize: bool,
    /// Routed expert output multiplier.
    pub routed_scaling_factor: f32,
    /// Initial dense layer count.
    pub first_k_dense_replace: i32,
    /// Sparse layer frequency after the dense prefix.
    #[serde(default = "default_moe_layer_freq")]
    pub moe_layer_freq: i32,
    /// Whether grouped top-k routing is enabled.
    #[serde(default = "default_true")]
    pub use_grouped_topk: bool,
    /// Router group count.
    #[serde(alias = "n_group")]
    pub num_expert_group: i32,
    /// Selected router group count.
    pub topk_group: i32,
    /// Whether embeddings and output projection are tied.
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Embedded MTP layer count; released checkpoints set this to zero.
    #[serde(default)]
    pub num_nextn_predict_layers: i32,
    /// Optional MLX affine/MXFP4 checkpoint metadata.
    #[serde(default)]
    pub quantization: Option<WeightQuantization>,
    /// Per-weight formats populated by GGUF preparation.
    #[serde(skip)]
    pub quantized_weight_configs: Option<HashMap<String, WeightQuantization>>,
    /// Whether MLA KV-B is stored as split per-head projections.
    #[serde(skip)]
    pub split_kv_b: bool,
}

/// Validated Kimi Linear text configuration.
#[derive(Debug, Clone)]
pub struct ModelArgs {
    /// Effective model type.
    pub model_type: String,
    /// Vocabulary size.
    pub vocab_size: i32,
    /// Transformer hidden width.
    pub hidden_size: i32,
    /// Decoder layer count.
    pub num_hidden_layers: i32,
    /// MLA query head count.
    pub num_attention_heads: i32,
    /// MLA key/value head count from source metadata.
    pub num_key_value_heads: i32,
    /// Dense SwiGLU width.
    pub intermediate_size: i32,
    /// Conventional head width metadata.
    pub head_dim: i32,
    /// Maximum context length.
    pub model_max_length: i32,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// RoPE base retained for compatible MLA construction.
    pub rope_theta: f32,
    /// Authoritative operator policy in decoder-layer order.
    pub layer_schedule: LayerSchedule<LayerPolicy>,
    /// KDA recurrent geometry.
    pub kda_config: KdaConfig,
    /// Routed expert count.
    pub num_experts: i32,
    /// Per-expert SwiGLU width.
    pub moe_intermediate_size: i32,
    /// MLA compressed latent width.
    pub kv_lora_rank: i32,
    /// Optional query LoRA width.
    pub q_lora_rank: Option<i32>,
    /// MLA non-positional width per head.
    pub qk_nope_head_dim: i32,
    /// MLA nominal positional width per head.
    pub qk_rope_head_dim: i32,
    /// MLA value width per head.
    pub v_head_dim: i32,
    /// Kimi MLA leaves its nominal positional subspace unrotated.
    pub mla_use_nope: bool,
    /// Experts selected per token.
    pub num_experts_per_token: i32,
    /// Shared expert count.
    pub num_shared_experts: i32,
    /// Router score transform.
    pub moe_router_activation_func: String,
    /// Whether selected expert weights are renormalized.
    pub moe_renormalize: bool,
    /// Routed expert output multiplier.
    pub routed_scaling_factor: f32,
    /// Whether grouped top-k routing is enabled.
    pub use_grouped_topk: bool,
    /// Router group count.
    pub num_expert_group: i32,
    /// Selected router group count.
    pub topk_group: i32,
    /// Whether embeddings and output projection are tied.
    pub tie_word_embeddings: bool,
    /// Embedded MTP layer count; released checkpoints set this to zero.
    pub num_nextn_predict_layers: i32,
    /// Optional MLX affine/MXFP4 checkpoint metadata.
    pub quantization: Option<WeightQuantization>,
    /// Per-weight formats populated by GGUF preparation.
    pub quantized_weight_configs: Option<HashMap<String, WeightQuantization>>,
    /// Whether MLA KV-B is stored as split per-head projections.
    pub split_kv_b: bool,
}

impl ModelArgsSource {
    fn normalize(self) -> Result<ModelArgs, Error> {
        let layer_schedule = kimi_layer_schedule(&self)?;
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
            layer_schedule,
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
            quantization: self.quantization,
            quantized_weight_configs: self.quantized_weight_configs,
            split_kv_b: self.split_kv_b,
        };
        args.validate()?;
        Ok(args)
    }
}

fn kimi_layer_schedule(source: &ModelArgsSource) -> Result<LayerSchedule<LayerPolicy>, Error> {
    let layers = usize::try_from(source.num_hidden_layers).map_err(|_| {
        Error::UnsupportedArchitecture(format!(
            "Kimi Linear num_hidden_layers must be positive, got {}",
            source.num_hidden_layers
        ))
    })?;
    if layers == 0 {
        return Err(Error::UnsupportedArchitecture(
            "Kimi Linear num_hidden_layers must be positive, got 0".into(),
        ));
    }
    if source.first_k_dense_replace < 0 || source.first_k_dense_replace > source.num_hidden_layers {
        return Err(Error::UnsupportedArchitecture(format!(
            "Kimi Linear first_k_dense_replace must be between zero and num_hidden_layers, got {}",
            source.first_k_dense_replace
        )));
    }
    if source.moe_layer_freq <= 0 {
        return Err(Error::UnsupportedArchitecture(format!(
            "Kimi Linear moe_layer_freq must be positive, got {}",
            source.moe_layer_freq
        )));
    }

    let mut attention = vec![None; layers];
    for (name, indexes, kind) in [
        (
            "KDA",
            &source.linear_attn_config.kda_layers,
            AttentionKind::Kda,
        ),
        (
            "MLA",
            &source.linear_attn_config.full_attn_layers,
            AttentionKind::Mla,
        ),
    ] {
        for &one_based in indexes {
            if one_based <= 0 || one_based > source.num_hidden_layers {
                return Err(Error::UnsupportedArchitecture(format!(
                    "Kimi Linear {name} layer index {one_based} is outside 1..={}",
                    source.num_hidden_layers
                )));
            }
            let slot = &mut attention[one_based as usize - 1];
            if slot.replace(kind).is_some() {
                return Err(Error::UnsupportedArchitecture(format!(
                    "Kimi Linear layer {one_based} occurs more than once"
                )));
            }
        }
    }
    if let Some(layer) = attention.iter().position(Option::is_none) {
        return Err(Error::UnsupportedArchitecture(format!(
            "Kimi Linear KDA and MLA layer lists do not define decoder layer {}",
            layer + 1
        )));
    }
    let policies = attention
        .into_iter()
        .enumerate()
        .map(|(layer, attention)| LayerPolicy {
            attention: attention.expect("validated complete Kimi attention layout"),
            feed_forward: if source.num_experts > 0
                && layer as i32 >= source.first_k_dense_replace
                && layer as i32 % source.moe_layer_freq == 0
            {
                FeedForwardPolicy::SparseMoe
            } else {
                FeedForwardPolicy::Dense
            },
        })
        .collect();
    LayerSchedule::new(layers, policies)
        .map_err(|error| Error::UnsupportedArchitecture(format!("Kimi Linear {error}")))
}

impl ModelArgs {
    /// Returns one validated layer policy without an out-of-range fallback.
    pub fn layer_policy(&self, layer: usize) -> Option<&LayerPolicy> {
        self.layer_schedule.get(layer)
    }

    /// Returns a stable ordered representation of the complete layer schedule.
    pub fn layer_schedule_fingerprint(&self) -> String {
        self.layer_schedule
            .iter()
            .map(|policy| {
                let attention = match policy.attention {
                    AttentionKind::Kda => "k",
                    AttentionKind::Mla => "m",
                };
                let feed_forward = match policy.feed_forward {
                    FeedForwardPolicy::Dense => "d",
                    FeedForwardPolicy::SparseMoe => "e",
                };
                format!("{attention}{feed_forward}")
            })
            .collect::<Vec<_>>()
            .join(",")
    }

    pub(crate) fn weight_quantization_for(&self, name: &str) -> Option<WeightQuantization> {
        self.quantized_weight_configs
            .as_ref()
            .and_then(|configs| configs.get(name).copied())
            .or(self.quantization)
    }

    fn weight_format_for(&self, name: &str) -> QwenWeightFormat {
        match self.weight_quantization_for(name) {
            Some(iq @ WeightQuantization::GgufIQuant { .. }) => QwenWeightFormat::IQuant(iq),
            Some(quantization) => QwenWeightFormat::Affine(quantization),
            None => QwenWeightFormat::Dense,
        }
    }

    fn unloaded_swiglu(
        &self,
        prefix: &str,
        intermediate_size: i32,
        stream: &Stream,
    ) -> Result<SwiGluMlp, Exception> {
        Ok(SwiGluMlp {
            gate_proj: unloaded_maybe_quantized_linear(
                self.hidden_size,
                intermediate_size,
                false,
                self.weight_quantization_for(&format!("{prefix}.gate_proj.weight")),
                stream,
            )?,
            down_proj: unloaded_maybe_quantized_linear(
                intermediate_size,
                self.hidden_size,
                false,
                self.weight_quantization_for(&format!("{prefix}.down_proj.weight")),
                stream,
            )?,
            up_proj: unloaded_maybe_quantized_linear(
                self.hidden_size,
                intermediate_size,
                false,
                self.weight_quantization_for(&format!("{prefix}.up_proj.weight")),
                stream,
            )?,
        })
    }

    pub(crate) fn validate(&self) -> Result<(), Error> {
        if self.model_type != "kimi_linear" {
            return Err(Error::UnsupportedModelType(self.model_type.clone()));
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
                return Err(Error::UnsupportedArchitecture(format!(
                    "Kimi Linear {name} must be positive, got {value}"
                )));
            }
        }
        if !self.mla_use_nope {
            return Err(Error::UnsupportedArchitecture(
                "Kimi Linear currently requires mla_use_nope=true".into(),
            ));
        }
        if self.q_lora_rank.is_some_and(|rank| rank <= 0)
            || self.rms_norm_eps <= 0.0
            || self.rope_theta <= 0.0
            || self.routed_scaling_factor <= 0.0
            || self.num_experts_per_token > self.num_experts
            || self.num_experts % self.num_expert_group != 0
            || self.topk_group > self.num_expert_group
            || self.num_experts_per_token
                > self.topk_group * (self.num_experts / self.num_expert_group)
        {
            return Err(Error::UnsupportedArchitecture(
                "invalid Kimi Linear MLA/MoE dimensions".into(),
            ));
        }
        if self.num_shared_experts != 1 {
            return Err(Error::UnsupportedArchitecture(format!(
                "Kimi Linear currently requires one shared expert, got {}",
                self.num_shared_experts
            )));
        }
        if self.moe_router_activation_func != "sigmoid" {
            return Err(Error::UnsupportedArchitecture(format!(
                "unsupported Kimi Linear router activation {:?}",
                self.moe_router_activation_func
            )));
        }
        if self.num_nextn_predict_layers != 0 {
            return Err(Error::UnsupportedArchitecture(
                "Kimi Linear MTP layers are not implemented".into(),
            ));
        }
        if self.layer_schedule.len() != self.num_hidden_layers as usize {
            return Err(Error::UnsupportedArchitecture(
                "Kimi Linear layer schedule does not match num_hidden_layers".into(),
            ));
        }
        if let Some(quantization) = self.quantization {
            quantization.validate()?;
        }
        Ok(())
    }

    fn deepseek_mla_args(&self) -> DeepSeekArgs {
        DeepSeekArgs {
            model_type: "deepseek_v3".into(),
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            moe_intermediate_size: self.moe_intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            vocab_size: self.vocab_size,
            rms_norm_eps: self.rms_norm_eps,
            max_position_embeddings: self.model_max_length,
            rope_theta: self.rope_theta,
            rope_scaling: None,
            q_lora_rank: self.q_lora_rank,
            kv_lora_rank: self.kv_lora_rank,
            qk_nope_head_dim: self.qk_nope_head_dim,
            qk_rope_head_dim: self.qk_rope_head_dim,
            v_head_dim: self.v_head_dim,
            layer_schedule: LayerSchedule::new(
                self.layer_schedule.len(),
                self.layer_schedule
                    .iter()
                    .map(|policy| match policy.feed_forward {
                        FeedForwardPolicy::Dense => DeepSeekLayerPolicy::DenseMlp,
                        FeedForwardPolicy::SparseMoe => DeepSeekLayerPolicy::SparseMoe,
                    })
                    .collect(),
            )
            .expect("validated Kimi layer schedule"),
            n_routed_experts: self.num_experts,
            n_shared_experts: self.num_shared_experts,
            num_experts_per_tok: self.num_experts_per_token,
            n_group: self.num_expert_group,
            topk_group: self.topk_group,
            topk_method: "noaux_tc".into(),
            scoring_func: "sigmoid".into(),
            norm_topk_prob: self.moe_renormalize,
            routed_scaling_factor: self.routed_scaling_factor,
            num_nextn_predict_layers: 0,
            quantization_config: Option::<DeepSeekQuantizationConfig>::None,
            quantization: self.quantization,
            quantized_weight_configs: self.quantized_weight_configs.clone(),
            split_kv_b: self.split_kv_b,
            tie_word_embeddings: self.tie_word_embeddings,
        }
    }
}

/// Validates a parsed Kimi Linear configuration.
pub(crate) fn validate_model_config_value(value: &Value) -> Result<(), Error> {
    model_args_from_config_value(value).map(|_| ())
}

/// Normalizes a Hugging Face configuration into authoritative Kimi layer geometry.
pub fn model_args_from_config_value(value: &Value) -> Result<ModelArgs, Error> {
    let source: ModelArgsSource = serde_json::from_value(value.clone()).map_err(|error| {
        Error::UnsupportedArchitecture(format!("invalid Kimi Linear config: {error}"))
    })?;
    source.normalize()
}

pub(crate) fn prompt_cache_layer_layout(
    args: &ModelArgs,
) -> Result<LayerSchedule<LayerCachePolicy>, Error> {
    prompt_cache_layer_layout_with_geometry(
        args,
        &args
            .layer_schedule
            .iter()
            .map(|policy| KimiLayerCacheGeometry {
                kda_heads: (policy.attention == AttentionKind::Kda)
                    .then_some(args.kda_config.num_heads),
            })
            .collect::<Vec<_>>(),
    )
}

/// Rank-local state geometry for one Kimi hybrid layer. MLA stores its
/// compressed latent before head expansion, while KDA state follows the
/// planner-owned head partition.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) struct KimiLayerCacheGeometry {
    pub kda_heads: Option<i32>,
}

pub(crate) fn prompt_cache_layer_layout_with_geometry(
    args: &ModelArgs,
    geometry: &[KimiLayerCacheGeometry],
) -> Result<LayerSchedule<LayerCachePolicy>, Error> {
    if geometry.len() != args.layer_schedule.len() {
        return Err(Error::UnsupportedArchitecture(format!(
            "Kimi Linear cache geometry has {} layers, expected {}",
            geometry.len(),
            args.layer_schedule.len()
        )));
    }
    let head_dim = args.kda_config.head_dim;
    let history = args
        .kda_config
        .short_conv_kernel_size
        .checked_sub(1)
        .ok_or_else(|| Error::UnsupportedArchitecture("invalid Kimi KDA kernel width".into()))?;
    let cache_error = |error: crate::core::cache::CachePolicyError| {
        Error::UnsupportedArchitecture(error.to_string())
    };
    let fixed = |value| StateTensorDimension::fixed(value).map_err(cache_error);
    let kda_tensors = |heads: i32| -> Result<Vec<StateTensorPolicy>, Error> {
        let projection = heads.checked_mul(head_dim).ok_or_else(|| {
            Error::UnsupportedArchitecture("Kimi KDA projection width overflow".into())
        })?;
        let mut tensors = Vec::with_capacity(4);
        for slot in 0..3 {
            tensors.push(
                StateTensorPolicy::new(
                    StateTensorRole::Convolution { slot },
                    vec![
                        StateTensorDimension::Batch,
                        fixed(history)?,
                        fixed(projection)?,
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
                    fixed(head_dim)?,
                    fixed(head_dim)?,
                ],
                StateTensorDtype::Float32,
                crate::MutableStateResidency::LayerScopedOffloadable,
            )
            .map_err(cache_error)?,
        );
        Ok(tensors)
    };
    let layers = args.num_hidden_layers as usize;
    let policies = args
        .layer_schedule
        .iter()
        .zip(geometry)
        .enumerate()
        .map(|(layer, (policy, geometry))| match (policy.attention, geometry.kda_heads) {
            (AttentionKind::Kda, Some(heads)) => {
                LayerCachePolicy::fixed_only(kda_tensors(heads)?).map_err(cache_error)
            }
            (AttentionKind::Mla, None) => LayerCachePolicy::compressed_latent_rotary(
                    AttentionPolicy::Full,
                    args.kv_lora_rank,
                    args.qk_rope_head_dim,
                )
                .map_err(cache_error),
            (attention, heads) => Err(Error::UnsupportedArchitecture(format!(
                "Kimi Linear cache geometry mismatch at layer {layer}: {attention:?} with KDA heads {heads:?}"
            ))),
        })
        .collect::<Result<Vec<_>, _>>()?;
    LayerSchedule::new(layers, policies)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

/// Loads and validates `config.json`.
pub fn get_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let file = std::fs::File::open(model_dir.as_ref().join("config.json"))?;
    model_args_from_config_value(&serde_json::from_reader(file)?)
}

/// Returns a deterministic cache identity including the complete ordered Kimi layer schedule.
pub(crate) fn prompt_cache_architecture_fingerprint(args: &ModelArgs) -> String {
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
            ("layer_schedule", args.layer_schedule_fingerprint()),
        ],
    )
}

#[derive(Debug, Clone, Default)]
/// KDA recurrent state for one decoder layer.
pub struct KdaCache {
    /// Q convolution state.
    pub q_conv: CausalConv1dCache,
    /// K convolution state.
    pub k_conv: CausalConv1dCache,
    /// V convolution state.
    pub v_conv: CausalConv1dCache,
    /// F32 recurrent delta state `[batch, heads, key_dim, value_dim]`.
    pub recurrent_state: Option<Array>,
}

#[derive(Debug, Clone)]
/// Per-layer Kimi cache.
pub enum LayerCache {
    /// KDA convolution and recurrent state.
    Kda(KdaCache),
    /// Compressed no-RoPE MLA state.
    Mla(CompressedLatentCache),
}

/// Borrowed hybrid operator state used by generalized execution runtimes.
pub(crate) enum OperatorCache<'a> {
    /// KDA convolution and recurrent state.
    Kda(&'a mut KdaCache),
    /// MLA compressed latent and rotary-key state.
    Mla(&'a mut CompressedLatentCache),
}

impl LayerCache {
    fn offset(&self) -> i32 {
        match self {
            Self::Kda(cache) => cache.q_conv.offset,
            Self::Mla(cache) => cache.offset(),
        }
    }

    pub(super) fn retained_arrays(&self) -> Vec<&Array> {
        match self {
            Self::Kda(cache) => cache
                .q_conv
                .state
                .iter()
                .chain(cache.k_conv.state.iter())
                .chain(cache.v_conv.state.iter())
                .chain(cache.recurrent_state.iter())
                .collect(),
            Self::Mla(cache) => cache
                .arrays()
                .into_iter()
                .flat_map(|(latent, key)| [latent, key])
                .collect(),
        }
    }
}

#[derive(Debug, Clone)]
/// Heterogeneous KDA/MLA generation cache.
pub struct Cache {
    /// One cache per decoder layer.
    pub layers: Vec<LayerCache>,
}

impl Cache {
    /// Creates an empty cache matching the configured hybrid layout.
    pub fn new(args: &ModelArgs) -> Self {
        Self {
            layers: args
                .layer_schedule
                .iter()
                .map(|policy| {
                    if policy.attention == AttentionKind::Kda {
                        LayerCache::Kda(KdaCache::default())
                    } else {
                        LayerCache::Mla(CompressedLatentCache::new())
                    }
                })
                .collect(),
        }
    }

    /// Creates device-resident or blockwise-paged MLA state while KDA keeps
    /// its bounded convolution and recurrent tensors resident.
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
                Self::new_with_manager(args, manager, rank)
            }
        }
    }

    fn new_with_manager(
        args: &ModelArgs,
        manager: CacheResidencyManager,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, Exception> {
        let layers = args
            .layer_schedule
            .iter()
            .enumerate()
            .map(|(layer, policy)| match policy.attention {
                AttentionKind::Kda => Ok(LayerCache::Kda(KdaCache::default())),
                AttentionKind::Mla => {
                    CompressedLatentCache::new_paged(manager.clone(), layer, rank)
                        .map(LayerCache::Mla)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { layers })
    }

    fn residency_manager(&self) -> Option<&CacheResidencyManager> {
        self.layers.iter().find_map(|layer| match layer {
            LayerCache::Kda(_) => None,
            LayerCache::Mla(cache) => cache.residency_manager(),
        })
    }

    /// Returns aggregate live MLA paging observations, if paging is enabled.
    pub fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        self.residency_manager()
            .map(|manager| {
                manager
                    .report()
                    .map_err(|error| Exception::custom(error.to_string()))
            })
            .transpose()
    }

    /// Returns the common number of consumed tokens.
    pub fn offset(&self) -> i32 {
        self.layers.first().map_or(0, LayerCache::offset)
    }

    /// Clears every recurrent and compressed-attention layer state.
    pub fn reset(&mut self) -> Result<(), Exception> {
        for layer in &mut self.layers {
            match layer {
                LayerCache::Kda(cache) => *cache = KdaCache::default(),
                LayerCache::Mla(cache) => cache.clear()?,
            }
        }
        Ok(())
    }

    /// Returns arrays retained by the hybrid cache.
    pub fn retained_arrays(&self) -> Vec<&Array> {
        self.layers
            .iter()
            .flat_map(LayerCache::retained_arrays)
            .collect()
    }

    pub(crate) fn validate(&self, schedule: &LayerSchedule<LayerPolicy>) -> Result<(), Error> {
        if self.layers.len() != schedule.len() {
            return Err(Error::UnsupportedArchitecture(format!(
                "Kimi Linear cache has {} layers, expected {}",
                self.layers.len(),
                schedule.len()
            )));
        }
        for (layer, (cache, policy)) in self.layers.iter().zip(schedule.iter()).enumerate() {
            let matches = matches!(
                (cache, policy.attention),
                (LayerCache::Kda(_), AttentionKind::Kda) | (LayerCache::Mla(_), AttentionKind::Mla)
            );
            if !matches {
                return Err(Error::UnsupportedArchitecture(format!(
                    "Kimi Linear cache policy mismatch at layer {layer}: expected {:?}",
                    policy.attention
                )));
            }
        }
        Ok(())
    }
}

#[allow(non_snake_case)]
#[derive(Debug, Clone, ModuleParameters)]
/// Kimi Delta Attention layer.
pub struct KimiDeltaAttention {
    /// Head count.
    pub num_heads: i32,
    /// Per-head key/value dimension.
    pub head_dim: i32,
    #[param]
    /// Query projection.
    pub q_proj: QwenLinear,
    #[param]
    /// Key projection.
    pub k_proj: QwenLinear,
    #[param]
    /// Value projection.
    pub v_proj: QwenLinear,
    #[param]
    /// Query causal convolution.
    pub q_conv1d: DepthwiseConv1d,
    #[param]
    /// Key causal convolution.
    pub k_conv1d: DepthwiseConv1d,
    #[param]
    /// Value causal convolution.
    pub v_conv1d: DepthwiseConv1d,
    #[param]
    /// Decay down projection.
    pub f_a_proj: QwenLinear,
    #[param]
    /// Decay up projection.
    pub f_b_proj: QwenLinear,
    #[param]
    /// Delta update-strength projection.
    pub b_proj: QwenLinear,
    #[param]
    /// Output-gate down projection.
    pub g_a_proj: QwenLinear,
    #[param]
    /// Output-gate up projection.
    pub g_b_proj: QwenLinear,
    #[param]
    /// Log transition rate.
    pub A_log: Param<Array>,
    #[param]
    /// Per-channel decay bias.
    pub dt_bias: Param<Array>,
    #[param]
    /// Per-head output normalization.
    pub o_norm: nn::RmsNorm,
    #[param]
    /// Output projection.
    pub o_proj: QwenLinear,
}

impl KimiDeltaAttention {
    pub(crate) fn new(args: &ModelArgs, layer: usize, stream: &Stream) -> Result<Self, Exception> {
        let heads = args.kda_config.num_heads;
        let head_dim = args.kda_config.head_dim;
        let projection = heads * head_dim;
        let prefix = format!("model.layers.{layer}.self_attn");
        let linear = |name: &str, input, output| {
            QwenLinear::new(
                input,
                output,
                false,
                args.weight_format_for(&format!("{prefix}.{name}.weight")),
                stream,
            )
        };
        Ok(Self {
            num_heads: heads,
            head_dim,
            q_proj: linear("q_proj", args.hidden_size, projection)?,
            k_proj: linear("k_proj", args.hidden_size, projection)?,
            v_proj: linear("v_proj", args.hidden_size, projection)?,
            q_conv1d: DepthwiseConv1d::new(
                projection,
                args.kda_config.short_conv_kernel_size,
                false,
                stream,
            )?,
            k_conv1d: DepthwiseConv1d::new(
                projection,
                args.kda_config.short_conv_kernel_size,
                false,
                stream,
            )?,
            v_conv1d: DepthwiseConv1d::new(
                projection,
                args.kda_config.short_conv_kernel_size,
                false,
                stream,
            )?,
            f_a_proj: linear("f_a_proj", args.hidden_size, head_dim)?,
            f_b_proj: linear("f_b_proj", head_dim, projection)?,
            b_proj: linear("b_proj", args.hidden_size, heads)?,
            g_a_proj: linear("g_a_proj", args.hidden_size, head_dim)?,
            g_b_proj: linear("g_b_proj", head_dim, projection)?,
            A_log: Param::<Array>::unloaded(&[1, 1, heads, 1], Dtype::Float32, stream)?,
            dt_bias: Param::<Array>::unloaded(&[projection], Dtype::Float32, stream)?,
            o_norm: nn::RmsNorm::unloaded(head_dim, args.rms_norm_eps, Dtype::Float32, stream)?,
            o_proj: linear("o_proj", projection, args.hidden_size)?,
        })
    }

    fn qk_normalize(&self, value: Array, query: bool, stream: &Stream) -> Result<Array, Exception> {
        let variance = mean_axis(&value.square(stream)?, -1, true, stream)?;
        let normalized = value.multiply(
            rsqrt(variance.add(Array::from_f32(1e-6), stream)?, stream)?,
            stream,
        )?;
        let scale = if query {
            1.0 / self.head_dim as f32
        } else {
            (self.head_dim as f32).sqrt().recip()
        };
        normalized.multiply(Array::from_f32(scale), stream)
    }

    pub(super) fn forward_impl(
        &mut self,
        x: &Array,
        mut cache: Option<&mut KdaCache>,
        stream: &Stream,
        prefix: &str,
        observer: &mut Option<&mut dyn ActivationObserver>,
    ) -> Result<Array, Exception> {
        let batch = x.dim(0);
        let sequence = x.dim(1);
        let projection = self.num_heads * self.head_dim;
        let q_projected = self.q_proj.forward(x, stream)?;
        let k_projected = self.k_proj.forward(x, stream)?;
        let v_projected = self.v_proj.forward(x, stream)?;
        if let Some(observer) = observer.as_deref_mut() {
            observer.observe(&format!("{prefix}.q_proj"), &q_projected)?;
            observer.observe(&format!("{prefix}.k_proj"), &k_projected)?;
            observer.observe(&format!("{prefix}.v_proj"), &v_projected)?;
        }
        let q = silu(
            causal_depthwise_conv1d(
                &self.q_conv1d,
                &q_projected,
                cache.as_deref_mut().map(|cache| &mut cache.q_conv),
                stream,
            )?,
            stream,
        )?;
        let k = silu(
            causal_depthwise_conv1d(
                &self.k_conv1d,
                &k_projected,
                cache.as_deref_mut().map(|cache| &mut cache.k_conv),
                stream,
            )?,
            stream,
        )?;
        let v = silu(
            causal_depthwise_conv1d(
                &self.v_conv1d,
                &v_projected,
                cache.as_deref_mut().map(|cache| &mut cache.v_conv),
                stream,
            )?,
            stream,
        )?;
        if let Some(observer) = observer.as_deref_mut() {
            observer.observe(&format!("{prefix}.q_conv1d"), &q)?;
            observer.observe(&format!("{prefix}.k_conv1d"), &k)?;
            observer.observe(&format!("{prefix}.v_conv1d"), &v)?;
        }
        let q = self.qk_normalize(
            q.reshape(&[batch, sequence, self.num_heads, self.head_dim], stream)?,
            true,
            stream,
        )?;
        let k = self.qk_normalize(
            k.reshape(&[batch, sequence, self.num_heads, self.head_dim], stream)?,
            false,
            stream,
        )?;
        let v = v.reshape(&[batch, sequence, self.num_heads, self.head_dim], stream)?;
        let decay_logits = self
            .f_b_proj
            .forward(&self.f_a_proj.forward(x, stream)?, stream)?
            .reshape(&[batch, sequence, self.num_heads, self.head_dim], stream)?;
        let dt_bias = self
            .dt_bias
            .reshape(&[1, 1, self.num_heads, self.head_dim], stream)?;
        let rate = exp(self.A_log.as_ref(), stream)?.multiply(Array::from_f32(-1.0), stream)?;
        let log_decay =
            nn::softplus(decay_logits.add(dt_bias, stream)?, stream)?.multiply(rate, stream)?;
        let beta = sigmoid(
            self.b_proj
                .forward(x, stream)?
                .reshape(&[batch, sequence, self.num_heads], stream)?,
            stream,
        )?;
        if let Some(observer) = observer.as_deref_mut() {
            observer.observe(&format!("{prefix}.log_decay"), &log_decay)?;
            observer.observe(&format!("{prefix}.beta"), &beta)?;
        }
        let initial_state = cache
            .as_ref()
            .and_then(|cache| cache.recurrent_state.clone());
        let (state, recurrent) =
            gated_delta_scan(&q, &k, &v, &log_decay, &beta, initial_state, stream)?;
        if let Some(cache) = cache {
            cache.recurrent_state = Some(state);
        }
        if let Some(observer) = observer.as_deref_mut() {
            observer.observe(&format!("{prefix}.recurrent_core"), &recurrent)?;
        }
        let gate = sigmoid(
            self.g_b_proj
                .forward(&self.g_a_proj.forward(x, stream)?, stream)?
                .reshape(&[batch, sequence, self.num_heads, self.head_dim], stream)?,
            stream,
        )?;
        let normalized = self
            .o_norm
            .forward(&recurrent, stream)?
            .multiply(gate, stream)?;
        let output = self.o_proj.forward(
            &normalized.reshape(&[batch, sequence, projection], stream)?,
            stream,
        )?;
        if let Some(observer) = observer.as_deref_mut() {
            observer.observe(&format!("{prefix}.gated_norm"), &normalized)?;
            observer.observe(&format!("{prefix}.o_proj"), &output)?;
        }
        Ok(output)
    }
}

#[derive(Debug, Clone)]
pub(crate) enum Attention {
    Kda(Box<KimiDeltaAttention>),
    Mla(Box<MultiHeadLatentAttention>),
}

impl ModuleParameters for Attention {
    fn num_parameters(&self) -> usize {
        match self {
            Self::Kda(value) => value.num_parameters(),
            Self::Mla(value) => value.num_parameters(),
        }
    }
    fn parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        match self {
            Self::Kda(value) => value.parameters(),
            Self::Mla(value) => value.parameters(),
        }
    }
    fn parameters_mut(&mut self) -> safemlx::module::ModuleParamMut<'_> {
        match self {
            Self::Kda(value) => value.parameters_mut(),
            Self::Mla(value) => value.parameters_mut(),
        }
    }
    fn trainable_parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        match self {
            Self::Kda(value) => value.trainable_parameters(),
            Self::Mla(value) => value.trainable_parameters(),
        }
    }
    fn freeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Kda(value) => value.freeze_parameters(recursive),
            Self::Mla(value) => value.freeze_parameters(recursive),
        }
    }
    fn unfreeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Kda(value) => value.unfreeze_parameters(recursive),
            Self::Mla(value) => value.unfreeze_parameters(recursive),
        }
    }
    fn all_frozen(&self) -> Option<bool> {
        match self {
            Self::Kda(value) => value.all_frozen(),
            Self::Mla(value) => value.all_frozen(),
        }
    }
    fn any_frozen(&self) -> Option<bool> {
        match self {
            Self::Kda(value) => value.any_frozen(),
            Self::Mla(value) => value.any_frozen(),
        }
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct SparseMoe {
    #[param]
    pub(crate) gate: TopKRouter,
    #[param]
    pub(crate) experts: PackedSwiGluExperts,
    #[param]
    pub(crate) shared_experts: SwiGluMlp,
}

impl SparseMoe {
    fn new_with_widths(
        args: &ModelArgs,
        layer: usize,
        routed_intermediate: i32,
        shared_intermediate: i32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let prefix = format!("model.layers.{layer}.mlp");
        let router_quantization = args.weight_quantization_for(&format!("{prefix}.gate.weight"));
        let gate_up_quantization =
            args.weight_quantization_for(&format!("{prefix}.experts.gate_up_proj"));
        let down_quantization =
            args.weight_quantization_for(&format!("{prefix}.experts.down_proj"));
        Ok(Self {
            gate: TopKRouter::new_with_quantization(
                TopKRouterConfig {
                    top_k: args.num_experts_per_token,
                    num_experts: args.num_experts,
                    hidden_size: args.hidden_size,
                    score_function: TopKRouterScoreFunction::Sigmoid,
                    norm_topk_prob: args.moe_renormalize,
                    normalization_epsilon: 1e-20,
                    routed_scaling_factor: args.routed_scaling_factor,
                    n_group: args.num_expert_group,
                    topk_group: args.topk_group,
                    score_correction_bias: true,
                },
                router_quantization,
                stream,
            )?,
            experts: PackedSwiGluExperts::new(
                args.num_experts,
                args.hidden_size,
                routed_intermediate,
                gate_up_quantization,
                down_quantization,
                stream,
            )?,
            shared_experts: args.unloaded_swiglu(
                &format!("{prefix}.shared_experts"),
                shared_intermediate,
                stream,
            )?,
        })
    }

    fn forward_impl(
        &mut self,
        input: &Array,
        stream: &Stream,
        prefix: &str,
        observer: &mut Option<&mut dyn ActivationObserver>,
    ) -> Result<Array, Exception> {
        let shape = input.shape().to_vec();
        let flat = input.reshape(&[-1, input.dim(-1)], stream)?;
        if let Some(observer) = observer.as_deref_mut() {
            let routing = self.gate.forward_with_observer(
                &flat,
                stream,
                &format!("{prefix}.gate"),
                observer,
            )?;
            let routed = self
                .experts
                .forward(&flat, &routing.indices, &routing.weights, stream)?;
            let shared = self.shared_experts.forward(&flat, stream)?;
            let combined = routed.add(&shared, stream)?;
            observer.observe(&format!("{prefix}.routed_experts"), &routed)?;
            observer.observe(&format!("{prefix}.shared_experts"), &shared)?;
            observer.observe_moe_routing(MoeRoutingObservation {
                prefix,
                selected_experts: &routing.indices,
                selected_scores: &routing.scores,
                routing_weights: &routing.weights,
                routed_output: &routed,
                local_routed_output: None,
                reduced_routed_output: Some(&routed),
                shared_output: Some(&shared),
                combined_output: Some(&combined),
                num_experts: self.gate.num_experts,
            })?;
            combined.reshape(&shape, stream)
        } else {
            let (indices, weights) = self.gate.forward(&flat, stream)?;
            let routed = self.experts.forward(&flat, &indices, &weights, stream)?;
            routed
                .add(self.shared_experts.forward(&flat, stream)?, stream)?
                .reshape(&shape, stream)
        }
    }

    fn forward_sparse_experts<F>(
        &mut self,
        input: &Array,
        stream: &Stream,
        mut execute: F,
    ) -> Result<Array, Exception>
    where
        F: FnMut(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let shape = input.shape().to_vec();
        let flat = input.reshape(&[-1, input.dim(-1)], stream)?;
        let (indices, weights) = self.gate.forward(&flat, stream)?;
        let routed = execute(&flat, &indices, &weights, stream)?;
        routed
            .add(self.shared_experts.forward(&flat, stream)?, stream)?
            .reshape(&shape, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_expert_parallel(
        &mut self,
        input: &Array,
        assignment: &crate::backend::mlx::architectures::distributed::expert::ExpertAssignment,
        group: &safemlx::distributed::Group,
        statistics: &mut crate::backend::mlx::architectures::distributed::expert::RoutingStatistics,
        stream: &Stream,
        observer: &mut Option<&mut dyn ActivationObserver>,
        prefix: &str,
    ) -> Result<Array, Exception> {
        let shape = input.shape().to_vec();
        let flat = input.reshape(&[-1, input.dim(-1)], stream)?;
        crate::backend::mlx::architectures::distributed::expert::materialize_timing_phase([&flat])?;
        let router_started = std::time::Instant::now();
        let (indices, selected_scores, weights) = if let Some(observer) = observer.as_deref_mut() {
            let routing = self.gate.forward_with_observer(
                &flat,
                stream,
                &format!("{prefix}.gate"),
                observer,
            )?;
            (routing.indices, Some(routing.scores), routing.weights)
        } else {
            let (indices, weights) = self.gate.forward(&flat, stream)?;
            (indices, None, weights)
        };
        crate::backend::mlx::architectures::distributed::expert::materialize_timing_phase([
            &indices, &weights,
        ])?;
        statistics.router_time += router_started.elapsed();
        let returned =
            crate::backend::mlx::architectures::distributed::expert::dispatch_replicated(
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
        let shared_started = std::time::Instant::now();
        let shared = self.shared_experts.forward(&flat, stream)?;
        crate::backend::mlx::architectures::distributed::expert::materialize_timing_phase([
            &shared,
        ])?;
        statistics.shared_expert_time += shared_started.elapsed();
        let combined = returned.reduced_output.add(&shared, stream)?;
        if let (Some(observer), Some(scores)) = (observer.as_deref_mut(), selected_scores.as_ref())
        {
            observer.observe_moe_routing(MoeRoutingObservation {
                prefix,
                selected_experts: &indices,
                selected_scores: scores,
                routing_weights: &weights,
                routed_output: &returned.reduced_output,
                local_routed_output: Some(&returned.local_output),
                reduced_routed_output: Some(&returned.reduced_output),
                shared_output: Some(&shared),
                combined_output: Some(&combined),
                num_experts: self.gate.num_experts,
            })?;
        }
        combined.reshape(&shape, stream)
    }
}

#[derive(Debug, Clone)]
pub(crate) enum FeedForward {
    Dense(Box<SwiGluMlp>),
    Moe(Box<SparseMoe>),
}

impl ModuleParameters for FeedForward {
    fn num_parameters(&self) -> usize {
        match self {
            Self::Dense(value) => value.num_parameters(),
            Self::Moe(value) => value.num_parameters(),
        }
    }
    fn parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        match self {
            Self::Dense(value) => value.parameters(),
            Self::Moe(value) => value.parameters(),
        }
    }
    fn parameters_mut(&mut self) -> safemlx::module::ModuleParamMut<'_> {
        match self {
            Self::Dense(value) => value.parameters_mut(),
            Self::Moe(value) => value.parameters_mut(),
        }
    }
    fn trainable_parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        match self {
            Self::Dense(value) => value.trainable_parameters(),
            Self::Moe(value) => value.trainable_parameters(),
        }
    }
    fn freeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Dense(value) => value.freeze_parameters(recursive),
            Self::Moe(value) => value.freeze_parameters(recursive),
        }
    }
    fn unfreeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Dense(value) => value.unfreeze_parameters(recursive),
            Self::Moe(value) => value.unfreeze_parameters(recursive),
        }
    }
    fn all_frozen(&self) -> Option<bool> {
        match self {
            Self::Dense(value) => value.all_frozen(),
            Self::Moe(value) => value.all_frozen(),
        }
    }
    fn any_frozen(&self) -> Option<bool> {
        match self {
            Self::Dense(value) => value.any_frozen(),
            Self::Moe(value) => value.any_frozen(),
        }
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// One Kimi Linear decoder layer.
pub struct DecoderLayer {
    #[param]
    pub(crate) self_attn: Attention,
    #[param]
    pub(crate) mlp: FeedForward,
    #[param]
    pub(crate) input_layernorm: nn::RmsNorm,
    #[param]
    pub(crate) post_attention_layernorm: nn::RmsNorm,
}

impl DecoderLayer {
    pub(crate) fn new(args: &ModelArgs, layer: usize, stream: &Stream) -> Result<Self, Exception> {
        Self::new_with_widths(
            args,
            layer,
            args.intermediate_size,
            args.moe_intermediate_size,
            args.moe_intermediate_size * args.num_shared_experts,
            stream,
        )
    }

    pub(super) fn new_with_widths(
        args: &ModelArgs,
        layer: usize,
        dense_intermediate: i32,
        routed_intermediate: i32,
        shared_intermediate: i32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let policy = args.layer_policy(layer).copied().ok_or_else(|| {
            Exception::custom(format!(
                "Kimi Linear layer schedule has no policy for layer {layer}"
            ))
        })?;
        let attention = match policy.attention {
            AttentionKind::Kda => {
                Attention::Kda(Box::new(KimiDeltaAttention::new(args, layer, stream)?))
            }
            AttentionKind::Mla => {
                Attention::Mla(Box::new(MultiHeadLatentAttention::new_with_nope(
                    &args.deepseek_mla_args(),
                    layer as i32,
                    true,
                    stream,
                )?))
            }
        };
        let mlp = match policy.feed_forward {
            FeedForwardPolicy::SparseMoe => FeedForward::Moe(Box::new(SparseMoe::new_with_widths(
                args,
                layer,
                routed_intermediate,
                shared_intermediate,
                stream,
            )?)),
            FeedForwardPolicy::Dense => {
                let prefix = format!("model.layers.{layer}.mlp");
                FeedForward::Dense(Box::new(args.unloaded_swiglu(
                    &prefix,
                    dense_intermediate,
                    stream,
                )?))
            }
        };
        Ok(Self {
            self_attn: attention,
            mlp,
            input_layernorm: nn::RmsNorm::unloaded(
                args.hidden_size,
                args.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?,
            post_attention_layernorm: nn::RmsNorm::unloaded(
                args.hidden_size,
                args.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?,
        })
    }

    /// Executes a Kimi hybrid layer with rank-local KDA/MLA heads and
    /// feed-forward intermediates.
    pub(crate) fn forward_tensor_parallel(
        &mut self,
        input: &Array,
        mask: Option<&Array>,
        cache: Option<&mut LayerCache>,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let normalized = self.input_layernorm.forward(input, stream)?;
        let mut observer = None;
        let attention = match (&mut self.self_attn, cache) {
            (Attention::Kda(attention), Some(LayerCache::Kda(cache))) => {
                let partial =
                    attention.forward_impl(&normalized, Some(cache), stream, "", &mut observer)?;
                safemlx::distributed::all_sum(&partial, group, stream)?
            }
            (Attention::Kda(attention), None) => {
                let partial =
                    attention.forward_impl(&normalized, None, stream, "", &mut observer)?;
                safemlx::distributed::all_sum(&partial, group, stream)?
            }
            (Attention::Mla(attention), Some(LayerCache::Mla(cache))) => {
                let partial = attention.forward_shared(
                    &normalized,
                    mask,
                    Some(cache),
                    stream,
                    "",
                    &mut observer,
                )?;
                safemlx::distributed::all_sum(&partial, group, stream)?
            }
            (Attention::Mla(attention), None) => {
                let partial =
                    attention.forward_shared(&normalized, mask, None, stream, "", &mut observer)?;
                safemlx::distributed::all_sum(&partial, group, stream)?
            }
            (Attention::Kda(_), Some(LayerCache::Mla(_)))
            | (Attention::Mla(_), Some(LayerCache::Kda(_))) => {
                return Err(Exception::custom(
                    "Kimi tensor-parallel cache kind mismatch",
                ));
            }
        };
        let hidden = input.add(attention, stream)?;
        let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
        let feed_forward = match &mut self.mlp {
            FeedForward::Dense(mlp) => mlp.forward(&normalized, stream)?,
            FeedForward::Moe(moe) => moe.forward_impl(&normalized, stream, "", &mut observer)?,
        };
        let feed_forward = safemlx::distributed::all_sum(&feed_forward, group, stream)?;
        hidden.add(feed_forward, stream)
    }

    pub(super) fn forward_impl(
        &mut self,
        input: &Array,
        mask: Option<&Array>,
        cache: Option<&mut LayerCache>,
        stream: &Stream,
        prefix: &str,
        observer: &mut Option<&mut dyn ActivationObserver>,
    ) -> Result<Array, Exception> {
        if let Some(observer) = observer.as_deref_mut() {
            observer.observe(&format!("{prefix}.input"), input)?;
        }
        let normalized = self.input_layernorm.forward(input, stream)?;
        let attention_prefix = format!("{prefix}.self_attn");
        let attention = match (&mut self.self_attn, cache) {
            (Attention::Kda(attention), Some(LayerCache::Kda(cache))) => attention.forward_impl(
                &normalized,
                Some(cache),
                stream,
                &attention_prefix,
                observer,
            )?,
            (Attention::Kda(attention), None) => {
                attention.forward_impl(&normalized, None, stream, &attention_prefix, observer)?
            }
            (Attention::Mla(attention), Some(LayerCache::Mla(cache))) => attention.forward_shared(
                &normalized,
                mask,
                Some(cache),
                stream,
                &attention_prefix,
                observer,
            )?,
            (Attention::Mla(attention), None) => attention.forward_shared(
                &normalized,
                mask,
                None,
                stream,
                &attention_prefix,
                observer,
            )?,
            _ => {
                return Err(Exception::custom(
                    "Kimi Linear cache layer kind does not match decoder layer",
                ))
            }
        };
        let hidden = input.add(attention, stream)?;
        let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
        let feed_forward = match &mut self.mlp {
            FeedForward::Dense(mlp) => {
                if let Some(observer) = observer.as_deref_mut() {
                    mlp.forward_with_observer(
                        &normalized,
                        stream,
                        &format!("{prefix}.mlp"),
                        observer,
                    )?
                } else {
                    mlp.forward(&normalized, stream)?
                }
            }
            FeedForward::Moe(moe) => {
                moe.forward_impl(&normalized, stream, &format!("{prefix}.mlp"), observer)?
            }
        };
        let output = hidden.add(feed_forward, stream)?;
        if let Some(observer) = observer.as_deref_mut() {
            observer.observe(&format!("{prefix}.output"), &output)?;
        }
        Ok(output)
    }

    /// Executes this layer from semantic operator state without requiring the
    /// resident model's family-specific cache container.
    pub(crate) fn forward_with_operator_cache(
        &mut self,
        input: &Array,
        mask: Option<&Array>,
        cache: Option<OperatorCache<'_>>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let normalized = self.input_layernorm.forward(input, stream)?;
        let mut observer = None;
        let attention = match (&mut self.self_attn, cache) {
            (Attention::Kda(attention), Some(OperatorCache::Kda(cache))) => {
                attention.forward_impl(&normalized, Some(cache), stream, "", &mut observer)?
            }
            (Attention::Kda(attention), None) => {
                attention.forward_impl(&normalized, None, stream, "", &mut observer)?
            }
            (Attention::Mla(attention), Some(OperatorCache::Mla(cache))) => attention
                .forward_shared(&normalized, mask, Some(cache), stream, "", &mut observer)?,
            (Attention::Mla(attention), None) => {
                attention.forward_shared(&normalized, mask, None, stream, "", &mut observer)?
            }
            _ => {
                return Err(Exception::custom(
                    "Kimi Linear operator cache does not match decoder layer",
                ))
            }
        };
        let hidden = input.add(attention, stream)?;
        let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
        let feed_forward = match &mut self.mlp {
            FeedForward::Dense(mlp) => mlp.forward(&normalized, stream)?,
            FeedForward::Moe(moe) => moe.forward_impl(&normalized, stream, "", &mut observer)?,
        };
        hidden.add(feed_forward, stream)
    }

    /// Executes a TP-sharded layer from semantic operator state.
    pub(crate) fn forward_tensor_parallel_with_operator_cache(
        &mut self,
        input: &Array,
        mask: Option<&Array>,
        cache: Option<OperatorCache<'_>>,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let normalized = self.input_layernorm.forward(input, stream)?;
        let mut observer = None;
        let attention = match (&mut self.self_attn, cache) {
            (Attention::Kda(attention), Some(OperatorCache::Kda(cache))) => {
                attention.forward_impl(&normalized, Some(cache), stream, "", &mut observer)?
            }
            (Attention::Mla(attention), Some(OperatorCache::Mla(cache))) => attention
                .forward_shared(&normalized, mask, Some(cache), stream, "", &mut observer)?,
            _ => {
                return Err(Exception::custom(
                    "Kimi Linear tensor-parallel operator cache does not match decoder layer",
                ))
            }
        };
        let attention = safemlx::distributed::all_sum(&attention, group, stream)?;
        let hidden = input.add(attention, stream)?;
        let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
        let partial = match &mut self.mlp {
            FeedForward::Dense(mlp) => mlp.forward(&normalized, stream)?,
            FeedForward::Moe(moe) => moe.forward_impl(&normalized, stream, "", &mut observer)?,
        };
        let feed_forward = safemlx::distributed::all_sum(&partial, group, stream)?;
        hidden.add(feed_forward, stream)
    }

    /// Executes TP-sharded KDA/MLA and dense/shared projections while an
    /// EP-scoped executor evaluates independently stored routed experts.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_tensor_with_expert_executor_and_operator_cache<F>(
        &mut self,
        input: &Array,
        mask: Option<&Array>,
        cache: Option<OperatorCache<'_>>,
        group: &safemlx::distributed::Group,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let normalized = self.input_layernorm.forward(input, stream)?;
        let mut observer = None;
        let attention = match (&mut self.self_attn, cache) {
            (Attention::Kda(attention), Some(OperatorCache::Kda(cache))) => {
                attention.forward_impl(&normalized, Some(cache), stream, "", &mut observer)?
            }
            (Attention::Mla(attention), Some(OperatorCache::Mla(cache))) => attention
                .forward_shared(&normalized, mask, Some(cache), stream, "", &mut observer)?,
            _ => {
                return Err(Exception::custom(
                    "Kimi Linear tensor/expert operator cache does not match decoder layer",
                ))
            }
        };
        let attention = safemlx::distributed::all_sum(&attention, group, stream)?;
        let hidden = input.add(attention, stream)?;
        let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
        let feed_forward = match &mut self.mlp {
            FeedForward::Dense(mlp) => {
                let partial = mlp.forward(&normalized, stream)?;
                safemlx::distributed::all_sum(&partial, group, stream)?
            }
            FeedForward::Moe(moe) => {
                let shape = normalized.shape().to_vec();
                let flat = normalized.reshape(&[-1, normalized.dim(-1)], stream)?;
                let (indices, weights) = moe.gate.forward(&flat, stream)?;
                let routed = execute(&flat, &indices, &weights, stream)?;
                let shared_partial = moe.shared_experts.forward(&flat, stream)?;
                let shared = safemlx::distributed::all_sum(&shared_partial, group, stream)?;
                routed.add(shared, stream)?.reshape(&shape, stream)?
            }
        };
        hidden.add(feed_forward, stream)
    }

    /// Executes replicated KDA/MLA and shared projections while a caller owns
    /// routed expert execution.
    pub(crate) fn forward_with_expert_executor_and_operator_cache<F>(
        &mut self,
        input: &Array,
        mask: Option<&Array>,
        cache: Option<OperatorCache<'_>>,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let normalized = self.input_layernorm.forward(input, stream)?;
        let mut observer = None;
        let attention = match (&mut self.self_attn, cache) {
            (Attention::Kda(attention), Some(OperatorCache::Kda(cache))) => {
                attention.forward_impl(&normalized, Some(cache), stream, "", &mut observer)?
            }
            (Attention::Mla(attention), Some(OperatorCache::Mla(cache))) => attention
                .forward_shared(&normalized, mask, Some(cache), stream, "", &mut observer)?,
            _ => {
                return Err(Exception::custom(
                    "Kimi Linear external-expert operator cache does not match decoder layer",
                ))
            }
        };
        let hidden = input.add(attention, stream)?;
        let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
        let feed_forward = match &mut self.mlp {
            FeedForward::Dense(mlp) => mlp.forward(&normalized, stream)?,
            FeedForward::Moe(moe) => {
                let shape = normalized.shape().to_vec();
                let flat = normalized.reshape(&[-1, normalized.dim(-1)], stream)?;
                let (indices, weights) = moe.gate.forward(&flat, stream)?;
                let routed = execute(&flat, &indices, &weights, stream)?;
                routed
                    .add(moe.shared_experts.forward(&flat, stream)?, stream)?
                    .reshape(&shape, stream)?
            }
        };
        hidden.add(feed_forward, stream)
    }

    /// Executes TP-sharded KDA/MLA, shared experts, and routed expert
    /// intermediates while EP owns the resident routed expert banks.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_tensor_expert_parallel_with_operator_cache(
        &mut self,
        input: &Array,
        mask: Option<&Array>,
        cache: Option<OperatorCache<'_>>,
        tensor_group: &safemlx::distributed::Group,
        assignment: &crate::backend::mlx::architectures::distributed::expert::ExpertAssignment,
        expert_group: &safemlx::distributed::Group,
        statistics: &mut crate::backend::mlx::architectures::distributed::expert::RoutingStatistics,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let normalized = self.input_layernorm.forward(input, stream)?;
        let mut observer = None;
        let attention = match (&mut self.self_attn, cache) {
            (Attention::Kda(attention), Some(OperatorCache::Kda(cache))) => {
                attention.forward_impl(&normalized, Some(cache), stream, "", &mut observer)?
            }
            (Attention::Mla(attention), Some(OperatorCache::Mla(cache))) => attention
                .forward_shared(&normalized, mask, Some(cache), stream, "", &mut observer)?,
            _ => {
                return Err(Exception::custom(
                    "Kimi Linear tensor/expert operator cache does not match decoder layer",
                ))
            }
        };
        let attention = safemlx::distributed::all_sum(&attention, tensor_group, stream)?;
        let hidden = input.add(attention, stream)?;
        let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
        let partial = match &mut self.mlp {
            FeedForward::Dense(mlp) => mlp.forward(&normalized, stream)?,
            FeedForward::Moe(moe) => moe.forward_expert_parallel(
                &normalized,
                assignment,
                expert_group,
                statistics,
                stream,
                &mut observer,
                "mlp",
            )?,
        };
        let feed_forward = safemlx::distributed::all_sum(&partial, tensor_group, stream)?;
        hidden.add(feed_forward, stream)
    }

    pub(super) fn forward_sparse_experts<F>(
        &mut self,
        input: &Array,
        mask: Option<&Array>,
        cache: Option<&mut LayerCache>,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        F: FnMut(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let normalized = self.input_layernorm.forward(input, stream)?;
        let mut observer = None;
        let attention =
            match (&mut self.self_attn, cache) {
                (Attention::Kda(attention), Some(LayerCache::Kda(cache))) => attention
                    .forward_impl(&normalized, Some(cache), stream, "self_attn", &mut observer)?,
                (Attention::Mla(attention), Some(LayerCache::Mla(cache))) => attention
                    .forward_shared(
                        &normalized,
                        mask,
                        Some(cache),
                        stream,
                        "self_attn",
                        &mut observer,
                    )?,
                _ => {
                    return Err(Exception::custom(
                        "Kimi Linear sparse layer requires a matching cache",
                    ))
                }
            };
        let hidden = input.add(attention, stream)?;
        let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
        let feed_forward = match &mut self.mlp {
            FeedForward::Dense(mlp) => mlp.forward(&normalized, stream)?,
            FeedForward::Moe(moe) => moe.forward_sparse_experts(&normalized, stream, execute)?,
        };
        hidden.add(feed_forward, stream)
    }

    /// Executes TP-sharded KDA/MLA and dense/shared projections while an
    /// EP-scoped executor evaluates routed experts.
    pub(crate) fn forward_tensor_with_expert_executor<F>(
        &mut self,
        input: &Array,
        mask: Option<&Array>,
        cache: Option<&mut LayerCache>,
        group: &safemlx::distributed::Group,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let normalized = self.input_layernorm.forward(input, stream)?;
        let mut observer = None;
        let attention =
            match (&mut self.self_attn, cache) {
                (Attention::Kda(attention), Some(LayerCache::Kda(cache))) => attention
                    .forward_impl(&normalized, Some(cache), stream, "self_attn", &mut observer)?,
                (Attention::Mla(attention), Some(LayerCache::Mla(cache))) => attention
                    .forward_shared(
                        &normalized,
                        mask,
                        Some(cache),
                        stream,
                        "self_attn",
                        &mut observer,
                    )?,
                _ => {
                    return Err(Exception::custom(
                        "Kimi Linear tensor/expert cache layer kind mismatch",
                    ))
                }
            };
        let attention = safemlx::distributed::all_sum(&attention, group, stream)?;
        let hidden = input.add(attention, stream)?;
        let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
        let feed_forward = match &mut self.mlp {
            FeedForward::Dense(mlp) => {
                let partial = mlp.forward(&normalized, stream)?;
                safemlx::distributed::all_sum(&partial, group, stream)?
            }
            FeedForward::Moe(moe) => {
                let shape = normalized.shape().to_vec();
                let flat = normalized.reshape(&[-1, normalized.dim(-1)], stream)?;
                let (indices, weights) = moe.gate.forward(&flat, stream)?;
                let routed = execute(&flat, &indices, &weights, stream)?;
                let shared_partial = moe.shared_experts.forward(&flat, stream)?;
                let shared = safemlx::distributed::all_sum(&shared_partial, group, stream)?;
                routed.add(shared, stream)?.reshape(&shape, stream)?
            }
        };
        hidden.add(feed_forward, stream)
    }

    /// Executes EP-local routed experts from semantic operator state.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_expert_parallel_with_operator_cache(
        &mut self,
        input: &Array,
        mask: Option<&Array>,
        cache: Option<OperatorCache<'_>>,
        assignment: &crate::backend::mlx::architectures::distributed::expert::ExpertAssignment,
        group: &safemlx::distributed::Group,
        statistics: &mut crate::backend::mlx::architectures::distributed::expert::RoutingStatistics,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let normalized = self.input_layernorm.forward(input, stream)?;
        let mut observer = None;
        let attention = match (&mut self.self_attn, cache) {
            (Attention::Kda(attention), Some(OperatorCache::Kda(cache))) => {
                attention.forward_impl(&normalized, Some(cache), stream, "", &mut observer)?
            }
            (Attention::Mla(attention), Some(OperatorCache::Mla(cache))) => attention
                .forward_shared(&normalized, mask, Some(cache), stream, "", &mut observer)?,
            _ => {
                return Err(Exception::custom(
                    "Kimi Linear expert-parallel operator cache does not match decoder layer",
                ))
            }
        };
        let hidden = input.add(attention, stream)?;
        let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
        let feed_forward = match &mut self.mlp {
            FeedForward::Dense(mlp) => mlp.forward(&normalized, stream)?,
            FeedForward::Moe(moe) => moe.forward_expert_parallel(
                &normalized,
                assignment,
                group,
                statistics,
                stream,
                &mut observer,
                "mlp",
            )?,
        };
        hidden.add(feed_forward, stream)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// Kimi Linear transformer body.
pub struct TextModel {
    #[param]
    /// Token embedding table.
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[param]
    /// Hybrid decoder layers.
    pub layers: Vec<DecoderLayer>,
    #[param]
    /// Final normalization.
    pub norm: nn::RmsNorm,
}

impl TextModel {
    fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            embed_tokens: unloaded_maybe_quantized_embedding(
                args.vocab_size,
                args.hidden_size,
                args.weight_quantization_for("model.embed_tokens.weight"),
                stream,
            )?,
            layers: args
                .layer_schedule
                .iter()
                .enumerate()
                .map(|(layer, _)| DecoderLayer::new(args, layer, stream))
                .collect::<Result<_, _>>()?,
            norm: nn::RmsNorm::unloaded(
                args.hidden_size,
                args.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?,
        })
    }
}

/// Input for a Kimi Linear forward pass.
pub struct ModelInput<'a> {
    /// Token ids `[batch, sequence]`.
    pub inputs: &'a Array,
    /// Optional attention mask.
    pub mask: Option<&'a Array>,
    /// Optional hybrid generation cache.
    pub cache: Option<&'a mut Cache>,
}

#[derive(Debug, Clone, ModuleParameters)]
/// Kimi Linear causal language model.
pub struct Model {
    /// Parsed architecture arguments.
    pub args: ModelArgs,
    #[param]
    /// Transformer body.
    pub model: TextModel,
    #[param]
    /// Optional untied output projection.
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    /// Creates an unloaded model.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(unloaded_maybe_quantized_linear(
                args.hidden_size,
                args.vocab_size,
                false,
                args.weight_quantization_for("lm_head.weight"),
                stream,
            )?)
        };
        Ok(Self {
            model: TextModel::new(&args, stream)?,
            lm_head,
            args,
        })
    }

    /// Creates an empty cache.
    pub fn new_cache(&self) -> Cache {
        Cache::new(&self.args)
    }

    /// Creates device-resident or blockwise-paged MLA state independently of
    /// parameter residency. Bounded KDA state remains device resident.
    pub fn new_cache_with_options(&self, policy: CacheResidencyPolicy) -> Result<Cache, Exception> {
        Cache::new_with_options(&self.args, policy)
    }

    /// Returns the stable model type.
    pub fn model_type(&self) -> &str {
        &self.args.model_type
    }

    pub(crate) fn save_prompt_cache_with_rank(
        cache: &mut Cache,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        rank: Option<CacheRankIdentity>,
    ) -> Result<PromptCacheManifest, Exception> {
        let end = i64::try_from(prefix_token_ids.len())
            .map_err(|_| Exception::custom("Kimi prompt length exceeds i64"))?;
        if cache.offset() as i64 != end {
            return Err(Exception::custom(
                "Kimi cache offset does not match the persisted prefix",
            ));
        }
        for layer in &mut cache.layers {
            if let LayerCache::Mla(cache) = layer {
                cache.finalize()?;
            }
        }
        let mut blocks = Vec::new();
        let mut state = Vec::new();
        for (layer, cache) in cache.layers.iter().enumerate() {
            match cache {
                LayerCache::Kda(cache) => {
                    for (slot, convolution) in [&cache.q_conv, &cache.k_conv, &cache.v_conv]
                        .into_iter()
                        .enumerate()
                    {
                        state.push(PromptCacheStateArray {
                            owner: StateTensorOwner::Layer(layer),
                            role: StateTensorRole::Convolution { slot: slot as u32 },
                            array: convolution.state.as_ref().ok_or_else(|| {
                                Exception::custom("Kimi KDA convolution state is missing")
                            })?,
                        });
                    }
                    state.push(PromptCacheStateArray {
                        owner: StateTensorOwner::Layer(layer),
                        role: StateTensorRole::Recurrent,
                        array: cache.recurrent_state.as_ref().ok_or_else(|| {
                            Exception::custom("Kimi KDA recurrent state is missing")
                        })?,
                    });
                }
                LayerCache::Mla(cache) => {
                    if !cache.is_paged() {
                        let (latent, rotary_key) = cache
                            .arrays()
                            .ok_or_else(|| Exception::custom("Kimi MLA cache state is missing"))?;
                        blocks.push(PromptCacheSnapshotBlock {
                            global_layer: layer,
                            start: 0,
                            end,
                            rank,
                            arrays: CacheBlockArrays::CompressedLatentRotary {
                                latent: latent.clone(),
                                rotary_key: rotary_key.clone(),
                            },
                        });
                    }
                }
            }
        }
        if let Some(manager) = cache.residency_manager() {
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
    ) -> Result<PromptCacheManifest, Exception> {
        Self::save_prompt_cache_with_rank(
            cache,
            destination,
            descriptor,
            prefix_token_ids,
            options,
            None,
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
        let layer_count = args.num_hidden_layers as usize;
        let identity = PromptCacheModelIdentity {
            model_family: "kimi_linear".into(),
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
        let (blocks, state, manifest) =
            open_prompt_cache_snapshot(directory, expected, &identity, prefix_token_ids, stream)
                .map_err(|error| Exception::custom(error.to_string()))?;
        let mut cache = Cache::new(args);
        let mut block_map = BTreeMap::new();
        for block in blocks {
            if block_map.insert(block.global_layer, block).is_some() {
                return Err(Exception::custom(
                    "Kimi resident prompt cache contains multiple blocks for one layer",
                ));
            }
        }
        let mut state_map = state
            .into_iter()
            .map(|tensor| ((tensor.owner, tensor.role), tensor.array))
            .collect::<BTreeMap<_, _>>();
        let end = i32::try_from(prefix_token_ids.len())
            .map_err(|_| Exception::custom("Kimi prompt length exceeds i32"))?;
        for (layer, layer_cache) in cache.layers.iter_mut().enumerate() {
            match layer_cache {
                LayerCache::Mla(cache) => {
                    let block = block_map.remove(&layer).ok_or_else(|| {
                        Exception::custom("Kimi MLA prompt-cache block is missing")
                    })?;
                    match block.arrays {
                        CacheBlockArrays::CompressedLatentRotary { latent, rotary_key } => {
                            cache.restore_resident(latent, rotary_key, end)?;
                        }
                        _ => return Err(Exception::custom("Kimi MLA prompt-cache kind mismatch")),
                    }
                }
                LayerCache::Kda(cache) => {
                    for (slot, convolution) in
                        [&mut cache.q_conv, &mut cache.k_conv, &mut cache.v_conv]
                            .into_iter()
                            .enumerate()
                    {
                        convolution.state = Some(
                            state_map
                                .remove(&(
                                    StateTensorOwner::Layer(layer),
                                    StateTensorRole::Convolution { slot: slot as u32 },
                                ))
                                .ok_or_else(|| {
                                    Exception::custom("Kimi KDA convolution state is missing")
                                })?,
                        );
                        convolution.offset = end;
                    }
                    cache.recurrent_state = Some(
                        state_map
                            .remove(&(StateTensorOwner::Layer(layer), StateTensorRole::Recurrent))
                            .ok_or_else(|| {
                                Exception::custom("Kimi KDA recurrent state is missing")
                            })?,
                    );
                }
            }
        }
        if !block_map.is_empty() || !state_map.is_empty() {
            return Err(Exception::custom("Kimi prompt cache has unexpected state"));
        }
        Ok((cache, manifest))
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
        let state = load_prompt_cache_state_tensors(directory, &manifest, stream)
            .map_err(|error| Exception::custom(error.to_string()))?;
        let mut state = state
            .into_iter()
            .map(|tensor| ((tensor.owner, tensor.role), tensor.array))
            .collect::<BTreeMap<_, _>>();
        let end = i32::try_from(prefix_token_ids.len())
            .map_err(|_| Exception::custom("Kimi prompt length exceeds i32"))?;
        let mut cache =
            Cache::new_with_manager(args, manager, identity.topology.cache_rank_identity())?;
        for (layer, layer_cache) in cache.layers.iter_mut().enumerate() {
            if let LayerCache::Kda(cache) = layer_cache {
                for (slot, convolution) in [&mut cache.q_conv, &mut cache.k_conv, &mut cache.v_conv]
                    .into_iter()
                    .enumerate()
                {
                    convolution.state = Some(
                        state
                            .remove(&(
                                StateTensorOwner::Layer(layer),
                                StateTensorRole::Convolution { slot: slot as u32 },
                            ))
                            .ok_or_else(|| {
                                Exception::custom("Kimi KDA convolution state is missing")
                            })?,
                    );
                    convolution.offset = end;
                }
                cache.recurrent_state = Some(
                    state
                        .remove(&(StateTensorOwner::Layer(layer), StateTensorRole::Recurrent))
                        .ok_or_else(|| Exception::custom("Kimi KDA recurrent state is missing"))?,
                );
            }
        }
        if !state.is_empty() {
            return Err(Exception::custom(
                "Kimi paged prompt cache has unexpected fixed state",
            ));
        }
        Ok((cache, manifest))
    }

    fn forward_logits_impl(
        &mut self,
        input: ModelInput<'_>,
        last_token_only: bool,
        stream: &Stream,
        mut observer: Option<&mut dyn ActivationObserver>,
    ) -> Result<Array, Exception> {
        let mut hidden = self.model.embed_tokens.forward(input.inputs, stream)?;
        if let Some(observer) = observer.as_deref_mut() {
            observer.observe("model.embed_tokens", &hidden)?;
        }
        let offset = input.cache.as_ref().map_or(0, |cache| cache.offset());
        let generated_mask = if input.mask.is_none() && hidden.dim(1) > 1 && offset > 0 {
            Some(create_causal_mask(
                hidden.dim(1),
                Some(offset),
                None,
                None,
                stream,
            )?)
        } else {
            None
        };
        let mask = input.mask.or(generated_mask.as_ref());
        if let Some(cache) = input.cache {
            cache
                .validate(&self.args.layer_schedule)
                .map_err(|error| Exception::custom(error.to_string()))?;
            for (index, (layer, cache)) in self
                .model
                .layers
                .iter_mut()
                .zip(&mut cache.layers)
                .enumerate()
            {
                hidden = layer.forward_impl(
                    &hidden,
                    mask,
                    Some(cache),
                    stream,
                    &format!("model.layers.{index}"),
                    &mut observer,
                )?;
            }
        } else {
            for (index, layer) in self.model.layers.iter_mut().enumerate() {
                hidden = layer.forward_impl(
                    &hidden,
                    mask,
                    None,
                    stream,
                    &format!("model.layers.{index}"),
                    &mut observer,
                )?;
            }
        }
        hidden = self.model.norm.forward(&hidden, stream)?;
        if last_token_only {
            hidden = hidden.try_index_device((.., -1, ..), stream)?;
        }
        let logits = project_logits_maybe_quantized(
            &mut self.lm_head,
            &mut self.model.embed_tokens,
            &hidden,
            stream,
        )?;
        if let Some(observer) = observer {
            observer.observe("lm_head.logits", &logits)?;
        }
        Ok(logits)
    }

    /// Runs a full logits pass.
    pub fn forward_logits(
        &mut self,
        input: ModelInput<'_>,
        last_token_only: bool,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.forward_logits_impl(input, last_token_only, stream, None)
    }

    /// Runs a full observed logits pass.
    pub fn forward_with_observer(
        &mut self,
        input: ModelInput<'_>,
        stream: &Stream,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        self.forward_logits_impl(input, false, stream, Some(observer))
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

    fn training_mode(&mut self, _mode: bool) {}
}

impl CausalLm<Cache> for Model {
    fn prefill_input_logits(
        &mut self,
        input: runtime_input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let tokens = runtime_input::text_token_ids(input, stream)?;
        self.forward_logits(
            ModelInput {
                inputs: &tokens,
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
        self.forward_logits(
            ModelInput {
                inputs: input_tokens,
                mask: None,
                cache: Some(cache),
            },
            true,
            stream,
        )
    }
}

/// Kimi Linear token generation iterator.
pub type Generate<'a, S = crate::backend::mlx::runtime::generation::sampler::DefaultSampler> =
    crate::backend::mlx::nn::generation::Generate<'a, Model, Cache, S>;

/// Kimi Linear model plus architecture-owned GGUF stop IDs.
pub(crate) struct PreparedKimiLinearGguf {
    pub(crate) args: ModelArgs,
    pub(crate) eos_token_ids: Vec<u32>,
}

pub(crate) fn prepare_gguf_checkpoint(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    quantization: Option<WeightQuantization>,
    _weights_stream: &Stream,
) -> Result<PreparedKimiLinearGguf, Error> {
    let architecture = gguf_string(metadata, "general.architecture")?;
    if architecture != "kimi-linear" {
        return Err(Error::UnsupportedArchitecture(format!(
            "GGUF architecture {architecture:?}; the Kimi Linear loader supports kimi-linear"
        )));
    }
    checkpoint
        .catalog()
        .translated_outputs(translate_gguf_weight_name)
        .map_err(safemlx::error::IoError::from)?;
    let mut args = model_args_from_gguf_catalog(checkpoint, metadata)?;
    let mut formats = gguf_quantization_configs(checkpoint, translate_gguf_weight_name)?;
    combine_expert_gate_up_formats(&mut formats, &args)?;
    if let Some(quantization) = quantization {
        args.quantization = Some(quantization);
        args.quantized_weight_configs = None;
    } else {
        args.quantized_weight_configs = Some(formats);
    }
    args.validate()?;
    Ok(PreparedKimiLinearGguf {
        args,
        eos_token_ids: crate::backend::mlx::gguf_eos_token_ids(metadata)?,
    })
}

fn combine_expert_gate_up_formats(
    formats: &mut HashMap<String, WeightQuantization>,
    args: &ModelArgs,
) -> Result<(), Error> {
    for (layer, policy) in args.layer_schedule.iter().enumerate() {
        if policy.feed_forward != FeedForwardPolicy::SparseMoe {
            continue;
        }
        let prefix = format!("model.layers.{layer}.mlp.experts");
        let gate = formats.remove(&format!("{prefix}.gate_proj"));
        let up = formats.remove(&format!("{prefix}.up_proj"));
        match (gate, up) {
            (Some(gate), Some(up)) if gate == up => {
                formats.insert(format!("{prefix}.gate_up_proj"), gate);
            }
            (None, None) => {}
            (gate, up) => {
                return Err(Error::UnsupportedArchitecture(format!(
                    "Kimi Linear GGUF layer {layer} routed gate/up formats must match; gate={gate:?}, up={up:?}"
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
fn normalize_gguf_weight(
    name: &str,
    value: Array,
    args: &ModelArgs,
    stream: &Stream,
) -> Result<Array, Error> {
    if name.ends_with(".q_conv1d.weight")
        || name.ends_with(".k_conv1d.weight")
        || name.ends_with(".v_conv1d.weight")
    {
        let expected = args.kda_config.num_heads * args.kda_config.head_dim;
        let kernel = args.kda_config.short_conv_kernel_size;
        if value.size() != (expected * kernel) as usize {
            return Err(Error::UnsupportedArchitecture(format!(
                "Kimi Linear convolution {name:?} has shape {:?}, expected {expected}x{kernel}",
                value.shape()
            )));
        }
        return Ok(value.reshape(&[expected, 1, kernel], stream)?);
    }
    if name.ends_with(".A_log") {
        let heads = args.kda_config.num_heads;
        if value.size() != heads as usize {
            return Err(Error::UnsupportedArchitecture(format!(
                "Kimi Linear GGUF ssm_a {name:?} has shape {:?}, expected {heads} values",
                value.shape()
            )));
        }
        let negative = value
            .lt(Array::from_f32(0.0), stream)?
            .all(false, stream)?
            .item::<bool>(stream);
        if !negative {
            return Err(Error::UnsupportedArchitecture(format!(
                "Kimi Linear GGUF ssm_a {name:?} must contain only negative values"
            )));
        }
        return Ok(value
            .negative(stream)?
            .log(stream)?
            .reshape(&[1, 1, heads, 1], stream)?);
    }
    Ok(value)
}

pub(crate) fn translate_gguf_weight_name(name: &str) -> String {
    for (source, target) in [
        ("token_embd", "model.embed_tokens"),
        ("output_norm", "model.norm"),
        ("output", "lm_head"),
    ] {
        if name == source || name.starts_with(&format!("{source}.")) {
            return name.replacen(source, target, 1);
        }
    }
    let Some(rest) = name.strip_prefix("blk.") else {
        return name.to_owned();
    };
    let Some((layer, parameter)) = rest.split_once('.') else {
        return name.to_owned();
    };
    for (source, target) in [
        ("ffn_gate_exps", "mlp.experts.gate_proj"),
        ("ffn_up_exps", "mlp.experts.up_proj"),
        ("ffn_down_exps", "mlp.experts.down_proj"),
    ] {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            let suffix = match parameter.strip_prefix(source).unwrap_or_default() {
                ".weight" => "",
                ".scales" => "_scales",
                ".biases" => "_biases",
                other => other,
            };
            return format!("model.layers.{layer}.{target}{suffix}");
        }
    }
    if matches!(parameter, "exp_probs_b.bias" | "ffn_exp_probs_b.bias") {
        return format!("model.layers.{layer}.mlp.gate.e_score_correction_bias");
    }
    for (source, target) in [
        ("attn_q", "self_attn.q_proj"),
        ("attn_k", "self_attn.k_proj"),
        ("attn_v", "self_attn.v_proj"),
        ("attn_q_a", "self_attn.q_a_proj"),
        ("attn_q_b", "self_attn.q_b_proj"),
        ("attn_kv_a_mqa", "self_attn.kv_a_proj_with_mqa"),
        ("attn_kv_b", "self_attn.kv_b_proj"),
        ("attn_k_b", "self_attn.k_b_proj"),
        ("attn_v_b", "self_attn.v_b_proj"),
        ("attn_q_a_norm", "self_attn.q_a_layernorm"),
        ("attn_kv_a_norm", "self_attn.kv_a_layernorm"),
        ("attn_output", "self_attn.o_proj"),
        ("ssm_conv1d_q", "self_attn.q_conv1d"),
        ("ssm_conv1d_k", "self_attn.k_conv1d"),
        ("ssm_conv1d_v", "self_attn.v_conv1d"),
        ("ssm_f_a", "self_attn.f_a_proj"),
        ("ssm_f_b", "self_attn.f_b_proj"),
        ("ssm_beta", "self_attn.b_proj"),
        ("ssm_g_a", "self_attn.g_a_proj"),
        ("ssm_g_b", "self_attn.g_b_proj"),
        ("ssm_norm", "self_attn.o_norm"),
        ("attn_norm", "input_layernorm"),
        ("ffn_norm", "post_attention_layernorm"),
        ("ffn_gate", "mlp.gate_proj"),
        ("ffn_up", "mlp.up_proj"),
        ("ffn_down", "mlp.down_proj"),
        ("ffn_gate_shexp", "mlp.shared_experts.gate_proj"),
        ("ffn_up_shexp", "mlp.shared_experts.up_proj"),
        ("ffn_down_shexp", "mlp.shared_experts.down_proj"),
        ("ffn_gate_inp", "mlp.gate"),
    ] {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            return format!(
                "model.layers.{layer}.{}",
                parameter.replacen(source, target, 1)
            );
        }
    }
    if parameter == "ssm_a" || parameter.starts_with("ssm_a.") {
        let suffix = parameter.strip_prefix("ssm_a").unwrap_or_default();
        let suffix = if suffix == ".weight" { "" } else { suffix };
        return format!("model.layers.{layer}.self_attn.A_log{suffix}");
    }
    if parameter == "ssm_dt.bias" || parameter == "ssm_dt" {
        return format!("model.layers.{layer}.self_attn.dt_bias");
    }
    name.to_owned()
}

/// Parses and validates Kimi Linear geometry using GGUF metadata and catalog names only.
pub(crate) fn model_args_from_gguf_catalog(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> Result<ModelArgs, Error> {
    let architecture = "kimi-linear";
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let num_hidden_layers = gguf_i32(metadata, &key("block_count"))?;
    let num_attention_heads = gguf_i32(metadata, &key("attention.head_count"))?;
    let full_attention_layers =
        gguf_attention_layers(checkpoint, metadata, num_hidden_layers, &key)?;
    let kda_layers = (1..=num_hidden_layers)
        .filter(|layer| !full_attention_layers.contains(layer))
        .collect::<Vec<_>>();
    let qk_rope_head_dim = gguf_i32(metadata, &key("rope.dimension_count"))?;
    let qk_head_dim = gguf_i32(metadata, &key("attention.key_length_mla"))?;
    let qk_nope_head_dim = qk_head_dim.checked_sub(qk_rope_head_dim).ok_or_else(|| {
        Error::UnsupportedArchitecture(format!(
            "Kimi Linear GGUF MLA key length {qk_head_dim} is smaller than nominal positional length {qk_rope_head_dim}"
        ))
    })?;
    let q_lora_rank = gguf_optional_i64(metadata, &key("attention.q_lora_rank"))?
        .map(i32::try_from)
        .transpose()
        .map_err(|_| Error::UnsupportedArchitecture("GGUF query LoRA rank exceeds i32".into()))?
        .filter(|rank| *rank > 0);
    let vocab_size = metadata
        .get("tokenizer.ggml.tokens")
        .and_then(GgufMetadataValue::as_strings)
        .map(|tokens| i32::try_from(tokens.len()))
        .transpose()
        .map_err(|_| Error::UnsupportedArchitecture("GGUF vocabulary exceeds i32".into()))?
        .unwrap_or(gguf_i32(metadata, &key("vocab_size"))?);
    let gating = gguf_optional_i64(metadata, &key("expert_gating_func"))?.unwrap_or(2);
    if gating != 2 {
        return Err(Error::UnsupportedArchitecture(format!(
            "Kimi Linear GGUF expert_gating_func {gating} is unsupported; expected sigmoid (2)"
        )));
    }
    let hidden_size = gguf_i32(metadata, &key("embedding_length"))?;
    ModelArgsSource {
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
        linear_attn_config: LinearAttentionConfigSource {
            full_attn_layers: full_attention_layers,
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
            .map_err(|_| {
                Error::UnsupportedArchitecture("leading dense layer count exceeds i32".into())
            })?,
        moe_layer_freq: 1,
        num_experts_per_token: gguf_i32(metadata, &key("expert_used_count"))?,
        num_shared_experts: gguf_optional_i64(metadata, &key("expert_shared_count"))?
            .unwrap_or(1)
            .try_into()
            .map_err(|_| {
                Error::UnsupportedArchitecture("shared expert count exceeds i32".into())
            })?,
        num_expert_group: gguf_optional_i64(metadata, &key("expert_group_count"))?
            .unwrap_or(1)
            .try_into()
            .map_err(|_| Error::UnsupportedArchitecture("expert group count exceeds i32".into()))?,
        topk_group: gguf_optional_i64(metadata, &key("expert_group_used_count"))?
            .unwrap_or(1)
            .try_into()
            .map_err(|_| {
                Error::UnsupportedArchitecture("expert group selection exceeds i32".into())
            })?,
        routed_scaling_factor: gguf_optional_f32(metadata, &key("expert_weights_scale"))?
            .unwrap_or(1.0),
        moe_renormalize: gguf_optional_bool(metadata, &key("expert_weights_norm"))?.unwrap_or(true),
        moe_router_activation_func: "sigmoid".into(),
        use_grouped_topk: true,
        tie_word_embeddings: !checkpoint.contains_gguf_tensor("output.weight"),
        num_nextn_predict_layers: 0,
        quantization: None,
        quantized_weight_configs: None,
        split_kv_b: checkpoint.any_gguf_tensor(|name| name.contains(".attn_k_b.")),
    }
    .normalize()
}

fn gguf_attention_layers(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    layers: i32,
    key: &impl Fn(&str) -> String,
) -> Result<Vec<i32>, Error> {
    let metadata_key = key("attention.head_count_kv");
    if let Some(array) = metadata
        .get(&metadata_key)
        .and_then(GgufMetadataValue::as_array)
    {
        let values = array.to_i64_vec().ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "GGUF metadata key {metadata_key:?} must be an integer array"
            ))
        })?;
        if values.len() != layers as usize {
            return Err(Error::UnsupportedArchitecture(format!(
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
        .filter(|layer| {
            !checkpoint.any_gguf_tensor(|name| name.starts_with(&format!("blk.{layer}.ssm_")))
        })
        .map(|layer| layer + 1)
        .collect())
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
            Error::UnsupportedArchitecture(format!("GGUF metadata key {key:?} has the wrong type"))
        }),
        None => Ok(None),
    }
}

fn gguf_optional_bool(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<Option<bool>, Error> {
    match metadata.get(key) {
        Some(GgufMetadataValue::Bool(value)) => Ok(Some(*value)),
        Some(value) => value.as_i64().map(|value| Some(value != 0)).ok_or_else(|| {
            Error::UnsupportedArchitecture(format!("GGUF metadata key {key:?} has the wrong type"))
        }),
        None => Ok(None),
    }
}

/// Loads the checkpoint tokenizer.
///
/// Native `tiktoken.model` loading is added by the tokenizer integration; a
/// generated `tokenizer.json` remains accepted for converted checkpoints.
pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let model_dir = model_dir.as_ref();
    let converted = model_dir.join("tokenizer.json");
    if converted.exists() {
        return Ok(Tokenizer::from_file(converted)?);
    }
    eredu_text::tiktoken::load_kimi_k2(model_dir).map_err(Error::Template)
}
