//! DeepSeek-V3 and DeepSeek-R1 decoder architecture.
//!
//! The implementation follows the released DeepSeek-V3 inference equations:
//! multi-token prefill reconstructs head-specific K/V transiently for fused
//! attention, while decode keeps only normalized latent KV and the rotary key
//! component in the cache.

use std::{collections::HashMap, path::Path};

use safemlx::{
    builder::Builder,
    error::Exception,
    fast::ScaledDotProductAttentionMask,
    macros::ModuleParameters,
    module::{Module, ModuleParameters, Param},
    native_quantization::{native_grouped_linear, NativeQuantizedTensor},
    nn,
    ops::{
        broadcast_to, concatenate_axis, einsum, gather_grouped_rows, grouped_matmul,
        indexing::{NewAxis, TryIndexOp},
        quantized_packed_dimension, r#where, softmax_axis, topk_route_plan, GgufCheckpoint,
        GgufMetadataValue,
    },
    quantization::MaybeQuantized,
    transforms::eval,
    Array, Dtype, Stream,
};
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::nn as common;
use crate::{
    api::{
        input as runtime_input,
        qwen3_5::{QwenLinear as Linear, QwenWeightFormat as WeightFormat},
    },
    nn::{
        generation::CausalLm,
        layers::silu,
        moe::{weighted_route_sum, TopKRouter, TopKRouterConfig, TopKRouterScoreFunction},
    },
};
use crate::{
    error::Error,
    nn::tensor::{
        create_causal_mask,
        rope::{initialize_rope, FloatOrString, RopeVariant},
    },
    runtime::attention::LayerSchedule,
    runtime::cache::residency::{
        derive_prompt_cache_architecture_fingerprint, open_prompt_cache,
        validate_prompt_cache_model_identity, CacheBlockArrays, CacheRankIdentity,
        CacheResidencyManager, CacheResidencyPolicy, CacheResidencyReport, PagedCacheOptions,
        PromptCacheDescriptor, PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
    },
    runtime::cache::{
        BlockwiseAttentionAccumulator, CompressedLatentCache, KeyValueAttentionBlock,
    },
    runtime::checkpoint::load::{gguf_quantization_configs, GgufTensorNames},
    runtime::checkpoint::quantization::WeightQuantization,
    runtime::execution::inspection::{ActivationObserver, MoeRoutingObservation},
};
type ObserverOption<'a> = Option<&'a mut dyn ActivationObserver>;

fn activation_name(prefix: &str, suffix: &str) -> String {
    if suffix.is_empty() {
        prefix.to_string()
    } else {
        format!("{prefix}.{suffix}")
    }
}

#[inline]
fn observe_activation(
    observer: &mut ObserverOption<'_>,
    prefix: &str,
    suffix: &str,
    value: &Array,
) -> Result<(), Exception> {
    if let Some(observer) = observer.as_mut() {
        observer.observe(&activation_name(prefix, suffix), value)?;
    }
    Ok(())
}

#[inline]
fn intervene_activation(
    observer: &mut ObserverOption<'_>,
    prefix: &str,
    suffix: &str,
    value: Array,
) -> Result<Array, Exception> {
    let Some(observer) = observer.as_mut() else {
        return Ok(value);
    };
    Ok(observer
        .intervene(&activation_name(prefix, suffix), &value)?
        .unwrap_or(value))
}

fn default_model_type() -> String {
    "deepseek_v3".into()
}

fn default_rms_norm_eps() -> f32 {
    1e-6
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

fn default_topk_method() -> String {
    "noaux_tc".into()
}

fn default_scoring_func() -> String {
    "sigmoid".into()
}

/// DeepSeek YaRN configuration used by released V3/R1 checkpoints.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct YarnConfig {
    /// Scaling type; released checkpoints use `yarn`.
    #[serde(alias = "rope_type")]
    pub r#type: String,
    /// Context extension factor.
    pub factor: f32,
    /// Original trained context length.
    pub original_max_position_embeddings: i32,
    /// YaRN fast correction rotations.
    #[serde(default = "default_beta_fast")]
    pub beta_fast: f32,
    /// YaRN slow correction rotations.
    #[serde(default = "default_beta_slow")]
    pub beta_slow: f32,
    /// Rotary concentration coefficient.
    #[serde(default = "default_float_one")]
    pub mscale: f32,
    /// Attention-scale coefficient.
    #[serde(default)]
    pub mscale_all_dim: f32,
}

fn default_beta_fast() -> f32 {
    32.0
}
fn default_beta_slow() -> f32 {
    1.0
}
fn default_float_one() -> f32 {
    1.0
}

impl YarnConfig {
    fn rope_config(&self) -> HashMap<String, FloatOrString> {
        HashMap::from([
            ("type".into(), FloatOrString::String(self.r#type.clone())),
            ("factor".into(), FloatOrString::Float(self.factor)),
            (
                "original_max_position_embeddings".into(),
                FloatOrString::Float(self.original_max_position_embeddings as f32),
            ),
            ("beta_fast".into(), FloatOrString::Float(self.beta_fast)),
            ("beta_slow".into(), FloatOrString::Float(self.beta_slow)),
            ("mscale".into(), FloatOrString::Float(self.mscale)),
            (
                "mscale_all_dim".into(),
                FloatOrString::Float(self.mscale_all_dim),
            ),
        ])
    }

    fn attention_multiplier(&self) -> f32 {
        if self.mscale_all_dim == 0.0 || self.factor <= 1.0 {
            1.0
        } else {
            let scale = 0.1 * self.mscale_all_dim * self.factor.ln() + 1.0;
            scale * scale
        }
    }
}

/// Published block-FP8 metadata.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct Fp8QuantizationConfig {
    /// Quantization method (`fp8`).
    pub quant_method: String,
    /// E4M3 storage format.
    pub fmt: String,
    /// Dynamic activation scaling marker.
    pub activation_scheme: String,
    /// Two-dimensional weight block.
    pub weight_block_size: Vec<i32>,
}

impl Fp8QuantizationConfig {
    pub(crate) fn validate(&self) -> Result<(), Error> {
        if self.quant_method != "fp8"
            || self.fmt != "e4m3"
            || self.activation_scheme != "dynamic"
            || self.weight_block_size.as_slice() != [128, 128]
        {
            return Err(Error::UnsupportedArchitecture(format!(
                "DeepSeek-V3 supports only dynamic E4M3 block-FP8 with weight_block_size [128, 128], got {self:?}"
            )));
        }
        Ok(())
    }
}

/// Quantization metadata accepted under Hugging Face's
/// `quantization_config` key.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum DeepSeekQuantizationConfig {
    /// Official DeepSeek block-FP8 metadata.
    Fp8(Fp8QuantizationConfig),
    /// MLX affine or MXFP4 metadata emitted by checkpoint conversion.
    Affine(WeightQuantization),
}

/// Feed-forward operator used by one DeepSeek-V3/R1 decoder layer.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum LayerPolicy {
    /// Dense SwiGLU feed-forward block.
    DenseMlp,
    /// Routed and shared expert feed-forward block.
    SparseMoe,
}

/// Source-format DeepSeek-V3/R1 text configuration.
#[derive(Debug, Clone, Deserialize)]
struct ModelArgsSource {
    /// Hugging Face model type.
    #[serde(default = "default_model_type")]
    pub model_type: String,
    /// Model width.
    pub hidden_size: i32,
    /// Dense SwiGLU width.
    pub intermediate_size: i32,
    /// Routed-expert SwiGLU width.
    pub moe_intermediate_size: i32,
    /// Decoder layer count, excluding MTP layers.
    pub num_hidden_layers: i32,
    /// MLA query head count.
    pub num_attention_heads: i32,
    /// Token vocabulary size.
    pub vocab_size: i32,
    /// RMS normalization epsilon.
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    /// Maximum configured context length.
    pub max_position_embeddings: i32,
    /// RoPE base.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// Optional YaRN extension settings.
    #[serde(default)]
    pub rope_scaling: Option<YarnConfig>,
    /// Query LoRA rank; `None` selects the direct `q_proj` form.
    #[serde(default)]
    pub q_lora_rank: Option<i32>,
    /// Compressed KV latent width.
    pub kv_lora_rank: i32,
    /// Non-positional query/key width per head.
    pub qk_nope_head_dim: i32,
    /// Rotary query/key width per head.
    pub qk_rope_head_dim: i32,
    /// Value width per head.
    pub v_head_dim: i32,
    first_k_dense_replace: i32,
    #[serde(default = "default_moe_layer_freq")]
    moe_layer_freq: i32,
    /// Routed expert count.
    pub n_routed_experts: i32,
    /// Shared expert count.
    #[serde(default = "default_one")]
    pub n_shared_experts: i32,
    /// Selected experts per token.
    pub num_experts_per_tok: i32,
    /// Expert routing group count.
    pub n_group: i32,
    /// Selected routing group count.
    pub topk_group: i32,
    /// Grouped top-k method.
    #[serde(default = "default_topk_method")]
    pub topk_method: String,
    /// Router score transform.
    #[serde(default = "default_scoring_func")]
    pub scoring_func: String,
    /// Normalize selected scores.
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,
    /// Final routed contribution multiplier.
    #[serde(default = "default_float_one")]
    pub routed_scaling_factor: f32,
    /// Appended multi-token-prediction layer count.
    #[serde(default)]
    pub num_nextn_predict_layers: i32,
    /// Native FP8 metadata.
    #[serde(default)]
    pub quantization_config: Option<DeepSeekQuantizationConfig>,
    /// Optional MLX affine checkpoint metadata.
    #[serde(default)]
    pub quantization: Option<WeightQuantization>,
    /// Per-weight affine settings for mixed-quantization GGUF tensors.
    #[serde(skip)]
    pub quantized_weight_configs: Option<HashMap<String, WeightQuantization>>,
    /// Whether the checkpoint stores per-head MLA K/V reconstruction separately.
    #[serde(skip)]
    pub split_kv_b: bool,
    /// Whether embedding and LM-head weights are tied.
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

/// Validated DeepSeek-V3/R1 text configuration.
#[derive(Debug, Clone)]
pub struct ModelArgs {
    /// Hugging Face model type.
    pub model_type: String,
    /// Model width.
    pub hidden_size: i32,
    /// Dense SwiGLU width.
    pub intermediate_size: i32,
    /// Routed-expert SwiGLU width.
    pub moe_intermediate_size: i32,
    /// Decoder layer count, excluding MTP layers.
    pub num_hidden_layers: i32,
    /// MLA query head count.
    pub num_attention_heads: i32,
    /// Token vocabulary size.
    pub vocab_size: i32,
    /// RMS normalization epsilon.
    pub rms_norm_eps: f32,
    /// Maximum configured context length.
    pub max_position_embeddings: i32,
    /// RoPE base.
    pub rope_theta: f32,
    /// Optional YaRN extension settings.
    pub rope_scaling: Option<YarnConfig>,
    /// Query LoRA rank; `None` selects the direct `q_proj` form.
    pub q_lora_rank: Option<i32>,
    /// Compressed KV latent width.
    pub kv_lora_rank: i32,
    /// Non-positional query/key width per head.
    pub qk_nope_head_dim: i32,
    /// Rotary query/key width per head.
    pub qk_rope_head_dim: i32,
    /// Value width per head.
    pub v_head_dim: i32,
    /// Authoritative feed-forward policy in decoder-layer order.
    pub layer_schedule: LayerSchedule<LayerPolicy>,
    /// Routed expert count.
    pub n_routed_experts: i32,
    /// Shared expert count.
    pub n_shared_experts: i32,
    /// Selected experts per token.
    pub num_experts_per_tok: i32,
    /// Expert routing group count.
    pub n_group: i32,
    /// Selected routing group count.
    pub topk_group: i32,
    /// Grouped top-k method.
    pub topk_method: String,
    /// Router score transform.
    pub scoring_func: String,
    /// Normalize selected scores.
    pub norm_topk_prob: bool,
    /// Final routed contribution multiplier.
    pub routed_scaling_factor: f32,
    /// Appended multi-token-prediction layer count.
    pub num_nextn_predict_layers: i32,
    /// Native FP8 metadata.
    pub quantization_config: Option<DeepSeekQuantizationConfig>,
    /// Optional MLX affine checkpoint metadata.
    pub quantization: Option<WeightQuantization>,
    /// Per-weight affine settings for mixed-quantization GGUF tensors.
    pub quantized_weight_configs: Option<HashMap<String, WeightQuantization>>,
    /// Whether the checkpoint stores per-head MLA K/V reconstruction separately.
    pub split_kv_b: bool,
    /// Whether embedding and LM-head weights are tied.
    pub tie_word_embeddings: bool,
}

impl ModelArgsSource {
    fn normalize(self) -> Result<ModelArgs, Error> {
        let layer_schedule = deepseek_layer_schedule(
            self.num_hidden_layers,
            self.first_k_dense_replace,
            self.moe_layer_freq,
        )?;
        let args = ModelArgs {
            model_type: self.model_type,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            moe_intermediate_size: self.moe_intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            vocab_size: self.vocab_size,
            rms_norm_eps: self.rms_norm_eps,
            max_position_embeddings: self.max_position_embeddings,
            rope_theta: self.rope_theta,
            rope_scaling: self.rope_scaling,
            q_lora_rank: self.q_lora_rank,
            kv_lora_rank: self.kv_lora_rank,
            qk_nope_head_dim: self.qk_nope_head_dim,
            qk_rope_head_dim: self.qk_rope_head_dim,
            v_head_dim: self.v_head_dim,
            layer_schedule,
            n_routed_experts: self.n_routed_experts,
            n_shared_experts: self.n_shared_experts,
            num_experts_per_tok: self.num_experts_per_tok,
            n_group: self.n_group,
            topk_group: self.topk_group,
            topk_method: self.topk_method,
            scoring_func: self.scoring_func,
            norm_topk_prob: self.norm_topk_prob,
            routed_scaling_factor: self.routed_scaling_factor,
            num_nextn_predict_layers: self.num_nextn_predict_layers,
            quantization_config: self.quantization_config,
            quantization: self.quantization,
            quantized_weight_configs: self.quantized_weight_configs,
            split_kv_b: self.split_kv_b,
            tie_word_embeddings: self.tie_word_embeddings,
        };
        args.validate()?;
        Ok(args)
    }
}

fn deepseek_layer_schedule(
    num_hidden_layers: i32,
    first_k_dense_replace: i32,
    moe_layer_freq: i32,
) -> Result<LayerSchedule<LayerPolicy>, Error> {
    let layer_count = usize::try_from(num_hidden_layers).map_err(|_| {
        Error::UnsupportedArchitecture(format!(
            "DeepSeek-V3 num_hidden_layers must be positive, got {num_hidden_layers}"
        ))
    })?;
    if layer_count == 0 {
        return Err(Error::UnsupportedArchitecture(
            "DeepSeek-V3 num_hidden_layers must be positive, got 0".into(),
        ));
    }
    if first_k_dense_replace < 0 || first_k_dense_replace > num_hidden_layers {
        return Err(Error::UnsupportedArchitecture(format!(
            "DeepSeek-V3 first_k_dense_replace must be between zero and num_hidden_layers, got {first_k_dense_replace}"
        )));
    }
    if moe_layer_freq <= 0 {
        return Err(Error::UnsupportedArchitecture(format!(
            "DeepSeek-V3 moe_layer_freq must be positive, got {moe_layer_freq}"
        )));
    }
    LayerSchedule::new(
        layer_count,
        (0..layer_count)
            .map(|layer| {
                if layer as i32 >= first_k_dense_replace && layer as i32 % moe_layer_freq == 0 {
                    LayerPolicy::SparseMoe
                } else {
                    LayerPolicy::DenseMlp
                }
            })
            .collect(),
    )
    .map_err(|error| Error::UnsupportedArchitecture(format!("DeepSeek-V3 {error}")))
}

impl ModelArgs {
    pub(crate) fn validate(&self) -> Result<(), Error> {
        if self.model_type != "deepseek_v3" {
            return Err(Error::UnsupportedModelType(self.model_type.clone()));
        }
        for (name, value) in [
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("moe_intermediate_size", self.moe_intermediate_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("vocab_size", self.vocab_size),
            ("max_position_embeddings", self.max_position_embeddings),
            ("kv_lora_rank", self.kv_lora_rank),
            ("qk_nope_head_dim", self.qk_nope_head_dim),
            ("qk_rope_head_dim", self.qk_rope_head_dim),
            ("v_head_dim", self.v_head_dim),
        ] {
            if value <= 0 {
                return Err(Error::UnsupportedArchitecture(format!(
                    "DeepSeek-V3 {name} must be positive, got {value}"
                )));
            }
        }
        if self.qk_rope_head_dim % 2 != 0 {
            return Err(Error::UnsupportedArchitecture(
                "DeepSeek-V3 qk_rope_head_dim must be even".into(),
            ));
        }
        if self.rms_norm_eps <= 0.0 || self.rope_theta <= 0.0 || self.routed_scaling_factor <= 0.0 {
            return Err(Error::UnsupportedArchitecture(
                "DeepSeek-V3 normalization, RoPE, and routed scaling values must be positive"
                    .into(),
            ));
        }
        if self.q_lora_rank.is_some_and(|rank| rank <= 0) {
            return Err(Error::UnsupportedArchitecture(
                "DeepSeek-V3 q_lora_rank must be positive or null".into(),
            ));
        }
        if self.layer_schedule.len() != self.num_hidden_layers as usize {
            return Err(Error::UnsupportedArchitecture(format!(
                "DeepSeek-V3 layer schedule has {} entries for {} decoder layers",
                self.layer_schedule.len(),
                self.num_hidden_layers
            )));
        }
        if self.n_routed_experts <= 0
            || self.n_shared_experts <= 0
            || self.num_experts_per_tok <= 0
            || self.num_experts_per_tok > self.n_routed_experts
            || self.n_group <= 0
            || self.n_routed_experts % self.n_group != 0
            || self.topk_group <= 0
            || self.topk_group > self.n_group
            || self.num_experts_per_tok > self.topk_group * (self.n_routed_experts / self.n_group)
        {
            return Err(Error::UnsupportedArchitecture(
                "invalid DeepSeek-V3 dense/MoE routing dimensions".into(),
            ));
        }
        if self.topk_method != "noaux_tc" {
            return Err(Error::UnsupportedArchitecture(format!(
                "unsupported DeepSeek-V3 topk_method {:?}; only noaux_tc is implemented",
                self.topk_method
            )));
        }
        if self.scoring_func != "sigmoid" {
            return Err(Error::UnsupportedArchitecture(format!(
                "unsupported DeepSeek-V3 scoring_func {:?}; only sigmoid is implemented",
                self.scoring_func
            )));
        }
        if self.tie_word_embeddings {
            return Err(Error::UnsupportedArchitecture(
                "tied DeepSeek-V3 embeddings are not supported by published V3/R1 checkpoints"
                    .into(),
            ));
        }
        if self.num_nextn_predict_layers < 0 {
            return Err(Error::UnsupportedArchitecture(
                "DeepSeek-V3 num_nextn_predict_layers cannot be negative".into(),
            ));
        }
        if let Some(rope) = &self.rope_scaling {
            if rope.r#type != "yarn"
                || rope.factor <= 0.0
                || rope.original_max_position_embeddings <= 0
            {
                return Err(Error::UnsupportedArchitecture(format!(
                    "unsupported DeepSeek-V3 RoPE scaling {:?}",
                    rope.r#type
                )));
            }
        }
        if let Some(fp8) = self.native_fp8_config() {
            fp8.validate()?;
        }
        if let Some(affine) = self.affine_quantization()? {
            affine.validate()?;
        }
        Ok(())
    }

    /// Returns one validated layer policy without an out-of-range fallback.
    pub fn layer_policy(&self, layer: usize) -> Option<&LayerPolicy> {
        self.layer_schedule.get(layer)
    }

    /// Returns a stable ordered representation of the complete layer schedule.
    pub fn layer_schedule_fingerprint(&self) -> String {
        self.layer_schedule
            .iter()
            .map(|policy| match policy {
                LayerPolicy::DenseMlp => "d",
                LayerPolicy::SparseMoe => "e",
            })
            .collect::<Vec<_>>()
            .join(",")
    }

    pub(crate) fn native_fp8_config(&self) -> Option<&Fp8QuantizationConfig> {
        match &self.quantization_config {
            Some(DeepSeekQuantizationConfig::Fp8(config)) => Some(config),
            Some(DeepSeekQuantizationConfig::Affine(_)) | None => None,
        }
    }

    pub(crate) fn affine_quantization(&self) -> Result<Option<WeightQuantization>, Error> {
        let config_affine = match &self.quantization_config {
            Some(DeepSeekQuantizationConfig::Affine(quantization)) => Some(*quantization),
            Some(DeepSeekQuantizationConfig::Fp8(_)) | None => None,
        };
        if self.native_fp8_config().is_some() && self.quantization.is_some() {
            return Err(Error::Quantization(
                "DeepSeek-V3 config cannot combine native block-FP8 and affine quantization metadata"
                    .into(),
            ));
        }
        match (self.quantization, config_affine) {
            (Some(left), Some(right)) if left != right => Err(Error::Quantization(format!(
                "DeepSeek-V3 quantization and quantization_config disagree: {left:?} versus {right:?}"
            ))),
            (Some(quantization), _) | (_, Some(quantization)) => Ok(Some(quantization)),
            (None, None) => Ok(None),
        }
    }

    fn weight_format_for(&self, weight_name: &str) -> WeightFormat {
        if self.native_fp8_config().is_some() {
            WeightFormat::Fp8
        } else if let Some(quantization) = self
            .affine_quantization()
            .expect("validated DeepSeek quantization metadata")
        {
            WeightFormat::Affine(quantization)
        } else if let Some(config) = self
            .quantized_weight_configs
            .as_ref()
            .and_then(|configs| configs.get(weight_name))
        {
            match *config {
                iq @ WeightQuantization::GgufIQuant { .. } => WeightFormat::IQuant(iq),
                affine => WeightFormat::Affine(affine),
            }
        } else {
            WeightFormat::Dense
        }
    }

    pub(crate) fn weight_quantization_for(&self, weight_name: &str) -> Option<WeightQuantization> {
        self.weight_format_for(weight_name).quantization()
    }
}

pub(crate) fn prompt_cache_architecture_fingerprint(args: &ModelArgs) -> String {
    let rope_scaling = args.rope_scaling.as_ref().map_or_else(
        || "none".to_string(),
        |config| {
            [
                format!("type={}", config.r#type),
                format!("factor={:08x}", config.factor.to_bits()),
                format!(
                    "original_max_position_embeddings={}",
                    config.original_max_position_embeddings
                ),
                format!("beta_fast={:08x}", config.beta_fast.to_bits()),
                format!("beta_slow={:08x}", config.beta_slow.to_bits()),
                format!("mscale={:08x}", config.mscale.to_bits()),
                format!("mscale_all_dim={:08x}", config.mscale_all_dim.to_bits()),
            ]
            .join(";")
        },
    );
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
        "deepseek_v3",
        [
            ("model_type", args.model_type.clone()),
            ("hidden_size", args.hidden_size.to_string()),
            ("intermediate_size", args.intermediate_size.to_string()),
            (
                "moe_intermediate_size",
                args.moe_intermediate_size.to_string(),
            ),
            ("num_hidden_layers", args.num_hidden_layers.to_string()),
            ("num_attention_heads", args.num_attention_heads.to_string()),
            ("vocab_size", args.vocab_size.to_string()),
            (
                "rms_norm_eps",
                format!("{:08x}", args.rms_norm_eps.to_bits()),
            ),
            (
                "max_position_embeddings",
                args.max_position_embeddings.to_string(),
            ),
            ("rope_theta", format!("{:08x}", args.rope_theta.to_bits())),
            ("rope_scaling", rope_scaling),
            ("q_lora_rank", format!("{:?}", args.q_lora_rank)),
            ("kv_lora_rank", args.kv_lora_rank.to_string()),
            ("qk_nope_head_dim", args.qk_nope_head_dim.to_string()),
            ("qk_rope_head_dim", args.qk_rope_head_dim.to_string()),
            ("v_head_dim", args.v_head_dim.to_string()),
            ("layer_schedule", args.layer_schedule_fingerprint()),
            ("n_routed_experts", args.n_routed_experts.to_string()),
            ("n_shared_experts", args.n_shared_experts.to_string()),
            ("num_experts_per_tok", args.num_experts_per_tok.to_string()),
            ("n_group", args.n_group.to_string()),
            ("topk_group", args.topk_group.to_string()),
            ("topk_method", args.topk_method.clone()),
            ("scoring_func", args.scoring_func.clone()),
            ("norm_topk_prob", args.norm_topk_prob.to_string()),
            (
                "routed_scaling_factor",
                format!("{:08x}", args.routed_scaling_factor.to_bits()),
            ),
            (
                "num_nextn_predict_layers",
                args.num_nextn_predict_layers.to_string(),
            ),
            (
                "quantization_config",
                format!("{:?}", args.quantization_config),
            ),
            ("quantization", format!("{:?}", args.quantization)),
            (
                "quantized_weight_configs",
                quantized_weight_configs.join(";"),
            ),
            ("split_kv_b", args.split_kv_b.to_string()),
            ("tie_word_embeddings", args.tie_word_embeddings.to_string()),
        ],
    )
}

/// One compressed MLA cache per decoder layer.
#[derive(Debug, Clone)]
pub struct Cache {
    /// Per-layer compressed latent state.
    pub layers: Vec<CompressedLatentCache>,
    /// Compressed latent state owned by checkpoint-embedded prediction layers.
    pub(crate) mtp_layers: Vec<CompressedLatentCache>,
}

impl Cache {
    pub(crate) fn new(layer_schedule: &LayerSchedule<LayerPolicy>) -> Self {
        Self {
            layers: layer_schedule
                .iter()
                .map(|_| CompressedLatentCache::new())
                .collect(),
            mtp_layers: Vec::new(),
        }
    }

    pub(crate) fn with_mtp_layers(mut self, count: usize) -> Self {
        self.mtp_layers = (0..count).map(|_| CompressedLatentCache::new()).collect();
        self
    }

    pub(crate) fn with_paged_mtp_layers(
        mut self,
        count: usize,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, Exception> {
        if count == 0 {
            return Ok(self);
        }
        let manager = self
            .layers
            .iter()
            .find_map(CompressedLatentCache::residency_manager)
            .cloned()
            .ok_or_else(|| {
                Exception::custom("DeepSeek paged MTP requires a shared cache manager")
            })?;
        let start = self.layers.len();
        self.mtp_layers = (0..count)
            .map(|index| CompressedLatentCache::new_paged(manager.clone(), start + index, rank))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self)
    }

    pub(crate) fn new_with_options(
        layer_schedule: &LayerSchedule<LayerPolicy>,
        policy: CacheResidencyPolicy,
    ) -> Result<Self, Exception> {
        Self::new_with_options_and_rank(layer_schedule, policy, None)
    }

    pub(crate) fn new_with_options_and_rank(
        layer_schedule: &LayerSchedule<LayerPolicy>,
        policy: CacheResidencyPolicy,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, Exception> {
        match policy {
            CacheResidencyPolicy::Device => Ok(Self::new(layer_schedule)),
            CacheResidencyPolicy::Paged(options) => {
                let manager = CacheResidencyManager::new(options)
                    .map_err(|error| Exception::custom(error.to_string()))?;
                Self::new_with_manager(layer_schedule, manager, rank)
            }
        }
    }

    fn new_with_manager(
        layer_schedule: &LayerSchedule<LayerPolicy>,
        manager: CacheResidencyManager,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, Exception> {
        let layers = (0..layer_schedule.len())
            .map(|layer| CompressedLatentCache::new_paged(manager.clone(), layer, rank))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            layers,
            mtp_layers: Vec::new(),
        })
    }

    /// Returns the common token offset.
    pub fn offset(&self) -> i32 {
        self.layers.first().map_or(0, CompressedLatentCache::offset)
    }

    pub(crate) fn reset(&mut self) -> Result<(), Exception> {
        if let Some(manager) = self
            .layers
            .iter()
            .chain(&self.mtp_layers)
            .find_map(CompressedLatentCache::residency_manager)
            .cloned()
        {
            manager
                .clear()
                .map_err(|error| Exception::custom(error.to_string()))?;
        }
        for cache in self.layers.iter_mut().chain(&mut self.mtp_layers) {
            cache.reset_local_after_manager_clear();
        }
        Ok(())
    }

    pub(crate) fn restore_target_checkpoint(
        &mut self,
        checkpoint: &Self,
        stream: &Stream,
    ) -> Result<(), Exception> {
        if self.layers.len() != checkpoint.layers.len() {
            return Err(Exception::custom(
                "DeepSeek target cache checkpoint has a different layer count",
            ));
        }
        for (cache, previous) in self.layers.iter_mut().zip(&checkpoint.layers) {
            cache.restore_checkpoint(previous, stream)?;
        }
        Ok(())
    }

    /// Returns aggregate compressed-cache residency observations.
    pub fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        self.layers
            .first()
            .and_then(CompressedLatentCache::residency_manager)
            .map(|manager| {
                manager
                    .report()
                    .map_err(|error| Exception::custom(error.to_string()))
            })
            .transpose()
    }

    /// Finalizes and atomically saves an immutable text prefix.
    pub fn save_prompt_cache(
        &mut self,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Exception> {
        for layer in &mut self.layers {
            layer.finalize()?;
        }
        let manager = self
            .layers
            .first()
            .and_then(CompressedLatentCache::residency_manager)
            .ok_or_else(|| {
                Exception::custom(
                    "prompt-cache persistence requires an explicitly configured paged compressed cache",
                )
            })?;
        manager
            .save_prompt_cache(destination, descriptor, prefix_token_ids, &[], options)
            .map_err(|error| Exception::custom(error.to_string()))
    }

    /// Catalogs compatible compressed prefix blocks without eager array loading.
    pub(crate) fn load_prompt_cache(
        layer_schedule: &LayerSchedule<LayerPolicy>,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        model: &PromptCacheModelIdentity,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
    ) -> Result<(Self, PromptCacheManifest), Exception> {
        let (manager, manifest) =
            open_prompt_cache(directory, expected, model, prefix_token_ids, options)
                .map_err(|error| Exception::custom(error.to_string()))?;
        Ok((
            Self::new_with_manager(
                layer_schedule,
                manager,
                model.topology.cache_rank_identity(),
            )?,
            manifest,
        ))
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// Per-head MLA reconstruction matrix used by modern `deepseek2` GGUFs.
pub struct PackedHeadProjection {
    /// Head count represented by the leading weight dimension.
    pub num_heads: i32,
    /// Optional per-weight affine encoding.
    pub affine: Option<WeightQuantization>,
    #[param]
    /// Weight shaped `[heads, output, input]` before affine packing.
    pub weight: Param<Array>,
    #[param]
    /// Affine scales.
    pub scales: Param<Option<Array>>,
    #[param]
    /// Affine biases.
    pub biases: Param<Option<Array>>,
}

impl PackedHeadProjection {
    fn new(
        num_heads: i32,
        input_dims: i32,
        output_dims: i32,
        format: WeightFormat,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let affine = format.affine();
        let packed_input = affine.map_or(input_dims, |quantization| {
            quantized_packed_dimension(input_dims, quantization.bits())
        });
        Ok(Self {
            num_heads,
            affine,
            weight: Param::<Array>::unloaded(
                &[num_heads, output_dims, packed_input],
                if affine.is_some() {
                    Dtype::Uint32
                } else {
                    Dtype::Float32
                },
                stream,
            )?,
            scales: if let Some(quantization) = affine {
                Param::<Option<Array>>::unloaded_some(
                    &[
                        num_heads,
                        output_dims,
                        input_dims / quantization.group_size(),
                    ],
                    if quantization == WeightQuantization::MxFp4 {
                        Dtype::Uint8
                    } else {
                        Dtype::Float32
                    },
                    stream,
                )?
            } else {
                Param::new(None)
            },
            biases: if let Some(quantization) = affine.filter(|q| q.has_biases()) {
                Param::<Option<Array>>::unloaded_some(
                    &[
                        num_heads,
                        output_dims,
                        input_dims / quantization.group_size(),
                    ],
                    Dtype::Float32,
                    stream,
                )?
            } else {
                Param::new(None)
            },
        })
    }

    fn forward(
        &mut self,
        input: &Array,
        transpose: bool,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let shape = input.shape();
        let routes = input.size() as i32 / input.dim(-1);
        let mut ids = Vec::with_capacity(routes as usize);
        for _ in 0..routes / self.num_heads {
            ids.extend(0..self.num_heads as u32);
        }
        let group_ids = Array::from_slice(&ids, &[routes]);
        let input = input.reshape(&[routes, input.dim(-1)], stream)?;
        let output = if let Some(affine) = self.affine {
            common::moe::packed_grouped_linear_with_transpose(
                &input,
                self.weight.as_ref(),
                self.scales.as_ref().as_ref().expect("packed head scales"),
                self.biases.as_ref().as_ref(),
                &group_ids,
                affine,
                transpose,
                stream,
            )?
        } else {
            let weight = if transpose {
                self.weight.as_ref().swap_axes(-1, -2, stream)?
            } else {
                self.weight.as_ref().clone()
            };
            grouped_matmul(&input, &weight, &group_ids, true, stream)?
        };
        let mut output_shape = shape.to_vec();
        *output_shape.last_mut().expect("head projection rank") = output.dim(-1);
        output.reshape(&output_shape, stream)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// DeepSeek Multi-head Latent Attention.
pub struct MultiHeadLatentAttention {
    /// Query head count.
    pub num_heads: i32,
    /// Non-positional width per head.
    pub qk_nope_head_dim: i32,
    /// Rotary width per head.
    pub qk_rope_head_dim: i32,
    /// Value width per head.
    pub v_head_dim: i32,
    /// Compressed latent width.
    pub kv_lora_rank: i32,
    /// Attention score scale.
    pub softmax_scale: f32,
    /// Whether to leave the nominal positional subspace unrotated.
    pub use_nope: bool,
    #[param]
    /// Direct query projection for compatible no-query-LoRA checkpoints.
    pub q_proj: Option<Linear>,
    #[param]
    /// Query LoRA down projection.
    pub q_a_proj: Option<Linear>,
    #[param]
    /// Query LoRA normalization.
    pub q_a_layernorm: Option<nn::RmsNorm>,
    #[param]
    /// Query LoRA up projection.
    pub q_b_proj: Option<Linear>,
    #[param]
    /// Combined compressed latent and shared rotary-key projection.
    pub kv_a_proj_with_mqa: Linear,
    #[param]
    /// Compressed latent normalization.
    pub kv_a_layernorm: nn::RmsNorm,
    #[param]
    /// Per-head non-positional key and value reconstruction.
    pub kv_b_proj: Option<Linear>,
    #[param]
    /// Split non-positional key reconstruction used by modern GGUFs.
    pub k_b_proj: Option<PackedHeadProjection>,
    #[param]
    /// Split value reconstruction used by modern GGUFs.
    pub v_b_proj: Option<PackedHeadProjection>,
    #[param]
    /// Attention output projection.
    pub o_proj: Linear,
    #[param]
    /// Rotary embedding applied only to the positional subspace.
    pub rope: RopeVariant,
}

impl MultiHeadLatentAttention {
    pub(crate) fn new(args: &ModelArgs, layer: i32, stream: &Stream) -> Result<Self, Exception> {
        Self::new_with_nope(args, layer, false, stream)
    }

    /// Creates MLA with an optional identity positional-subspace policy.
    pub(crate) fn new_with_nope(
        args: &ModelArgs,
        layer: i32,
        use_nope: bool,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let prefix = format!("model.layers.{layer}.self_attn");
        let format =
            |projection: &str| args.weight_format_for(&format!("{prefix}.{projection}.weight"));
        let q_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim;
        let (q_proj, q_a_proj, q_a_layernorm, q_b_proj) = match args.q_lora_rank {
            Some(rank) => (
                None,
                Some(Linear::new(
                    args.hidden_size,
                    rank,
                    false,
                    format("q_a_proj"),
                    stream,
                )?),
                Some(nn::RmsNorm::unloaded(
                    rank,
                    args.rms_norm_eps,
                    Dtype::Float32,
                    stream,
                )?),
                Some(Linear::new(
                    rank,
                    args.num_attention_heads * q_head_dim,
                    false,
                    format("q_b_proj"),
                    stream,
                )?),
            ),
            None => (
                Some(Linear::new(
                    args.hidden_size,
                    args.num_attention_heads * q_head_dim,
                    false,
                    format("q_proj"),
                    stream,
                )?),
                None,
                None,
                None,
            ),
        };
        let rope_config = args.rope_scaling.as_ref().map(YarnConfig::rope_config);
        let scale = (q_head_dim as f32).sqrt().recip()
            * args
                .rope_scaling
                .as_ref()
                .map_or(1.0, YarnConfig::attention_multiplier);
        Ok(Self {
            num_heads: args.num_attention_heads,
            qk_nope_head_dim: args.qk_nope_head_dim,
            qk_rope_head_dim: args.qk_rope_head_dim,
            v_head_dim: args.v_head_dim,
            kv_lora_rank: args.kv_lora_rank,
            softmax_scale: scale,
            use_nope,
            q_proj,
            q_a_proj,
            q_a_layernorm,
            q_b_proj,
            kv_a_proj_with_mqa: Linear::new(
                args.hidden_size,
                args.kv_lora_rank + args.qk_rope_head_dim,
                false,
                format("kv_a_proj_with_mqa"),
                stream,
            )?,
            kv_a_layernorm: nn::RmsNorm::unloaded(
                args.kv_lora_rank,
                args.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?,
            kv_b_proj: if args.split_kv_b {
                None
            } else {
                Some(Linear::new(
                    args.kv_lora_rank,
                    args.num_attention_heads * (args.qk_nope_head_dim + args.v_head_dim),
                    false,
                    format("kv_b_proj"),
                    stream,
                )?)
            },
            k_b_proj: if args.split_kv_b {
                Some(PackedHeadProjection::new(
                    args.num_attention_heads,
                    args.qk_nope_head_dim,
                    args.kv_lora_rank,
                    format("k_b_proj"),
                    stream,
                )?)
            } else {
                None
            },
            v_b_proj: if args.split_kv_b {
                Some(PackedHeadProjection::new(
                    args.num_attention_heads,
                    args.kv_lora_rank,
                    args.v_head_dim,
                    format("v_b_proj"),
                    stream,
                )?)
            } else {
                None
            },
            o_proj: Linear::new(
                args.num_attention_heads * args.v_head_dim,
                args.hidden_size,
                false,
                format("o_proj"),
                stream,
            )?,
            rope: initialize_rope(
                args.qk_rope_head_dim,
                args.rope_theta,
                false,
                &rope_config,
                args.max_position_embeddings,
                stream,
            )?,
        })
    }

    fn project_queries(
        &mut self,
        x: &Array,
        stream: &Stream,
        prefix: &str,
        observer: &mut ObserverOption<'_>,
    ) -> Result<Array, Exception> {
        if let Some(q_proj) = &mut self.q_proj {
            let query = q_proj.forward(x, stream)?;
            observe_activation(observer, prefix, "q_proj", &query)?;
            Ok(query)
        } else {
            let q = self
                .q_a_proj
                .as_mut()
                .expect("query LoRA down projection")
                .forward(x, stream)?;
            observe_activation(observer, prefix, "q_a_proj", &q)?;
            let q = self
                .q_a_layernorm
                .as_mut()
                .expect("query LoRA norm")
                .forward(&q, stream)?;
            observe_activation(observer, prefix, "q_a_layernorm", &q)?;
            let q = self
                .q_b_proj
                .as_mut()
                .expect("query LoRA up projection")
                .forward(&q, stream)?;
            observe_activation(observer, prefix, "q_b_proj", &q)?;
            Ok(q)
        }
    }

    fn reconstruct_keys_values(
        &mut self,
        latent: &Array,
        rotary_key: &Array,
        stream: &Stream,
        prefix: &str,
        observer: &mut ObserverOption<'_>,
    ) -> Result<(Array, Array), Exception> {
        let batch = latent.dim(0);
        let sequence = latent.dim(1);
        let (k_nope, values) = if let Some(kv_b_proj) = &mut self.kv_b_proj {
            let kv_projected = kv_b_proj.forward(latent, stream)?;
            observe_activation(observer, prefix, "kv_b_proj", &kv_projected)?;
            let kv = kv_projected.reshape(
                &[
                    batch,
                    sequence,
                    self.num_heads,
                    self.qk_nope_head_dim + self.v_head_dim,
                ],
                stream,
            )?;
            (
                kv.try_index_device((.., .., .., ..self.qk_nope_head_dim), stream)?,
                kv.try_index_device((.., .., .., self.qk_nope_head_dim..), stream)?
                    .transpose_axes(&[0, 2, 1, 3], stream)?,
            )
        } else {
            let latent_heads = broadcast_to(
                latent.try_index_device((.., .., NewAxis, ..), stream)?,
                &[batch, sequence, self.num_heads, self.kv_lora_rank],
                stream,
            )?;
            let k_nope = self
                .k_b_proj
                .as_mut()
                .expect("split MLA key projection")
                .forward(&latent_heads, false, stream)?;
            observe_activation(observer, prefix, "k_b_proj", &k_nope)?;
            let values = self
                .v_b_proj
                .as_mut()
                .expect("split MLA value projection")
                .forward(&latent_heads, true, stream)?
                .transpose_axes(&[0, 2, 1, 3], stream)?;
            observe_activation(observer, prefix, "v_b_proj", &values)?;
            (k_nope, values)
        };
        observe_activation(observer, prefix, "keys_nope", &k_nope)?;
        observe_activation(observer, prefix, "values", &values)?;
        let keys = concatenate_axis(
            &[
                k_nope,
                broadcast_to(
                    rotary_key.try_index_device((.., .., NewAxis, ..), stream)?,
                    &[batch, sequence, self.num_heads, self.qk_rope_head_dim],
                    stream,
                )?,
            ],
            -1,
            stream,
        )?
        .transpose_axes(&[0, 2, 1, 3], stream)?;
        observe_activation(observer, prefix, "keys", &keys)?;
        Ok((keys, values))
    }

    #[allow(clippy::unnecessary_unwrap)]
    fn forward_impl(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut CompressedLatentCache>,
        stream: &Stream,
        prefix: &str,
        observer: &mut ObserverOption<'_>,
    ) -> Result<Array, Exception> {
        observe_activation(observer, prefix, "input", x)?;
        let b = x.dim(0);
        let l = x.dim(1);
        let q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim;
        let q = self
            .project_queries(x, stream, prefix, observer)?
            .reshape(&[b, l, self.num_heads, q_head_dim], stream)?;
        observe_activation(observer, prefix, "queries", &q)?;
        let q_nope = q.try_index_device((.., .., .., ..self.qk_nope_head_dim), stream)?;
        observe_activation(observer, prefix, "queries_nope", &q_nope)?;
        let q_pe = q
            .try_index_device((.., .., .., self.qk_nope_head_dim..), stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
        observe_activation(observer, prefix, "queries_rope_input", &q_pe)?;

        let kv = self.kv_a_proj_with_mqa.forward(x, stream)?;
        observe_activation(observer, prefix, "kv_a_proj_with_mqa", &kv)?;
        let latent_raw = kv.try_index_device((.., .., ..self.kv_lora_rank), stream)?;
        observe_activation(observer, prefix, "latent_raw", &latent_raw)?;
        let latent = latent_raw;
        let latent = self.kv_a_layernorm.forward(&latent, stream)?;
        observe_activation(observer, prefix, "kv_a_layernorm", &latent)?;
        let k_pe = kv
            .try_index_device((.., .., self.kv_lora_rank..), stream)?
            .try_index_device((.., NewAxis, .., ..), stream)?;
        observe_activation(observer, prefix, "keys_rope_input", &k_pe)?;

        let offset = cache.as_ref().map_or(0, |cache| cache.offset());
        let q_pe = if self.use_nope {
            q_pe
        } else {
            self.rope.forward(
                nn::RopeInputBuilder::new(&q_pe).offset(offset).build()?,
                stream,
            )?
        };
        observe_activation(observer, prefix, "queries_rope", &q_pe)?;
        let k_pe = if self.use_nope {
            k_pe
        } else {
            self.rope.forward(
                nn::RopeInputBuilder::new(&k_pe).offset(offset).build()?,
                stream,
            )?
        };
        observe_activation(observer, prefix, "keys_rope", &k_pe)?;
        let new_k_pe = k_pe.try_index_device((.., 0, .., ..), stream)?;

        let mut paged_block_ids = None;
        let mut paged_tail = None;
        let mut paged_manager = None;
        let mut paged_global_layer = None;
        let (cached_latent, cached_k_pe) = if let Some(cache) = cache {
            let updated = cache.update_and_fetch(latent.clone(), new_k_pe.clone(), stream)?;
            if cache.is_paged() {
                if observer.is_some() {
                    return Err(Exception::custom(
                        "attention-probability inspection is unavailable for paged compressed-latent attention",
                    ));
                }
                paged_block_ids = cache.paged_block_ids()?;
                paged_tail = cache.paged_tail_block();
                paged_manager = cache.residency_manager().cloned();
                paged_global_layer = cache.paged_global_layer();
            }
            updated
        } else {
            (latent.clone(), new_k_pe.clone())
        };
        observe_activation(observer, prefix, "latent_cache", &cached_latent)?;
        observe_activation(observer, prefix, "rotary_key_cache", &cached_k_pe)?;
        if let Some(mask) = mask {
            observe_activation(observer, prefix, "attention_mask", mask)?;
        }

        // Every multi-token prefill reconstructs K/V transiently and stays on
        // MLX's fused attention path. Initial prefill uses the compact causal
        // mode; cached chunks use the explicit offset-aware mask constructed by
        // `TextModel`. Persistent state remains compressed and head-independent.
        let attended = if let Some(block_ids) = paged_block_ids {
            let queries = concatenate_axis(
                &[q_nope, q_pe.transpose_axes(&[0, 2, 1, 3], stream)?],
                -1,
                stream,
            )?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
            let manager = paged_manager.expect("paged cache manager captured with block ids");
            let mut accumulator = BlockwiseAttentionAccumulator::new(
                &queries,
                self.softmax_scale,
                mask,
                offset as i64,
                None,
                0,
                None,
                offset as i64 + l as i64,
                stream,
            )?;
            let mut reconstructed_scratch = 0u64;
            let mut scanned_blocks = 0u64;
            let mut scanned_bytes = 0u64;
            let mut blocks = manager
                .prefetch_blocks(block_ids, stream)
                .map_err(|error| Exception::custom(error.to_string()))?;
            while let Some(lease) = blocks
                .next_block()
                .map_err(|error| Exception::custom(error.to_string()))?
            {
                let id = lease.id();
                let (latent, rotary_key) = match lease.arrays() {
                    CacheBlockArrays::CompressedLatentRotary { latent, rotary_key } => {
                        (latent.clone(), rotary_key.clone())
                    }
                    _ => {
                        return Err(Exception::custom(
                            "paged compressed cache found an incompatible block representation",
                        ))
                    }
                };
                let mut no_observer = None;
                let (keys, values) = self.reconstruct_keys_values(
                    &latent,
                    &rotary_key,
                    stream,
                    prefix,
                    &mut no_observer,
                )?;
                reconstructed_scratch =
                    reconstructed_scratch.max(keys.nbytes() as u64 + values.nbytes() as u64);
                let block = KeyValueAttentionBlock::unleased(id.start, id.end, keys, values);
                scanned_blocks += 1;
                scanned_bytes += lease.bytes();
                accumulator.accumulate(&block, stream)?;
                accumulator.submit()?;
                drop(lease);
            }
            if let Some(block) = paged_tail {
                let mut no_observer = None;
                let (keys, values) = self.reconstruct_keys_values(
                    &block.latent,
                    &block.rotary_key,
                    stream,
                    prefix,
                    &mut no_observer,
                )?;
                reconstructed_scratch =
                    reconstructed_scratch.max(keys.nbytes() as u64 + values.nbytes() as u64);
                let kv_block =
                    KeyValueAttentionBlock::unleased(block.start, block.end, keys, values);
                scanned_blocks += 1;
                scanned_bytes += block.bytes;
                accumulator.accumulate(&kv_block, stream)?;
            }
            let output = accumulator.finish(stream)?;
            eval([&output])?;
            manager
                .record_attention_scan(
                    paged_global_layer.expect("paged cache layer captured with block ids"),
                    l > 1,
                    scanned_blocks,
                    scanned_bytes,
                    reconstructed_scratch,
                )
                .map_err(|error| Exception::custom(error.to_string()))?;
            output.transpose_axes(&[0, 2, 1, 3], stream)?
        } else if l > 1 {
            let (keys, values) = self.reconstruct_keys_values(
                &cached_latent,
                &cached_k_pe,
                stream,
                prefix,
                observer,
            )?;
            let queries = concatenate_axis(
                &[q_nope, q_pe.transpose_axes(&[0, 2, 1, 3], stream)?],
                -1,
                stream,
            )?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
            observe_activation(observer, prefix, "queries_combined", &queries)?;
            if observer.is_some() {
                let generated_causal_mask = if mask.is_none() {
                    Some(create_causal_mask(l, Some(offset), None, None, stream)?)
                } else {
                    None
                };
                if let Some(mask) = generated_causal_mask.as_ref() {
                    observe_activation(observer, prefix, "attention_mask", mask)?;
                }
                let probability_mask = mask.or(generated_causal_mask.as_ref());
                let probabilities = common::attention::attention_probabilities(
                    &queries,
                    &keys,
                    self.softmax_scale,
                    probability_mask,
                    stream,
                )?;
                observe_activation(observer, prefix, "attention_probs", &probabilities)?;
            }
            safemlx::fast::scaled_dot_product_attention(
                queries,
                keys,
                values,
                self.softmax_scale,
                Some(match mask {
                    Some(mask) => ScaledDotProductAttentionMask::Array(mask),
                    None => ScaledDotProductAttentionMask::Causal,
                }),
                None,
                stream,
            )?
            .transpose_axes(&[0, 2, 1, 3], stream)?
        } else {
            if self.kv_b_proj.is_none() {
                let q_latent = self
                    .k_b_proj
                    .as_mut()
                    .expect("split MLA key projection")
                    .forward(&q_nope, true, stream)?;
                observe_activation(observer, prefix, "queries_latent", &q_latent)?;
                let mut scores = einsum("blhc,btc->bhlt", [&q_latent, &cached_latent], stream)?
                    .add(
                        einsum("bhlr,btr->bhlt", [&q_pe, &cached_k_pe], stream)?,
                        stream,
                    )?
                    .multiply(Array::from_f32(self.softmax_scale), stream)?;
                if let Some(mask) = mask {
                    if mask.dtype() == Dtype::Bool {
                        scores = r#where(
                            mask,
                            &scores,
                            Array::from_f32(scores.dtype().finfo_min()? as f32),
                            stream,
                        )?;
                    } else {
                        scores = scores.add(mask, stream)?;
                    }
                }
                observe_activation(observer, prefix, "attention_scores", &scores)?;
                let probabilities = softmax_axis(scores, -1, true, stream)?;
                observe_activation(observer, prefix, "attention_probs", &probabilities)?;
                let context = einsum("bhlt,btc->blhc", [&probabilities, &cached_latent], stream)?;
                observe_activation(observer, prefix, "latent_context", &context)?;
                let values = self
                    .v_b_proj
                    .as_mut()
                    .expect("split MLA value projection")
                    .forward(&context, true, stream)?;
                observe_activation(observer, prefix, "v_b_proj", &values)?;
                values
            } else {
                let kv_b_proj = self.kv_b_proj.as_mut().expect("fused MLA projection");
                let fp8_group_ids = kv_b_proj.weight_scale_inv.as_ref().as_ref().map(|_| {
                    let mut ids = Vec::with_capacity((b * l * self.num_heads) as usize);
                    for _ in 0..b * l {
                        ids.extend(0..self.num_heads as u32);
                    }
                    Array::from_slice(&ids, &[b * l * self.num_heads])
                });

                let mut absorbed_weight = None;
                let q_latent = if let (Some(scale), Some(group_ids)) = (
                    kv_b_proj.weight_scale_inv.as_ref().as_ref(),
                    fp8_group_ids.as_ref(),
                ) {
                    common::fp8::segmented_transposed_linear(
                        &q_nope
                            .reshape(&[b * l * self.num_heads, self.qk_nope_head_dim], stream)?,
                        kv_b_proj.weight.as_ref(),
                        scale,
                        group_ids,
                        self.qk_nope_head_dim + self.v_head_dim,
                        0,
                        stream,
                    )?
                    .reshape(&[b, l, self.num_heads, self.kv_lora_rank], stream)?
                } else {
                    let weight = kv_b_proj.dequantized_weight(stream)?.reshape(
                        &[
                            self.num_heads,
                            self.qk_nope_head_dim + self.v_head_dim,
                            self.kv_lora_rank,
                        ],
                        stream,
                    )?;
                    let wk = weight.try_index_device((.., ..self.qk_nope_head_dim, ..), stream)?;
                    let q_latent = einsum("blhd,hdc->blhc", [&q_nope, &wk], stream)?;
                    absorbed_weight = Some(weight);
                    q_latent
                };
                observe_activation(observer, prefix, "queries_latent", &q_latent)?;
                let mut scores = einsum("blhc,btc->bhlt", [&q_latent, &cached_latent], stream)?
                    .add(
                        einsum("bhlr,btr->bhlt", [&q_pe, &cached_k_pe], stream)?,
                        stream,
                    )?
                    .multiply(Array::from_f32(self.softmax_scale), stream)?;
                if let Some(mask) = mask {
                    if mask.dtype() == Dtype::Bool {
                        scores = r#where(
                            mask,
                            &scores,
                            Array::from_f32(scores.dtype().finfo_min()? as f32),
                            stream,
                        )?;
                    } else {
                        scores = scores.add(mask, stream)?;
                    }
                }
                observe_activation(observer, prefix, "attention_scores", &scores)?;
                let probabilities = softmax_axis(scores, -1, true, stream)?;
                observe_activation(observer, prefix, "attention_probs", &probabilities)?;
                let context = einsum("bhlt,btc->blhc", [&probabilities, &cached_latent], stream)?;
                observe_activation(observer, prefix, "latent_context", &context)?;
                if let (Some(scale), Some(group_ids)) = (
                    kv_b_proj.weight_scale_inv.as_ref().as_ref(),
                    fp8_group_ids.as_ref(),
                ) {
                    common::fp8::segmented_linear(
                        &context.reshape(&[b * l * self.num_heads, self.kv_lora_rank], stream)?,
                        kv_b_proj.weight.as_ref(),
                        scale,
                        group_ids,
                        self.qk_nope_head_dim + self.v_head_dim,
                        self.qk_nope_head_dim,
                        self.v_head_dim,
                        stream,
                    )?
                    .reshape(&[b, l, self.num_heads, self.v_head_dim], stream)?
                } else {
                    let weight = absorbed_weight.expect("dense absorbed MLA weight initialized");
                    let wv = weight.try_index_device((.., self.qk_nope_head_dim.., ..), stream)?;
                    einsum("blhc,hvc->blhv", [&context, &wv], stream)?
                }
            }
        };
        observe_activation(observer, prefix, "attention", &attended)?;
        let attended = attended.reshape(&[b, l, self.num_heads * self.v_head_dim], stream)?;
        observe_activation(observer, prefix, "o_proj_input", &attended)?;
        let output = self.o_proj.forward(&attended, stream)?;
        observe_activation(observer, prefix, "o_proj", &output)?;
        Ok(output)
    }

    /// Runs shared MLA with optional activation observation.
    pub(crate) fn forward_shared(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut CompressedLatentCache>,
        stream: &Stream,
        prefix: &str,
        observer: &mut Option<&mut dyn ActivationObserver>,
    ) -> Result<Array, Exception> {
        self.forward_impl(x, mask, cache, stream, prefix, observer)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// Standard DeepSeek SwiGLU MLP.
pub struct Mlp {
    #[param]
    /// Gating projection.
    pub gate_proj: Linear,
    #[param]
    /// Value projection.
    pub up_proj: Linear,
    #[param]
    /// Output projection.
    pub down_proj: Linear,
}

impl Mlp {
    fn new(
        args: &ModelArgs,
        prefix: &str,
        intermediate_size: i32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: Linear::new(
                args.hidden_size,
                intermediate_size,
                false,
                args.weight_format_for(&format!("{prefix}.gate_proj.weight")),
                stream,
            )?,
            up_proj: Linear::new(
                args.hidden_size,
                intermediate_size,
                false,
                args.weight_format_for(&format!("{prefix}.up_proj.weight")),
                stream,
            )?,
            down_proj: Linear::new(
                intermediate_size,
                args.hidden_size,
                false,
                args.weight_format_for(&format!("{prefix}.down_proj.weight")),
                stream,
            )?,
        })
    }

    fn forward_impl(
        &mut self,
        x: &Array,
        stream: &Stream,
        prefix: &str,
        observer: &mut ObserverOption<'_>,
    ) -> Result<Array, Exception> {
        observe_activation(observer, prefix, "input", x)?;
        let gate = self.gate_proj.forward(x, stream)?;
        observe_activation(observer, prefix, "gate_proj", &gate)?;
        let gate = silu(gate, stream)?;
        observe_activation(observer, prefix, "gate_activation", &gate)?;
        let up = self.up_proj.forward(x, stream)?;
        observe_activation(observer, prefix, "up_proj", &up)?;
        let gated = gate.multiply(up, stream)?;
        observe_activation(observer, prefix, "gated", &gated)?;
        let output = self.down_proj.forward(&gated, stream)?;
        observe_activation(observer, prefix, "down_proj", &output)?;
        observe_activation(observer, prefix, "output", &output)?;
        Ok(output)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// Packed runtime bank for checkpoint-split DeepSeek routed experts.
pub struct RoutedExperts {
    /// Expert count.
    pub num_experts: i32,
    /// Expert intermediate width.
    pub intermediate_size: i32,
    /// Native block-FP8 storage marker.
    pub use_fp8: bool,
    /// Optional affine encoding for the gate projection.
    pub gate_affine: Option<WeightQuantization>,
    /// Optional affine encoding for the up projection.
    pub up_affine: Option<WeightQuantization>,
    /// Optional affine encoding for the down projection.
    pub down_affine: Option<WeightQuantization>,
    /// Optional checkpoint-native IQ encoding for the gate projection.
    pub gate_iquant: Option<WeightQuantization>,
    /// Optional checkpoint-native IQ encoding for the up projection.
    pub up_iquant: Option<WeightQuantization>,
    /// Optional checkpoint-native IQ encoding for the down projection.
    pub down_iquant: Option<WeightQuantization>,
    #[param]
    /// Packed gate weights `[experts, intermediate, hidden]`.
    pub gate_proj: Param<Option<Array>>,
    #[param]
    /// Packed gate inverse scales.
    pub gate_proj_scale_inv: Param<Option<Array>>,
    #[param]
    /// Packed gate affine scales.
    pub gate_proj_scales: Param<Option<Array>>,
    #[param]
    /// Packed gate affine biases.
    pub gate_proj_biases: Param<Option<Array>>,
    #[param]
    /// Packed up weights `[experts, intermediate, hidden]`.
    pub up_proj: Param<Option<Array>>,
    #[param]
    /// Packed up inverse scales.
    pub up_proj_scale_inv: Param<Option<Array>>,
    #[param]
    /// Packed up affine scales.
    pub up_proj_scales: Param<Option<Array>>,
    #[param]
    /// Packed up affine biases.
    pub up_proj_biases: Param<Option<Array>>,
    #[param]
    /// Packed down weights `[experts, hidden, intermediate]`.
    pub down_proj: Param<Option<Array>>,
    #[param]
    /// Packed down inverse scales.
    pub down_proj_scale_inv: Param<Option<Array>>,
    #[param]
    /// Packed down affine scales.
    pub down_proj_scales: Param<Option<Array>>,
    #[param]
    /// Packed down affine biases.
    pub down_proj_biases: Param<Option<Array>>,
}

impl RoutedExperts {
    fn new(args: &ModelArgs, layer: i32) -> Result<Self, Exception> {
        let prefix = format!("model.layers.{layer}.mlp.experts");
        let split = |format: WeightFormat| match format.quantization() {
            Some(iq @ WeightQuantization::GgufIQuant { .. }) => (None, Some(iq)),
            affine => (affine, None),
        };
        let (gate_affine, gate_iquant) =
            split(args.weight_format_for(&format!("{prefix}.gate_proj")));
        let (up_affine, up_iquant) = split(args.weight_format_for(&format!("{prefix}.up_proj")));
        let (down_affine, down_iquant) =
            split(args.weight_format_for(&format!("{prefix}.down_proj")));
        Ok(Self {
            num_experts: args.n_routed_experts,
            intermediate_size: args.moe_intermediate_size,
            use_fp8: args.native_fp8_config().is_some(),
            gate_affine,
            up_affine,
            down_affine,
            gate_iquant,
            up_iquant,
            down_iquant,
            gate_proj: Param::new(None),
            gate_proj_scale_inv: Param::new(None),
            gate_proj_scales: Param::new(None),
            gate_proj_biases: Param::new(None),
            up_proj: Param::new(None),
            up_proj_scale_inv: Param::new(None),
            up_proj_scales: Param::new(None),
            up_proj_biases: Param::new(None),
            down_proj: Param::new(None),
            down_proj_scale_inv: Param::new(None),
            down_proj_scales: Param::new(None),
            down_proj_biases: Param::new(None),
        })
    }

    fn initialize_unloaded_banks(
        &mut self,
        args: &ModelArgs,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let expert_weight = |output: i32,
                             input: i32,
                             affine: Option<WeightQuantization>,
                             iquant: Option<WeightQuantization>|
         -> Result<Param<Option<Array>>, Exception> {
            let (packed_input, dtype) = if let Some(iquant) = iquant {
                let (ggml_type, _) = iquant.gguf_iquant().expect("IQ expert format");
                let (block_values, block_bytes) = ggml_type
                    .block_and_bytes()
                    .expect("canonical IQ block geometry");
                (
                    input / block_values as i32 * block_bytes as i32,
                    Dtype::Uint8,
                )
            } else if let Some(quantization) = affine {
                (
                    quantized_packed_dimension(input, quantization.bits()),
                    Dtype::Uint32,
                )
            } else if args.native_fp8_config().is_some() {
                (input, Dtype::Uint8)
            } else {
                (input, Dtype::Float32)
            };
            Param::<Option<Array>>::unloaded_some(
                &[args.n_routed_experts, output, packed_input],
                dtype,
                stream,
            )
        };
        let fp8_scale = |output: i32, input: i32| {
            if args.native_fp8_config().is_some() {
                Param::<Option<Array>>::unloaded_some(
                    &[
                        args.n_routed_experts,
                        (output + 127) / 128,
                        (input + 127) / 128,
                    ],
                    Dtype::Float32,
                    stream,
                )
            } else {
                Ok(Param::new(None))
            }
        };
        let affine_component = |output: i32,
                                input: i32,
                                affine: Option<WeightQuantization>,
                                biases: bool|
         -> Result<Param<Option<Array>>, Exception> {
            if let Some(quantization) =
                affine.filter(|quantization| !biases || quantization.has_biases())
            {
                Param::<Option<Array>>::unloaded_some(
                    &[
                        args.n_routed_experts,
                        output,
                        input / quantization.group_size(),
                    ],
                    Dtype::Float32,
                    stream,
                )
            } else {
                Ok(Param::new(None))
            }
        };
        self.gate_proj = expert_weight(
            args.moe_intermediate_size,
            args.hidden_size,
            self.gate_affine,
            self.gate_iquant,
        )?;
        self.gate_proj_scale_inv = fp8_scale(args.moe_intermediate_size, args.hidden_size)?;
        self.gate_proj_scales = affine_component(
            args.moe_intermediate_size,
            args.hidden_size,
            self.gate_affine,
            false,
        )?;
        self.gate_proj_biases = affine_component(
            args.moe_intermediate_size,
            args.hidden_size,
            self.gate_affine,
            true,
        )?;
        self.up_proj = expert_weight(
            args.moe_intermediate_size,
            args.hidden_size,
            self.up_affine,
            self.up_iquant,
        )?;
        self.up_proj_scale_inv = fp8_scale(args.moe_intermediate_size, args.hidden_size)?;
        self.up_proj_scales = affine_component(
            args.moe_intermediate_size,
            args.hidden_size,
            self.up_affine,
            false,
        )?;
        self.up_proj_biases = affine_component(
            args.moe_intermediate_size,
            args.hidden_size,
            self.up_affine,
            true,
        )?;
        self.down_proj = expert_weight(
            args.hidden_size,
            args.moe_intermediate_size,
            self.down_affine,
            self.down_iquant,
        )?;
        self.down_proj_scale_inv = fp8_scale(args.hidden_size, args.moe_intermediate_size)?;
        self.down_proj_scales = affine_component(
            args.hidden_size,
            args.moe_intermediate_size,
            self.down_affine,
            false,
        )?;
        self.down_proj_biases = affine_component(
            args.hidden_size,
            args.moe_intermediate_size,
            self.down_affine,
            true,
        )?;
        Ok(())
    }

    /// Creates an unloaded compact bank preserving the layer's checkpoint format.
    pub(crate) fn new_compact(
        args: &ModelArgs,
        layer: i32,
        num_experts: i32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Self::new_compact_with_width(args, layer, num_experts, args.moe_intermediate_size, stream)
    }

    /// Creates an unloaded compact bank with a rank-local expert width.
    pub(crate) fn new_compact_with_width(
        args: &ModelArgs,
        layer: i32,
        num_experts: i32,
        intermediate_size: i32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let mut compact_args = args.clone();
        compact_args.n_routed_experts = num_experts;
        compact_args.moe_intermediate_size = intermediate_size;
        let mut bank = Self::new(&compact_args, layer)?;
        bank.initialize_unloaded_banks(&compact_args, stream)?;
        Ok(bank)
    }

    #[allow(clippy::too_many_arguments)]
    fn projection(
        input: &Array,
        weight: &Array,
        fp8_scale: Option<&Array>,
        affine_scales: Option<&Array>,
        affine_biases: Option<&Array>,
        affine: Option<WeightQuantization>,
        iquant: Option<WeightQuantization>,
        logical_shape: &[i32],
        group_ids: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if let Some(iquant) = iquant {
            let (ggml_type, endian) = iquant.gguf_iquant().expect("IQ expert format");
            let native = NativeQuantizedTensor::from_iq_array(
                weight.clone(),
                logical_shape,
                ggml_type,
                endian,
            )?;
            native_grouped_linear(input, &native, group_ids, stream)
        } else if let Some(affine) = affine {
            common::moe::packed_grouped_linear(
                input,
                weight,
                affine_scales.expect("affine routed-expert scales loaded"),
                affine_biases,
                group_ids,
                affine,
                stream,
            )
        } else if let Some(scale) = fp8_scale {
            common::fp8::grouped_linear(input, weight, scale, group_ids, stream)
        } else {
            grouped_matmul(
                input,
                &weight.swap_axes(-1, -2, stream)?,
                group_ids,
                true,
                stream,
            )
        }
    }

    fn forward_impl(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
        stream: &Stream,
        prefix: &str,
        observer: &mut ObserverOption<'_>,
    ) -> Result<Array, Exception> {
        let num_tokens = hidden_states.dim(0);
        observe_activation(observer, prefix, "input", hidden_states)?;
        let plan = topk_route_plan(top_k_index, self.num_experts, stream)?;
        let hidden = gather_grouped_rows(hidden_states, &plan, stream)?;
        observe_activation(observer, prefix, "expert_major_input", &hidden)?;
        let gate = Self::projection(
            &hidden,
            self.gate_proj
                .as_ref()
                .as_ref()
                .expect("routed gate expert bank loaded"),
            self.gate_proj_scale_inv.as_ref().as_ref(),
            self.gate_proj_scales.as_ref().as_ref(),
            self.gate_proj_biases.as_ref().as_ref(),
            self.gate_affine,
            self.gate_iquant,
            &[
                self.num_experts,
                self.intermediate_size,
                hidden_states.dim(-1),
            ],
            &plan.sorted_group_ids,
            stream,
        )?;
        observe_activation(observer, prefix, "gate_proj", &gate)?;
        let up = Self::projection(
            &hidden,
            self.up_proj
                .as_ref()
                .as_ref()
                .expect("routed up expert bank loaded"),
            self.up_proj_scale_inv.as_ref().as_ref(),
            self.up_proj_scales.as_ref().as_ref(),
            self.up_proj_biases.as_ref().as_ref(),
            self.up_affine,
            self.up_iquant,
            &[
                self.num_experts,
                self.intermediate_size,
                hidden_states.dim(-1),
            ],
            &plan.sorted_group_ids,
            stream,
        )?;
        observe_activation(observer, prefix, "up_proj", &up)?;
        let activated = silu(gate, stream)?.multiply(up, stream)?;
        observe_activation(observer, prefix, "activated", &activated)?;
        let output = Self::projection(
            &activated,
            self.down_proj
                .as_ref()
                .as_ref()
                .expect("routed down expert bank loaded"),
            self.down_proj_scale_inv.as_ref().as_ref(),
            self.down_proj_scales.as_ref().as_ref(),
            self.down_proj_biases.as_ref().as_ref(),
            self.down_affine,
            self.down_iquant,
            &[
                self.num_experts,
                hidden_states.dim(-1),
                self.intermediate_size,
            ],
            &plan.sorted_group_ids,
            stream,
        )?;
        observe_activation(observer, prefix, "down_proj", &output)?;
        let output = weighted_route_sum(output, top_k_weights, &plan, num_tokens, stream)?;
        observe_activation(observer, prefix, "output", &output)?;
        Ok(output)
    }

    /// Executes a compact bank-local route table and reduces it to one output
    /// row per compact input row. This is the adapter entry point used by the
    /// architecture-independent expert-parallel dispatcher.
    pub fn forward_local(
        &mut self,
        hidden_states: &Array,
        local_expert_ids: &Array,
        route_weights: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let mut observer = None;
        self.forward_impl(
            hidden_states,
            local_expert_ids,
            route_weights,
            stream,
            "",
            &mut observer,
        )
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// DeepSeek MoE block containing routed and shared experts.
pub struct Moe {
    #[param]
    /// Exact noaux grouped router.
    pub gate: TopKRouter,
    #[param]
    /// Packed routed expert bank.
    pub experts: RoutedExperts,
    #[param]
    /// Shared expert MLP.
    pub shared_experts: Mlp,
}

impl Moe {
    fn new(args: &ModelArgs, layer: i32, stream: &Stream) -> Result<Self, Exception> {
        Self::new_with_widths(
            args,
            layer,
            args.moe_intermediate_size,
            args.moe_intermediate_size * args.n_shared_experts,
            false,
            stream,
        )
    }

    fn new_with_widths(
        args: &ModelArgs,
        layer: i32,
        routed_intermediate: i32,
        shared_intermediate: i32,
        initialize_experts: bool,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let mut expert_args = args.clone();
        expert_args.moe_intermediate_size = routed_intermediate;
        let mut experts = RoutedExperts::new(&expert_args, layer)?;
        if initialize_experts {
            experts.initialize_unloaded_banks(&expert_args, stream)?;
        }
        Ok(Self {
            gate: TopKRouter::new(
                TopKRouterConfig {
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
            experts,
            shared_experts: Mlp::new(
                args,
                &format!("model.layers.{layer}.mlp.shared_experts"),
                shared_intermediate,
                stream,
            )?,
        })
    }

    fn forward_impl(
        &mut self,
        x: &Array,
        stream: &Stream,
        prefix: &str,
        observer: &mut ObserverOption<'_>,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let flat = x.reshape(&[-1, x.dim(-1)], stream)?;
        observe_activation(observer, prefix, "input_flat", &flat)?;
        let (indices, selected_scores, weights) = if let Some(observer) = observer.as_deref_mut() {
            let gate_prefix = activation_name(prefix, "gate");
            let routing = self
                .gate
                .forward_with_observer(&flat, stream, &gate_prefix, observer)?;
            (routing.indices, Some(routing.scores), routing.weights)
        } else {
            let (indices, weights) = self.gate.forward(&flat, stream)?;
            (indices, None, weights)
        };
        let experts_prefix = if observer.is_some() {
            activation_name(prefix, "experts")
        } else {
            String::new()
        };
        let routed = self.experts.forward_impl(
            &flat,
            &indices,
            &weights,
            stream,
            &experts_prefix,
            observer,
        )?;
        observe_activation(observer, prefix, "routed_expert_output", &routed)?;
        let shared_prefix = if observer.is_some() {
            activation_name(prefix, "shared_experts")
        } else {
            String::new()
        };
        let shared = self
            .shared_experts
            .forward_impl(&flat, stream, &shared_prefix, observer)?;
        observe_activation(observer, prefix, "shared_expert_output", &shared)?;
        let combined = routed.add(&shared, stream)?;
        observe_activation(observer, prefix, "combined_flat", &combined)?;
        if let Some(observer) = observer.as_deref_mut() {
            observer.observe_moe_routing(MoeRoutingObservation {
                prefix,
                selected_experts: &indices,
                selected_scores: selected_scores
                    .as_ref()
                    .expect("observed routing scores initialized"),
                routing_weights: &weights,
                routed_output: &routed,
                local_routed_output: None,
                reduced_routed_output: Some(&routed),
                shared_output: Some(&shared),
                combined_output: Some(&combined),
                num_experts: self.gate.num_experts,
            })?;
        }
        let output = combined.reshape(shape, stream)?;
        observe_activation(observer, prefix, "output", &output)?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_expert_parallel(
        &mut self,
        x: &Array,
        assignment: &crate::architectures::distributed::expert::ExpertAssignment,
        group: &safemlx::distributed::Group,
        statistics: &mut crate::architectures::distributed::expert::RoutingStatistics,
        prefix: &str,
        mut observer: ObserverOption<'_>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let flat = x.reshape(&[-1, x.dim(-1)], stream)?;
        observe_activation(&mut observer, prefix, "input_flat", &flat)?;
        crate::architectures::distributed::expert::materialize_timing_phase([&flat])?;
        let moe_started = std::time::Instant::now();
        let previous_moe_time = statistics.total_time;
        let router_started = std::time::Instant::now();
        let (indices, selected_scores, weights) = if let Some(observer) = observer.as_deref_mut() {
            let routing = self.gate.forward_with_observer(
                &flat,
                stream,
                &activation_name(prefix, "gate"),
                observer,
            )?;
            (routing.indices, Some(routing.scores), routing.weights)
        } else {
            let (indices, weights) = self.gate.forward(&flat, stream)?;
            (indices, None, weights)
        };
        let mut router_outputs = vec![&indices, &weights];
        if let Some(scores) = selected_scores.as_ref() {
            router_outputs.push(scores);
        }
        crate::architectures::distributed::expert::materialize_timing_phase(router_outputs)?;
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
        observe_activation(
            &mut observer,
            prefix,
            "routed_expert_local_output",
            &returned.local_output,
        )?;
        observe_activation(
            &mut observer,
            prefix,
            "routed_expert_reduced_output",
            &returned.reduced_output,
        )?;
        // Shared experts are replicated and deliberately added after the
        // routed all-sum so their contribution is applied exactly once.
        let shared_started = std::time::Instant::now();
        let shared = self.shared_experts.forward_impl(
            &flat,
            stream,
            &activation_name(prefix, "shared_experts"),
            &mut observer,
        )?;
        crate::architectures::distributed::expert::materialize_timing_phase([&shared])?;
        statistics.shared_expert_time += shared_started.elapsed();
        observe_activation(&mut observer, prefix, "shared_expert_output", &shared)?;
        let combined = returned.reduced_output.add(&shared, stream)?;
        observe_activation(&mut observer, prefix, "combined_flat", &combined)?;
        if let Some(observer) = observer {
            observer.observe_moe_routing(MoeRoutingObservation {
                prefix,
                selected_experts: &indices,
                selected_scores: selected_scores
                    .as_ref()
                    .expect("observed EP routing scores initialized"),
                routing_weights: &weights,
                routed_output: &returned.reduced_output,
                local_routed_output: Some(&returned.local_output),
                reduced_routed_output: Some(&returned.reduced_output),
                shared_output: Some(&shared),
                combined_output: Some(&combined),
                num_experts: self.gate.num_experts,
            })?;
        }
        let output = combined.reshape(shape, stream)?;
        crate::architectures::distributed::expert::materialize_timing_phase([&output])?;
        statistics.total_time = previous_moe_time + moe_started.elapsed();
        Ok(output)
    }
}

#[derive(Debug, Clone)]
/// Dense or sparse feed-forward layer.
pub enum FeedForward {
    /// Dense SwiGLU.
    Dense(Box<Mlp>),
    /// Routed plus shared DeepSeekMoE.
    Moe(Box<Moe>),
}

impl ModuleParameters for FeedForward {
    fn num_parameters(&self) -> usize {
        match self {
            Self::Dense(mlp) => mlp.num_parameters(),
            Self::Moe(moe) => moe.num_parameters(),
        }
    }

    fn parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        match self {
            Self::Dense(mlp) => mlp.parameters(),
            Self::Moe(moe) => moe.parameters(),
        }
    }

    fn parameters_mut(&mut self) -> safemlx::module::ModuleParamMut<'_> {
        match self {
            Self::Dense(mlp) => mlp.parameters_mut(),
            Self::Moe(moe) => moe.parameters_mut(),
        }
    }

    fn trainable_parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        match self {
            Self::Dense(mlp) => mlp.trainable_parameters(),
            Self::Moe(moe) => moe.trainable_parameters(),
        }
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Dense(mlp) => mlp.freeze_parameters(recursive),
            Self::Moe(moe) => moe.freeze_parameters(recursive),
        }
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Dense(mlp) => mlp.unfreeze_parameters(recursive),
            Self::Moe(moe) => moe.unfreeze_parameters(recursive),
        }
    }

    fn all_frozen(&self) -> Option<bool> {
        match self {
            Self::Dense(mlp) => mlp.all_frozen(),
            Self::Moe(moe) => moe.all_frozen(),
        }
    }

    fn any_frozen(&self) -> Option<bool> {
        match self {
            Self::Dense(mlp) => mlp.any_frozen(),
            Self::Moe(moe) => moe.any_frozen(),
        }
    }
}

impl FeedForward {
    fn new(args: &ModelArgs, layer: i32, stream: &Stream) -> Result<Self, Exception> {
        let index = usize::try_from(layer)
            .ok()
            .and_then(|index| args.layer_policy(index))
            .ok_or_else(|| Exception::custom("DeepSeek-V3 decoder layer is out of range"))?;
        match index {
            LayerPolicy::SparseMoe => Ok(Self::Moe(Box::new(Moe::new(args, layer, stream)?))),
            LayerPolicy::DenseMlp => Ok(Self::Dense(Box::new(Mlp::new(
                args,
                &format!("model.layers.{layer}.mlp"),
                args.intermediate_size,
                stream,
            )?))),
        }
    }

    fn forward_impl(
        &mut self,
        x: &Array,
        stream: &Stream,
        prefix: &str,
        observer: &mut ObserverOption<'_>,
    ) -> Result<Array, Exception> {
        match self {
            Self::Dense(mlp) => mlp.forward_impl(x, stream, prefix, observer),
            Self::Moe(moe) => moe.forward_impl(x, stream, prefix, observer),
        }
    }

    fn is_moe(&self) -> bool {
        matches!(self, Self::Moe(_))
    }

    pub(crate) fn moe_mut(&mut self) -> Option<&mut Moe> {
        match self {
            Self::Moe(moe) => Some(moe),
            Self::Dense(_) => None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_expert_parallel(
        &mut self,
        x: &Array,
        assignment: &crate::architectures::distributed::expert::ExpertAssignment,
        group: &safemlx::distributed::Group,
        statistics: &mut crate::architectures::distributed::expert::RoutingStatistics,
        prefix: &str,
        observer: ObserverOption<'_>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        match self {
            Self::Dense(mlp) => {
                let mut observer = observer;
                mlp.forward_impl(x, stream, prefix, &mut observer)
            }
            Self::Moe(moe) => moe.forward_expert_parallel(
                x, assignment, group, statistics, prefix, observer, stream,
            ),
        }
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// One DeepSeek-V3 decoder block.
pub struct DecoderLayer {
    #[param]
    /// MLA sublayer.
    pub self_attn: MultiHeadLatentAttention,
    #[param]
    /// Dense or MoE feed-forward sublayer.
    pub mlp: FeedForward,
    #[param]
    /// Pre-attention RMSNorm.
    pub input_layernorm: nn::RmsNorm,
    #[param]
    /// Pre-MLP RMSNorm.
    pub post_attention_layernorm: nn::RmsNorm,
}

impl DecoderLayer {
    pub(crate) fn new(args: &ModelArgs, layer: i32, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            self_attn: MultiHeadLatentAttention::new(args, layer, stream)?,
            mlp: FeedForward::new(args, layer, stream)?,
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

    pub(crate) fn new_layerwise(
        args: &ModelArgs,
        layer: i32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let mut block = Self::new(args, layer, stream)?;
        if let FeedForward::Moe(moe) = &mut block.mlp {
            moe.experts.initialize_unloaded_banks(args, stream)?;
        }
        Ok(block)
    }

    pub(crate) fn new_parallel_layerwise(
        args: &ModelArgs,
        layer: i32,
        attention_heads: i32,
        dense_intermediate: i32,
        routed_intermediate: i32,
        shared_intermediate: i32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let mut attention_args = args.clone();
        attention_args.num_attention_heads = attention_heads;
        let index = usize::try_from(layer)
            .map_err(|_| Exception::custom("DeepSeek parallel layer index is negative"))?;
        let policy = args.layer_policy(index).ok_or_else(|| {
            Exception::custom("DeepSeek parallel layer is outside the decoder schedule")
        })?;
        let mlp = match policy {
            LayerPolicy::DenseMlp => FeedForward::Dense(Box::new(Mlp::new(
                args,
                &format!("model.layers.{layer}.mlp"),
                dense_intermediate,
                stream,
            )?)),
            LayerPolicy::SparseMoe => FeedForward::Moe(Box::new(Moe::new_with_widths(
                args,
                layer,
                routed_intermediate,
                shared_intermediate,
                true,
                stream,
            )?)),
        };
        Ok(Self {
            self_attn: MultiHeadLatentAttention::new(&attention_args, layer, stream)?,
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

    fn forward_impl(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut CompressedLatentCache>,
        stream: &Stream,
        prefix: &str,
        observer: &mut ObserverOption<'_>,
    ) -> Result<Array, Exception> {
        observe_activation(observer, prefix, "input", x)?;
        observe_activation(observer, prefix, "residual_before_attention", x)?;
        let normalized = self.input_layernorm.forward(x, stream)?;
        observe_activation(observer, prefix, "input_layernorm", &normalized)?;
        let attention_prefix = if observer.is_some() {
            activation_name(prefix, "self_attn")
        } else {
            String::new()
        };
        let attention = self.self_attn.forward_impl(
            &normalized,
            mask,
            cache,
            stream,
            &attention_prefix,
            observer,
        )?;
        observe_activation(observer, prefix, "self_attn_output", &attention)?;
        observe_activation(observer, prefix, "residual_delta_attention", &attention)?;
        let hidden = x.add(attention, stream)?;
        observe_activation(observer, prefix, "post_attention_residual", &hidden)?;
        observe_activation(observer, prefix, "residual_after_attention", &hidden)?;
        let is_moe = self.mlp.is_moe();
        observe_activation(
            observer,
            prefix,
            if is_moe {
                "residual_before_moe"
            } else {
                "residual_before_mlp"
            },
            &hidden,
        )?;
        let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
        observe_activation(observer, prefix, "post_attention_layernorm", &normalized)?;
        let mlp_prefix = if observer.is_some() {
            activation_name(prefix, "mlp")
        } else {
            String::new()
        };
        let feed_forward = self
            .mlp
            .forward_impl(&normalized, stream, &mlp_prefix, observer)?;
        observe_activation(
            observer,
            prefix,
            if is_moe { "moe_output" } else { "mlp_output" },
            &feed_forward,
        )?;
        observe_activation(
            observer,
            prefix,
            if is_moe {
                "residual_delta_moe"
            } else {
                "residual_delta_mlp"
            },
            &feed_forward,
        )?;
        let output = hidden.add(feed_forward, stream)?;
        let output = intervene_activation(observer, prefix, "output", output)?;
        observe_activation(observer, prefix, "output", &output)?;
        observe_activation(
            observer,
            prefix,
            if is_moe {
                "residual_after_moe"
            } else {
                "residual_after_mlp"
            },
            &output,
        )?;
        Ok(output)
    }

    pub(crate) fn forward_stage(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut CompressedLatentCache>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let mut observer = None;
        self.forward_impl(x, mask, cache, stream, "", &mut observer)
    }

    pub(crate) fn forward_stage_with_observer(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut CompressedLatentCache>,
        stream: &Stream,
        prefix: &str,
        observer: &mut dyn ActivationObserver,
    ) -> Result<Array, Exception> {
        let mut observer = Some(observer);
        self.forward_impl(x, mask, cache, stream, prefix, &mut observer)
    }

    /// Executes a block while delegating routed-expert evaluation to a compact bank.
    pub(crate) fn forward_sparse_experts<F>(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut CompressedLatentCache>,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let normalized = self.input_layernorm.forward(x, stream)?;
        let mut observer = None;
        let attention =
            self.self_attn
                .forward_impl(&normalized, mask, cache, stream, "", &mut observer)?;
        let hidden = x.add(attention, stream)?;
        let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
        let feed_forward = match &mut self.mlp {
            FeedForward::Dense(mlp) => mlp.forward_impl(&normalized, stream, "", &mut observer)?,
            FeedForward::Moe(moe) => {
                let shape = normalized.shape();
                let flat = normalized.reshape(&[-1, normalized.dim(-1)], stream)?;
                let (indices, weights) = moe.gate.forward(&flat, stream)?;
                let routed = execute(&flat, &indices, &weights, stream)?;
                let shared = moe
                    .shared_experts
                    .forward_impl(&flat, stream, "", &mut observer)?;
                routed.add(shared, stream)?.reshape(shape, stream)?
            }
        };
        hidden.add(feed_forward, stream)
    }

    /// Executes TP-sharded MLA and dense/shared projections while delegating
    /// routed experts to an EP-scoped executor.
    pub(crate) fn forward_tensor_with_expert_executor<F>(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut CompressedLatentCache>,
        group: &safemlx::distributed::Group,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let normalized = self.input_layernorm.forward(x, stream)?;
        let mut observer = None;
        let attention =
            self.self_attn
                .forward_impl(&normalized, mask, cache, stream, "", &mut observer)?;
        let attention = safemlx::distributed::all_sum(&attention, group, stream)?;
        let hidden = x.add(attention, stream)?;
        let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
        let feed_forward = match &mut self.mlp {
            FeedForward::Dense(mlp) => {
                let partial = mlp.forward_impl(&normalized, stream, "", &mut observer)?;
                safemlx::distributed::all_sum(&partial, group, stream)?
            }
            FeedForward::Moe(moe) => {
                let shape = normalized.shape();
                let flat = normalized.reshape(&[-1, normalized.dim(-1)], stream)?;
                let (indices, weights) = moe.gate.forward(&flat, stream)?;
                let routed = execute(&flat, &indices, &weights, stream)?;
                let shared_partial =
                    moe.shared_experts
                        .forward_impl(&flat, stream, "", &mut observer)?;
                let shared = safemlx::distributed::all_sum(&shared_partial, group, stream)?;
                routed.add(shared, stream)?.reshape(shape, stream)?
            }
        };
        hidden.add(feed_forward, stream)
    }

    /// Executes a rank-local tensor-parallel block.
    ///
    /// The layer must have head/intermediate projections constructed with
    /// local dimensions and row projections loaded with input-axis shards.
    /// Attention and feed-forward residual deltas are reduced exactly once.
    pub(crate) fn forward_tensor_parallel(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut CompressedLatentCache>,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let normalized = self.input_layernorm.forward(x, stream)?;
        let mut observer = None;
        let attention =
            self.self_attn
                .forward_impl(&normalized, mask, cache, stream, "", &mut observer)?;
        let attention = safemlx::distributed::all_sum(&attention, group, stream)?;
        let hidden = x.add(attention, stream)?;
        let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
        let feed_forward = self
            .mlp
            .forward_impl(&normalized, stream, "", &mut observer)?;
        let feed_forward = safemlx::distributed::all_sum(&feed_forward, group, stream)?;
        hidden.add(feed_forward, stream)
    }

    /// Executes TP-sharded MLA and feed-forward projections while EP owns the
    /// rank-local routed expert banks.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_tensor_expert_parallel(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut CompressedLatentCache>,
        tensor_group: &safemlx::distributed::Group,
        assignment: &crate::architectures::distributed::expert::ExpertAssignment,
        expert_group: &safemlx::distributed::Group,
        statistics: &mut crate::architectures::distributed::expert::RoutingStatistics,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let normalized = self.input_layernorm.forward(x, stream)?;
        let mut observer = None;
        let attention =
            self.self_attn
                .forward_impl(&normalized, mask, cache, stream, "", &mut observer)?;
        let attention = safemlx::distributed::all_sum(&attention, tensor_group, stream)?;
        let hidden = x.add(attention, stream)?;
        let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
        let partial = self.mlp.forward_expert_parallel(
            &normalized,
            assignment,
            expert_group,
            statistics,
            "",
            None,
            stream,
        )?;
        let feed_forward = safemlx::distributed::all_sum(&partial, tensor_group, stream)?;
        hidden.add(feed_forward, stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_expert_parallel(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut CompressedLatentCache>,
        assignment: &crate::architectures::distributed::expert::ExpertAssignment,
        group: &safemlx::distributed::Group,
        statistics: &mut crate::architectures::distributed::expert::RoutingStatistics,
        prefix: &str,
        observer: ObserverOption<'_>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let normalized = self.input_layernorm.forward(x, stream)?;
        let mut attention_observer = None;
        let attention = self.self_attn.forward_impl(
            &normalized,
            mask,
            cache,
            stream,
            "",
            &mut attention_observer,
        )?;
        let hidden = x.add(attention, stream)?;
        let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
        let feed_forward = self.mlp.forward_expert_parallel(
            &normalized,
            assignment,
            group,
            statistics,
            &activation_name(prefix, "mlp"),
            observer,
            stream,
        )?;
        hidden.add(feed_forward, stream)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// DeepSeek-V3 transformer body.
pub struct TextModel {
    #[param]
    /// Token embedding table.
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[param]
    /// Decoder blocks.
    pub layers: Vec<DecoderLayer>,
    #[param]
    /// Final RMSNorm.
    pub norm: nn::RmsNorm,
}

impl TextModel {
    fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            embed_tokens: common::linear::unloaded_maybe_quantized_embedding(
                args.vocab_size,
                args.hidden_size,
                args.weight_quantization_for("model.embed_tokens.weight"),
                stream,
            )?,
            layers: args
                .layer_schedule
                .iter()
                .enumerate()
                .map(|(layer, _)| DecoderLayer::new(args, layer as i32, stream))
                .collect::<Result<_, _>>()?,
            norm: nn::RmsNorm::unloaded(
                args.hidden_size,
                args.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?,
        })
    }

    fn forward_impl(
        &mut self,
        input: ModelInput<'_>,
        stream: &Stream,
        observer: &mut ObserverOption<'_>,
    ) -> Result<Array, Exception> {
        let mut hidden = self.embed_tokens.forward(input.inputs, stream)?;
        observe_activation(observer, "model", "embed_tokens", &hidden)?;
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
        if let Some(mask) = mask {
            observe_activation(observer, "model", "attention_mask", mask)?;
        }
        if let Some(cache) = input.cache {
            if cache.layers.len() != self.layers.len() {
                return Err(Exception::custom(
                    "DeepSeek-V3 cache layer count does not match model",
                ));
            }
            for (layer_index, (layer, layer_cache)) in
                self.layers.iter_mut().zip(&mut cache.layers).enumerate()
            {
                let prefix = if observer.is_some() {
                    format!("model.layers.{layer_index}")
                } else {
                    String::new()
                };
                hidden = layer.forward_impl(
                    &hidden,
                    mask,
                    Some(layer_cache),
                    stream,
                    &prefix,
                    observer,
                )?;
            }
        } else {
            for (layer_index, layer) in self.layers.iter_mut().enumerate() {
                let prefix = if observer.is_some() {
                    format!("model.layers.{layer_index}")
                } else {
                    String::new()
                };
                hidden = layer.forward_impl(&hidden, mask, None, stream, &prefix, observer)?;
            }
        }
        hidden = self.norm.forward(&hidden, stream)?;
        observe_activation(observer, "model", "norm", &hidden)?;
        observe_activation(observer, "model", "output", &hidden)?;
        Ok(hidden)
    }
}

/// Input for a DeepSeek-V3 forward pass.
pub struct ModelInput<'a> {
    /// Token ids `[batch, sequence]`.
    pub inputs: &'a Array,
    /// Optional additive or boolean attention mask.
    pub mask: Option<&'a Array>,
    /// Optional compressed MLA cache.
    pub cache: Option<&'a mut Cache>,
}

#[derive(Debug, Clone, ModuleParameters)]
/// DeepSeek-V3/R1 causal language model.
pub struct Model {
    /// Parsed architecture arguments.
    pub args: ModelArgs,
    #[param]
    /// Transformer body.
    pub model: TextModel,
    #[param]
    /// Untied language-model head.
    pub lm_head: MaybeQuantized<nn::Linear>,
}

impl Model {
    /// Creates an unloaded model.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        args.validate()
            .map_err(|error| Exception::custom(error.to_string()))?;
        Ok(Self {
            model: TextModel::new(&args, stream)?,
            lm_head: common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                args.vocab_size,
                false,
                args.weight_quantization_for("lm_head.weight"),
                stream,
            )?,
            args,
        })
    }

    /// Returns an empty compressed cache.
    pub fn new_cache(&self) -> Cache {
        Cache::new(&self.args.layer_schedule)
    }

    /// Creates a device-resident or explicitly bounded paged compressed cache.
    pub fn new_cache_with_options(&self, policy: CacheResidencyPolicy) -> Result<Cache, Exception> {
        Cache::new_with_options(&self.args.layer_schedule, policy)
    }

    /// Lazily catalogs a compatible persisted compressed prefix.
    pub fn load_prompt_cache(
        &self,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
    ) -> Result<(Cache, PromptCacheManifest), Exception> {
        let layer_count = self.args.layer_schedule.len();
        let identity = PromptCacheModelIdentity {
            model_family: "deepseek_v3".into(),
            effective_model_type: self.args.model_type.clone(),
            architecture_fingerprint: prompt_cache_architecture_fingerprint(&self.args),
            layer_count,
            global_layer_start: 0,
            global_layer_end: layer_count,
            sink_tokens: 0,
            layer_prefix_offsets: vec![0; layer_count],
            topology: Default::default(),
            layer_layout: PromptCacheModelIdentity::compressed_layouts(
                layer_count,
                self.args.kv_lora_rank,
                self.args.qk_rope_head_dim,
            )
            .map_err(|error| Exception::custom(error.to_string()))?,
        };
        validate_prompt_cache_model_identity(expected, &identity)
            .map_err(|error| Exception::custom(error.to_string()))?;
        Cache::load_prompt_cache(
            &self.args.layer_schedule,
            directory,
            expected,
            &identity,
            prefix_token_ids,
            options,
        )
    }

    /// Returns the dispatched model type.
    pub fn model_type(&self) -> &str {
        &self.args.model_type
    }

    pub(crate) fn forward_logits(
        &mut self,
        input: ModelInput<'_>,
        last_token_only: bool,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let mut observer = None;
        self.forward_logits_impl(input, last_token_only, stream, &mut observer)
    }

    fn forward_logits_impl(
        &mut self,
        input: ModelInput<'_>,
        last_token_only: bool,
        stream: &Stream,
        observer: &mut ObserverOption<'_>,
    ) -> Result<Array, Exception> {
        let hidden = self.model.forward_impl(input, stream, observer)?;
        let hidden = if last_token_only {
            hidden.try_index_device((.., -1, ..), stream)?
        } else {
            hidden
        };
        let logits = self.lm_head.forward(&hidden, stream)?;
        observe_activation(observer, "lm_head", "logits", &logits)?;
        Ok(logits)
    }

    /// Runs the normal DeepSeek forward path with detailed runtime observation.
    pub fn forward_with_observer(
        &mut self,
        input: ModelInput<'_>,
        stream: &Stream,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        let mut observer: ObserverOption<'_> = Some(observer);
        self.forward_logits_impl(input, false, stream, &mut observer)
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

/// DeepSeek token-generation iterator.
pub type Generate<'a, S = crate::runtime::generation::sampler::DefaultSampler> =
    common::generation::Generate<'a, Model, Cache, S>;

fn parse_config_value(value: Value) -> Result<ModelArgs, Error> {
    let source: ModelArgsSource = serde_json::from_value(value.clone()).map_err(|error| {
        Error::UnsupportedArchitecture(format!("invalid DeepSeek-V3 config: {error}"))
    })?;
    let args = source.normalize()?;
    if value
        .get("architectures")
        .and_then(Value::as_array)
        .is_some_and(|architectures| {
            !architectures
                .iter()
                .any(|name| name.as_str() == Some("DeepseekV3ForCausalLM"))
        })
    {
        return Err(Error::UnsupportedArchitecture(
            "DeepSeek-V3 config does not declare DeepseekV3ForCausalLM".into(),
        ));
    }
    if value
        .get("attention_bias")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return Err(Error::UnsupportedArchitecture(
            "DeepSeek-V3 attention_bias=true is not supported".into(),
        ));
    }
    if value
        .get("attention_dropout")
        .and_then(Value::as_f64)
        .unwrap_or(0.0)
        != 0.0
    {
        return Err(Error::UnsupportedArchitecture(
            "DeepSeek-V3 inference requires attention_dropout=0".into(),
        ));
    }
    if value
        .get("hidden_act")
        .and_then(Value::as_str)
        .is_some_and(|activation| activation != "silu")
    {
        return Err(Error::UnsupportedArchitecture(
            "DeepSeek-V3 supports only hidden_act=silu".into(),
        ));
    }
    if value.get("ep_size").and_then(Value::as_i64).unwrap_or(1) != 1 {
        return Err(Error::UnsupportedArchitecture(
            "tensor-local loading supports only DeepSeek-V3 ep_size=1 checkpoints".into(),
        ));
    }
    if value
        .get("num_key_value_heads")
        .and_then(Value::as_i64)
        .is_some_and(|heads| heads != args.num_attention_heads as i64)
    {
        return Err(Error::UnsupportedArchitecture(
            "DeepSeek-V3 num_key_value_heads must equal num_attention_heads for MLA checkpoint compatibility".into(),
        ));
    }
    Ok(args)
}

/// Parses the same validated architecture arguments used by loading without
/// opening a model directory or constructing an MLX module tree.
pub fn model_args_from_config_value(config: &Value) -> Result<ModelArgs, Error> {
    parse_config_value(config.clone())
}

/// Parses and validates `config.json` from a model directory.
pub fn get_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let file = std::fs::File::open(model_dir.as_ref().join("config.json"))?;
    model_args_from_config_value(&serde_json::from_reader(file)?)
}

pub(crate) fn validate_model_config_value(config: &Value) -> Result<(), Error> {
    model_args_from_config_value(config).map(|_| ())
}

pub(crate) struct PreparedDeepSeekGguf {
    pub(crate) args: ModelArgs,
    pub(crate) eos_token_ids: Vec<u32>,
}

pub(crate) fn prepare_gguf_checkpoint(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    quantization: Option<WeightQuantization>,
    _weights_stream: &Stream,
) -> Result<PreparedDeepSeekGguf, Error> {
    let mut args = model_args_from_gguf_catalog(checkpoint, metadata)?;
    args.quantized_weight_configs = Some(gguf_quantization_configs(
        checkpoint,
        translate_gguf_weight_name,
    )?);
    if let Some(quantization) = quantization {
        args.quantization = Some(quantization);
        args.quantization_config = None;
        args.quantized_weight_configs = None;
    }
    args.validate()?;

    let eos_token_ids = crate::api::gguf_eos_token_ids(metadata)?;
    Ok(PreparedDeepSeekGguf {
        args,
        eos_token_ids,
    })
}

/// Parses the same pure GGUF catalog geometry used by loading, without
/// converting tensor payloads or constructing an MLX stream.
pub(crate) fn model_args_from_gguf_catalog(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> Result<ModelArgs, Error> {
    let architecture = gguf_string(metadata, "general.architecture")?;
    if architecture != "deepseek2" {
        return Err(Error::UnsupportedArchitecture(format!(
            "GGUF architecture {architecture:?}; the DeepSeek-V3 loader supports deepseek2"
        )));
    }
    checkpoint
        .catalog()
        .translated_outputs(translate_gguf_weight_name)
        .map_err(safemlx::error::IoError::from)?;
    let args = args_from_gguf(checkpoint, metadata)?;
    args.validate()?;
    Ok(args)
}

fn args_from_gguf(
    arrays: &impl GgufTensorNames,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> Result<ModelArgs, Error> {
    let architecture = "deepseek2";
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let qk_rope_head_dim = gguf_i32(metadata, &key("rope.dimension_count"))?;
    let qk_head_dim = gguf_i32(metadata, &key("attention.key_length_mla"))?;
    let qk_nope_head_dim = qk_head_dim.checked_sub(qk_rope_head_dim).ok_or_else(|| {
        Error::UnsupportedArchitecture(format!(
            "DeepSeek GGUF MLA key length {qk_head_dim} is smaller than rotary length {qk_rope_head_dim}"
        ))
    })?;
    let q_lora_rank = gguf_optional_i64(metadata, &key("attention.q_lora_rank"))?
        .map(i32::try_from)
        .transpose()
        .map_err(|_| Error::UnsupportedArchitecture("GGUF query LoRA rank exceeds i32".into()))?
        .filter(|rank| *rank > 0);
    let rope_scaling = match gguf_optional_string(metadata, &key("rope.scaling.type"))? {
        None => None,
        Some(scaling) if scaling == "none" || scaling == "default" => None,
        Some(scaling) if scaling == "yarn" => Some(YarnConfig {
            r#type: "yarn".into(),
            factor: gguf_f32(metadata, &key("rope.scaling.factor"))?,
            original_max_position_embeddings: gguf_i32(
                metadata,
                &key("rope.scaling.original_context_length"),
            )?,
            beta_fast: gguf_optional_f32(metadata, &key("rope.scaling.yarn_beta_fast"))?
                .unwrap_or_else(default_beta_fast),
            beta_slow: gguf_optional_f32(metadata, &key("rope.scaling.yarn_beta_slow"))?
                .unwrap_or_else(default_beta_slow),
            mscale: 1.0,
            mscale_all_dim: gguf_optional_f32(metadata, &key("rope.scaling.yarn_log_multiplier"))?
                .map(|value| value / 0.1)
                .unwrap_or(1.0),
        }),
        Some(scaling) => {
            return Err(Error::UnsupportedArchitecture(format!(
                "DeepSeek GGUF RoPE scaling {scaling:?} is unsupported"
            )))
        }
    };
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
            ))
        }
        None => gguf_i32(metadata, &key("vocab_size"))?,
    };
    let gating = gguf_optional_i64(metadata, &key("expert_gating_func"))?.unwrap_or(2);
    if gating != 2 {
        return Err(Error::UnsupportedArchitecture(format!(
            "DeepSeek GGUF expert_gating_func {gating} is unsupported; expected sigmoid (2)"
        )));
    }
    ModelArgsSource {
        model_type: "deepseek_v3".into(),
        hidden_size: gguf_i32(metadata, &key("embedding_length"))?,
        intermediate_size: gguf_i32(metadata, &key("feed_forward_length"))?,
        moe_intermediate_size: gguf_i32(metadata, &key("expert_feed_forward_length"))?,
        num_hidden_layers: gguf_i32(metadata, &key("block_count"))?,
        num_attention_heads: gguf_i32(metadata, &key("attention.head_count"))?,
        vocab_size,
        rms_norm_eps: gguf_f32(metadata, &key("attention.layer_norm_rms_epsilon"))?,
        max_position_embeddings: gguf_i32(metadata, &key("context_length"))?,
        rope_theta: gguf_optional_f32(metadata, &key("rope.freq_base"))?
            .unwrap_or_else(default_rope_theta),
        rope_scaling,
        q_lora_rank,
        kv_lora_rank: gguf_i32(metadata, &key("attention.kv_lora_rank"))?,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim: gguf_i32(metadata, &key("attention.value_length_mla"))?,
        first_k_dense_replace: gguf_i32(metadata, &key("leading_dense_block_count"))?,
        moe_layer_freq: 1,
        n_routed_experts: gguf_i32(metadata, &key("expert_count"))?,
        n_shared_experts: gguf_optional_i64(metadata, &key("expert_shared_count"))?
            .map(i32::try_from)
            .transpose()
            .map_err(|_| {
                Error::UnsupportedArchitecture("GGUF shared expert count exceeds i32".into())
            })?
            .unwrap_or(1),
        num_experts_per_tok: gguf_i32(metadata, &key("expert_used_count"))?,
        n_group: gguf_i32(metadata, &key("expert_group_count"))?,
        topk_group: gguf_i32(metadata, &key("expert_group_used_count"))?,
        topk_method: "noaux_tc".into(),
        scoring_func: "sigmoid".into(),
        norm_topk_prob: gguf_optional_bool(metadata, &key("expert_weights_norm"))?.unwrap_or(true),
        routed_scaling_factor: gguf_optional_f32(metadata, &key("expert_weights_scale"))?
            .unwrap_or(1.0),
        num_nextn_predict_layers: 0,
        quantization_config: None,
        quantization: None,
        quantized_weight_configs: None,
        split_kv_b: arrays.any_gguf_tensor(|name| name.contains(".attn_k_b.")),
        tie_word_embeddings: false,
    }
    .normalize()
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
        return name.to_string();
    };
    let Some((layer, parameter)) = rest.split_once('.') else {
        return name.to_string();
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
        ("attn_q_a", "self_attn.q_a_proj"),
        ("attn_q_b", "self_attn.q_b_proj"),
        ("attn_kv_a_mqa", "self_attn.kv_a_proj_with_mqa"),
        ("attn_kv_b", "self_attn.kv_b_proj"),
        ("attn_k_b", "self_attn.k_b_proj"),
        ("attn_v_b", "self_attn.v_b_proj"),
        ("attn_q_a_norm", "self_attn.q_a_layernorm"),
        ("attn_kv_a_norm", "self_attn.kv_a_layernorm"),
        ("attn_output", "self_attn.o_proj"),
        ("attn_norm", "input_layernorm"),
        ("ffn_norm", "post_attention_layernorm"),
        ("ffn_gate", "mlp.gate_proj"),
        ("ffn_up", "mlp.up_proj"),
        ("ffn_down", "mlp.down_proj"),
        ("ffn_gate_shexp", "mlp.shared_experts.gate_proj"),
        ("ffn_up_shexp", "mlp.shared_experts.up_proj"),
        ("ffn_down_shexp", "mlp.shared_experts.down_proj"),
        ("ffn_gate_inp", "mlp.gate"),
        ("rope_freqs", "rope_freqs"),
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

fn gguf_string(metadata: &HashMap<String, GgufMetadataValue>, key: &str) -> Result<String, Error> {
    gguf_optional_string(metadata, key)?.ok_or_else(|| {
        Error::UnsupportedArchitecture(format!("GGUF metadata is missing required key {key:?}"))
    })
}

fn gguf_optional_string(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<Option<String>, Error> {
    match metadata.get(key) {
        Some(GgufMetadataValue::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(Error::UnsupportedArchitecture(format!(
            "GGUF metadata key {key:?} has the wrong type"
        ))),
        None => Ok(None),
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

/// Loads the official `tokenizer.json`.
pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    Tokenizer::from_file(model_dir.as_ref().join("tokenizer.json")).map_err(Error::Other)
}

#[cfg(test)]
mod tests {
    use super::{
        deepseek_layer_schedule, parse_config_value, prompt_cache_architecture_fingerprint, Cache,
        FeedForward, LayerPolicy, Model, ModelArgs, ModelInput,
    };
    use crate::{
        api::{LoadedModel, ModelKind},
        architectures::deepseek_v3::layerwise::{
            load_deepseek_v3_layerwise_model, DeepSeekV3LayerwiseModel,
        },
        error::Error,
        runtime::cache::residency::{CacheResidencyPolicy, PagedCacheOptions},
        runtime::cache::CompressedLatentCache,
        runtime::execution::inspection::{ActivationObserver, MoeRoutingObservation},
    };
    use safemlx::{
        error::Exception,
        host_transfer_capacity_upper_bound,
        module::{Module, ModuleParameters, Param},
        ops::{indexing::TryIndexOp, ones_dtype, zeros_dtype},
        transforms::eval,
        Array, Device, DeviceType, Dtype, ExecutionContext, HostTransferPolicy,
    };
    use serde_json::{json, Value};
    use std::{collections::HashMap, fs, path::Path, time::SystemTime};

    fn load_fully_resident(
        model_dir: impl AsRef<Path>,
        quantization: Option<crate::runtime::checkpoint::quantization::WeightQuantization>,
        stream: &safemlx::Stream,
    ) -> Result<DeepSeekV3LayerwiseModel, Error> {
        let weights = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        load_deepseek_v3_layerwise_model(
            model_dir,
            crate::runtime::execution::layerwise::LayerWeightResidency::FullyResident,
            quantization,
            stream,
            weights.stream(),
        )
    }

    fn load_gguf_fully_resident(
        path: impl AsRef<Path>,
        quantization: Option<crate::runtime::checkpoint::quantization::WeightQuantization>,
        stream: &safemlx::Stream,
    ) -> Result<DeepSeekV3LayerwiseModel, Error> {
        let checkpoint = safemlx::ops::GgufCheckpoint::open(path)?;
        let metadata = crate::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        let weights = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        crate::architectures::deepseek_v3::layerwise::load_deepseek_v3_gguf_layerwise_model(
            &checkpoint,
            &metadata,
            crate::WeightResidency::fully_resident(),
            quantization,
            stream,
            weights.stream(),
        )
        .map(|(model, _)| model)
    }

    fn tiny_config_value(q_lora_rank: Option<i32>) -> Value {
        json!({
            "architectures": ["DeepseekV3ForCausalLM"],
            "model_type": "deepseek_v3",
            "hidden_size": 8,
            "intermediate_size": 16,
            "moe_intermediate_size": 4,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "vocab_size": 32,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 128,
            "rope_theta": 10000,
            "q_lora_rank": q_lora_rank,
            "kv_lora_rank": 4,
            "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 2,
            "v_head_dim": 2,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "n_group": 2,
            "topk_group": 1,
            "topk_method": "noaux_tc",
            "scoring_func": "sigmoid",
            "norm_topk_prob": true,
            "routed_scaling_factor": 1.5,
            "num_nextn_predict_layers": 0,
            "tie_word_embeddings": false,
            "eos_token_id": 1
        })
    }

    fn tiny_args(q_lora_rank: Option<i32>) -> ModelArgs {
        parse_config_value(tiny_config_value(q_lora_rank)).unwrap()
    }

    fn tiny_fp8_args() -> ModelArgs {
        let mut value = tiny_config_value(Some(4));
        value.as_object_mut().unwrap().insert(
            "quantization_config".into(),
            json!({"activation_scheme":"dynamic","fmt":"e4m3","quant_method":"fp8","weight_block_size":[128,128]}),
        );
        parse_config_value(value).unwrap()
    }

    fn affine_config_value() -> Value {
        let mut value = tiny_config_value(Some(32));
        let object = value.as_object_mut().unwrap();
        for (key, value) in [
            ("hidden_size", 32),
            ("intermediate_size", 64),
            ("moe_intermediate_size", 32),
            ("kv_lora_rank", 32),
            ("qk_nope_head_dim", 32),
            ("qk_rope_head_dim", 8),
            ("v_head_dim", 16),
        ] {
            object.insert(key.into(), json!(value));
        }
        value
    }

    fn test_context() -> ExecutionContext {
        ExecutionContext::new(Device::new(DeviceType::Gpu, 0))
    }

    fn temp_dir() -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "safemlx-deepseek-v3-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn initialize_dense_model(model: &mut Model, stream: &safemlx::Stream) {
        for layer in &mut model.model.layers {
            if let FeedForward::Moe(moe) = &mut layer.mlp {
                let e = model.args.n_routed_experts;
                let h = model.args.hidden_size;
                let i = model.args.moe_intermediate_size;
                moe.experts.gate_proj = Param::new(Some(
                    Array::full::<f32>(&[e, i, h], Array::from_f32(0.01), stream).unwrap(),
                ));
                moe.experts.up_proj = Param::new(Some(
                    Array::full::<f32>(&[e, i, h], Array::from_f32(0.01), stream).unwrap(),
                ));
                moe.experts.down_proj = Param::new(Some(
                    Array::full::<f32>(&[e, h, i], Array::from_f32(0.01), stream).unwrap(),
                ));
            }
        }
        for (name, parameter) in model.parameters_mut().flatten() {
            let shape = parameter.shape().to_vec();
            let dtype = parameter.dtype();
            *parameter =
                if name.ends_with("layernorm.weight") || name.as_ref() == "model.norm.weight" {
                    ones_dtype(&shape, dtype, stream).unwrap()
                } else if dtype == Dtype::Float32 {
                    Array::full::<f32>(&shape, Array::from_f32(0.01), stream).unwrap()
                } else {
                    zeros_dtype(&shape, dtype, stream).unwrap()
                };
        }
    }

    fn initialize_fp8_model(model: &mut Model, stream: &safemlx::Stream) {
        for layer in &mut model.model.layers {
            if let FeedForward::Moe(moe) = &mut layer.mlp {
                let e = model.args.n_routed_experts;
                let h = model.args.hidden_size;
                let i = model.args.moe_intermediate_size;
                moe.experts.gate_proj = Param::new(Some(
                    Array::full::<u8>(&[e, i, h], Array::from_slice(&[0x38u8], &[]), stream)
                        .unwrap(),
                ));
                moe.experts.up_proj = Param::new(Some(
                    Array::full::<u8>(&[e, i, h], Array::from_slice(&[0x38u8], &[]), stream)
                        .unwrap(),
                ));
                moe.experts.down_proj = Param::new(Some(
                    Array::full::<u8>(&[e, h, i], Array::from_slice(&[0x38u8], &[]), stream)
                        .unwrap(),
                ));
                moe.experts.gate_proj_scale_inv =
                    Param::new(Some(Array::ones::<f32>(&[e, 1, 1], stream).unwrap()));
                moe.experts.up_proj_scale_inv =
                    Param::new(Some(Array::ones::<f32>(&[e, 1, 1], stream).unwrap()));
                moe.experts.down_proj_scale_inv =
                    Param::new(Some(Array::ones::<f32>(&[e, 1, 1], stream).unwrap()));
            }
        }
        for (name, parameter) in model.parameters_mut().flatten() {
            let shape = parameter.shape().to_vec();
            let dtype = parameter.dtype();
            *parameter = if dtype == Dtype::Uint8 {
                Array::full::<u8>(&shape, Array::from_slice(&[0x38u8], &[]), stream).unwrap()
            } else if name.ends_with("layernorm.weight")
                || name.as_ref() == "model.norm.weight"
                || name.ends_with("weight_scale_inv")
                || name.ends_with("_scale_inv")
            {
                ones_dtype(&shape, dtype, stream).unwrap()
            } else {
                Array::full::<f32>(&shape, Array::from_f32(0.01), stream).unwrap()
            };
        }
    }

    #[derive(Default)]
    struct DetailedObserver {
        names: Vec<String>,
        routing_observations: usize,
        interventions: Vec<String>,
    }

    impl ActivationObserver for DetailedObserver {
        fn observe(&mut self, name: &str, _value: &Array) -> Result<(), Exception> {
            self.names.push(name.to_string());
            Ok(())
        }

        fn intervene(&mut self, name: &str, _value: &Array) -> Result<Option<Array>, Exception> {
            self.interventions.push(name.to_string());
            Ok(None)
        }

        fn observe_moe_routing(
            &mut self,
            routing: MoeRoutingObservation<'_>,
        ) -> Result<(), Exception> {
            assert_eq!(routing.prefix, "model.layers.1.mlp");
            assert_eq!(routing.num_experts, 4);
            assert_eq!(routing.selected_experts.dim(-1), 2);
            assert_eq!(
                routing.selected_scores.shape(),
                routing.routing_weights.shape()
            );
            assert_eq!(routing.routed_output.dim(-1), 8);
            assert!(routing.shared_output.is_some());
            assert!(routing.combined_output.is_some());
            self.routing_observations += 1;
            Ok(())
        }
    }

    fn checkpoint_arrays(model: &Model, stream: &safemlx::Stream) -> Vec<(String, Array)> {
        let mut arrays = Vec::new();
        for (key, value) in model.parameters().flatten() {
            let key = key.to_string();
            let packed_projection =
                ["gate_proj", "up_proj", "down_proj"]
                    .into_iter()
                    .find_map(|projection| {
                        [
                            ("", "weight"),
                            ("_scale_inv", "weight_scale_inv"),
                            ("_scales", "scales"),
                            ("_biases", "biases"),
                        ]
                        .into_iter()
                        .find_map(
                            |(packed_suffix, checkpoint_component)| {
                                key.ends_with(&format!(".mlp.experts.{projection}{packed_suffix}"))
                                    .then_some((projection, packed_suffix, checkpoint_component))
                            },
                        )
                    });
            if let Some((projection, packed_suffix, checkpoint_component)) = packed_projection {
                let suffix = format!(".experts.{projection}{packed_suffix}");
                let prefix = key.strip_suffix(&suffix).unwrap();
                for expert in 0..model.args.n_routed_experts {
                    arrays.push((
                        format!("{prefix}.experts.{expert}.{projection}.{checkpoint_component}"),
                        value.try_index_device(expert, stream).unwrap(),
                    ));
                }
            } else {
                arrays.push((key, value.clone()));
            }
        }
        arrays
    }

    fn gguf_weight_name(name: &str) -> String {
        for (target, source) in [
            ("model.embed_tokens", "token_embd"),
            ("model.norm", "output_norm"),
            ("lm_head", "output"),
        ] {
            if name == target || name.starts_with(&format!("{target}.")) {
                return name.replacen(target, source, 1);
            }
        }
        let rest = name.strip_prefix("model.layers.").unwrap();
        let (layer, parameter) = rest.split_once('.').unwrap();
        for (target, source) in [
            ("mlp.experts.gate_proj", "ffn_gate_exps"),
            ("mlp.experts.up_proj", "ffn_up_exps"),
            ("mlp.experts.down_proj", "ffn_down_exps"),
        ] {
            if parameter == target || parameter.starts_with(&format!("{target}_")) {
                let suffix = match parameter.strip_prefix(target).unwrap_or_default() {
                    "" => ".weight",
                    "_scales" => ".scales",
                    "_biases" => ".biases",
                    other => other,
                };
                return format!("blk.{layer}.{source}{suffix}");
            }
        }
        if parameter == "mlp.gate.e_score_correction_bias" {
            return format!("blk.{layer}.exp_probs_b.bias");
        }
        for (target, source) in [
            ("self_attn.q_proj", "attn_q"),
            ("self_attn.q_a_proj", "attn_q_a"),
            ("self_attn.q_b_proj", "attn_q_b"),
            ("self_attn.kv_a_proj_with_mqa", "attn_kv_a_mqa"),
            ("self_attn.kv_b_proj", "attn_kv_b"),
            ("self_attn.q_a_layernorm", "attn_q_a_norm"),
            ("self_attn.kv_a_layernorm", "attn_kv_a_norm"),
            ("self_attn.o_proj", "attn_output"),
            ("input_layernorm", "attn_norm"),
            ("post_attention_layernorm", "ffn_norm"),
            ("mlp.shared_experts.gate_proj", "ffn_gate_shexp"),
            ("mlp.shared_experts.up_proj", "ffn_up_shexp"),
            ("mlp.shared_experts.down_proj", "ffn_down_shexp"),
            ("mlp.gate", "ffn_gate_inp"),
            ("mlp.gate_proj", "ffn_gate"),
            ("mlp.up_proj", "ffn_up"),
            ("mlp.down_proj", "ffn_down"),
        ] {
            if parameter == target || parameter.starts_with(&format!("{target}.")) {
                return format!("blk.{layer}.{}", parameter.replacen(target, source, 1));
            }
        }
        panic!("unmapped DeepSeek test parameter {name}")
    }

    fn gguf_metadata() -> HashMap<String, safemlx::ops::GgufMetadataValue> {
        use safemlx::ops::GgufMetadataValue as M;
        HashMap::from([
            ("general.architecture".into(), M::String("deepseek2".into())),
            ("deepseek2.block_count".into(), M::Uint32(2)),
            ("deepseek2.context_length".into(), M::Uint32(128)),
            ("deepseek2.embedding_length".into(), M::Uint32(32)),
            ("deepseek2.feed_forward_length".into(), M::Uint32(64)),
            ("deepseek2.attention.head_count".into(), M::Uint32(2)),
            (
                "deepseek2.attention.layer_norm_rms_epsilon".into(),
                M::Float32(1e-6),
            ),
            ("deepseek2.rope.freq_base".into(), M::Float32(10_000.0)),
            ("deepseek2.rope.dimension_count".into(), M::Uint32(8)),
            ("deepseek2.expert_used_count".into(), M::Uint32(2)),
            ("deepseek2.expert_group_count".into(), M::Uint32(2)),
            ("deepseek2.expert_group_used_count".into(), M::Uint32(1)),
            ("deepseek2.expert_gating_func".into(), M::Uint32(2)),
            ("deepseek2.leading_dense_block_count".into(), M::Uint32(1)),
            ("deepseek2.vocab_size".into(), M::Uint32(32)),
            ("deepseek2.attention.q_lora_rank".into(), M::Uint32(32)),
            ("deepseek2.attention.kv_lora_rank".into(), M::Uint32(32)),
            ("deepseek2.attention.key_length_mla".into(), M::Uint32(40)),
            ("deepseek2.attention.value_length_mla".into(), M::Uint32(16)),
            ("deepseek2.expert_feed_forward_length".into(), M::Uint32(32)),
            ("deepseek2.expert_count".into(), M::Uint32(4)),
            ("deepseek2.expert_shared_count".into(), M::Uint32(1)),
            ("deepseek2.expert_weights_scale".into(), M::Float32(1.5)),
            ("deepseek2.expert_weights_norm".into(), M::Bool(true)),
        ])
    }

    fn gguf_arrays(model: &Model, stream: &safemlx::Stream) -> HashMap<String, Array> {
        let mut arrays = HashMap::new();
        for (name, value) in model.parameters().flatten() {
            if name.ends_with(".self_attn.kv_b_proj.weight") {
                let gguf_name = gguf_weight_name(&name);
                let prefix = gguf_name.strip_suffix("attn_kv_b.weight").unwrap();
                let weight = value
                    .reshape(
                        &[
                            model.args.num_attention_heads,
                            model.args.qk_nope_head_dim + model.args.v_head_dim,
                            model.args.kv_lora_rank,
                        ],
                        stream,
                    )
                    .unwrap();
                arrays.insert(
                    format!("{prefix}attn_k_b.weight"),
                    weight
                        .try_index_device((.., ..model.args.qk_nope_head_dim, ..), stream)
                        .unwrap()
                        .swap_axes(-1, -2, stream)
                        .unwrap(),
                );
                arrays.insert(
                    format!("{prefix}attn_v_b.weight"),
                    weight
                        .try_index_device((.., model.args.qk_nope_head_dim.., ..), stream)
                        .unwrap(),
                );
            } else {
                arrays.insert(gguf_weight_name(&name), value.clone());
            }
        }
        arrays
    }

    fn save_fixture(
        dir: &Path,
        source: &Model,
        stream: &safemlx::Stream,
        omit: Option<&str>,
        extras: Vec<(String, Array)>,
    ) {
        let mut config = if source.args.hidden_size == 32 {
            affine_config_value()
        } else {
            tiny_config_value(Some(4))
        };
        if source.args.native_fp8_config().is_some() {
            config.as_object_mut().unwrap().insert(
                "quantization_config".into(),
                json!({"activation_scheme":"dynamic","fmt":"e4m3","quant_method":"fp8","weight_block_size":[128,128]}),
            );
        } else if let Some(affine) = source.args.affine_quantization().unwrap() {
            let metadata = serde_json::to_value(affine).unwrap();
            config
                .as_object_mut()
                .unwrap()
                .insert("quantization".into(), metadata.clone());
            config
                .as_object_mut()
                .unwrap()
                .insert("quantization_config".into(), metadata);
        }
        fs::write(
            dir.join("config.json"),
            serde_json::to_vec_pretty(&config).unwrap(),
        )
        .unwrap();
        let mut arrays = checkpoint_arrays(source, stream)
            .into_iter()
            .filter(|(key, _)| omit != Some(key.as_str()))
            .collect::<Vec<_>>();
        arrays.extend(extras);
        eval(arrays.iter().map(|(_, value)| value)).unwrap();
        Array::save_safetensors(
            arrays.iter().map(|(key, value)| (key.as_str(), value)),
            None,
            dir.join("model.safetensors"),
        )
        .unwrap();
    }

    #[test]
    fn parses_published_v3_0324_and_r1_0528_configuration() {
        let value = json!({
            "architectures": ["DeepseekV3ForCausalLM"],
            "attention_bias": false,
            "attention_dropout": 0.0,
            "ep_size": 1,
            "first_k_dense_replace": 3,
            "hidden_act": "silu",
            "hidden_size": 7168,
            "intermediate_size": 18432,
            "kv_lora_rank": 512,
            "max_position_embeddings": 163840,
            "model_type": "deepseek_v3",
            "moe_intermediate_size": 2048,
            "moe_layer_freq": 1,
            "n_group": 8,
            "n_routed_experts": 256,
            "n_shared_experts": 1,
            "norm_topk_prob": true,
            "num_attention_heads": 128,
            "num_key_value_heads": 128,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 61,
            "num_nextn_predict_layers": 1,
            "q_lora_rank": 1536,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "quantization_config": {"activation_scheme":"dynamic","fmt":"e4m3","quant_method":"fp8","weight_block_size":[128,128]},
            "rms_norm_eps": 1e-6,
            "rope_scaling": {"beta_fast":32,"beta_slow":1,"factor":40,"mscale":1.0,"mscale_all_dim":1.0,"original_max_position_embeddings":4096,"type":"yarn"},
            "rope_theta": 10000,
            "routed_scaling_factor": 2.5,
            "scoring_func": "sigmoid",
            "tie_word_embeddings": false,
            "topk_group": 4,
            "topk_method": "noaux_tc",
            "v_head_dim": 128,
            "vocab_size": 129280
        });
        let args = parse_config_value(value.clone()).unwrap();
        assert_eq!(args.num_hidden_layers, 61);
        assert_eq!(args.q_lora_rank, Some(1536));
        assert_eq!(args.num_nextn_predict_layers, 1);
        assert_eq!(
            args.layer_schedule.iter().copied().collect::<Vec<_>>(),
            std::iter::repeat_n(LayerPolicy::DenseMlp, 3)
                .chain(std::iter::repeat_n(LayerPolicy::SparseMoe, 58))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            crate::api::resolve_model_config(&value).ok(),
            Some(crate::api::ResolvedModelConfig {
                kind: ModelKind::DeepSeekV3,
                model_type: "deepseek_v3".into(),
                effective_model_type: "deepseek_v3".into(),
            })
        );
    }

    #[test]
    fn hf_dense_prefix_and_frequency_normalize_once_into_ordered_policy() {
        let schedule = deepseek_layer_schedule(6, 1, 2).unwrap();
        assert_eq!(
            schedule.iter().copied().collect::<Vec<_>>(),
            vec![
                LayerPolicy::DenseMlp,
                LayerPolicy::DenseMlp,
                LayerPolicy::SparseMoe,
                LayerPolicy::DenseMlp,
                LayerPolicy::SparseMoe,
                LayerPolicy::DenseMlp,
            ]
        );
        assert!(deepseek_layer_schedule(0, 0, 1).is_err());
        assert!(deepseek_layer_schedule(2, -1, 1).is_err());
        assert!(deepseek_layer_schedule(2, 3, 1).is_err());
        assert!(deepseek_layer_schedule(2, 1, 0).is_err());
    }

    #[test]
    fn arbitrary_internal_schedule_drives_layers_cache_and_identity_exactly() {
        let mut first = tiny_args(Some(4));
        first.num_hidden_layers = 4;
        first.layer_schedule = crate::runtime::attention::LayerSchedule::new(
            4,
            vec![
                LayerPolicy::SparseMoe,
                LayerPolicy::DenseMlp,
                LayerPolicy::SparseMoe,
                LayerPolicy::DenseMlp,
            ],
        )
        .unwrap();
        first.validate().unwrap();
        let mut second = first.clone();
        second.layer_schedule = crate::runtime::attention::LayerSchedule::new(
            4,
            vec![
                LayerPolicy::DenseMlp,
                LayerPolicy::SparseMoe,
                LayerPolicy::SparseMoe,
                LayerPolicy::DenseMlp,
            ],
        )
        .unwrap();
        second.validate().unwrap();

        assert_eq!(first.layer_schedule_fingerprint(), "e,d,e,d");
        assert_ne!(
            prompt_cache_architecture_fingerprint(&first),
            prompt_cache_architecture_fingerprint(&second)
        );
        assert_eq!(Cache::new(&first.layer_schedule).layers.len(), 4);
    }

    #[test]
    fn inspection_and_loading_reject_invalid_hf_schedule_with_the_same_diagnostic() {
        let mut value = tiny_config_value(Some(4));
        value["moe_layer_freq"] = json!(0);
        let loading = parse_config_value(value.clone()).unwrap_err().to_string();
        let inspection = crate::api::resolve_model_config(&value)
            .unwrap_err()
            .to_string();
        assert_eq!(inspection, loading);
        assert!(loading.contains("moe_layer_freq must be positive"));
    }

    #[test]
    fn arbitrary_internal_schedule_constructs_the_declared_feed_forward_operators() {
        let context = test_context();
        let stream = context.stream();
        let mut args = tiny_args(Some(4));
        args.layer_schedule = crate::runtime::attention::LayerSchedule::new(
            2,
            vec![LayerPolicy::SparseMoe, LayerPolicy::DenseMlp],
        )
        .unwrap();
        args.validate().unwrap();
        let model = Model::new(args, stream).unwrap();
        assert!(matches!(model.model.layers[0].mlp, FeedForward::Moe(_)));
        assert!(matches!(model.model.layers[1].mlp, FeedForward::Dense(_)));
    }

    #[test]
    fn parses_converter_affine_metadata_from_both_config_keys() {
        use crate::runtime::checkpoint::quantization::{AffineQuantization, WeightQuantization};
        let quantization = WeightQuantization::Affine(AffineQuantization::new(32, 4).unwrap());
        let metadata = serde_json::to_value(quantization).unwrap();
        let mut value = affine_config_value();
        value
            .as_object_mut()
            .unwrap()
            .insert("quantization".into(), metadata.clone());
        value
            .as_object_mut()
            .unwrap()
            .insert("quantization_config".into(), metadata);
        let args = parse_config_value(value).unwrap();
        assert_eq!(args.affine_quantization().unwrap(), Some(quantization));
        assert!(args.native_fp8_config().is_none());
        assert_eq!(
            args.weight_format_for("model.layers.0.self_attn.q_a_proj.weight"),
            super::WeightFormat::Affine(quantization)
        );
    }

    #[test]
    fn parameter_tree_tracks_query_variant_and_dense_to_moe_transition() {
        let context = test_context();
        let stream = context.stream();
        let lora = Model::new(tiny_args(Some(4)), stream).unwrap();
        let keys = lora.parameters().flatten();
        assert!(keys.contains_key("model.layers.0.self_attn.q_a_proj.weight"));
        assert!(keys.contains_key("model.layers.0.self_attn.q_a_layernorm.weight"));
        assert!(keys.contains_key("model.layers.0.self_attn.q_b_proj.weight"));
        assert!(!keys.contains_key("model.layers.0.self_attn.q_proj.weight"));
        assert!(matches!(lora.model.layers[0].mlp, FeedForward::Dense(_)));
        assert!(matches!(lora.model.layers[1].mlp, FeedForward::Moe(_)));
        assert!(keys.contains_key("model.layers.1.mlp.gate.weight"));
        assert!(keys.contains_key("model.layers.1.mlp.gate.e_score_correction_bias"));

        let direct = Model::new(tiny_args(None), stream).unwrap();
        let keys = direct.parameters().flatten();
        assert!(keys.contains_key("model.layers.0.self_attn.q_proj.weight"));
        assert!(!keys.contains_key("model.layers.0.self_attn.q_a_proj.weight"));
    }

    #[test]
    fn detailed_runtime_observer_preserves_logits_and_reports_deepseek_internals() {
        let context = test_context();
        let stream = context.stream();
        let mut plain = Model::new(tiny_args(Some(4)), stream).unwrap();
        initialize_dense_model(&mut plain, stream);
        let directory = temp_dir();
        save_fixture(&directory, &plain, stream, None, Vec::new());
        let weights = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let generalized =
            crate::architectures::deepseek_v3::layerwise::load_deepseek_v3_layerwise_model(
                &directory,
                crate::LayerWeightResidency::FullyResident,
                None,
                stream,
                weights.stream(),
            )
            .unwrap();
        let mut observed = crate::api::Model::DeepSeekV3(generalized);
        let input = Array::from_slice(&[1i32, 2, 3], &[1, 3]);
        let mut plain_cache = plain.new_cache();
        let mut observed_cache = observed.new_cache();

        let expected = plain
            .forward(
                ModelInput {
                    inputs: &input,
                    mask: None,
                    cache: Some(&mut plain_cache),
                },
                stream,
            )
            .unwrap();
        let mut observer = DetailedObserver::default();
        let actual = observed
            .forward_with_observer(&input, None, &mut observed_cache, stream, &mut observer)
            .unwrap();

        let max_error = actual
            .subtract(expected, stream)
            .unwrap()
            .abs(stream)
            .unwrap()
            .max(None, stream)
            .unwrap()
            .item::<f32>(stream);
        assert!(
            max_error < 1e-5,
            "observed forward changed logits by {max_error}"
        );
        let crate::api::ModelCache::DeepSeekV3(observed_cache) = observed_cache else {
            panic!("DeepSeek model returned the wrong cache type")
        };
        assert_eq!(plain_cache.offset(), observed_cache.offset());
        for expected_name in [
            "model.layers.0.input",
            "model.layers.0.input_layernorm",
            "model.layers.0.self_attn.q_a_proj",
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.output",
            "model.layers.1.input",
            "model.layers.1.mlp.gate.top_k_experts",
            "model.layers.1.mlp.routed_expert_output",
            "model.layers.1.mlp.shared_expert_output",
            "model.layers.1.output",
            "model.logits",
        ] {
            assert!(
                observer.names.iter().any(|name| name == expected_name),
                "missing activation {expected_name}"
            );
        }
        assert_eq!(observer.routing_observations, 1);
        assert_eq!(
            observer.interventions,
            ["model.layers.0.output", "model.layers.1.output"]
        );
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn compressed_cache_shapes_do_not_include_attention_heads() {
        let context = test_context();
        let stream = context.stream();
        let mut cache = CompressedLatentCache::new();
        let latent = Array::zeros::<f32>(&[2, 3, 4], stream).unwrap();
        let rotary = Array::zeros::<f32>(&[2, 3, 2], stream).unwrap();
        let (latent, rotary) = cache.update_and_fetch(latent, rotary, stream).unwrap();
        assert_eq!(latent.shape(), &[2, 3, 4]);
        assert_eq!(rotary.shape(), &[2, 3, 2]);
        assert_eq!(cache.offset(), 3);
        assert_eq!(cache.capacity(), 256);

        let latent = Array::zeros::<f32>(&[2, 1, 4], stream).unwrap();
        let rotary = Array::zeros::<f32>(&[2, 1, 2], stream).unwrap();
        let (latent, rotary) = cache.update_and_fetch(latent, rotary, stream).unwrap();
        assert_eq!(latent.shape(), &[2, 4, 4]);
        assert_eq!(rotary.shape(), &[2, 4, 2]);
        assert_eq!(cache.capacity(), 256);

        let latent = Array::from_slice(&vec![3.0f32; 2 * 253 * 4], &[2, 253, 4]);
        let rotary = Array::from_slice(&vec![4.0f32; 2 * 253 * 2], &[2, 253, 2]);
        let (latent, rotary) = cache.update_and_fetch(latent, rotary, stream).unwrap();
        assert_eq!(latent.shape(), &[2, 257, 4]);
        assert_eq!(rotary.shape(), &[2, 257, 2]);
        assert_eq!(cache.offset(), 257);
        assert_eq!(cache.capacity(), 512);
        assert_eq!(
            latent
                .try_index_device((0, 0, 0), stream)
                .unwrap()
                .item::<f32>(stream),
            0.0
        );
        assert_eq!(
            latent
                .try_index_device((1, -1, -1), stream)
                .unwrap()
                .item::<f32>(stream),
            3.0
        );
        assert_eq!(
            rotary
                .try_index_device((1, -1, -1), stream)
                .unwrap()
                .item::<f32>(stream),
            4.0
        );
    }

    #[test]
    fn cached_prefill_decode_matches_uncached_reference() {
        let context = test_context();
        let stream = context.stream();
        let mut cached = Model::new(tiny_args(Some(4)), stream).unwrap();
        initialize_dense_model(&mut cached, stream);
        let mut reference = cached.clone();
        let prompt = Array::from_slice(&[1i32, 2, 3], &[1, 3]);
        let decode = Array::from_slice(&[4i32], &[1, 1]);
        let mut cache = cached.new_cache();
        cached
            .forward(
                ModelInput {
                    inputs: &prompt,
                    mask: None,
                    cache: Some(&mut cache),
                },
                stream,
            )
            .unwrap();
        let cached_logits = cached
            .forward(
                ModelInput {
                    inputs: &decode,
                    mask: None,
                    cache: Some(&mut cache),
                },
                stream,
            )
            .unwrap();
        assert_eq!(cache.offset(), 4);
        let full = Array::from_slice(&[1i32, 2, 3, 4], &[1, 4]);
        let reference_logits = reference
            .forward(
                ModelInput {
                    inputs: &full,
                    mask: None,
                    cache: None,
                },
                stream,
            )
            .unwrap()
            .try_index_device((.., -1.., ..), stream)
            .unwrap();
        let max_error = cached_logits
            .subtract(reference_logits, stream)
            .unwrap()
            .abs(stream)
            .unwrap()
            .max(None, stream)
            .unwrap()
            .item::<f32>(stream);
        assert!(max_error < 1e-4, "cached MLA error {max_error}");
        let (latent, rotary) = cache.layers[0].arrays().unwrap();
        assert_eq!(latent.shape(), &[1, 4, 4]);
        assert_eq!(rotary.shape(), &[1, 4, 2]);
    }

    #[test]
    fn paged_compressed_prefill_and_decode_match_resident_cache() {
        let context = test_context();
        let stream = context.stream();
        let mut resident = Model::new(tiny_args(Some(4)), stream).unwrap();
        initialize_dense_model(&mut resident, stream);
        let mut paged = resident.clone();
        let prompt = Array::from_slice(&[1i32, 2, 3], &[1, 3]);
        let mut resident_cache = resident.new_cache();
        let host_budget = [32usize, 16]
            .into_iter()
            .map(|bytes| {
                host_transfer_capacity_upper_bound(bytes, HostTransferPolicy::Transfer).unwrap()
                    as u64
            })
            .sum();
        let options = PagedCacheOptions::new(2, 192, host_budget, 1)
            .unwrap()
            .with_full_attention(true);
        let mut paged_cache = paged
            .new_cache_with_options(CacheResidencyPolicy::Paged(options))
            .unwrap();

        for model_cache in [&mut resident_cache, &mut paged_cache] {
            let model = if model_cache.layers[0].is_paged() {
                &mut paged
            } else {
                &mut resident
            };
            model
                .forward(
                    ModelInput {
                        inputs: &prompt,
                        mask: None,
                        cache: Some(model_cache),
                    },
                    stream,
                )
                .unwrap();
        }

        for token in [4i32, 5] {
            let decode = Array::from_slice(&[token], &[1, 1]);
            let expected = resident
                .forward(
                    ModelInput {
                        inputs: &decode,
                        mask: None,
                        cache: Some(&mut resident_cache),
                    },
                    stream,
                )
                .unwrap();
            let actual = paged
                .forward(
                    ModelInput {
                        inputs: &decode,
                        mask: None,
                        cache: Some(&mut paged_cache),
                    },
                    stream,
                )
                .unwrap();
            let max_error = actual
                .subtract(expected, stream)
                .unwrap()
                .abs(stream)
                .unwrap()
                .max(None, stream)
                .unwrap()
                .item::<f32>(stream);
            assert!(max_error < 1e-4, "paged MLA error {max_error}");
        }
        assert_eq!(paged_cache.offset(), 5);
        let report = paged_cache.residency_report().unwrap().unwrap();
        assert_eq!(report.logical_cached_tokens, 5);
        assert!(report.compressed_latent_blocks >= 4);
        assert!(report.peak_device_bytes <= 192);
    }

    #[test]
    fn cached_chunked_prefill_matches_uncached_reference() {
        let context = test_context();
        let stream = context.stream();
        let mut cached = Model::new(tiny_args(Some(4)), stream).unwrap();
        initialize_dense_model(&mut cached, stream);
        let mut reference = cached.clone();
        let first = Array::from_slice(&[1i32, 2], &[1, 2]);
        let second = Array::from_slice(&[3i32, 4], &[1, 2]);
        let mut cache = cached.new_cache();
        cached
            .forward(
                ModelInput {
                    inputs: &first,
                    mask: None,
                    cache: Some(&mut cache),
                },
                stream,
            )
            .unwrap();
        let cached_logits = cached
            .forward(
                ModelInput {
                    inputs: &second,
                    mask: None,
                    cache: Some(&mut cache),
                },
                stream,
            )
            .unwrap();

        let full = Array::from_slice(&[1i32, 2, 3, 4], &[1, 4]);
        let reference_logits = reference
            .forward(
                ModelInput {
                    inputs: &full,
                    mask: None,
                    cache: None,
                },
                stream,
            )
            .unwrap()
            .try_index_device((.., 2.., ..), stream)
            .unwrap();
        let max_error = cached_logits
            .subtract(reference_logits, stream)
            .unwrap()
            .abs(stream)
            .unwrap()
            .max(None, stream)
            .unwrap()
            .item::<f32>(stream);
        assert!(max_error < 1e-4, "chunked MLA error {max_error}");
        assert_eq!(cache.offset(), 4);
        assert_eq!(cache.layers[0].capacity(), 256);
    }

    #[test]
    fn strict_loading_rejects_unconfigured_mtp_prefix_and_dispatches_loaded_model() {
        let context = test_context();
        let weights_context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let weights_stream = weights_context.stream();
        let mut source = Model::new(tiny_args(Some(4)), stream).unwrap();
        initialize_dense_model(&mut source, stream);
        let dir = temp_dir();
        save_fixture(&dir, &source, stream, None, Vec::new());
        fs::write(
            dir.join("tokenizer.json"),
            r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<unk>":0,"hello":1},"unk_token":"<unk>"}}"#,
        )
        .unwrap();
        fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"chat_template":"{{ messages[0]['content'] }}","eos_token":"<unk>"}"#,
        )
        .unwrap();

        let loaded = load_fully_resident(&dir, None, stream).unwrap();
        assert_eq!(loaded.args().model_type, "deepseek_v3");
        let loaded = LoadedModel::load(&dir, stream, weights_stream).unwrap();
        assert_eq!(loaded.model_type(), "deepseek_v3");
        assert_eq!(loaded.eos_token_ids(), &[1]);
        assert!(loaded.has_chat_template());

        save_fixture(
            &dir,
            &source,
            stream,
            None,
            vec![(
                "model.layers.3.eh_proj.weight".into(),
                Array::zeros::<f32>(&[8, 16], stream).unwrap(),
            )],
        );
        let error = load_fully_resident(&dir, None, stream).err().unwrap();
        assert!(!error.to_string().is_empty());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn strict_loading_rejects_missing_and_unexpected_text_weights() {
        let context = test_context();
        let stream = context.stream();
        let mut source = Model::new(tiny_args(Some(4)), stream).unwrap();
        initialize_dense_model(&mut source, stream);
        let dir = temp_dir();
        save_fixture(&dir, &source, stream, Some("lm_head.weight"), Vec::new());
        let error = load_fully_resident(&dir, None, stream).err().unwrap();
        assert!(!error.to_string().is_empty());

        save_fixture(
            &dir,
            &source,
            stream,
            None,
            vec![(
                "model.layers.0.unexpected.weight".into(),
                Array::zeros::<f32>(&[1], stream).unwrap(),
            )],
        );
        let error = load_fully_resident(&dir, None, stream).err().unwrap();
        assert!(!error.to_string().is_empty());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn strict_loads_and_runs_checkpoint_shaped_block_fp8_experts() {
        let context = test_context();
        let stream = context.stream();
        let mut source = Model::new(tiny_fp8_args(), stream).unwrap();
        initialize_fp8_model(&mut source, stream);
        let dir = temp_dir();
        save_fixture(&dir, &source, stream, None, Vec::new());
        let mut loaded = load_fully_resident(&dir, None, stream).unwrap();
        let mut cache = loaded.new_cache();
        let prefill = loaded
            .forward(&Array::from_slice(&[1i32, 2], &[1, 2]), &mut cache, stream)
            .unwrap();
        assert_eq!(prefill.shape(), &[1, 2, 32]);
        let chunk = loaded
            .forward(&Array::from_slice(&[3i32, 4], &[1, 2]), &mut cache, stream)
            .unwrap();
        assert_eq!(chunk.shape(), &[1, 2, 32]);
        assert_eq!(cache.offset(), 4);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn noaux_grouped_router_uses_bias_only_for_selection() {
        use crate::nn::moe::{TopKRouter, TopKRouterConfig, TopKRouterScoreFunction};
        let context = test_context();
        let stream = context.stream();
        let mut router = TopKRouter::new(
            TopKRouterConfig {
                top_k: 2,
                num_experts: 8,
                hidden_size: 1,
                score_function: TopKRouterScoreFunction::Sigmoid,
                norm_topk_prob: true,
                normalization_epsilon: 1e-20,
                routed_scaling_factor: 2.0,
                n_group: 2,
                topk_group: 1,
                score_correction_bias: true,
            },
            stream,
        )
        .unwrap();
        router.weight = Param::new(Array::from_slice(
            &[4.0f32, 3.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0],
            &[8, 1],
        ));
        router.e_score_correction_bias = Param::new(Some(Array::from_slice(
            &[0.0f32, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0],
            &[8],
        )));
        let (indices, weights) = router
            .forward(&Array::from_slice(&[1.0f32], &[1, 1]), stream)
            .unwrap();
        let mut selected = vec![
            indices
                .try_index_device((0, 0), stream)
                .unwrap()
                .item::<u32>(stream),
            indices
                .try_index_device((0, 1), stream)
                .unwrap()
                .item::<u32>(stream),
        ];
        selected.sort();
        assert_eq!(selected, vec![4, 5]);
        let total = weights.sum(None, stream).unwrap().item::<f32>(stream);
        assert!((total - 2.0).abs() < 1e-5);
    }

    #[test]
    #[ignore = "set DEEPSEEK_V3_REFERENCE_DIR to a fixture generated by scripts/deepseek_v3_transformers_fixture.py"]
    fn official_transformers_tiny_fixture_logits() {
        let dir = std::env::var_os("DEEPSEEK_V3_REFERENCE_DIR")
            .map(std::path::PathBuf::from)
            .expect("DEEPSEEK_V3_REFERENCE_DIR");
        let fixture: Value = serde_json::from_reader(
            fs::File::open(dir.join("reference.json")).expect("reference.json"),
        )
        .unwrap();
        let ids = fixture["input_ids"][0]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_i64().unwrap() as i32)
            .collect::<Vec<_>>();
        let expected = fixture["logits"][0]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|token| token.as_array().unwrap())
            .map(|value| value.as_f64().unwrap() as f32)
            .collect::<Vec<_>>();
        let context = test_context();
        let stream = context.stream();
        let mut model = load_fully_resident(&dir, None, stream).unwrap();
        let mut cache = model.new_cache();
        let logits = model
            .forward(
                &Array::from_slice(&ids, &[1, ids.len() as i32]),
                &mut cache,
                stream,
            )
            .unwrap();
        let expected = Array::from_slice(&expected, logits.shape());
        let max_error = logits
            .subtract(expected, stream)
            .unwrap()
            .abs(stream)
            .unwrap()
            .max(None, stream)
            .unwrap()
            .item::<f32>(stream);
        assert!(max_error < 2e-4, "official fixture error {max_error}");
    }

    #[test]
    fn affine_and_mxfp4_on_load_run_through_canonical_loader() {
        use crate::runtime::checkpoint::quantization::{AffineQuantization, WeightQuantization};
        let context = test_context();
        let stream = context.stream();
        let mut source =
            Model::new(parse_config_value(affine_config_value()).unwrap(), stream).unwrap();
        initialize_dense_model(&mut source, stream);
        let dir = temp_dir();
        save_fixture(&dir, &source, stream, None, Vec::new());
        let quantization = WeightQuantization::Affine(AffineQuantization::new(32, 4).unwrap());
        let mut quantized = load_fully_resident(&dir, Some(quantization), stream).unwrap();

        let mut cache = quantized.new_cache();
        let logits = quantized
            .forward(&Array::from_slice(&[1i32, 2], &[1, 2]), &mut cache, stream)
            .unwrap();
        assert_eq!(logits.shape(), &[1, 2, 32]);

        let mut mxfp4 = load_fully_resident(&dir, Some(WeightQuantization::MxFp4), stream).unwrap();
        let mut cache = mxfp4.new_cache();
        let logits = mxfp4
            .forward(&Array::from_slice(&[1i32, 2], &[1, 2]), &mut cache, stream)
            .unwrap();
        assert_eq!(logits.shape(), &[1, 2, 32]);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn loads_dense_and_mixed_quantized_deepseek2_gguf_checkpoints() {
        use crate::runtime::checkpoint::quantization::AffineQuantization;
        use safemlx_gguf::GgmlType;
        let context = test_context();
        let stream = context.stream();
        let mut source =
            Model::new(parse_config_value(affine_config_value()).unwrap(), stream).unwrap();
        initialize_dense_model(&mut source, stream);

        let arrays = gguf_arrays(&source, stream);
        assert_eq!(
            super::translate_gguf_weight_name("blk.1.ffn_gate_exps.scales"),
            "model.layers.1.mlp.experts.gate_proj_scales"
        );
        assert_eq!(
            super::translate_gguf_weight_name("blk.1.attn_kv_a_mqa.weight"),
            "model.layers.1.self_attn.kv_a_proj_with_mqa.weight"
        );
        let metadata = gguf_metadata();
        let dense_fixture = crate::test_utils::SyntheticGguf::dense(&arrays, &metadata);
        let mut dense = load_gguf_fully_resident(dense_fixture.path(), None, stream).unwrap();
        assert_eq!(
            dense
                .args()
                .layer_schedule
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![LayerPolicy::DenseMlp, LayerPolicy::SparseMoe]
        );
        let mut cache = dense.new_cache();
        let logits = dense
            .forward(&Array::from_slice(&[1i32, 2], &[1, 2]), &mut cache, stream)
            .unwrap();
        assert_eq!(logits.shape(), &[1, 2, 32]);

        let q4 = AffineQuantization::new(32, 4).unwrap();
        let mut on_load =
            load_gguf_fully_resident(dense_fixture.path(), Some(q4.into()), stream).unwrap();
        let mut cache = on_load.new_cache();
        let logits = on_load
            .forward(&Array::from_slice(&[1i32], &[1, 1]), &mut cache, stream)
            .unwrap();
        assert_eq!(logits.shape(), &[1, 1, 32]);

        let packed_fixture = crate::test_utils::SyntheticGguf::with_packed_tensors(
            &arrays,
            &metadata,
            |name, value| {
                (name.ends_with(".weight")
                    && !name.ends_with("ffn_gate_inp.weight")
                    && value.ndim() >= 2
                    && value.dtype().is_float())
                .then_some(if name.contains("ffn_up_exps") {
                    GgmlType::Q8_0
                } else {
                    GgmlType::Q4_0
                })
            },
        );
        let mut quantized = load_gguf_fully_resident(packed_fixture.path(), None, stream).unwrap();
        let mut cache = quantized.new_cache();
        let logits = quantized
            .forward(&Array::from_slice(&[1i32, 2], &[1, 2]), &mut cache, stream)
            .unwrap();
        assert_eq!(logits.shape(), &[1, 2, 32]);
    }

    #[test]
    fn native_fp8_on_load_transcoding_is_rejected() {
        use crate::runtime::checkpoint::quantization::{AffineQuantization, WeightQuantization};
        let context = test_context();
        let stream = context.stream();
        let mut source = Model::new(tiny_fp8_args(), stream).unwrap();
        initialize_fp8_model(&mut source, stream);
        let dir = temp_dir();
        save_fixture(&dir, &source, stream, None, Vec::new());
        let error = load_fully_resident(
            &dir,
            Some(WeightQuantization::Affine(
                AffineQuantization::new(32, 4).unwrap(),
            )),
            stream,
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains("block-FP8"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    #[ignore = "set DEEPSEEK_V3_GGUF to a local llama.cpp deepseek2 checkpoint"]
    fn real_local_gguf_prefill_decode() {
        let file = std::env::var_os("DEEPSEEK_V3_GGUF")
            .map(std::path::PathBuf::from)
            .expect("DEEPSEEK_V3_GGUF");
        let context = test_context();
        let stream = context.stream();
        let mut model = load_gguf_fully_resident(&file, None, stream).unwrap();
        let mut cache = model.new_cache();
        let prefill = model
            .forward(&Array::from_slice(&[0i32, 1], &[1, 2]), &mut cache, stream)
            .unwrap();
        assert_eq!(prefill.dim(1), 2);
        let decode = model
            .forward(&Array::from_slice(&[2i32], &[1, 1]), &mut cache, stream)
            .unwrap();
        assert_eq!(decode.shape(), &[1, 1, model.args().vocab_size]);
        assert_eq!(cache.offset(), 3);
    }

    #[test]
    #[ignore = "set DEEPSEEK_V3_MODEL_DIR to an official local V3/R1 checkpoint"]
    fn real_local_checkpoint_prefill_decode() {
        let dir = std::env::var_os("DEEPSEEK_V3_MODEL_DIR")
            .map(std::path::PathBuf::from)
            .expect("DEEPSEEK_V3_MODEL_DIR");
        let context = test_context();
        let stream = context.stream();
        let mut model = load_fully_resident(&dir, None, stream).unwrap();
        let mut cache = model.new_cache();
        let logits = model
            .forward(&Array::from_slice(&[0i32, 1], &[1, 2]), &mut cache, stream)
            .unwrap();
        assert_eq!(logits.dim(-1), model.args().vocab_size);
        assert_eq!(cache.offset(), 2);
    }
}
