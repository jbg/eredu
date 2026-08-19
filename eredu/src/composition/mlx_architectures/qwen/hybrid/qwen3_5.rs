//! Dense and MoE Qwen3.5 text and vision-language implementation and loader.

use eredu_checkpoint::AffineQuantization;
use eredu_checkpoint::WeightQuantization;
use eredu_nn::RopeValue;
use eredu_runtime::CausalModel;

use safemlx::{
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, ModuleParameters, Param},
    native_quantization::{native_grouped_linear, NativeQuantizedTensor},
    nn,
    ops::{
        broadcast_to, concatenate_axis, conv1d, exp, gather_grouped_rows, grouped_matmul,
        indexing::{NewAxis, TryIndexOp},
        matmul, quantized_matmul_with_mode, quantized_packed_dimension, sigmoid, sum_axis,
        topk_route_plan, zeros, GgufCheckpoint, GgufMetadataValue, QuantizationMode,
    },
    quantization::MaybeQuantized,
    transforms::eval,
    Array, Dtype, Stream,
};
use serde::{Deserialize, Deserializer};
use serde_json::Value;
use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap},
    path::Path,
    time::Instant,
};
use tokenizers::Tokenizer;

use crate::composition::mlx::structural::GgufArchitectureValidation;
use crate::core::cache::{
    derive_prompt_cache_architecture_fingerprint, PromptCacheDescriptor, PromptCacheManifest,
    PromptCacheModelIdentity, PromptCacheOptions,
};

pub use crate::backend::mlx::nn::generation::sample;
use crate::composition::mlx_architectures::qwen::vl::vision::VisionConfigSource;
#[cfg(test)]
pub(crate) use crate::composition::mlx_architectures::qwen::vl::vision::{
    reverse_permutation, vision_window_index,
};
pub use crate::composition::mlx_architectures::qwen::vl::vision::{
    QwenVisionAttention, QwenVisionBlock, QwenVisionMlp, QwenVisionPatchEmbed,
    QwenVisionPatchMerger, QwenVisionPatchProjection, QwenVisionRmsNorm, QwenVisionTransformer,
    VisionConfig,
};
use crate::composition::mlx_architectures::qwen::vl::{
    model as qwen_vl, vision::grid_thw_from_array,
};

use crate::{
    backend::mlx::error::Error,
    backend::mlx::nn::tensor::{
        create_attention_mask,
        rope::{initialize_rope, validate_rope_scaling_config, RopeVariant},
        AttentionMask,
    },
    backend::mlx::nn::{
        self as common, attention::attention_probabilities, layers::silu,
        linear::project_logits_maybe_quantized, moe::TopKRouterScoreFunction,
    },
    backend::mlx::runtime::cache::{
        residency::{
            open_prompt_cache_snapshot, save_prompt_cache_snapshot, CacheBlockArrays,
            CacheResidencyManager, CacheResidencyReport, PromptCacheSnapshotBlock,
            PromptCacheStateArray,
        },
        ConcatKeyValueCache, KeyValueCache, LiveKeyValueCache,
    },
    backend::mlx::runtime::checkpoint::load::{
        gguf_metadata, gguf_quantization_configs, GgufTensorNames,
    },
    backend::mlx::runtime::execution::inspection::{ActivationObserver, MoeRoutingObservation},
    backend::mlx::runtime::media::input as runtime_input,
    core::attention::{AttentionPolicy, LayerSchedule},
    core::cache::{
        CacheRankIdentity, LayerCachePolicy, StateTensorDimension, StateTensorDtype,
        StateTensorOwner, StateTensorPolicy, StateTensorRole,
    },
};
use eredu_runtime::CacheResidencyPool;
use eredu_runtime::{CacheResidencyPolicy, PagedCacheOptions};

#[cfg(test)]
#[cfg(test)]
use crate::backend::mlx::runtime::checkpoint::load::{
    for_each_safetensor_array, load_array_strict, safetensors_files, StrictLoadConfig,
    StrictLoadReport,
};

#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
/// Stateful operator policy for one Qwen3.5 or Qwen3-Next decoder layer.
pub enum LayerPolicy {
    /// Recurrent linear-attention layer.
    LinearAttention,
    /// Self-attention layer with its exact attention policy.
    SelfAttention(AttentionPolicy),
}

#[derive(Debug, Clone, Copy)]
enum LayerPolicySource {
    LinearAttention,
    FullAttention,
}

impl LayerPolicySource {
    const fn normalize(self) -> LayerPolicy {
        match self {
            Self::LinearAttention => LayerPolicy::LinearAttention,
            Self::FullAttention => LayerPolicy::SelfAttention(AttentionPolicy::Full),
        }
    }
}

impl<'de> Deserialize<'de> for LayerPolicySource {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        match String::deserialize(deserializer)?.as_str() {
            "linear_attention" => Ok(Self::LinearAttention),
            "full_attention" => Ok(Self::FullAttention),
            other => Err(serde::de::Error::custom(format!(
                "unsupported Qwen hybrid layer policy {other:?}"
            ))),
        }
    }
}

const ROUTED_EXPERT_CHUNK_THRESHOLD: i32 = 64;
const ROUTED_EXPERT_CHUNK_TOKENS: i32 = 32;

#[derive(Debug, Clone, Default)]
/// Profiling counters accumulated by Qwen3.5 MoE when profiling is enabled.
pub struct PerfStats {
    /// Time spent evaluating token embeddings.
    pub embed_s: f64,
    /// Time spent evaluating full-attention layers.
    pub full_attention_s: f64,
    /// Time spent evaluating linear-attention layers.
    pub linear_attention_s: f64,
    /// Time spent evaluating MoE routing.
    pub moe_router_s: f64,
    /// Time spent evaluating the shared expert.
    pub moe_shared_s: f64,
    /// Time spent evaluating routed experts.
    pub moe_routed_s: f64,
    /// Time spent combining MoE outputs.
    pub moe_combine_s: f64,
    /// Time spent evaluating final normalization.
    pub final_norm_s: f64,
    /// Time spent projecting hidden states to logits.
    pub lm_head_s: f64,
    /// Time spent materializing the prefill state dependency.
    pub prefill_state_dependency_s: f64,
}

impl PerfStats {
    /// Returns the sum of all profiled component durations.
    pub fn component_total_s(&self) -> f64 {
        self.embed_s
            + self.full_attention_s
            + self.linear_attention_s
            + self.moe_router_s
            + self.moe_shared_s
            + self.moe_routed_s
            + self.moe_combine_s
            + self.final_norm_s
            + self.lm_head_s
            + self.prefill_state_dependency_s
    }

    fn add(&mut self, component: PerfComponent, elapsed_s: f64) {
        match component {
            PerfComponent::Embed => self.embed_s += elapsed_s,
            PerfComponent::FullAttention => self.full_attention_s += elapsed_s,
            PerfComponent::LinearAttention => self.linear_attention_s += elapsed_s,
            PerfComponent::MoeRouter => self.moe_router_s += elapsed_s,
            PerfComponent::MoeShared => self.moe_shared_s += elapsed_s,
            PerfComponent::MoeRouted => self.moe_routed_s += elapsed_s,
            PerfComponent::MoeCombine => self.moe_combine_s += elapsed_s,
            PerfComponent::FinalNorm => self.final_norm_s += elapsed_s,
            PerfComponent::LmHead => self.lm_head_s += elapsed_s,
            PerfComponent::PrefillStateDependency => {
                self.prefill_state_dependency_s += elapsed_s;
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum PerfComponent {
    Embed,
    FullAttention,
    LinearAttention,
    MoeRouter,
    MoeShared,
    MoeRouted,
    MoeCombine,
    FinalNorm,
    LmHead,
    PrefillStateDependency,
}

thread_local! {
    static PERF_STATS: RefCell<Option<PerfStats>> = const { RefCell::new(None) };
}

/// Enables or disables per-thread Qwen3.5 MoE profiling.
pub fn set_perf_profiling(enabled: bool) {
    PERF_STATS.with(|stats| {
        *stats.borrow_mut() = enabled.then(PerfStats::default);
    });
}

/// Resets per-thread Qwen3.5 MoE profiling counters.
pub fn reset_perf_stats() {
    PERF_STATS.with(|stats| {
        if let Some(stats) = stats.borrow_mut().as_mut() {
            *stats = PerfStats::default();
        }
    });
}

/// Returns the current per-thread profiling counters, if profiling is enabled.
pub fn perf_stats() -> Option<PerfStats> {
    PERF_STATS.with(|stats| stats.borrow().clone())
}

fn profile_arrays(component: PerfComponent, arrays: &[&Array]) -> Result<(), Exception> {
    let enabled = PERF_STATS.with(|stats| stats.borrow().is_some());
    if !enabled {
        return Ok(());
    }

    let start = Instant::now();
    eval(arrays.iter().copied())?;
    let elapsed_s = start.elapsed().as_secs_f64();
    PERF_STATS.with(|stats| {
        if let Some(stats) = stats.borrow_mut().as_mut() {
            stats.add(component, elapsed_s);
        }
    });
    Ok(())
}

fn profile_array(component: PerfComponent, array: &Array) -> Result<(), Exception> {
    profile_arrays(component, &[array])
}

#[derive(Debug, Clone)]
/// Normalized dense or MoE Qwen3.5 text configuration used by this loader.
pub struct ModelArgs {
    /// Effective text model type.
    pub model_type: String,
    /// Token vocabulary size.
    pub vocab_size: i32,
    /// Transformer hidden size.
    pub hidden_size: i32,
    /// Number of decoder layers.
    pub num_hidden_layers: i32,
    /// Number of embedded multi-token-prediction layers.
    pub mtp_num_hidden_layers: i32,
    /// Number of full-attention query heads.
    pub num_attention_heads: i32,
    /// Number of full-attention key/value heads.
    pub num_key_value_heads: i32,
    /// Full-attention head dimension.
    pub head_dim: i32,
    /// Maximum configured sequence length.
    pub max_position_embeddings: i32,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// Whether logits use tied input embeddings.
    pub tie_word_embeddings: bool,
    /// Whether full-attention projections include bias terms.
    pub attention_bias: bool,
    /// Activation function name from the config.
    pub hidden_act: String,
    /// Causal convolution kernel width in linear-attention layers.
    pub linear_conv_kernel_dim: i32,
    /// Key head dimension in linear-attention layers.
    pub linear_key_head_dim: i32,
    /// Value head dimension in linear-attention layers.
    pub linear_value_head_dim: i32,
    /// Number of key heads in linear-attention layers.
    pub linear_num_key_heads: i32,
    /// Number of value heads in linear-attention layers.
    pub linear_num_value_heads: i32,
    /// Dense SwiGLU intermediate size. Zero for MoE checkpoints.
    pub intermediate_size: i32,
    /// Routed-expert intermediate size.
    pub moe_intermediate_size: i32,
    /// Shared-expert intermediate size.
    pub shared_expert_intermediate_size: i32,
    /// Number of experts selected per token.
    pub num_experts_per_tok: i32,
    /// Total number of routed experts.
    pub num_experts: i32,
    /// Whether top-k routing probabilities are normalized.
    pub norm_topk_prob: bool,
    /// Authoritative stateful-operator policy in decoder-layer order.
    pub layer_schedule: LayerSchedule<LayerPolicy>,
    /// RoPE parameter overrides.
    pub rope_parameters: Option<HashMap<String, Value>>,
    /// RoPE scaling configuration.
    pub rope_scaling: Option<HashMap<String, Value>>,
    /// Optional FP8 quantization configuration.
    pub quantization_config: Option<QwenFp8QuantizationConfig>,
    /// Optional MLX affine or MXFP4 metadata for standard text weights.
    pub quantization: Option<WeightQuantization>,
    /// Exact GGUF affine settings keyed by runtime weight name.
    pub quantized_weight_configs: Option<HashMap<String, WeightQuantization>>,
}

/// Rank-local execution geometry for one hybrid decoder layer.
///
/// Attention/recurrent heads and feed-forward intermediates are independent
/// logical partition domains. Keeping both dimensions explicit prevents
/// packed checkpoint shapes from becoming an accidental source of runtime
/// geometry.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) struct ParallelLayerGeometry {
    pub attention: ParallelAttentionGeometry,
    pub feed_forward: ParallelFeedForwardGeometry,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum ParallelAttentionGeometry {
    Full { query_heads: i32, kv_heads: i32 },
    Linear { key_heads: i32, value_heads: i32 },
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum ParallelFeedForwardGeometry {
    Dense {
        intermediate: i32,
    },
    Moe {
        routed_intermediate: i32,
        shared_intermediate: i32,
    },
}

impl ParallelLayerGeometry {
    fn resident(args: &ModelArgs, layer: usize) -> Result<Self, Error> {
        let attention = match args.layer_schedule.get(layer).copied() {
            Some(LayerPolicy::SelfAttention(AttentionPolicy::Full)) => {
                ParallelAttentionGeometry::Full {
                    query_heads: args.num_attention_heads,
                    kv_heads: args.num_key_value_heads,
                }
            }
            Some(LayerPolicy::LinearAttention) => ParallelAttentionGeometry::Linear {
                key_heads: args.linear_num_key_heads,
                value_heads: args.linear_num_value_heads,
            },
            Some(LayerPolicy::SelfAttention(AttentionPolicy::Sliding { .. })) => {
                return Err(Error::UnsupportedArchitecture(
                    "Qwen hybrid execution does not support sliding self-attention".into(),
                ));
            }
            None => {
                return Err(Error::UnsupportedArchitecture(format!(
                    "Qwen hybrid layer schedule has no decoder layer {layer}"
                )));
            }
        };
        let feed_forward = if args.is_moe() {
            ParallelFeedForwardGeometry::Moe {
                routed_intermediate: args.moe_intermediate_size,
                shared_intermediate: args.shared_expert_intermediate_size,
            }
        } else {
            ParallelFeedForwardGeometry::Dense {
                intermediate: args.intermediate_size,
            }
        };
        Ok(Self {
            attention,
            feed_forward,
        })
    }

    fn local_args(self, args: &ModelArgs) -> ModelArgs {
        let mut local = args.clone();
        match self.attention {
            ParallelAttentionGeometry::Full {
                query_heads,
                kv_heads,
            } => {
                local.num_attention_heads = query_heads;
                local.num_key_value_heads = kv_heads;
            }
            ParallelAttentionGeometry::Linear {
                key_heads,
                value_heads,
            } => {
                local.linear_num_key_heads = key_heads;
                local.linear_num_value_heads = value_heads;
            }
        }
        match self.feed_forward {
            ParallelFeedForwardGeometry::Dense { intermediate } => {
                local.intermediate_size = intermediate;
            }
            ParallelFeedForwardGeometry::Moe {
                routed_intermediate,
                shared_intermediate,
            } => {
                local.moe_intermediate_size = routed_intermediate;
                local.shared_expert_intermediate_size = shared_intermediate;
            }
        }
        local
    }
}

#[derive(Debug, Clone, Deserialize)]
struct ModelArgsSource {
    #[serde(default = "default_text_model_type")]
    model_type: String,
    vocab_size: i32,
    hidden_size: i32,
    num_hidden_layers: i32,
    #[serde(default, alias = "num_nextn_predict_layers")]
    mtp_num_hidden_layers: i32,
    num_attention_heads: i32,
    num_key_value_heads: i32,
    #[serde(default = "default_head_dim")]
    head_dim: i32,
    max_position_embeddings: i32,
    #[serde(default = "default_rms_norm_eps")]
    rms_norm_eps: f32,
    #[serde(default = "default_true")]
    tie_word_embeddings: bool,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default = "default_hidden_act")]
    hidden_act: String,
    #[serde(default = "default_linear_conv_kernel_dim")]
    linear_conv_kernel_dim: i32,
    #[serde(default = "default_linear_key_head_dim")]
    linear_key_head_dim: i32,
    #[serde(default = "default_linear_value_head_dim")]
    linear_value_head_dim: i32,
    #[serde(default = "default_linear_num_key_heads")]
    linear_num_key_heads: i32,
    #[serde(default = "default_linear_num_value_heads")]
    linear_num_value_heads: i32,
    #[serde(default)]
    intermediate_size: i32,
    #[serde(default = "default_moe_intermediate_size")]
    moe_intermediate_size: i32,
    #[serde(default = "default_shared_expert_intermediate_size")]
    shared_expert_intermediate_size: i32,
    #[serde(default = "default_num_experts_per_tok")]
    num_experts_per_tok: i32,
    #[serde(default = "default_num_experts")]
    num_experts: i32,
    #[serde(default)]
    norm_topk_prob: bool,
    #[serde(default)]
    layer_types: Vec<LayerPolicySource>,
    #[serde(default)]
    rope_parameters: Option<HashMap<String, Value>>,
    #[serde(default)]
    rope_scaling: Option<HashMap<String, Value>>,
    #[serde(default, deserialize_with = "deserialize_optional_fp8")]
    quantization_config: Option<QwenFp8QuantizationConfig>,
    #[serde(default)]
    quantization: Option<WeightQuantization>,
}

impl ModelArgsSource {
    fn normalize(self, layer_schedule: LayerSchedule<LayerPolicy>) -> ModelArgs {
        ModelArgs {
            model_type: self.model_type,
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            mtp_num_hidden_layers: self.mtp_num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps: self.rms_norm_eps,
            tie_word_embeddings: self.tie_word_embeddings,
            attention_bias: self.attention_bias,
            hidden_act: self.hidden_act,
            linear_conv_kernel_dim: self.linear_conv_kernel_dim,
            linear_key_head_dim: self.linear_key_head_dim,
            linear_value_head_dim: self.linear_value_head_dim,
            linear_num_key_heads: self.linear_num_key_heads,
            linear_num_value_heads: self.linear_num_value_heads,
            intermediate_size: self.intermediate_size,
            moe_intermediate_size: self.moe_intermediate_size,
            shared_expert_intermediate_size: self.shared_expert_intermediate_size,
            num_experts_per_tok: self.num_experts_per_tok,
            num_experts: self.num_experts,
            norm_topk_prob: self.norm_topk_prob,
            layer_schedule,
            rope_parameters: self.rope_parameters,
            rope_scaling: self.rope_scaling,
            quantization_config: self.quantization_config,
            quantization: self.quantization,
            quantized_weight_configs: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
/// FP8 quantization settings supported by the Qwen3.5 MoE loader.
pub struct QwenFp8QuantizationConfig {
    /// Quantization method, expected to be `fp8`.
    pub quant_method: String,
    /// FP8 format, expected to be `e4m3`.
    pub fmt: String,
    /// Activation quantization scheme, expected to be `dynamic`.
    pub activation_scheme: String,
    #[serde(default)]
    /// FP8 weight block size.
    pub weight_block_size: Option<Vec<i32>>,
    #[serde(default)]
    /// Module names excluded from quantization.
    pub modules_to_not_convert: Vec<String>,
}

impl QwenFp8QuantizationConfig {
    pub(crate) fn validate_supported(&self) -> Result<(), Error> {
        if self.quant_method != "fp8" {
            return Err(Error::UnsupportedArchitecture(format!(
                "unsupported Qwen3.5-MoE quantization method '{}'",
                self.quant_method
            )));
        }
        if self.fmt != "e4m3" {
            return Err(Error::UnsupportedArchitecture(format!(
                "unsupported Qwen3.5-MoE FP8 format '{}'",
                self.fmt
            )));
        }
        if self.activation_scheme != "dynamic" {
            return Err(Error::UnsupportedArchitecture(format!(
                "unsupported Qwen3.5-MoE FP8 activation scheme '{}'",
                self.activation_scheme
            )));
        }
        match self.weight_block_size.as_deref() {
            Some([128, 128]) => Ok(()),
            Some(other) => Err(Error::UnsupportedArchitecture(format!(
                "unsupported Qwen3.5-MoE FP8 weight block size {other:?}"
            ))),
            None => Err(Error::UnsupportedArchitecture(
                "Qwen3.5-MoE FP8 config is missing weight_block_size".to_string(),
            )),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct TopLevelConfig {
    model_type: String,
    #[serde(default)]
    text_config: Option<ModelArgsSource>,
    #[serde(default)]
    vision_config: Option<VisionConfigSource>,
    #[serde(default, deserialize_with = "deserialize_optional_fp8")]
    quantization_config: Option<QwenFp8QuantizationConfig>,
    #[serde(default)]
    quantization: Option<WeightQuantization>,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
    #[serde(default)]
    image_token_id: Option<i32>,
    #[serde(default)]
    video_token_id: Option<i32>,
}

fn default_true() -> bool {
    true
}

fn deserialize_optional_fp8<'de, D>(
    deserializer: D,
) -> Result<Option<QwenFp8QuantizationConfig>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<Value>::deserialize(deserializer)?;
    match value {
        Some(value) if value.get("mode").is_some() => Ok(None),
        Some(value) => serde_json::from_value(value)
            .map(Some)
            .map_err(serde::de::Error::custom),
        None => Ok(None),
    }
}

fn default_text_model_type() -> String {
    "qwen3_5_moe_text".to_string()
}

fn default_hidden_act() -> String {
    "silu".to_string()
}

fn default_head_dim() -> i32 {
    256
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}

fn default_linear_conv_kernel_dim() -> i32 {
    4
}

fn default_linear_key_head_dim() -> i32 {
    128
}

fn default_linear_value_head_dim() -> i32 {
    128
}

fn default_linear_num_key_heads() -> i32 {
    16
}

fn default_linear_num_value_heads() -> i32 {
    32
}

fn default_moe_intermediate_size() -> i32 {
    512
}

fn default_shared_expert_intermediate_size() -> i32 {
    512
}

fn default_num_experts_per_tok() -> i32 {
    8
}

fn default_num_experts() -> i32 {
    256
}

fn float_config_value(config: &Option<HashMap<String, Value>>, key: &str) -> Option<f32> {
    config.as_ref().and_then(|config| {
        config.get(key).and_then(|value| match value {
            Value::Number(v) => v.as_f64().map(|v| v as f32),
            Value::String(s) => s.parse().ok(),
            _ => None,
        })
    })
}

fn string_config_value<'a>(
    config: &'a Option<HashMap<String, Value>>,
    key: &str,
) -> Option<&'a str> {
    config.as_ref().and_then(|config| {
        config.get(key).and_then(|value| match value {
            Value::String(s) => Some(s.as_str()),
            _ => None,
        })
    })
}

fn rope_config_value(config: Option<HashMap<String, Value>>) -> Option<HashMap<String, RopeValue>> {
    config.map(|config| {
        config
            .into_iter()
            .filter_map(|(key, value)| {
                let value = match value {
                    Value::Number(v) => v.as_f64().map(|v| RopeValue::Float(v as f32)),
                    Value::String(s) => Some(RopeValue::String(s)),
                    _ => None,
                }?;
                Some((key, value))
            })
            .collect()
    })
}

fn ceil_div(lhs: i32, rhs: i32) -> i32 {
    (lhs + rhs - 1) / rhs
}

impl ModelArgs {
    pub(crate) fn uses_fp8(&self) -> bool {
        self.quantization_config.is_some()
    }

    pub(crate) fn quantization_for(&self, weight_name: &str) -> Option<WeightQuantization> {
        self.quantized_weight_configs
            .as_ref()
            .and_then(|configs| configs.get(weight_name).copied())
    }

    fn weight_format_for(&self, weight_name: &str, fallback: QwenWeightFormat) -> QwenWeightFormat {
        self.quantization_for(weight_name)
            .map(|format| match format {
                iq @ WeightQuantization::GgufIQuant { .. } => QwenWeightFormat::IQuant(iq),
                affine => QwenWeightFormat::Affine(affine),
            })
            .unwrap_or(fallback)
    }

    pub(crate) fn is_moe(&self) -> bool {
        self.num_experts > 0
    }

    fn rope_theta(&self) -> f32 {
        float_config_value(&self.rope_parameters, "rope_theta")
            .or_else(|| float_config_value(&self.rope_scaling, "rope_theta"))
            .unwrap_or(1_000_000.0)
    }

    fn rope_config(&self) -> Option<HashMap<String, RopeValue>> {
        rope_config_value(
            self.rope_parameters
                .clone()
                .or_else(|| self.rope_scaling.clone()),
        )
    }

    fn partial_rotary_factor(&self) -> f32 {
        float_config_value(&self.rope_parameters, "partial_rotary_factor")
            .or_else(|| float_config_value(&self.rope_scaling, "partial_rotary_factor"))
            .unwrap_or(0.25)
    }

    fn rope_dims(&self) -> i32 {
        let rope_type = string_config_value(&self.rope_parameters, "rope_type")
            .or_else(|| string_config_value(&self.rope_scaling, "rope_type"))
            .unwrap_or("default");
        if rope_type == "proportional" {
            self.head_dim
        } else {
            ((self.head_dim as f32 * self.partial_rotary_factor()).round() as i32)
                .clamp(2, self.head_dim)
        }
    }
}

fn canonical_config_map(value: &Option<HashMap<String, Value>>) -> String {
    value.as_ref().map_or_else(String::new, |value| {
        value
            .iter()
            .map(|(key, value)| (key.clone(), value.to_string()))
            .collect::<BTreeMap<_, _>>()
            .into_iter()
            .map(|(key, value)| format!("{key}={value}"))
            .collect::<Vec<_>>()
            .join(";")
    })
}

pub(crate) fn prompt_cache_architecture_fingerprint(args: &ModelArgs) -> String {
    derive_prompt_cache_architecture_fingerprint(
        "qwen_hybrid",
        [
            ("model_type", args.model_type.clone()),
            ("layers", args.num_hidden_layers.to_string()),
            (
                "layer_types",
                args.layer_schedule
                    .iter()
                    .map(|policy| format!("{policy:?}"))
                    .collect::<Vec<_>>()
                    .join(","),
            ),
            ("kv_heads", args.num_key_value_heads.to_string()),
            ("head_dim", args.head_dim.to_string()),
            ("linear_conv", args.linear_conv_kernel_dim.to_string()),
            ("linear_key_dim", args.linear_key_head_dim.to_string()),
            ("linear_value_dim", args.linear_value_head_dim.to_string()),
            ("linear_key_heads", args.linear_num_key_heads.to_string()),
            (
                "linear_value_heads",
                args.linear_num_value_heads.to_string(),
            ),
            ("max_positions", args.max_position_embeddings.to_string()),
            (
                "rope_parameters",
                canonical_config_map(&args.rope_parameters),
            ),
            ("rope_scaling", canonical_config_map(&args.rope_scaling)),
        ],
    )
}

pub(crate) fn prompt_cache_layer_layout(
    args: &ModelArgs,
) -> Result<LayerSchedule<LayerCachePolicy>, Error> {
    let geometry = (0..args.layer_schedule.len())
        .map(|layer| ParallelLayerGeometry::resident(args, layer))
        .collect::<Result<Vec<_>, _>>()?;
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
    let history = args.linear_conv_kernel_dim.checked_sub(1).ok_or_else(|| {
        Error::UnsupportedArchitecture("invalid Qwen linear convolution kernel".into())
    })?;
    let layers = args.num_hidden_layers as usize;
    if geometry.len() != layers {
        return Err(Error::UnsupportedArchitecture(format!(
            "Qwen hybrid cache geometry has {} layers, expected {layers}",
            geometry.len()
        )));
    }
    let policies = args
        .layer_schedule
        .iter()
        .copied()
        .zip(geometry.iter().copied())
        .enumerate()
        .map(|(layer, (policy, geometry))| match (policy, geometry.attention) {
            (
                LayerPolicy::SelfAttention(attention),
                ParallelAttentionGeometry::Full { kv_heads, .. },
            ) => LayerCachePolicy::key_value(attention, kv_heads, args.head_dim)
                .map_err(cache_error),
            (
                LayerPolicy::LinearAttention,
                ParallelAttentionGeometry::Linear {
                    key_heads,
                    value_heads,
                },
            ) => {
                let key_dim = key_heads.checked_mul(args.linear_key_head_dim).ok_or_else(|| {
                    Error::UnsupportedArchitecture("Qwen linear key width overflow".into())
                })?;
                let value_dim = value_heads
                    .checked_mul(args.linear_value_head_dim)
                    .ok_or_else(|| {
                        Error::UnsupportedArchitecture("Qwen linear value width overflow".into())
                    })?;
                let conv_dim = key_dim
                    .checked_mul(2)
                    .and_then(|value| value.checked_add(value_dim))
                    .ok_or_else(|| {
                        Error::UnsupportedArchitecture(
                            "Qwen linear convolution width overflow".into(),
                        )
                    })?;
                LayerCachePolicy::fixed_only(vec![
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
                StateTensorPolicy::new(
                    StateTensorRole::Recurrent,
                    vec![
                        StateTensorDimension::Batch,
                        fixed(value_heads)?,
                        fixed(args.linear_key_head_dim)?,
                        fixed(args.linear_value_head_dim)?,
                    ],
                    StateTensorDtype::Float32,
                    crate::MutableStateResidency::LayerScopedOffloadable,
                )
                .map_err(cache_error)?,
            ])
                .map_err(cache_error)
            }
            (policy, geometry) => Err(Error::UnsupportedArchitecture(format!(
                "Qwen hybrid cache geometry {geometry:?} does not match layer {layer} policy {policy:?}"
            ))),
        })
        .collect::<Result<Vec<_>, _>>()?;
    LayerSchedule::new(layers, policies)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum QwenWeightFormat {
    Dense,
    Fp8,
    /// Native block-FP8 with unsigned E8M0 scale bytes.
    Fp8E8M0,
    Affine(WeightQuantization),
    IQuant(WeightQuantization),
}

impl QwenWeightFormat {
    pub(crate) fn for_text(args: &ModelArgs, affine: Option<WeightQuantization>) -> Self {
        match affine.or(args.quantization) {
            Some(iq @ WeightQuantization::GgufIQuant { .. }) => Self::IQuant(iq),
            Some(affine) => Self::Affine(affine),
            None if args.uses_fp8() => Self::Fp8,
            None => Self::Dense,
        }
    }

    pub(crate) fn affine(self) -> Option<WeightQuantization> {
        match self {
            Self::Affine(affine) => Some(affine),
            Self::Dense | Self::Fp8 | Self::Fp8E8M0 | Self::IQuant(_) => None,
        }
    }

    pub(crate) fn quantization(self) -> Option<WeightQuantization> {
        match self {
            Self::Affine(quantization) | Self::IQuant(quantization) => Some(quantization),
            Self::Dense | Self::Fp8 | Self::Fp8E8M0 => None,
        }
    }

    fn iquant(self) -> Option<WeightQuantization> {
        match self {
            Self::IQuant(iq) => Some(iq),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// Linear layer that can hold dense, Qwen FP8, or MLX affine weights.
pub struct QwenLinear {
    /// Input feature dimension.
    pub input_dims: i32,
    /// Output feature dimension.
    pub output_dims: i32,
    #[param]
    /// Weight tensor.
    pub weight: Param<Array>,
    #[param]
    /// Optional FP8 inverse scale tensor.
    pub weight_scale_inv: Param<Option<Array>>,
    #[param]
    /// Optional affine quantization scales.
    pub scales: Param<Option<Array>>,
    #[param]
    /// Optional affine quantization biases.
    pub biases: Param<Option<Array>>,
    #[param]
    /// Optional bias tensor.
    pub bias: Param<Option<Array>>,
    /// Affine quantization group size, or zero for dense/FP8 storage.
    pub group_size: i32,
    /// Affine quantization bit width, or zero for dense/FP8 storage.
    pub bits: i32,
    /// Quantized weight encoding for packed storage.
    pub mode: QuantizationMode,
    /// Checkpoint-native IQ encoding and byte order.
    pub iquant: Option<WeightQuantization>,
}

impl QwenLinear {
    pub(crate) fn new(
        input_dims: i32,
        output_dims: i32,
        bias: bool,
        format: QwenWeightFormat,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        if let Some(quantization) = format.affine() {
            if input_dims <= 0 || input_dims % quantization.group_size() != 0 {
                return Err(Exception::custom(format!(
                    "Qwen affine linear input dimension {input_dims} is not divisible by group size {}",
                    quantization.group_size()
                )));
            }
        }
        let (weight_shape, weight_dtype) = match format {
            QwenWeightFormat::Dense => (vec![output_dims, input_dims], Dtype::Float32),
            QwenWeightFormat::Fp8 | QwenWeightFormat::Fp8E8M0 => {
                (vec![output_dims, input_dims], Dtype::Uint8)
            }
            QwenWeightFormat::Affine(quantization) => (
                vec![
                    output_dims,
                    quantized_packed_dimension(input_dims, quantization.bits()),
                ],
                Dtype::Uint32,
            ),
            QwenWeightFormat::IQuant(quantization) => {
                let (ggml_type, _) = quantization.gguf_iquant().expect("IQ format");
                let (block_values, block_bytes) = ggml_type.block_and_bytes().unwrap();
                (
                    vec![
                        output_dims,
                        input_dims / block_values as i32 * block_bytes as i32,
                    ],
                    Dtype::Uint8,
                )
            }
        };
        Ok(Self {
            input_dims,
            output_dims,
            weight: Param::<Array>::unloaded(&weight_shape, weight_dtype, stream)?,
            weight_scale_inv: if matches!(format, QwenWeightFormat::Fp8 | QwenWeightFormat::Fp8E8M0)
            {
                Param::<Option<Array>>::unloaded_some(
                    &[ceil_div(output_dims, 128), ceil_div(input_dims, 128)],
                    if format == QwenWeightFormat::Fp8E8M0 {
                        Dtype::Uint8
                    } else {
                        Dtype::Float32
                    },
                    stream,
                )?
            } else {
                Param::new(None)
            },
            scales: if let Some(quantization) = format.affine() {
                Param::<Option<Array>>::unloaded_some(
                    &[output_dims, input_dims / quantization.group_size()],
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
            biases: if let Some(quantization) = format.affine().filter(|q| q.has_biases()) {
                Param::<Option<Array>>::unloaded_some(
                    &[output_dims, input_dims / quantization.group_size()],
                    Dtype::Float32,
                    stream,
                )?
            } else {
                Param::new(None)
            },
            bias: if bias {
                Param::<Option<Array>>::unloaded_some(&[output_dims], Dtype::Float32, stream)?
            } else {
                Param::new(None)
            },
            group_size: format.affine().map_or(0, WeightQuantization::group_size),
            bits: format.affine().map_or(0, WeightQuantization::bits),
            mode: format.affine().map_or(
                QuantizationMode::Affine,
                crate::backend::mlx::runtime::checkpoint::quantization::mlx_quantization_mode,
            ),
            iquant: format.iquant(),
        })
    }

    fn new_dense_with_weight_dtype(
        input_dims: i32,
        output_dims: i32,
        bias: bool,
        weight_dtype: Dtype,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let mut linear = Self::new(
            input_dims,
            output_dims,
            bias,
            QwenWeightFormat::Dense,
            stream,
        )?;
        linear.weight = Param::<Array>::unloaded(&[output_dims, input_dims], weight_dtype, stream)?;
        Ok(linear)
    }

    pub(crate) fn forward(&mut self, input: &Array, stream: &Stream) -> Result<Array, Exception> {
        let mut output = if let Some(iquant) = self.iquant {
            let (ggml_type, endian) = iquant.gguf_iquant().expect("IQ format");
            safemlx::native_quantization::NativeQuantizedTensor::from_iq_array(
                self.weight.value.clone(),
                &[self.output_dims, self.input_dims],
                ggml_type,
                endian,
            )?
            .linear(input, true, stream)?
        } else if let Some(scales) = self.scales.as_ref() {
            quantized_matmul_with_mode(
                input,
                self.weight.as_ref(),
                scales,
                self.biases.as_ref().as_ref(),
                true,
                self.group_size,
                self.bits,
                self.mode,
                stream,
            )?
        } else if let Some(scale) = self.weight_scale_inv.as_ref() {
            common::fp8::linear(input, self.weight.as_ref(), scale, stream)?
        } else {
            matmul(input, self.weight.as_ref().transpose(stream)?, stream)?
        };
        if let Some(bias) = self.bias.as_ref() {
            output = output.add(bias, stream)?;
        }
        Ok(output)
    }

    fn forward_pair(
        first: &mut Self,
        second: &mut Self,
        input: &Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        if first.scales.as_ref().is_none() && second.scales.as_ref().is_none() {
            if let (Some(first_scale), Some(second_scale)) = (
                first.weight_scale_inv.as_ref(),
                second.weight_scale_inv.as_ref(),
            ) {
                let (mut first_output, mut second_output) = common::fp8::linear_pair(
                    input,
                    first.weight.as_ref(),
                    first_scale,
                    second.weight.as_ref(),
                    second_scale,
                    stream,
                )?;
                if let Some(bias) = first.bias.as_ref() {
                    first_output = first_output.add(bias, stream)?;
                }
                if let Some(bias) = second.bias.as_ref() {
                    second_output = second_output.add(bias, stream)?;
                }
                return Ok((first_output, second_output));
            }
        }
        Ok((
            first.forward(input, stream)?,
            second.forward(input, stream)?,
        ))
    }

    fn training_mode(&mut self, _mode: bool) {}

    pub(crate) fn dequantized_weight(&self, stream: &Stream) -> Result<Array, Exception> {
        if let Some(iquant) = self.iquant {
            let (ggml_type, endian) = iquant.gguf_iquant().expect("IQ format");
            safemlx::native_quantization::NativeQuantizedTensor::from_iq_array(
                self.weight.value.clone(),
                &[self.output_dims, self.input_dims],
                ggml_type,
                endian,
            )?
            .dequantize(stream)
        } else if let Some(scales) = self.scales.as_ref() {
            safemlx::ops::dequantize_with_mode(
                self.weight.as_ref(),
                scales,
                self.biases.as_ref().as_ref(),
                self.group_size,
                self.bits,
                self.mode,
                stream,
            )
        } else if let Some(scale) = self.weight_scale_inv.as_ref() {
            common::fp8::dequantize(self.weight.as_ref(), scale, stream)
        } else {
            Ok(self.weight.as_ref().clone())
        }
    }
}

#[derive(Debug, Clone)]
/// Heterogeneous cache for Qwen3.5 MoE layers.
pub struct Cache {
    /// One cache entry per transformer layer.
    pub layers: Vec<LayerCache>,
    /// Full-attention caches owned by the embedded MTP layers.
    pub(crate) mtp_layers: Vec<LayerCache>,
}

impl Cache {
    /// Creates an empty cache matching the layer pattern in `args`.
    pub fn new(args: &ModelArgs) -> Result<Self, Error> {
        validate_text_model_args(args, "Qwen hybrid cache")?;
        Ok(Self {
            layers: args
                .layer_schedule
                .iter()
                .map(|policy| match policy {
                    LayerPolicy::SelfAttention(AttentionPolicy::Full) => LayerCache::FullAttention(
                        LiveKeyValueCache::resident(ConcatKeyValueCache::new()),
                    ),
                    LayerPolicy::SelfAttention(AttentionPolicy::Sliding { .. }) => unreachable!(),
                    LayerPolicy::LinearAttention => {
                        LayerCache::LinearAttention(LinearAttentionCache::default())
                    }
                })
                .collect(),
            mtp_layers: (0..args.mtp_num_hidden_layers)
                .map(|_| {
                    LayerCache::FullAttention(LiveKeyValueCache::resident(
                        ConcatKeyValueCache::new(),
                    ))
                })
                .collect(),
        })
    }

    pub(crate) fn new_paged(
        args: &ModelArgs,
        options: PagedCacheOptions,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, Exception> {
        validate_text_model_args(args, "Qwen hybrid paged cache")
            .map_err(|error| Exception::custom(error.to_string()))?;
        let manager = CacheResidencyManager::new(options)
            .map_err(|error| Exception::custom(error.to_string()))?;
        let layers = args
            .layer_schedule
            .iter()
            .copied()
            .enumerate()
            .map(|(layer, policy)| match policy {
                LayerPolicy::SelfAttention(AttentionPolicy::Full) => {
                    LiveKeyValueCache::paged(manager.clone(), layer, None, 0, rank)
                        .map(LayerCache::FullAttention)
                }
                LayerPolicy::LinearAttention => {
                    Ok(LayerCache::LinearAttention(LinearAttentionCache::default()))
                }
                LayerPolicy::SelfAttention(AttentionPolicy::Sliding { .. }) => {
                    unreachable!("validated Qwen hybrid schedules use full attention")
                }
            })
            .collect::<Result<Vec<_>, Exception>>()?;
        let mtp_start = args.num_hidden_layers as usize;
        let mtp_layers = (0..args.mtp_num_hidden_layers as usize)
            .map(|index| {
                LiveKeyValueCache::paged(manager.clone(), mtp_start + index, None, 0, rank)
                    .map(LayerCache::FullAttention)
            })
            .collect::<Result<Vec<_>, Exception>>()?;
        Ok(Self { layers, mtp_layers })
    }

    /// Returns aggregate live full-attention paging observations, if enabled.
    pub fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        self.layers
            .iter()
            .chain(&self.mtp_layers)
            .find_map(|layer| match layer {
                LayerCache::FullAttention(cache) => cache.residency_report().transpose(),
                LayerCache::LinearAttention(_) => None,
            })
            .transpose()
    }

    /// Returns the aggregate process pool for paged full-attention state.
    pub fn residency_pool(&self) -> Option<&CacheResidencyPool> {
        self.layers
            .iter()
            .chain(&self.mtp_layers)
            .find_map(|layer| match layer {
                LayerCache::FullAttention(cache) => {
                    cache.manager().map(CacheResidencyManager::pool)
                }
                LayerCache::LinearAttention(_) => None,
            })
    }

    pub(crate) fn offset(&self) -> i32 {
        self.layers
            .iter()
            .map(|layer| match layer {
                LayerCache::FullAttention(cache) => cache.offset(),
                LayerCache::LinearAttention(cache) => cache.offset,
            })
            .next()
            .unwrap_or(0)
    }

    pub(crate) fn reset(&mut self) -> Result<(), Exception> {
        if let Some(manager) = self
            .layers
            .iter()
            .chain(&self.mtp_layers)
            .find_map(|layer| match layer {
                LayerCache::FullAttention(cache) => cache.manager().cloned(),
                LayerCache::LinearAttention(_) => None,
            })
        {
            manager
                .clear()
                .map_err(|error| Exception::custom(error.to_string()))?;
        }
        for layer in &mut self.layers {
            match layer {
                LayerCache::FullAttention(cache) => cache.reset_local_after_manager_clear(),
                LayerCache::LinearAttention(cache) => *cache = LinearAttentionCache::default(),
            }
        }
        for cache in &mut self.mtp_layers {
            if let LayerCache::FullAttention(cache) = cache {
                cache.reset_local_after_manager_clear();
            }
        }
        Ok(())
    }

    fn prefill_state_dependency(&self, stream: &Stream) -> Result<Option<Array>, Exception> {
        let mut dependency: Option<Array> = None;
        for layer in &self.layers {
            match layer {
                LayerCache::FullAttention(cache) => {
                    for array in cache.retained_arrays() {
                        let term = array.sum(None, stream)?;
                        dependency = Some(match dependency {
                            Some(acc) => acc.add(term, stream)?,
                            None => term,
                        });
                    }
                }
                LayerCache::LinearAttention(cache) => {
                    if let Some(conv_state) = &cache.conv_state {
                        let term = conv_state.sum(None, stream)?;
                        dependency = Some(match dependency {
                            Some(acc) => acc.add(term, stream)?,
                            None => term,
                        });
                    }
                    if let Some(recurrent_state) = &cache.recurrent_state {
                        let term = recurrent_state.sum(None, stream)?;
                        dependency = Some(match dependency {
                            Some(acc) => acc.add(term, stream)?,
                            None => term,
                        });
                    }
                }
            }
        }
        dependency
            .map(|dependency| dependency.multiply(Array::from_f32(0.0), stream))
            .transpose()
    }
}

#[derive(Debug, Clone)]
/// Per-layer cache for a Qwen3.5 MoE layer.
pub enum LayerCache {
    /// Full-attention key/value cache.
    FullAttention(LiveKeyValueCache),
    /// Linear-attention convolution and recurrent cache.
    LinearAttention(LinearAttentionCache),
}

impl LayerCache {
    pub(crate) fn retained_arrays(&self) -> Vec<&Array> {
        match self {
            Self::FullAttention(cache) => cache.retained_arrays(),
            Self::LinearAttention(cache) => cache
                .conv_state
                .iter()
                .chain(cache.recurrent_state.iter())
                .collect(),
        }
    }
}

#[derive(Debug, Clone, Default)]
/// Cache state for recurrent linear-attention layers.
pub struct LinearAttentionCache {
    /// Cached causal-convolution state.
    pub conv_state: Option<Array>,
    /// Cached recurrent attention state.
    pub recurrent_state: Option<Array>,
    /// Number of tokens consumed by the layer.
    pub offset: i32,
}

#[derive(Debug, Clone, ModuleParameters)]
/// Qwen3Next RMSNorm variant with learned offset scale.
pub struct Qwen3NextRmsNorm {
    #[param]
    /// Learned scale offset.
    pub weight: Param<Array>,
    /// Numerical epsilon.
    pub eps: f32,
}

impl Qwen3NextRmsNorm {
    /// Creates an unloaded RMSNorm layer.
    pub fn new(dim: i32, eps: f32, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            weight: Param::<Array>::unloaded(&[dim], Dtype::Float32, stream)?,
            eps,
        })
    }

    /// Applies normalization.
    pub fn forward(&self, x: &Array, stream: &Stream) -> Result<Array, Exception> {
        let variance = safemlx::ops::mean_axis(&x.square(stream)?, -1, true, stream)?;
        let normalized = x.multiply(
            safemlx::ops::rsqrt(variance.add(Array::from_f32(self.eps), stream)?, stream)?,
            stream,
        )?;
        let scale = self.weight.as_ref().add(Array::from_f32(1.0), stream)?;
        normalized.multiply(scale, stream)
    }

    /// Sets training mode.
    pub fn training_mode(&mut self, _mode: bool) {}
}

#[derive(Debug, Clone, ModuleParameters)]
/// Gated Qwen3Next RMSNorm used by linear attention.
pub struct Qwen3NextRmsNormGated {
    #[param]
    /// Learned scale.
    pub weight: Param<Array>,
    /// Numerical epsilon.
    pub eps: f32,
}

impl Qwen3NextRmsNormGated {
    /// Creates an unloaded gated RMSNorm layer.
    pub fn new(dim: i32, eps: f32, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            weight: Param::<Array>::unloaded(&[dim], Dtype::Float32, stream)?,
            eps,
        })
    }

    /// Applies normalization and SiLU gate modulation.
    pub fn forward(&self, x: &Array, gate: &Array, stream: &Stream) -> Result<Array, Exception> {
        let variance = safemlx::ops::mean_axis(&x.square(stream)?, -1, true, stream)?;
        let normalized = x.multiply(
            safemlx::ops::rsqrt(variance.add(Array::from_f32(self.eps), stream)?, stream)?,
            stream,
        )?;
        normalized
            .multiply(&*self.weight, stream)?
            .multiply(silu(gate.clone(), stream)?, stream)
    }

    /// Sets training mode.
    pub fn training_mode(&mut self, _mode: bool) {}
}

#[derive(Debug, Clone, ModuleParameters)]
/// Full self-attention layer in Qwen3.5 MoE.
pub struct FullAttention {
    /// Number of query heads.
    pub n_heads: i32,
    /// Number of key/value heads.
    pub n_kv_heads: i32,
    /// Per-head dimension.
    pub head_dim: i32,
    /// Attention scaling factor.
    pub scale: f32,
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
    /// Output projection.
    pub o_proj: QwenLinear,
    #[param]
    /// Query normalization.
    pub q_norm: Qwen3NextRmsNorm,
    #[param]
    /// Key normalization.
    pub k_norm: Qwen3NextRmsNorm,
    #[param]
    /// Rotary position embedding module.
    pub rope: RopeVariant,
}

impl FullAttention {
    /// Creates an unloaded full-attention layer.
    pub fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        Self::new_with_format(args, None, QwenWeightFormat::for_text(args, None), stream)
    }

    fn new_with_format(
        args: &ModelArgs,
        prefix: Option<&str>,
        format: QwenWeightFormat,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let hidden = args.hidden_size;
        let n_heads = args.num_attention_heads;
        let n_kv_heads = args.num_key_value_heads;
        let head_dim = args.head_dim;
        let projection_format = |name: &str| {
            prefix.map_or(format, |prefix| {
                args.weight_format_for(&format!("{prefix}.{name}.weight"), format)
            })
        };
        let q_proj = QwenLinear::new(
            hidden,
            n_heads * head_dim * 2,
            args.attention_bias,
            projection_format("q_proj"),
            stream,
        )?;
        let k_proj = QwenLinear::new(
            hidden,
            n_kv_heads * head_dim,
            args.attention_bias,
            projection_format("k_proj"),
            stream,
        )?;
        let v_proj = QwenLinear::new(
            hidden,
            n_kv_heads * head_dim,
            args.attention_bias,
            projection_format("v_proj"),
            stream,
        )?;
        let o_proj = QwenLinear::new(
            n_heads * head_dim,
            hidden,
            args.attention_bias,
            projection_format("o_proj"),
            stream,
        )?;
        let rope_config = args.rope_config();
        let rope = initialize_rope(
            args.rope_dims(),
            args.rope_theta(),
            false,
            &rope_config,
            args.max_position_embeddings,
            stream,
        )?;
        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            scale: (head_dim as f32).sqrt().recip(),
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm: Qwen3NextRmsNorm::new(head_dim, args.rms_norm_eps, stream)?,
            k_norm: Qwen3NextRmsNorm::new(head_dim, args.rms_norm_eps, stream)?,
            rope,
        })
    }
}

/// Input for a Qwen3.5 full-attention layer.
pub struct FullAttentionInput<'a> {
    /// Hidden states.
    pub x: &'a Array,
    /// Optional attention mask.
    pub mask: Option<&'a Array>,
    /// Optional key/value cache.
    pub cache: Option<&'a mut dyn KeyValueCache>,
}

impl Module<FullAttentionInput<'_>> for FullAttention {
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(
        &mut self,
        input: FullAttentionInput<'_>,
        stream: &Stream,
    ) -> Result<Self::Output, Self::Error> {
        let FullAttentionInput { x, mask, mut cache } = input;
        let shape = x.shape();
        let B = shape[0];
        let L = shape[1];
        let q_proj = self
            .q_proj
            .forward(x, stream)?
            .reshape(&[B, L, self.n_heads, 2 * self.head_dim], stream)?;
        let query = q_proj.try_index_device((.., .., .., ..self.head_dim), stream)?;
        let gate = q_proj
            .try_index_device((.., .., .., self.head_dim..), stream)?
            .reshape(&[B, L, self.n_heads * self.head_dim], stream)?;
        let mut query = self
            .q_norm
            .forward(&query, stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
        let mut key = self
            .k_norm
            .forward(
                &self
                    .k_proj
                    .forward(x, stream)?
                    .reshape(&[B, L, self.n_kv_heads, self.head_dim], stream)?,
                stream,
            )?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
        let mut value = self
            .v_proj
            .forward(x, stream)?
            .reshape(&[B, L, self.n_kv_heads, self.head_dim], stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?;

        if let Some(cache) = cache.as_mut() {
            let offset = cache.offset();
            query = self.rope.forward(
                nn::RopeInputBuilder::new(&query).offset(offset).build()?,
                stream,
            )?;
            key = self.rope.forward(
                nn::RopeInputBuilder::new(&key).offset(offset).build()?,
                stream,
            )?;
            (key, value) = cache.update_for_attention(key, value, stream)?;
        } else {
            query = self.rope.forward(nn::RopeInput::new(&query), stream)?;
            key = self.rope.forward(nn::RopeInput::new(&key), stream)?;
        }

        let out = crate::backend::mlx::nn::attention::finish_attention(
            query, key, value, cache, self.scale, mask, B, L, stream,
        )?
        .multiply(sigmoid(gate, stream)?, stream)?;
        self.o_proj.forward(&out, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        self.k_proj.training_mode(mode);
        self.v_proj.training_mode(mode);
        self.o_proj.training_mode(mode);
        self.q_norm.training_mode(mode);
        self.k_norm.training_mode(mode);
        <RopeVariant as Module<nn::RopeInput>>::training_mode(&mut self.rope, mode);
    }
}

impl FullAttention {
    /// Forward pass that reports full-attention activations to an observer.
    pub fn forward_with_observer(
        &mut self,
        input: FullAttentionInput<'_>,
        stream: &Stream,
        prefix: &str,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        let FullAttentionInput { x, mask, mut cache } = input;
        let shape = x.shape();
        let b = shape[0];
        let l = shape[1];
        let q_proj = self
            .q_proj
            .forward(x, stream)?
            .reshape(&[b, l, self.n_heads, 2 * self.head_dim], stream)?;
        observer.observe(&format!("{prefix}.q_proj"), &q_proj)?;
        let query = q_proj.try_index_device((.., .., .., ..self.head_dim), stream)?;
        let gate = q_proj
            .try_index_device((.., .., .., self.head_dim..), stream)?
            .reshape(&[b, l, self.n_heads * self.head_dim], stream)?;
        observer.observe(&format!("{prefix}.gate"), &gate)?;
        let mut query = self
            .q_norm
            .forward(&query, stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
        observer.observe(&format!("{prefix}.q_norm"), &query)?;
        let mut key = self
            .k_norm
            .forward(
                &self
                    .k_proj
                    .forward(x, stream)?
                    .reshape(&[b, l, self.n_kv_heads, self.head_dim], stream)?,
                stream,
            )?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
        observer.observe(&format!("{prefix}.k_norm"), &key)?;
        let mut value = self
            .v_proj
            .forward(x, stream)?
            .reshape(&[b, l, self.n_kv_heads, self.head_dim], stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
        observer.observe(&format!("{prefix}.values"), &value)?;

        if let Some(cache) = cache.as_mut() {
            let offset = cache.offset();
            query = self.rope.forward(
                nn::RopeInputBuilder::new(&query).offset(offset).build()?,
                stream,
            )?;
            key = self.rope.forward(
                nn::RopeInputBuilder::new(&key).offset(offset).build()?,
                stream,
            )?;
            (key, value) = cache.update_and_fetch(key, value, stream)?;
        } else {
            query = self.rope.forward(nn::RopeInput::new(&query), stream)?;
            key = self.rope.forward(nn::RopeInput::new(&key), stream)?;
        }
        observer.observe(&format!("{prefix}.queries_rope"), &query)?;
        observer.observe(&format!("{prefix}.keys_rope"), &key)?;
        observer.observe(&format!("{prefix}.values_cache"), &value)?;
        let attention_probs = attention_probabilities(&query, &key, self.scale, mask, stream)?;
        observer.observe(&format!("{prefix}.attention_probs"), &attention_probs)?;

        let out = crate::backend::mlx::nn::tensor::scaled_dot_product_attention(
            query, key, value, cache, self.scale, mask, stream,
        )?
        .transpose_axes(&[0, 2, 1, 3], stream)?
        .reshape(&[b, l, -1], stream)?;
        observer.observe(&format!("{prefix}.attention"), &out)?;
        let gated = out.multiply(sigmoid(gate, stream)?, stream)?;
        observer.observe(&format!("{prefix}.attention_gated"), &gated)?;
        let output = self.o_proj.forward(&gated, stream)?;
        observer.observe(&format!("{prefix}.o_proj"), &output)?;
        Ok(output)
    }

    pub(crate) fn forward_tensor_parallel(
        &mut self,
        input: FullAttentionInput<'_>,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let mut partial = self.forward(input, stream)?;
        if let Some(bias) = self.o_proj.bias.as_ref() {
            partial = partial.subtract(bias, stream)?;
        }
        let mut output = safemlx::distributed::all_sum(&partial, group, stream)?;
        if let Some(bias) = self.o_proj.bias.as_ref() {
            output = output.add(bias, stream)?;
        }
        Ok(output)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// Depthwise one-dimensional convolution parameters.
pub struct DepthwiseConv1d {
    #[param]
    /// Convolution weights.
    pub weight: Param<Array>,
}

impl DepthwiseConv1d {
    /// Creates an unloaded depthwise convolution.
    pub fn new(channels: i32, kernel_size: i32, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            weight: Param::<Array>::unloaded(&[channels, 1, kernel_size], Dtype::Float32, stream)?,
        })
    }
}

#[allow(non_snake_case)]
#[derive(Debug, Clone, ModuleParameters)]
/// Recurrent linear-attention layer used by Qwen3.5 MoE.
pub struct LinearAttention {
    /// Number of value heads.
    pub num_v_heads: i32,
    /// Number of key heads.
    pub num_k_heads: i32,
    /// Key head dimension.
    pub head_k_dim: i32,
    /// Value head dimension.
    pub head_v_dim: i32,
    /// Total key dimension.
    pub key_dim: i32,
    /// Total value dimension.
    pub value_dim: i32,
    /// Convolution input dimension.
    pub conv_dim: i32,
    /// Causal convolution kernel size.
    pub conv_kernel_size: i32,
    #[param]
    /// Depthwise causal convolution.
    pub conv1d: DepthwiseConv1d,
    #[param]
    /// Joint query/key/value projection.
    pub in_proj_qkv: QwenLinear,
    #[param]
    /// Output gate projection.
    pub in_proj_z: QwenLinear,
    #[param]
    /// Beta projection.
    pub in_proj_b: QwenLinear,
    #[param]
    /// Delta projection.
    pub in_proj_a: QwenLinear,
    #[param]
    /// Delta bias.
    pub dt_bias: Param<Array>,
    #[param]
    /// Log transition parameter.
    pub A_log: Param<Array>,
    #[param]
    /// Gated normalization.
    pub norm: Qwen3NextRmsNormGated,
    #[param]
    /// Output projection.
    pub out_proj: QwenLinear,
}

impl LinearAttention {
    /// Creates an unloaded linear-attention layer.
    pub fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        Self::new_with_format(args, None, QwenWeightFormat::for_text(args, None), stream)
    }

    fn new_with_format(
        args: &ModelArgs,
        prefix: Option<&str>,
        format: QwenWeightFormat,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let num_v_heads = args.linear_num_value_heads;
        let num_k_heads = args.linear_num_key_heads;
        let head_k_dim = args.linear_key_head_dim;
        let head_v_dim = args.linear_value_head_dim;
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_dim = key_dim * 2 + value_dim;
        let projection_size_qkv = key_dim * 2 + value_dim;
        let projection_format = |name: &str, fallback: QwenWeightFormat| {
            prefix.map_or(fallback, |prefix| {
                args.weight_format_for(&format!("{prefix}.{name}.weight"), fallback)
            })
        };
        // Official native-FP8 Qwen3-Next checkpoints deliberately keep the
        // fused BA projection dense BF16. Its layerwise recipes split that
        // tensor without casting, so the unloaded destinations must advertise
        // the checkpoint dtype rather than the generic dense F32 default.
        let ba_weight_dtype = if format == QwenWeightFormat::Fp8 {
            Dtype::Bfloat16
        } else {
            Dtype::Float32
        };
        Ok(Self {
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_dim,
            conv_kernel_size: args.linear_conv_kernel_dim,
            conv1d: DepthwiseConv1d::new(conv_dim, args.linear_conv_kernel_dim, stream)?,
            in_proj_qkv: QwenLinear::new(
                args.hidden_size,
                projection_size_qkv,
                false,
                projection_format("in_proj_qkv", format),
                stream,
            )?,
            in_proj_z: QwenLinear::new(
                args.hidden_size,
                value_dim,
                false,
                projection_format("in_proj_z", format),
                stream,
            )?,
            in_proj_b: match projection_format("in_proj_b", QwenWeightFormat::Dense) {
                QwenWeightFormat::Dense => QwenLinear::new_dense_with_weight_dtype(
                    args.hidden_size,
                    num_v_heads,
                    false,
                    ba_weight_dtype,
                    stream,
                )?,
                affine => QwenLinear::new(args.hidden_size, num_v_heads, false, affine, stream)?,
            },
            in_proj_a: match projection_format("in_proj_a", QwenWeightFormat::Dense) {
                QwenWeightFormat::Dense => QwenLinear::new_dense_with_weight_dtype(
                    args.hidden_size,
                    num_v_heads,
                    false,
                    ba_weight_dtype,
                    stream,
                )?,
                affine => QwenLinear::new(args.hidden_size, num_v_heads, false, affine, stream)?,
            },
            dt_bias: Param::new(Array::from_slice(
                &vec![1.0f32; num_v_heads as usize],
                &[num_v_heads],
            )),
            A_log: Param::new(Array::from_slice(
                &vec![0.0f32; num_v_heads as usize],
                &[num_v_heads],
            )),
            norm: Qwen3NextRmsNormGated::new(head_v_dim, args.rms_norm_eps, stream)?,
            out_proj: QwenLinear::new(
                value_dim,
                args.hidden_size,
                false,
                projection_format("out_proj", format),
                stream,
            )?,
        })
    }

    #[allow(non_snake_case)]
    fn depthwise_causal_conv(
        &self,
        mixed_qkv: &Array,
        cache: Option<&mut LinearAttentionCache>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let shape = mixed_qkv.shape();
        let B = shape[0];
        let L = shape[1];
        let C = shape[2];
        let state_len = self.conv_kernel_size - 1;
        let state = cache
            .as_ref()
            .and_then(|cache| cache.conv_state.clone())
            .unwrap_or(zeros::<f32>(&[B, state_len, C], stream)?);
        let padded = concatenate_axis(&[state, mixed_qkv.clone()], 1, stream)?;
        if let Some(cache) = cache {
            cache.conv_state = Some(padded.try_index_device((.., L.., ..), stream)?);
            cache.offset += L;
        }

        if L > 1 {
            let weight = self.conv1d.weight.swap_axes(1, 2, stream)?;
            let out = conv1d(&padded, &weight, Some(1), Some(0), Some(1), Some(C), stream)?;
            return silu(out, stream);
        }

        let mut out: Option<Array> = None;
        for k in 0..self.conv_kernel_size {
            let window = padded.try_index_device((.., k..k + L, ..), stream)?;
            let weight = self
                .conv1d
                .weight
                .try_index_device((.., 0, k), stream)?
                .reshape(&[1, 1, C], stream)?;
            let term = window.multiply(weight, stream)?;
            out = Some(match out {
                Some(acc) => acc.add(term, stream)?,
                None => term,
            });
        }
        silu(out.expect("conv kernel must have at least one tap"), stream)
    }

    fn l2norm(x: Array, stream: &Stream) -> Result<Array, Exception> {
        let denom =
            sum_axis(&x.square(stream)?, -1, true, stream)?.add(Array::from_f32(1e-6), stream)?;
        x.multiply(safemlx::ops::rsqrt(denom, stream)?, stream)
    }

    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn recurrent_delta_rule(
        &self,
        query: Array,
        key: Array,
        value: Array,
        g: Array,
        beta: Array,
        cache: Option<&mut LinearAttentionCache>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let shape = query.shape();
        let B = shape[0];
        let H = shape[2];
        let KD = shape[3];
        let VD = value.shape()[3];
        let scale = (KD as f32).sqrt().recip();
        let query = query.multiply(Array::from_f32(scale), stream)?;
        let state = cache
            .as_ref()
            .and_then(|cache| cache.recurrent_state.clone())
            .unwrap_or(zeros::<f32>(&[B, H, KD, VD], stream)?);
        let (new_state, out) = crate::backend::mlx::nn::gated_delta::gated_delta_scan(
            &query,
            &key,
            &value,
            &g,
            &beta,
            Some(state),
            stream,
        )?;
        if let Some(cache) = cache {
            cache.recurrent_state = Some(new_state);
        }
        Ok(out)
    }
}

/// Input for a Qwen3.5 linear-attention layer.
pub struct LinearAttentionInput<'a> {
    /// Hidden states.
    pub x: &'a Array,
    /// Optional linear-attention cache.
    pub cache: Option<&'a mut LinearAttentionCache>,
}

impl Module<LinearAttentionInput<'_>> for LinearAttention {
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(
        &mut self,
        input: LinearAttentionInput<'_>,
        stream: &Stream,
    ) -> Result<Self::Output, Self::Error> {
        let LinearAttentionInput { x, mut cache } = input;
        let shape = x.shape();
        let B = shape[0];
        let L = shape[1];
        let (mixed_qkv, z) =
            QwenLinear::forward_pair(&mut self.in_proj_qkv, &mut self.in_proj_z, x, stream)?;
        let z = z.reshape(&[B, L, self.num_v_heads, self.head_v_dim], stream)?;
        let b = self.in_proj_b.forward(x, stream)?;
        let a = self.in_proj_a.forward(x, stream)?;
        let mixed_qkv = self.depthwise_causal_conv(&mixed_qkv, cache.as_deref_mut(), stream)?;
        let query = mixed_qkv
            .try_index_device((.., .., ..self.key_dim), stream)?
            .reshape(&[B, L, self.num_k_heads, self.head_k_dim], stream)?;
        let key = mixed_qkv
            .try_index_device((.., .., self.key_dim..2 * self.key_dim), stream)?
            .reshape(&[B, L, self.num_k_heads, self.head_k_dim], stream)?;
        let mut value = mixed_qkv
            .try_index_device((.., .., 2 * self.key_dim..), stream)?
            .reshape(&[B, L, self.num_v_heads, self.head_v_dim], stream)?;
        let mut query = Self::l2norm(query, stream)?;
        let mut key = Self::l2norm(key, stream)?;
        let beta = sigmoid(b, stream)?;
        let dt_bias = self.dt_bias.reshape(&[1, 1, self.num_v_heads], stream)?;
        let g = nn::softplus(a.add(dt_bias, stream)?, stream)?.multiply(
            exp(self.A_log.as_ref(), stream)?.multiply(Array::from_f32(-1.0), stream)?,
            stream,
        )?;

        let repeats = self.num_v_heads / self.num_k_heads;
        if repeats > 1 {
            let expanded_query = query.try_index_device((.., .., .., NewAxis, ..), stream)?;
            query = broadcast_to(
                &expanded_query,
                &[B, L, self.num_k_heads, repeats, self.head_k_dim],
                stream,
            )?
            .reshape(&[B, L, self.num_v_heads, self.head_k_dim], stream)?;
            let expanded_key = key.try_index_device((.., .., .., NewAxis, ..), stream)?;
            key = broadcast_to(
                &expanded_key,
                &[B, L, self.num_k_heads, repeats, self.head_k_dim],
                stream,
            )?
            .reshape(&[B, L, self.num_v_heads, self.head_k_dim], stream)?;
        }

        value = value.as_dtype(x.dtype(), stream)?;
        let core = self.recurrent_delta_rule(query, key, value, g, beta, cache, stream)?;
        let z_shape = z.shape().to_vec();
        let core = core.reshape(&[-1, self.head_v_dim], stream)?;
        let z = z.reshape(&[-1, self.head_v_dim], stream)?;
        let out = self
            .norm
            .forward(&core, &z, stream)?
            .reshape(&z_shape, stream)?
            .reshape(&[B, L, self.value_dim], stream)?;
        self.out_proj.forward(&out, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.in_proj_qkv.training_mode(mode);
        self.in_proj_z.training_mode(mode);
        self.in_proj_b.training_mode(mode);
        self.in_proj_a.training_mode(mode);
        self.norm.training_mode(mode);
        self.out_proj.training_mode(mode);
    }
}

impl LinearAttention {
    /// Executes rank-local recurrent heads and reduces the row-parallel output
    /// projection exactly once.
    pub(crate) fn forward_tensor_parallel(
        &mut self,
        input: LinearAttentionInput<'_>,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let partial = self.forward(input, stream)?;
        safemlx::distributed::all_sum(&partial, group, stream)
    }
}

impl LinearAttention {
    /// Forward pass that reports recurrent linear-attention internals.
    #[allow(non_snake_case)]
    pub fn forward_with_observer(
        &mut self,
        input: LinearAttentionInput<'_>,
        stream: &Stream,
        prefix: &str,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        let LinearAttentionInput { x, mut cache } = input;
        let shape = x.shape();
        let B = shape[0];
        let L = shape[1];
        let (mixed_qkv, z) =
            QwenLinear::forward_pair(&mut self.in_proj_qkv, &mut self.in_proj_z, x, stream)?;
        observer.observe(&format!("{prefix}.in_proj_qkv"), &mixed_qkv)?;
        let z = z.reshape(&[B, L, self.num_v_heads, self.head_v_dim], stream)?;
        observer.observe(&format!("{prefix}.z_proj"), &z)?;
        let b = self.in_proj_b.forward(x, stream)?;
        observer.observe(&format!("{prefix}.beta_proj"), &b)?;
        let a = self.in_proj_a.forward(x, stream)?;
        observer.observe(&format!("{prefix}.a_proj"), &a)?;
        let mixed_qkv = self.depthwise_causal_conv(&mixed_qkv, cache.as_deref_mut(), stream)?;
        observer.observe(&format!("{prefix}.causal_conv"), &mixed_qkv)?;

        let query = mixed_qkv
            .try_index_device((.., .., ..self.key_dim), stream)?
            .reshape(&[B, L, self.num_k_heads, self.head_k_dim], stream)?;
        observer.observe(&format!("{prefix}.query_raw"), &query)?;
        let key = mixed_qkv
            .try_index_device((.., .., self.key_dim..2 * self.key_dim), stream)?
            .reshape(&[B, L, self.num_k_heads, self.head_k_dim], stream)?;
        observer.observe(&format!("{prefix}.key_raw"), &key)?;
        let mut value = mixed_qkv
            .try_index_device((.., .., 2 * self.key_dim..), stream)?
            .reshape(&[B, L, self.num_v_heads, self.head_v_dim], stream)?;
        observer.observe(&format!("{prefix}.value"), &value)?;
        let mut query = Self::l2norm(query, stream)?;
        observer.observe(&format!("{prefix}.query_l2norm"), &query)?;
        let mut key = Self::l2norm(key, stream)?;
        observer.observe(&format!("{prefix}.key_l2norm"), &key)?;
        let beta = sigmoid(b, stream)?;
        observer.observe(&format!("{prefix}.beta"), &beta)?;
        let dt_bias = self.dt_bias.reshape(&[1, 1, self.num_v_heads], stream)?;
        let g = nn::softplus(a.add(dt_bias, stream)?, stream)?.multiply(
            exp(self.A_log.as_ref(), stream)?.multiply(Array::from_f32(-1.0), stream)?,
            stream,
        )?;
        observer.observe(&format!("{prefix}.decay"), &g)?;

        let repeats = self.num_v_heads / self.num_k_heads;
        if repeats > 1 {
            let expanded_query = query.try_index_device((.., .., .., NewAxis, ..), stream)?;
            query = broadcast_to(
                &expanded_query,
                &[B, L, self.num_k_heads, repeats, self.head_k_dim],
                stream,
            )?
            .reshape(&[B, L, self.num_v_heads, self.head_k_dim], stream)?;
            observer.observe(&format!("{prefix}.query_repeated"), &query)?;
            let expanded_key = key.try_index_device((.., .., .., NewAxis, ..), stream)?;
            key = broadcast_to(
                &expanded_key,
                &[B, L, self.num_k_heads, repeats, self.head_k_dim],
                stream,
            )?
            .reshape(&[B, L, self.num_v_heads, self.head_k_dim], stream)?;
            observer.observe(&format!("{prefix}.key_repeated"), &key)?;
        }

        value = value.as_dtype(x.dtype(), stream)?;
        let core = self.recurrent_delta_rule(query, key, value, g, beta, cache, stream)?;
        observer.observe(&format!("{prefix}.recurrent_core"), &core)?;
        let z_shape = z.shape().to_vec();
        let core = core.reshape(&[-1, self.head_v_dim], stream)?;
        observer.observe(&format!("{prefix}.recurrent_core_flat"), &core)?;
        let z = z.reshape(&[-1, self.head_v_dim], stream)?;
        observer.observe(&format!("{prefix}.z_flat"), &z)?;
        let normalized = self.norm.forward(&core, &z, stream)?;
        observer.observe(&format!("{prefix}.gated_norm"), &normalized)?;
        let out = normalized
            .reshape(&z_shape, stream)?
            .reshape(&[B, L, self.value_dim], stream)?;
        observer.observe(&format!("{prefix}.pre_out_proj"), &out)?;
        let output = self.out_proj.forward(&out, stream)?;
        observer.observe(&format!("{prefix}.out_proj"), &output)?;
        Ok(output)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// Dense SwiGLU MLP used by dense layers and shared experts.
pub struct Mlp {
    #[param]
    /// Gate projection.
    pub gate_proj: QwenLinear,
    #[param]
    /// Up projection.
    pub up_proj: QwenLinear,
    #[param]
    /// Down projection.
    pub down_proj: QwenLinear,
}

impl Mlp {
    fn new(
        dim: i32,
        hidden_dim: i32,
        bias: bool,
        args: &ModelArgs,
        prefix: Option<&str>,
        format: QwenWeightFormat,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let projection_format = |name: &str| {
            prefix.map_or(format, |prefix| {
                args.weight_format_for(&format!("{prefix}.{name}.weight"), format)
            })
        };
        Ok(Self {
            gate_proj: QwenLinear::new(
                dim,
                hidden_dim,
                bias,
                projection_format("gate_proj"),
                stream,
            )?,
            up_proj: QwenLinear::new(dim, hidden_dim, bias, projection_format("up_proj"), stream)?,
            down_proj: QwenLinear::new(
                hidden_dim,
                dim,
                bias,
                projection_format("down_proj"),
                stream,
            )?,
        })
    }

    fn forward(&mut self, input: &Array, stream: &Stream) -> Result<Array, Exception> {
        let down_proj_input = silu(self.gate_proj.forward(input, stream)?, stream)?
            .multiply(self.up_proj.forward(input, stream)?, stream)?;
        self.down_proj.forward(&down_proj_input, stream)
    }

    fn forward_with_observer(
        &mut self,
        input: &Array,
        stream: &Stream,
        prefix: &str,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        let gate = self.gate_proj.forward(input, stream)?;
        observer.observe(&format!("{prefix}.gate_proj"), &gate)?;
        let up = self.up_proj.forward(input, stream)?;
        observer.observe(&format!("{prefix}.up_proj"), &up)?;
        let hidden = silu(gate, stream)?.multiply(up, stream)?;
        observer.observe(&format!("{prefix}.down_proj_input"), &hidden)?;
        let output = self.down_proj.forward(&hidden, stream)?;
        observer.observe(&format!("{prefix}.down_proj"), &output)?;
        Ok(output)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// Routed expert bank for Qwen3.5 MoE.
pub struct Experts {
    /// Number of experts.
    pub num_experts: i32,
    /// Model hidden dimension.
    pub hidden_dim: i32,
    /// Expert intermediate dimension.
    pub intermediate_dim: i32,
    /// Whether expert weights are stored as FP8.
    pub use_fp8: bool,
    /// Optional affine quantization for the packed gate/up bank.
    pub gate_up_affine: Option<WeightQuantization>,
    /// Optional affine quantization for the packed down bank.
    pub down_affine: Option<WeightQuantization>,
    /// Optional checkpoint-native IQ encoding for the packed gate/up bank.
    pub gate_up_iquant: Option<WeightQuantization>,
    /// Optional checkpoint-native IQ encoding for the packed down bank.
    pub down_iquant: Option<WeightQuantization>,
    #[param]
    /// Packed gate and up projection weights for all experts.
    pub gate_up_proj: Param<Array>,
    #[param]
    /// Optional FP8 inverse scales for gate/up projection weights.
    pub gate_up_proj_scale_inv: Param<Option<Array>>,
    #[param]
    /// Optional affine scales for gate/up projection weights.
    pub gate_up_proj_scales: Param<Option<Array>>,
    #[param]
    /// Optional affine biases for gate/up projection weights.
    pub gate_up_proj_biases: Param<Option<Array>>,
    #[param]
    /// Down projection weights for all experts.
    pub down_proj: Param<Array>,
    #[param]
    /// Optional FP8 inverse scales for down projection weights.
    pub down_proj_scale_inv: Param<Option<Array>>,
    #[param]
    /// Optional affine scales for down projection weights.
    pub down_proj_scales: Param<Option<Array>>,
    #[param]
    /// Optional affine biases for down projection weights.
    pub down_proj_biases: Param<Option<Array>>,
}

type AffineExpertProjectionParams = (Param<Array>, Param<Option<Array>>, Param<Option<Array>>);

impl Experts {
    /// Creates an unloaded routed expert bank.
    pub fn new(args: &ModelArgs, layer_idx: usize, stream: &Stream) -> Result<Self, Exception> {
        let prefix = format!("model.layers.{layer_idx}.mlp.experts");
        Self::new_with_format(
            args,
            &prefix,
            QwenWeightFormat::for_text(args, None),
            stream,
        )
    }

    fn new_with_format(
        args: &ModelArgs,
        prefix: &str,
        format: QwenWeightFormat,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let gate_up_quantization = args
            .quantization_for(&format!("{prefix}.gate_up_proj"))
            .or_else(|| format.quantization());
        let down_quantization = args
            .quantization_for(&format!("{prefix}.down_proj"))
            .or_else(|| format.quantization());
        let (gate_up_affine, gate_up_iquant) = match gate_up_quantization {
            Some(iq @ WeightQuantization::GgufIQuant { .. }) => (None, Some(iq)),
            affine => (affine, None),
        };
        let (down_affine, down_iquant) = match down_quantization {
            Some(iq @ WeightQuantization::GgufIQuant { .. }) => (None, Some(iq)),
            affine => (affine, None),
        };
        let use_fp8 = format == QwenWeightFormat::Fp8;
        let expert_weight_dtype = if use_fp8 {
            Dtype::Uint8
        } else {
            Dtype::Float32
        };
        let projection = |out_features: i32,
                          in_features: i32,
                          affine: Option<WeightQuantization>,
                          iquant: Option<WeightQuantization>|
         -> Result<AffineExpertProjectionParams, Exception> {
            if let Some(iquant) = iquant {
                let (ggml_type, _) = iquant.gguf_iquant().expect("IQ expert format");
                let (block_values, block_bytes) = ggml_type
                    .block_and_bytes()
                    .expect("canonical IQ block geometry");
                Ok((
                    Param::<Array>::unloaded(
                        &[
                            args.num_experts,
                            out_features,
                            in_features / block_values as i32 * block_bytes as i32,
                        ],
                        Dtype::Uint8,
                        stream,
                    )?,
                    Param::new(None),
                    Param::new(None),
                ))
            } else if let Some(affine) = affine {
                Ok((
                    Param::<Array>::unloaded(
                        &[
                            args.num_experts,
                            out_features,
                            quantized_packed_dimension(in_features, affine.bits()),
                        ],
                        Dtype::Uint32,
                        stream,
                    )?,
                    Param::<Option<Array>>::unloaded_some(
                        &[
                            args.num_experts,
                            out_features,
                            in_features / affine.group_size(),
                        ],
                        if affine == WeightQuantization::MxFp4 {
                            Dtype::Uint8
                        } else {
                            Dtype::Float16
                        },
                        stream,
                    )?,
                    if affine.has_biases() {
                        Param::<Option<Array>>::unloaded_some(
                            &[
                                args.num_experts,
                                out_features,
                                in_features / affine.group_size(),
                            ],
                            Dtype::Float16,
                            stream,
                        )?
                    } else {
                        Param::new(None)
                    },
                ))
            } else {
                Ok((
                    Param::<Array>::unloaded(
                        &[args.num_experts, out_features, in_features],
                        expert_weight_dtype,
                        stream,
                    )?,
                    Param::new(None),
                    Param::new(None),
                ))
            }
        };
        let (gate_up_proj, gate_up_proj_scales, gate_up_proj_biases) = projection(
            2 * args.moe_intermediate_size,
            args.hidden_size,
            gate_up_affine,
            gate_up_iquant,
        )?;
        let (down_proj, down_proj_scales, down_proj_biases) = projection(
            args.hidden_size,
            args.moe_intermediate_size,
            down_affine,
            down_iquant,
        )?;
        Ok(Self {
            num_experts: args.num_experts,
            hidden_dim: args.hidden_size,
            intermediate_dim: args.moe_intermediate_size,
            use_fp8,
            gate_up_affine,
            down_affine,
            gate_up_iquant,
            down_iquant,
            gate_up_proj,
            gate_up_proj_scale_inv: if use_fp8 {
                Param::<Option<Array>>::unloaded_some(
                    &[
                        args.num_experts,
                        ceil_div(2 * args.moe_intermediate_size, 128),
                        ceil_div(args.hidden_size, 128),
                    ],
                    Dtype::Float32,
                    stream,
                )?
            } else {
                Param::new(None)
            },
            gate_up_proj_scales,
            gate_up_proj_biases,
            down_proj,
            down_proj_scale_inv: if use_fp8 {
                Param::<Option<Array>>::unloaded_some(
                    &[
                        args.num_experts,
                        ceil_div(args.hidden_size, 128),
                        ceil_div(args.moe_intermediate_size, 128),
                    ],
                    Dtype::Float32,
                    stream,
                )?
            } else {
                Param::new(None)
            },
            down_proj_scales,
            down_proj_biases,
        })
    }

    /// Evaluates routed experts for flattened token hidden states.
    pub fn forward(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if self.use_fp8
            || self.gate_up_affine.is_some()
            || self.down_affine.is_some()
            || self.gate_up_iquant.is_some()
            || self.down_iquant.is_some()
        {
            return self.forward_expert_major_chunk(
                hidden_states,
                top_k_index,
                top_k_weights,
                stream,
            );
        }

        let num_tokens = hidden_states.shape()[0];
        let top_k = top_k_index.shape()[1];
        let selected_gate_up = self
            .gate_up_proj
            .as_ref()
            .take_axis(top_k_index, 0, stream)?;
        let hidden = hidden_states.try_index_device((.., NewAxis, NewAxis, ..), stream)?;
        let gate_up = matmul(&hidden, selected_gate_up.swap_axes(-1, -2, stream)?, stream)?
            .reshape(&[num_tokens, top_k, 2 * self.intermediate_dim], stream)?;
        let gate = gate_up.try_index_device((.., .., ..self.intermediate_dim), stream)?;
        let up = gate_up.try_index_device((.., .., self.intermediate_dim..), stream)?;
        let current = silu(gate, stream)?.multiply(up, stream)?;

        let selected_down = self.down_proj.as_ref().take_axis(top_k_index, 0, stream)?;
        let current = matmul(
            current.try_index_device((.., .., NewAxis, ..), stream)?,
            selected_down.swap_axes(-1, -2, stream)?,
            stream,
        )?
        .reshape(&[num_tokens, top_k, self.hidden_dim], stream)?;
        let weighted = current.multiply(
            top_k_weights.try_index_device((.., .., NewAxis), stream)?,
            stream,
        )?;
        sum_axis(&weighted, -2, false, stream)
    }

    /// Evaluates routed experts in chunks for long prefill inputs.
    pub fn forward_chunked(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let num_tokens = hidden_states.shape()[0];
        if num_tokens <= ROUTED_EXPERT_CHUNK_THRESHOLD {
            return self.forward(hidden_states, top_k_index, top_k_weights, stream);
        }

        let mut outputs = Vec::with_capacity(
            ((num_tokens + ROUTED_EXPERT_CHUNK_TOKENS - 1) / ROUTED_EXPERT_CHUNK_TOKENS)
                .try_into()
                .expect("number of MoE chunks must fit in usize"),
        );
        let mut start = 0;
        while start < num_tokens {
            let end = (start + ROUTED_EXPERT_CHUNK_TOKENS).min(num_tokens);
            let hidden_chunk = hidden_states.try_index_device((start..end, ..), stream)?;
            let expert_chunk = top_k_index.try_index_device((start..end, ..), stream)?;
            let weight_chunk = top_k_weights.try_index_device((start..end, ..), stream)?;
            outputs.push(self.forward_expert_major_chunk(
                &hidden_chunk,
                &expert_chunk,
                &weight_chunk,
                stream,
            )?);
            start = end;
        }
        concatenate_axis(&outputs, 0, stream)
    }

    /// Evaluates routed experts while reporting per-route expert internals.
    pub fn forward_chunked_with_observer(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
        stream: &Stream,
        prefix: &str,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        let num_tokens = hidden_states.shape()[0];
        if num_tokens <= ROUTED_EXPERT_CHUNK_THRESHOLD {
            return self.forward_with_observer(
                hidden_states,
                top_k_index,
                top_k_weights,
                stream,
                prefix,
                observer,
            );
        }

        let mut outputs = Vec::with_capacity(
            ((num_tokens + ROUTED_EXPERT_CHUNK_TOKENS - 1) / ROUTED_EXPERT_CHUNK_TOKENS)
                .try_into()
                .expect("number of MoE chunks must fit in usize"),
        );
        let mut start = 0;
        let mut chunk = 0;
        while start < num_tokens {
            let end = (start + ROUTED_EXPERT_CHUNK_TOKENS).min(num_tokens);
            let hidden_chunk = hidden_states.try_index_device((start..end, ..), stream)?;
            observer.observe(&format!("{prefix}.chunks.{chunk}.input"), &hidden_chunk)?;
            let expert_chunk = top_k_index.try_index_device((start..end, ..), stream)?;
            let weight_chunk = top_k_weights.try_index_device((start..end, ..), stream)?;
            outputs.push(self.forward_expert_major_chunk_with_observer(
                &hidden_chunk,
                &expert_chunk,
                &weight_chunk,
                stream,
                &format!("{prefix}.chunks.{chunk}"),
                observer,
            )?);
            start = end;
            chunk += 1;
        }
        let output = concatenate_axis(&outputs, 0, stream)?;
        observer.observe(&format!("{prefix}.chunked_output"), &output)?;
        Ok(output)
    }

    /// Evaluates routed experts for flattened token hidden states with observer hooks.
    pub fn forward_with_observer(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
        stream: &Stream,
        prefix: &str,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        observer.observe(&format!("{prefix}.input"), hidden_states)?;
        observer.observe(&format!("{prefix}.top_k_experts"), top_k_index)?;
        observer.observe(&format!("{prefix}.top_k_weights"), top_k_weights)?;
        if self.use_fp8
            || self.gate_up_affine.is_some()
            || self.down_affine.is_some()
            || self.gate_up_iquant.is_some()
            || self.down_iquant.is_some()
        {
            return self.forward_expert_major_chunk_with_observer(
                hidden_states,
                top_k_index,
                top_k_weights,
                stream,
                prefix,
                observer,
            );
        }

        let num_tokens = hidden_states.shape()[0];
        let top_k = top_k_index.shape()[1];
        let selected_gate_up = self
            .gate_up_proj
            .as_ref()
            .take_axis(top_k_index, 0, stream)?;
        observer.observe(
            &format!("{prefix}.selected_gate_up_weight"),
            &selected_gate_up,
        )?;
        let hidden = hidden_states.try_index_device((.., NewAxis, NewAxis, ..), stream)?;
        let gate_up = matmul(&hidden, selected_gate_up.swap_axes(-1, -2, stream)?, stream)?
            .reshape(&[num_tokens, top_k, 2 * self.intermediate_dim], stream)?;
        observer.observe(&format!("{prefix}.gate_up_proj"), &gate_up)?;
        let gate = gate_up.try_index_device((.., .., ..self.intermediate_dim), stream)?;
        observer.observe(&format!("{prefix}.gate_proj"), &gate)?;
        let up = gate_up.try_index_device((.., .., self.intermediate_dim..), stream)?;
        observer.observe(&format!("{prefix}.up_proj"), &up)?;
        let gate_activation = silu(gate, stream)?;
        observer.observe(&format!("{prefix}.gate_activation"), &gate_activation)?;
        let current = gate_activation.multiply(up, stream)?;
        observer.observe(&format!("{prefix}.down_proj_input"), &current)?;

        let selected_down = self.down_proj.as_ref().take_axis(top_k_index, 0, stream)?;
        observer.observe(&format!("{prefix}.selected_down_weight"), &selected_down)?;
        let route_output = matmul(
            current.try_index_device((.., .., NewAxis, ..), stream)?,
            selected_down.swap_axes(-1, -2, stream)?,
            stream,
        )?
        .reshape(&[num_tokens, top_k, self.hidden_dim], stream)?;
        observer.observe(&format!("{prefix}.route_output"), &route_output)?;
        let weighted = route_output.multiply(
            top_k_weights.try_index_device((.., .., NewAxis), stream)?,
            stream,
        )?;
        observer.observe(&format!("{prefix}.weighted_route_output"), &weighted)?;
        let output = sum_axis(&weighted, -2, false, stream)?;
        observer.observe(&format!("{prefix}.output"), &output)?;
        Ok(output)
    }

    fn forward_expert_major_chunk(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let num_tokens = hidden_states.shape()[0];
        let plan = topk_route_plan(top_k_index, self.num_experts, stream)?;
        let hidden = gather_grouped_rows(hidden_states, &plan, stream)?;
        let gate_up = if let Some(iquant) = self.gate_up_iquant {
            let (ggml_type, endian) = iquant.gguf_iquant().expect("IQ expert format");
            let native = NativeQuantizedTensor::from_iq_array(
                self.gate_up_proj.value.clone(),
                &[self.num_experts, 2 * self.intermediate_dim, self.hidden_dim],
                ggml_type,
                endian,
            )?;
            native_grouped_linear(&hidden, &native, &plan.sorted_group_ids, stream)?
        } else if let Some(affine) = self.gate_up_affine {
            common::moe::packed_grouped_linear(
                &hidden,
                self.gate_up_proj.as_ref(),
                self.gate_up_proj_scales
                    .as_ref()
                    .as_ref()
                    .expect("affine gate/up scales"),
                self.gate_up_proj_biases.as_ref().as_ref(),
                &plan.sorted_group_ids,
                affine,
                stream,
            )?
        } else if let Some(scale) = self.gate_up_proj_scale_inv.as_ref() {
            common::fp8::grouped_linear(
                &hidden,
                self.gate_up_proj.as_ref(),
                scale,
                &plan.sorted_group_ids,
                stream,
            )?
        } else {
            let gate_up_weights = self.gate_up_proj.as_ref().swap_axes(-1, -2, stream)?;
            grouped_matmul(
                &hidden,
                &gate_up_weights,
                &plan.sorted_group_ids,
                true,
                stream,
            )?
        };
        let gate = gate_up.try_index_device((.., ..self.intermediate_dim), stream)?;
        let up = gate_up.try_index_device((.., self.intermediate_dim..), stream)?;
        let current = silu(gate, stream)?.multiply(up, stream)?;

        let current = if let Some(iquant) = self.down_iquant {
            let (ggml_type, endian) = iquant.gguf_iquant().expect("IQ expert format");
            let native = NativeQuantizedTensor::from_iq_array(
                self.down_proj.value.clone(),
                &[self.num_experts, self.hidden_dim, self.intermediate_dim],
                ggml_type,
                endian,
            )?;
            native_grouped_linear(&current, &native, &plan.sorted_group_ids, stream)?
        } else if let Some(affine) = self.down_affine {
            common::moe::packed_grouped_linear(
                &current,
                self.down_proj.as_ref(),
                self.down_proj_scales
                    .as_ref()
                    .as_ref()
                    .expect("affine down scales"),
                self.down_proj_biases.as_ref().as_ref(),
                &plan.sorted_group_ids,
                affine,
                stream,
            )?
        } else if let Some(scale) = self.down_proj_scale_inv.as_ref() {
            common::fp8::grouped_linear(
                &current,
                self.down_proj.as_ref(),
                scale,
                &plan.sorted_group_ids,
                stream,
            )?
        } else {
            let down_weights = self.down_proj.as_ref().swap_axes(-1, -2, stream)?;
            grouped_matmul(
                &current,
                &down_weights,
                &plan.sorted_group_ids,
                true,
                stream,
            )?
        };
        common::moe::weighted_route_sum(current, top_k_weights, &plan, num_tokens, stream)
    }

    fn forward_expert_major_chunk_with_observer(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
        stream: &Stream,
        prefix: &str,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        let num_tokens = hidden_states.shape()[0];
        let plan = topk_route_plan(top_k_index, self.num_experts, stream)?;
        observer.observe(&format!("{prefix}.route_indices"), &plan.route_indices)?;
        observer.observe(&format!("{prefix}.token_indices"), &plan.token_indices)?;
        observer.observe(&format!("{prefix}.slot_indices"), &plan.slot_indices)?;
        observer.observe(
            &format!("{prefix}.sorted_group_ids"),
            &plan.sorted_group_ids,
        )?;
        let hidden = gather_grouped_rows(hidden_states, &plan, stream)?;
        observer.observe(&format!("{prefix}.expert_major_input"), &hidden)?;
        let gate_up = if let Some(iquant) = self.gate_up_iquant {
            let (ggml_type, endian) = iquant.gguf_iquant().expect("IQ expert format");
            let native = NativeQuantizedTensor::from_iq_array(
                self.gate_up_proj.value.clone(),
                &[self.num_experts, 2 * self.intermediate_dim, self.hidden_dim],
                ggml_type,
                endian,
            )?;
            native_grouped_linear(&hidden, &native, &plan.sorted_group_ids, stream)?
        } else if let Some(affine) = self.gate_up_affine {
            common::moe::packed_grouped_linear(
                &hidden,
                self.gate_up_proj.as_ref(),
                self.gate_up_proj_scales
                    .as_ref()
                    .as_ref()
                    .expect("affine gate/up scales"),
                self.gate_up_proj_biases.as_ref().as_ref(),
                &plan.sorted_group_ids,
                affine,
                stream,
            )?
        } else if let Some(scale) = self.gate_up_proj_scale_inv.as_ref() {
            common::fp8::grouped_linear(
                &hidden,
                self.gate_up_proj.as_ref(),
                scale,
                &plan.sorted_group_ids,
                stream,
            )?
        } else {
            let gate_up_weights = self.gate_up_proj.as_ref().swap_axes(-1, -2, stream)?;
            grouped_matmul(
                &hidden,
                &gate_up_weights,
                &plan.sorted_group_ids,
                true,
                stream,
            )?
        };
        observer.observe(&format!("{prefix}.expert_major_gate_up_proj"), &gate_up)?;
        let gate = gate_up.try_index_device((.., ..self.intermediate_dim), stream)?;
        observer.observe(&format!("{prefix}.expert_major_gate_proj"), &gate)?;
        let up = gate_up.try_index_device((.., self.intermediate_dim..), stream)?;
        observer.observe(&format!("{prefix}.expert_major_up_proj"), &up)?;
        let gate_activation = silu(gate, stream)?;
        observer.observe(
            &format!("{prefix}.expert_major_gate_activation"),
            &gate_activation,
        )?;
        let current = gate_activation.multiply(up, stream)?;
        observer.observe(&format!("{prefix}.expert_major_down_proj_input"), &current)?;

        let route_output = if let Some(iquant) = self.down_iquant {
            let (ggml_type, endian) = iquant.gguf_iquant().expect("IQ expert format");
            let native = NativeQuantizedTensor::from_iq_array(
                self.down_proj.value.clone(),
                &[self.num_experts, self.hidden_dim, self.intermediate_dim],
                ggml_type,
                endian,
            )?;
            native_grouped_linear(&current, &native, &plan.sorted_group_ids, stream)?
        } else if let Some(affine) = self.down_affine {
            common::moe::packed_grouped_linear(
                &current,
                self.down_proj.as_ref(),
                self.down_proj_scales
                    .as_ref()
                    .as_ref()
                    .expect("affine down scales"),
                self.down_proj_biases.as_ref().as_ref(),
                &plan.sorted_group_ids,
                affine,
                stream,
            )?
        } else if let Some(scale) = self.down_proj_scale_inv.as_ref() {
            common::fp8::grouped_linear(
                &current,
                self.down_proj.as_ref(),
                scale,
                &plan.sorted_group_ids,
                stream,
            )?
        } else {
            let down_weights = self.down_proj.as_ref().swap_axes(-1, -2, stream)?;
            grouped_matmul(
                &current,
                &down_weights,
                &plan.sorted_group_ids,
                true,
                stream,
            )?
        };
        observer.observe(
            &format!("{prefix}.expert_major_route_output"),
            &route_output,
        )?;
        let output = common::moe::weighted_route_sum(
            route_output,
            top_k_weights,
            &plan,
            num_tokens,
            stream,
        )?;
        observer.observe(&format!("{prefix}.output"), &output)?;
        Ok(output)
    }

    /// Sets training mode.
    pub fn training_mode(&mut self, _mode: bool) {}
}

/// Top-k router for Qwen3.5 MoE experts.
pub type TopKRouter = common::moe::TopKRouter;

#[derive(Debug, Clone, ModuleParameters)]
/// Sparse MoE block with routed experts plus a shared expert.
pub struct SparseMoeBlock {
    #[param]
    /// Top-k router.
    pub gate: TopKRouter,
    #[param]
    /// Routed expert bank.
    pub experts: Experts,
    #[param]
    /// Shared expert MLP.
    pub shared_expert: Mlp,
    #[param]
    /// Gate applied to the shared expert output.
    pub shared_expert_gate: QwenLinear,
}

impl SparseMoeBlock {
    /// Creates an unloaded sparse MoE block.
    pub fn new(args: &ModelArgs, layer_idx: usize, stream: &Stream) -> Result<Self, Exception> {
        let prefix = format!("model.layers.{layer_idx}.mlp");
        Self::new_with_format(
            args,
            &prefix,
            QwenWeightFormat::for_text(args, None),
            stream,
        )
    }

    fn new_with_format(
        args: &ModelArgs,
        prefix: &str,
        format: QwenWeightFormat,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Ok(Self {
            gate: TopKRouter::new_with_quantization(
                common::moe::TopKRouterConfig {
                    top_k: args.num_experts_per_tok,
                    num_experts: args.num_experts,
                    hidden_size: args.hidden_size,
                    score_function: TopKRouterScoreFunction::Softmax,
                    norm_topk_prob: true,
                    normalization_epsilon: 0.0,
                    routed_scaling_factor: 1.0,
                    n_group: 1,
                    topk_group: 1,
                    score_correction_bias: false,
                },
                args.quantization_for(&format!("{prefix}.gate.weight")),
                stream,
            )?,
            experts: Experts::new_with_format(args, &format!("{prefix}.experts"), format, stream)?,
            shared_expert: Mlp::new(
                args.hidden_size,
                args.shared_expert_intermediate_size,
                false,
                args,
                Some(&format!("{prefix}.shared_expert")),
                format,
                stream,
            )?,
            shared_expert_gate: QwenLinear::new(
                args.hidden_size,
                1,
                false,
                args.weight_format_for(
                    &format!("{prefix}.shared_expert_gate.weight"),
                    QwenWeightFormat::Dense,
                ),
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
        let shared = self.shared_expert.forward(&flat, stream)?.multiply(
            sigmoid(self.shared_expert_gate.forward(&flat, stream)?, stream)?,
            stream,
        )?;
        let (selected_experts, routing_weights) = self.gate.forward(&flat, stream)?;
        let routed = execute(&flat, &selected_experts, &routing_weights, stream)?;
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
        let mut shared = self.shared_expert.forward(&flat, stream)?;
        if let Some(bias) = self.shared_expert.down_proj.bias.as_ref().as_ref() {
            shared = shared.subtract(bias, stream)?;
        }
        shared = safemlx::distributed::all_sum(&shared, group, stream)?;
        if let Some(bias) = self.shared_expert.down_proj.bias.as_ref().as_ref() {
            shared = shared.add(bias, stream)?;
        }
        shared = shared.multiply(
            sigmoid(self.shared_expert_gate.forward(&flat, stream)?, stream)?,
            stream,
        )?;
        let (selected_experts, routing_weights) = self.gate.forward(&flat, stream)?;
        let routed = execute(&flat, &selected_experts, &routing_weights, stream)?;
        routed.add(shared, stream)?.reshape(shape, stream)
    }

    pub(crate) fn forward_expert_parallel(
        &mut self,
        hidden_states: &Array,
        assignment: &crate::composition::mlx_architectures::distributed::expert::ExpertAssignment,
        group: &safemlx::distributed::Group,
        statistics: &mut crate::composition::mlx_architectures::distributed::expert::RoutingStatistics,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let shape = hidden_states.shape();
        let flat = hidden_states.reshape(&[-1, shape[2]], stream)?;
        let shared = self.shared_expert.forward(&flat, stream)?.multiply(
            sigmoid(self.shared_expert_gate.forward(&flat, stream)?, stream)?,
            stream,
        )?;
        let router_started = std::time::Instant::now();
        let (selected_experts, routing_weights) = self.gate.forward(&flat, stream)?;
        statistics.router_time += router_started.elapsed();
        let returned =
            crate::composition::mlx_architectures::distributed::expert::dispatch_replicated(
                &flat,
                &selected_experts,
                &routing_weights,
                assignment,
                &mut self.experts,
                group,
                stream,
            )
            .map_err(|error| Exception::custom(error.to_string()))?;
        statistics.accumulate(&returned.statistics);
        returned
            .reduced_output
            .add(shared, stream)?
            .reshape(shape, stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_tensor_expert_parallel(
        &mut self,
        hidden_states: &Array,
        tensor_group: &safemlx::distributed::Group,
        assignment: &crate::composition::mlx_architectures::distributed::expert::ExpertAssignment,
        expert_group: &safemlx::distributed::Group,
        statistics: &mut crate::composition::mlx_architectures::distributed::expert::RoutingStatistics,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let shape = hidden_states.shape();
        let flat = hidden_states.reshape(&[-1, shape[2]], stream)?;
        let mut shared = self.shared_expert.forward(&flat, stream)?;
        if let Some(bias) = self.shared_expert.down_proj.bias.as_ref().as_ref() {
            shared = shared.subtract(bias, stream)?;
        }
        shared = safemlx::distributed::all_sum(&shared, tensor_group, stream)?;
        if let Some(bias) = self.shared_expert.down_proj.bias.as_ref().as_ref() {
            shared = shared.add(bias, stream)?;
        }
        shared = shared.multiply(
            sigmoid(self.shared_expert_gate.forward(&flat, stream)?, stream)?,
            stream,
        )?;
        let router_started = std::time::Instant::now();
        let (selected_experts, routing_weights) = self.gate.forward(&flat, stream)?;
        statistics.router_time += router_started.elapsed();
        let returned =
            crate::composition::mlx_architectures::distributed::expert::dispatch_replicated(
                &flat,
                &selected_experts,
                &routing_weights,
                assignment,
                &mut self.experts,
                expert_group,
                stream,
            )
            .map_err(|error| Exception::custom(error.to_string()))?;
        statistics.accumulate(&returned.statistics);
        let routed = safemlx::distributed::all_sum(&returned.reduced_output, tensor_group, stream)?;
        routed.add(shared, stream)?.reshape(shape, stream)
    }

    /// Forward pass that reports router and expert activations to an observer.
    pub fn forward_with_observer(
        &mut self,
        hidden_states: &Array,
        stream: &Stream,
        prefix: &str,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        let shape = hidden_states.shape();
        let b = shape[0];
        let l = shape[1];
        let h = shape[2];
        let flat = hidden_states.reshape(&[-1, h], stream)?;
        observer.observe(&format!("{prefix}.input_flat"), &flat)?;

        let shared_gate = sigmoid(self.shared_expert_gate.forward(&flat, stream)?, stream)?;
        observer.observe(&format!("{prefix}.shared_expert_gate"), &shared_gate)?;
        let shared = self
            .shared_expert
            .forward(&flat, stream)?
            .multiply(shared_gate, stream)?;
        observer.observe(&format!("{prefix}.shared_expert_output"), &shared)?;
        profile_array(PerfComponent::MoeShared, &shared)?;

        let routing =
            self.gate
                .forward_with_observer(&flat, stream, &format!("{prefix}.gate"), observer)?;
        let selected_experts = routing.indices;
        let selected_scores = routing.scores;
        let routing_weights = routing.weights;
        profile_arrays(
            PerfComponent::MoeRouter,
            &[&selected_experts, &routing_weights],
        )?;

        let routed = self.experts.forward_chunked_with_observer(
            &flat,
            &selected_experts,
            &routing_weights,
            stream,
            &format!("{prefix}.experts"),
            observer,
        )?;
        observer.observe(&format!("{prefix}.routed_expert_output"), &routed)?;
        profile_array(PerfComponent::MoeRouted, &routed)?;

        let combined = routed.add(&shared, stream)?;
        observer.observe(&format!("{prefix}.combined_flat"), &combined)?;
        observer.observe_moe_routing(MoeRoutingObservation {
            prefix,
            selected_experts: &selected_experts,
            selected_scores: &selected_scores,
            routing_weights: &routing_weights,
            routed_output: &routed,
            local_routed_output: None,
            reduced_routed_output: Some(&routed),
            shared_output: Some(&shared),
            combined_output: Some(&combined),
            num_experts: self.gate.num_experts,
        })?;
        let output = combined.reshape(&[b, l, h], stream)?;
        observer.observe(&format!("{prefix}.output"), &output)?;
        profile_array(PerfComponent::MoeCombine, &output)?;
        Ok(output)
    }
}

impl Module<&Array> for SparseMoeBlock {
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(
        &mut self,
        hidden_states: &Array,
        stream: &Stream,
    ) -> Result<Self::Output, Self::Error> {
        let shape = hidden_states.shape();
        let B = shape[0];
        let L = shape[1];
        let H = shape[2];
        let flat = hidden_states.reshape(&[-1, H], stream)?;
        let shared = self.shared_expert.forward(&flat, stream)?.multiply(
            sigmoid(self.shared_expert_gate.forward(&flat, stream)?, stream)?,
            stream,
        )?;
        profile_array(PerfComponent::MoeShared, &shared)?;
        let (selected_experts, routing_weights) = self.gate.forward(&flat, stream)?;
        profile_arrays(
            PerfComponent::MoeRouter,
            &[&selected_experts, &routing_weights],
        )?;
        let routed =
            self.experts
                .forward_chunked(&flat, &selected_experts, &routing_weights, stream)?;
        profile_array(PerfComponent::MoeRouted, &routed)?;
        let output = routed.add(shared, stream)?.reshape(&[B, L, H], stream)?;
        profile_array(PerfComponent::MoeCombine, &output)?;
        Ok(output)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate.training_mode(mode);
        self.experts.training_mode(mode);
        self.shared_expert.training_mode(mode);
        self.shared_expert_gate.training_mode(mode);
    }
}

#[derive(Debug, Clone)]
/// Dense or sparse-MoE feed-forward layer stored under the checkpoint-native `mlp` namespace.
pub enum FeedForward {
    /// Dense SwiGLU MLP.
    Dense(Box<Mlp>),
    /// Sparse mixture-of-experts block.
    Moe(Box<SparseMoeBlock>),
}

impl FeedForward {
    fn new(
        args: &ModelArgs,
        prefix: &str,
        format: QwenWeightFormat,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        if args.is_moe() {
            Ok(Self::Moe(Box::new(SparseMoeBlock::new_with_format(
                args,
                &format!("{prefix}.mlp"),
                format,
                stream,
            )?)))
        } else {
            Ok(Self::Dense(Box::new(Mlp::new(
                args.hidden_size,
                args.intermediate_size,
                false,
                args,
                Some(&format!("{prefix}.mlp")),
                format,
                stream,
            )?)))
        }
    }

    fn is_moe(&self) -> bool {
        matches!(self, Self::Moe(_))
    }

    fn forward(&mut self, input: &Array, stream: &Stream) -> Result<Array, Exception> {
        match self {
            Self::Dense(mlp) => mlp.forward(input, stream),
            Self::Moe(moe) => moe.forward(input, stream),
        }
    }

    pub(crate) fn forward_with_expert_executor<F>(
        &mut self,
        input: &Array,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        match self {
            Self::Dense(mlp) => mlp.forward(input, stream),
            Self::Moe(moe) => moe.forward_with_expert_executor(input, stream, execute),
        }
    }

    fn forward_with_observer(
        &mut self,
        input: &Array,
        stream: &Stream,
        prefix: &str,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        match self {
            Self::Dense(mlp) => mlp.forward_with_observer(input, stream, prefix, observer),
            Self::Moe(moe) => moe.forward_with_observer(input, stream, prefix, observer),
        }
    }

    fn training_mode(&mut self, mode: bool) {
        match self {
            Self::Dense(mlp) => mlp.training_mode(mode),
            Self::Moe(moe) => moe.training_mode(mode),
        }
    }
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

#[derive(Debug, Clone, ModuleParameters)]
/// Qwen3.5 transformer block.
pub struct TransformerBlock {
    /// Layer policy.
    pub layer_policy: LayerPolicy,
    #[param]
    /// Full-attention layer when selected by `layer_policy`.
    pub self_attn: Option<FullAttention>,
    #[param]
    /// Linear-attention layer when selected by `layer_policy`.
    pub linear_attn: Option<LinearAttention>,
    #[param]
    /// Dense or sparse-MoE feed-forward block.
    pub mlp: FeedForward,
    #[param]
    /// Pre-attention normalization.
    pub input_layernorm: Qwen3NextRmsNorm,
    #[param]
    /// Pre-feed-forward normalization.
    pub post_attention_layernorm: Qwen3NextRmsNorm,
}

impl TransformerBlock {
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
        let h = self.input_layernorm.forward(x, stream)?;
        let h = match (self.layer_policy, cache) {
            (
                LayerPolicy::SelfAttention(AttentionPolicy::Full),
                Some(LayerCache::FullAttention(cache)),
            ) => self
                .self_attn
                .as_mut()
                .expect("full attention layer")
                .forward(
                    FullAttentionInput {
                        x: &h,
                        mask,
                        cache: Some(cache),
                    },
                    stream,
                )?,
            (LayerPolicy::SelfAttention(AttentionPolicy::Full), None) => self
                .self_attn
                .as_mut()
                .expect("full attention layer")
                .forward(
                    FullAttentionInput {
                        x: &h,
                        mask,
                        cache: None,
                    },
                    stream,
                )?,
            (LayerPolicy::LinearAttention, Some(LayerCache::LinearAttention(cache))) => self
                .linear_attn
                .as_mut()
                .expect("linear attention layer")
                .forward(
                    LinearAttentionInput {
                        x: &h,
                        cache: Some(cache),
                    },
                    stream,
                )?,
            (LayerPolicy::LinearAttention, None) => self
                .linear_attn
                .as_mut()
                .expect("linear attention layer")
                .forward(LinearAttentionInput { x: &h, cache: None }, stream)?,
            (policy, Some(_)) => {
                return Err(Exception::custom(format!(
                    "Qwen hybrid cache kind does not match layer policy {policy:?}"
                )))
            }
            (LayerPolicy::SelfAttention(AttentionPolicy::Sliding { .. }), _) => {
                return Err(Exception::custom(
                    "Qwen hybrid execution does not support sliding self-attention",
                ))
            }
        };
        let h = residual.add(h, stream)?;
        let residual = h.clone();
        let post_normed = self.post_attention_layernorm.forward(&h, stream)?;
        let h = self
            .mlp
            .forward_with_expert_executor(&post_normed, stream, execute)?;
        residual.add(h, stream)
    }

    /// Executes local full/recurrent-attention heads and local dense/expert
    /// intermediates.
    pub(crate) fn forward_tensor_parallel(
        &mut self,
        input: BlockInput<'_>,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let BlockInput { x, mask, cache } = input;
        let normalized = self.input_layernorm.forward(x, stream)?;
        let attention = match (self.layer_policy, cache) {
            (
                LayerPolicy::SelfAttention(AttentionPolicy::Full),
                Some(LayerCache::FullAttention(cache)),
            ) => self
                .self_attn
                .as_mut()
                .expect("full attention layer")
                .forward_tensor_parallel(
                    FullAttentionInput {
                        x: &normalized,
                        mask,
                        cache: Some(cache),
                    },
                    group,
                    stream,
                )?,
            (LayerPolicy::SelfAttention(AttentionPolicy::Full), None) => self
                .self_attn
                .as_mut()
                .expect("full attention layer")
                .forward_tensor_parallel(
                    FullAttentionInput {
                        x: &normalized,
                        mask,
                        cache: None,
                    },
                    group,
                    stream,
                )?,
            (LayerPolicy::LinearAttention, Some(LayerCache::LinearAttention(cache))) => self
                .linear_attn
                .as_mut()
                .expect("linear attention layer")
                .forward_tensor_parallel(
                    LinearAttentionInput {
                        x: &normalized,
                        cache: Some(cache),
                    },
                    group,
                    stream,
                )?,
            (LayerPolicy::LinearAttention, None) => self
                .linear_attn
                .as_mut()
                .expect("linear attention layer")
                .forward_tensor_parallel(
                    LinearAttentionInput {
                        x: &normalized,
                        cache: None,
                    },
                    group,
                    stream,
                )?,
            (policy, Some(_)) => {
                return Err(Exception::custom(format!(
                    "Qwen hybrid tensor-parallel cache kind does not match layer policy {policy:?}"
                )))
            }
            (LayerPolicy::SelfAttention(AttentionPolicy::Sliding { .. }), _) => {
                return Err(Exception::custom(
                    "Qwen hybrid tensor parallelism does not support sliding attention",
                ))
            }
        };
        let hidden = x.add(attention, stream)?;
        let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
        let mut partial = self.mlp.forward(&normalized, stream)?;
        let ordinary_bias = match &self.mlp {
            FeedForward::Dense(mlp) => mlp.down_proj.bias.as_ref().as_ref(),
            FeedForward::Moe(_) => None,
        };
        if let Some(bias) = ordinary_bias {
            partial = partial.subtract(bias, stream)?;
        }
        let mut feed_forward = safemlx::distributed::all_sum(&partial, group, stream)?;
        if let Some(bias) = ordinary_bias {
            feed_forward = feed_forward.add(bias, stream)?;
        }
        hidden.add(feed_forward, stream)
    }

    /// Executes TP-local attention and shared projections while routed
    /// experts are evaluated by the matching EP subgroup.
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
        let normalized = self.input_layernorm.forward(x, stream)?;
        let attention = match (self.layer_policy, cache) {
            (
                LayerPolicy::SelfAttention(AttentionPolicy::Full),
                Some(LayerCache::FullAttention(cache)),
            ) => self
                .self_attn
                .as_mut()
                .expect("full attention layer")
                .forward_tensor_parallel(
                    FullAttentionInput {
                        x: &normalized,
                        mask,
                        cache: Some(cache),
                    },
                    group,
                    stream,
                )?,
            (LayerPolicy::SelfAttention(AttentionPolicy::Full), None) => self
                .self_attn
                .as_mut()
                .expect("full attention layer")
                .forward_tensor_parallel(
                    FullAttentionInput {
                        x: &normalized,
                        mask,
                        cache: None,
                    },
                    group,
                    stream,
                )?,
            (LayerPolicy::LinearAttention, Some(LayerCache::LinearAttention(cache))) => self
                .linear_attn
                .as_mut()
                .expect("linear attention layer")
                .forward_tensor_parallel(
                    LinearAttentionInput {
                        x: &normalized,
                        cache: Some(cache),
                    },
                    group,
                    stream,
                )?,
            (LayerPolicy::LinearAttention, None) => self
                .linear_attn
                .as_mut()
                .expect("linear attention layer")
                .forward_tensor_parallel(
                    LinearAttentionInput {
                        x: &normalized,
                        cache: None,
                    },
                    group,
                    stream,
                )?,
            (policy, Some(_)) => {
                return Err(Exception::custom(format!(
                    "Qwen hybrid tensor/expert cache kind does not match layer policy {policy:?}"
                )))
            }
            (LayerPolicy::SelfAttention(AttentionPolicy::Sliding { .. }), _) => {
                return Err(Exception::custom(
                    "Qwen hybrid tensor/expert execution does not support sliding attention",
                ))
            }
        };
        let hidden = x.add(attention, stream)?;
        let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
        let feed_forward = match &mut self.mlp {
            FeedForward::Moe(moe) => {
                moe.forward_tensor_with_expert_executor(&normalized, group, stream, execute)?
            }
            FeedForward::Dense(_) => {
                return Err(Exception::custom(
                    "Qwen hybrid TP+EP requires routed MoE decoder layers",
                ))
            }
        };
        hidden.add(feed_forward, stream)
    }

    /// Creates an unloaded transformer block.
    pub fn new(args: &ModelArgs, layer_idx: usize, stream: &Stream) -> Result<Self, Exception> {
        Self::new_with_format(
            args,
            layer_idx,
            QwenWeightFormat::for_text(args, None),
            stream,
        )
    }

    pub(crate) fn new_parallel_layerwise(
        args: &ModelArgs,
        layer_idx: usize,
        geometry: ParallelLayerGeometry,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let local = geometry.local_args(args);
        Self::new(&local, layer_idx, stream).map_err(Into::into)
    }

    pub(crate) fn new_mtp_parallel_layerwise(
        args: &ModelArgs,
        layer_idx: usize,
        geometry: ParallelLayerGeometry,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let local = geometry.local_args(args);
        Self::new_mtp_with_format(
            &local,
            layer_idx,
            QwenWeightFormat::for_text(&local, None),
            stream,
        )
        .map_err(Into::into)
    }

    fn new_with_format(
        args: &ModelArgs,
        layer_idx: usize,
        format: QwenWeightFormat,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let layer_policy = *args.layer_schedule.get(layer_idx).ok_or_else(|| {
            Exception::custom(format!(
                "Qwen hybrid layer schedule is missing decoder layer {layer_idx}"
            ))
        })?;
        let prefix = format!("model.layers.{layer_idx}");
        Ok(Self {
            layer_policy,
            self_attn: if layer_policy == LayerPolicy::SelfAttention(AttentionPolicy::Full) {
                Some(FullAttention::new_with_format(
                    args,
                    Some(&format!("{prefix}.self_attn")),
                    format,
                    stream,
                )?)
            } else {
                None
            },
            linear_attn: if layer_policy == LayerPolicy::LinearAttention {
                Some(LinearAttention::new_with_format(
                    args,
                    Some(&format!("{prefix}.linear_attn")),
                    format,
                    stream,
                )?)
            } else {
                None
            },
            mlp: FeedForward::new(args, &prefix, format, stream)?,
            input_layernorm: Qwen3NextRmsNorm::new(args.hidden_size, args.rms_norm_eps, stream)?,
            post_attention_layernorm: Qwen3NextRmsNorm::new(
                args.hidden_size,
                args.rms_norm_eps,
                stream,
            )?,
        })
    }

    fn new_mtp_with_format(
        args: &ModelArgs,
        layer_idx: usize,
        format: QwenWeightFormat,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let prefix = format!("mtp.layers.{layer_idx}");
        Ok(Self {
            layer_policy: LayerPolicy::SelfAttention(AttentionPolicy::Full),
            self_attn: Some(FullAttention::new_with_format(
                args,
                Some(&format!("{prefix}.self_attn")),
                format,
                stream,
            )?),
            linear_attn: None,
            mlp: FeedForward::new(args, &prefix, format, stream)?,
            input_layernorm: Qwen3NextRmsNorm::new(args.hidden_size, args.rms_norm_eps, stream)?,
            post_attention_layernorm: Qwen3NextRmsNorm::new(
                args.hidden_size,
                args.rms_norm_eps,
                stream,
            )?,
        })
    }
}

/// Input for a Qwen3.5 transformer block.
pub struct BlockInput<'a> {
    /// Hidden states.
    pub x: &'a Array,
    /// Optional attention mask.
    pub mask: Option<&'a Array>,
    /// Optional layer cache.
    pub cache: Option<&'a mut LayerCache>,
}

/// Borrowed hybrid operator state used by generalized execution runtimes.
pub(crate) enum OperatorCache<'a> {
    /// Full-attention key/value state.
    FullAttention(&'a mut dyn KeyValueCache),
    /// Linear-attention convolution and recurrent state.
    LinearAttention(&'a mut LinearAttentionCache),
}

impl TransformerBlock {
    /// Executes one text block from semantic operator state without requiring
    /// the resident model's family-specific cache container.
    pub(crate) fn forward_with_operator_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<OperatorCache<'_>>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let residual = x;
        let h = self.input_layernorm.forward(x, stream)?;
        let h = match (self.layer_policy, cache) {
            (
                LayerPolicy::SelfAttention(AttentionPolicy::Full),
                Some(OperatorCache::FullAttention(cache)),
            ) => self
                .self_attn
                .as_mut()
                .expect("full attention layer")
                .forward(
                    FullAttentionInput {
                        x: &h,
                        mask,
                        cache: Some(cache),
                    },
                    stream,
                )?,
            (LayerPolicy::SelfAttention(AttentionPolicy::Full), None) => self
                .self_attn
                .as_mut()
                .expect("full attention layer")
                .forward(
                    FullAttentionInput {
                        x: &h,
                        mask,
                        cache: None,
                    },
                    stream,
                )?,
            (LayerPolicy::LinearAttention, Some(OperatorCache::LinearAttention(cache))) => self
                .linear_attn
                .as_mut()
                .expect("linear attention layer")
                .forward(
                    LinearAttentionInput {
                        x: &h,
                        cache: Some(cache),
                    },
                    stream,
                )?,
            (LayerPolicy::LinearAttention, None) => self
                .linear_attn
                .as_mut()
                .expect("linear attention layer")
                .forward(LinearAttentionInput { x: &h, cache: None }, stream)?,
            (LayerPolicy::SelfAttention(AttentionPolicy::Sliding { .. }), _) => {
                return Err(Exception::custom(
                    "Qwen hybrid execution does not support sliding self-attention",
                ))
            }
            (policy, Some(_)) => {
                return Err(Exception::custom(format!(
                    "Qwen hybrid operator cache does not match layer policy {policy:?}"
                )))
            }
        };
        let h = residual.add(h, stream)?;
        let residual = h.clone();
        let normalized = self.post_attention_layernorm.forward(&h, stream)?;
        residual.add(self.mlp.forward(&normalized, stream)?, stream)
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
        let normalized = self.input_layernorm.forward(x, stream)?;
        let attention = match (self.layer_policy, cache) {
            (
                LayerPolicy::SelfAttention(AttentionPolicy::Full),
                Some(OperatorCache::FullAttention(cache)),
            ) => self
                .self_attn
                .as_mut()
                .expect("full attention layer")
                .forward(
                    FullAttentionInput {
                        x: &normalized,
                        mask,
                        cache: Some(cache),
                    },
                    stream,
                )?,
            (LayerPolicy::SelfAttention(AttentionPolicy::Full), None) => self
                .self_attn
                .as_mut()
                .expect("full attention layer")
                .forward(
                    FullAttentionInput {
                        x: &normalized,
                        mask,
                        cache: None,
                    },
                    stream,
                )?,
            (LayerPolicy::LinearAttention, Some(OperatorCache::LinearAttention(cache))) => self
                .linear_attn
                .as_mut()
                .expect("linear attention layer")
                .forward(
                    LinearAttentionInput {
                        x: &normalized,
                        cache: Some(cache),
                    },
                    stream,
                )?,
            (LayerPolicy::LinearAttention, None) => self
                .linear_attn
                .as_mut()
                .expect("linear attention layer")
                .forward(
                    LinearAttentionInput {
                        x: &normalized,
                        cache: None,
                    },
                    stream,
                )?,
            (LayerPolicy::SelfAttention(AttentionPolicy::Sliding { .. }), _) => {
                return Err(Exception::custom(
                    "Qwen hybrid external-expert execution does not support sliding attention",
                ))
            }
            (policy, Some(_)) => {
                return Err(Exception::custom(format!(
                "Qwen hybrid external-expert operator cache does not match layer policy {policy:?}"
            )))
            }
        };
        let hidden = x.add(attention, stream)?;
        let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
        let feed_forward = match &mut self.mlp {
            FeedForward::Moe(moe) => {
                moe.forward_with_expert_executor(&normalized, stream, execute)?
            }
            FeedForward::Dense(_) => {
                return Err(Exception::custom(
                    "Qwen hybrid external expert execution requires routed MoE decoder layers",
                ))
            }
        };
        hidden.add(feed_forward, stream)
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
        let normalized = self.input_layernorm.forward(x, stream)?;
        let attention = match (self.layer_policy, cache) {
            (
                LayerPolicy::SelfAttention(AttentionPolicy::Full),
                Some(OperatorCache::FullAttention(cache)),
            ) => self
                .self_attn
                .as_mut()
                .expect("full attention layer")
                .forward_tensor_parallel(
                    FullAttentionInput {
                        x: &normalized,
                        mask,
                        cache: Some(cache),
                    },
                    group,
                    stream,
                )?,
            (LayerPolicy::SelfAttention(AttentionPolicy::Full), None) => self
                .self_attn
                .as_mut()
                .expect("full attention layer")
                .forward_tensor_parallel(
                    FullAttentionInput {
                        x: &normalized,
                        mask,
                        cache: None,
                    },
                    group,
                    stream,
                )?,
            (LayerPolicy::LinearAttention, Some(OperatorCache::LinearAttention(cache))) => self
                .linear_attn
                .as_mut()
                .expect("linear attention layer")
                .forward_tensor_parallel(
                    LinearAttentionInput {
                        x: &normalized,
                        cache: Some(cache),
                    },
                    group,
                    stream,
                )?,
            (LayerPolicy::LinearAttention, None) => self
                .linear_attn
                .as_mut()
                .expect("linear attention layer")
                .forward_tensor_parallel(
                    LinearAttentionInput {
                        x: &normalized,
                        cache: None,
                    },
                    group,
                    stream,
                )?,
            (LayerPolicy::SelfAttention(AttentionPolicy::Sliding { .. }), _) => {
                return Err(Exception::custom(
                    "Qwen hybrid tensor/external-expert execution does not support sliding attention",
                ))
            }
            (policy, Some(_)) => {
                return Err(Exception::custom(format!(
                    "Qwen hybrid tensor/external-expert operator cache does not match layer policy {policy:?}"
                )))
            }
        };
        let hidden = x.add(attention, stream)?;
        let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
        let feed_forward = match &mut self.mlp {
            FeedForward::Moe(moe) => {
                moe.forward_tensor_with_expert_executor(&normalized, group, stream, execute)?
            }
            FeedForward::Dense(_) => return Err(Exception::custom(
                "Qwen hybrid tensor/external expert execution requires routed MoE decoder layers",
            )),
        };
        hidden.add(feed_forward, stream)
    }

    pub(crate) fn forward_tensor_parallel_with_operator_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<OperatorCache<'_>>,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let normalized = self.input_layernorm.forward(x, stream)?;
        let attention = match (self.layer_policy, cache) {
            (
                LayerPolicy::SelfAttention(AttentionPolicy::Full),
                Some(OperatorCache::FullAttention(cache)),
            ) => self
                .self_attn
                .as_mut()
                .expect("full attention layer")
                .forward_tensor_parallel(
                    FullAttentionInput {
                        x: &normalized,
                        mask,
                        cache: Some(cache),
                    },
                    group,
                    stream,
                )?,
            (LayerPolicy::SelfAttention(AttentionPolicy::Full), None) => self
                .self_attn
                .as_mut()
                .expect("full attention layer")
                .forward_tensor_parallel(
                    FullAttentionInput {
                        x: &normalized,
                        mask,
                        cache: None,
                    },
                    group,
                    stream,
                )?,
            (LayerPolicy::LinearAttention, Some(OperatorCache::LinearAttention(cache))) => self
                .linear_attn
                .as_mut()
                .expect("linear attention layer")
                .forward_tensor_parallel(
                    LinearAttentionInput {
                        x: &normalized,
                        cache: Some(cache),
                    },
                    group,
                    stream,
                )?,
            (LayerPolicy::LinearAttention, None) => self
                .linear_attn
                .as_mut()
                .expect("linear attention layer")
                .forward_tensor_parallel(
                    LinearAttentionInput {
                        x: &normalized,
                        cache: None,
                    },
                    group,
                    stream,
                )?,
            (LayerPolicy::SelfAttention(AttentionPolicy::Sliding { .. }), _) => {
                return Err(Exception::custom(
                    "Qwen hybrid tensor parallelism does not support sliding attention",
                ))
            }
            (policy, Some(_)) => {
                return Err(Exception::custom(format!(
                "Qwen hybrid tensor-parallel operator cache does not match layer policy {policy:?}"
            )))
            }
        };
        let hidden = x.add(attention, stream)?;
        let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
        let mut partial = self.mlp.forward(&normalized, stream)?;
        let ordinary_bias = match &self.mlp {
            FeedForward::Dense(mlp) => mlp.down_proj.bias.as_ref().as_ref(),
            FeedForward::Moe(_) => None,
        };
        if let Some(bias) = ordinary_bias {
            partial = partial.subtract(bias, stream)?;
        }
        let mut feed_forward = safemlx::distributed::all_sum(&partial, group, stream)?;
        if let Some(bias) = ordinary_bias {
            feed_forward = feed_forward.add(bias, stream)?;
        }
        hidden.add(feed_forward, stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_expert_parallel_with_operator_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<OperatorCache<'_>>,
        assignment: &crate::composition::mlx_architectures::distributed::expert::ExpertAssignment,
        group: &safemlx::distributed::Group,
        statistics: &mut crate::composition::mlx_architectures::distributed::expert::RoutingStatistics,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let normalized = self.input_layernorm.forward(x, stream)?;
        let attention = match (self.layer_policy, cache) {
            (
                LayerPolicy::SelfAttention(AttentionPolicy::Full),
                Some(OperatorCache::FullAttention(cache)),
            ) => self
                .self_attn
                .as_mut()
                .expect("full attention layer")
                .forward(
                    FullAttentionInput {
                        x: &normalized,
                        mask,
                        cache: Some(cache),
                    },
                    stream,
                )?,
            (LayerPolicy::SelfAttention(AttentionPolicy::Full), None) => self
                .self_attn
                .as_mut()
                .expect("full attention layer")
                .forward(
                    FullAttentionInput {
                        x: &normalized,
                        mask,
                        cache: None,
                    },
                    stream,
                )?,
            (LayerPolicy::LinearAttention, Some(OperatorCache::LinearAttention(cache))) => self
                .linear_attn
                .as_mut()
                .expect("linear attention layer")
                .forward(
                    LinearAttentionInput {
                        x: &normalized,
                        cache: Some(cache),
                    },
                    stream,
                )?,
            (LayerPolicy::LinearAttention, None) => self
                .linear_attn
                .as_mut()
                .expect("linear attention layer")
                .forward(
                    LinearAttentionInput {
                        x: &normalized,
                        cache: None,
                    },
                    stream,
                )?,
            (LayerPolicy::SelfAttention(AttentionPolicy::Sliding { .. }), _) => {
                return Err(Exception::custom(
                    "Qwen hybrid expert parallelism does not support sliding attention",
                ))
            }
            (policy, Some(_)) => {
                return Err(Exception::custom(format!(
                "Qwen hybrid expert-parallel operator cache does not match layer policy {policy:?}"
            )))
            }
        };
        let hidden = x.add(attention, stream)?;
        let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
        let feed_forward = match &mut self.mlp {
            FeedForward::Moe(moe) => {
                moe.forward_expert_parallel(&normalized, assignment, group, statistics, stream)?
            }
            FeedForward::Dense(_) => {
                return Err(Exception::custom(
                    "Qwen hybrid PP+EP requires routed MoE decoder layers",
                ))
            }
        };
        hidden.add(feed_forward, stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_tensor_expert_parallel_with_operator_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<OperatorCache<'_>>,
        tensor_group: &safemlx::distributed::Group,
        assignment: &crate::composition::mlx_architectures::distributed::expert::ExpertAssignment,
        expert_group: &safemlx::distributed::Group,
        statistics: &mut crate::composition::mlx_architectures::distributed::expert::RoutingStatistics,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let normalized = self.input_layernorm.forward(x, stream)?;
        let attention = match (self.layer_policy, cache) {
            (
                LayerPolicy::SelfAttention(AttentionPolicy::Full),
                Some(OperatorCache::FullAttention(cache)),
            ) => self
                .self_attn
                .as_mut()
                .expect("full attention layer")
                .forward_tensor_parallel(
                    FullAttentionInput {
                        x: &normalized,
                        mask,
                        cache: Some(cache),
                    },
                    tensor_group,
                    stream,
                )?,
            (LayerPolicy::SelfAttention(AttentionPolicy::Full), None) => self
                .self_attn
                .as_mut()
                .expect("full attention layer")
                .forward_tensor_parallel(
                    FullAttentionInput {
                        x: &normalized,
                        mask,
                        cache: None,
                    },
                    tensor_group,
                    stream,
                )?,
            (LayerPolicy::LinearAttention, Some(OperatorCache::LinearAttention(cache))) => self
                .linear_attn
                .as_mut()
                .expect("linear attention layer")
                .forward_tensor_parallel(
                    LinearAttentionInput {
                        x: &normalized,
                        cache: Some(cache),
                    },
                    tensor_group,
                    stream,
                )?,
            (LayerPolicy::LinearAttention, None) => self
                .linear_attn
                .as_mut()
                .expect("linear attention layer")
                .forward_tensor_parallel(
                    LinearAttentionInput {
                        x: &normalized,
                        cache: None,
                    },
                    tensor_group,
                    stream,
                )?,
            (LayerPolicy::SelfAttention(AttentionPolicy::Sliding { .. }), _) => {
                return Err(Exception::custom(
                    "Qwen hybrid tensor/expert parallelism does not support sliding attention",
                ))
            }
            (policy, Some(_)) => {
                return Err(Exception::custom(format!(
                "Qwen hybrid tensor/expert operator cache does not match layer policy {policy:?}"
            )))
            }
        };
        let hidden = x.add(attention, stream)?;
        let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
        let feed_forward = match &mut self.mlp {
            FeedForward::Moe(moe) => moe.forward_tensor_expert_parallel(
                &normalized,
                tensor_group,
                assignment,
                expert_group,
                statistics,
                stream,
            )?,
            FeedForward::Dense(_) => {
                return Err(Exception::custom(
                    "Qwen hybrid TP+PP+EP requires routed MoE decoder layers",
                ))
            }
        };
        hidden.add(feed_forward, stream)
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
        let h = self.input_layernorm.forward(x, stream)?;
        let h = match (self.layer_policy, cache) {
            (
                LayerPolicy::SelfAttention(AttentionPolicy::Full),
                Some(LayerCache::FullAttention(cache)),
            ) => self
                .self_attn
                .as_mut()
                .expect("full attention layer")
                .forward(
                    FullAttentionInput {
                        x: &h,
                        mask,
                        cache: Some(cache),
                    },
                    stream,
                )?,
            (LayerPolicy::SelfAttention(AttentionPolicy::Full), None) => self
                .self_attn
                .as_mut()
                .expect("full attention layer")
                .forward(
                    FullAttentionInput {
                        x: &h,
                        mask,
                        cache: None,
                    },
                    stream,
                )?,
            (LayerPolicy::LinearAttention, Some(LayerCache::LinearAttention(cache))) => self
                .linear_attn
                .as_mut()
                .expect("linear attention layer")
                .forward(
                    LinearAttentionInput {
                        x: &h,
                        cache: Some(cache),
                    },
                    stream,
                )?,
            (LayerPolicy::LinearAttention, None) => self
                .linear_attn
                .as_mut()
                .expect("linear attention layer")
                .forward(LinearAttentionInput { x: &h, cache: None }, stream)?,
            (policy, Some(_)) => {
                return Err(Exception::custom(format!(
                    "Qwen hybrid cache kind does not match layer policy {policy:?}"
                )))
            }
            (LayerPolicy::SelfAttention(AttentionPolicy::Sliding { .. }), _) => {
                return Err(Exception::custom(
                    "Qwen hybrid execution does not support sliding self-attention",
                ))
            }
        };
        match self.layer_policy {
            LayerPolicy::SelfAttention(AttentionPolicy::Full) => {
                profile_array(PerfComponent::FullAttention, &h)?
            }
            LayerPolicy::LinearAttention => profile_array(PerfComponent::LinearAttention, &h)?,
            LayerPolicy::SelfAttention(AttentionPolicy::Sliding { .. }) => unreachable!(),
        }
        let h = residual.add(h, stream)?;
        let residual = h.clone();
        let post_normed = self.post_attention_layernorm.forward(&h, stream)?;
        let h = self.mlp.forward(&post_normed, stream)?;
        residual.add(h, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        if let Some(full_attention) = &mut self.self_attn {
            full_attention.training_mode(mode);
        }
        if let Some(linear_attention) = &mut self.linear_attn {
            linear_attention.training_mode(mode);
        }
        self.mlp.training_mode(mode);
        self.input_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
    }
}

impl TransformerBlock {
    /// Forward pass that reports Qwen3.5 MoE block activations to an observer.
    pub fn forward_with_observer(
        &mut self,
        input: BlockInput<'_>,
        stream: &Stream,
        prefix: &str,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        let BlockInput { x, mask, cache } = input;
        observer.observe(&format!("{prefix}.input"), x)?;
        observer.observe(&format!("{prefix}.residual_before_attention"), x)?;
        let residual = x;
        let h = self.input_layernorm.forward(x, stream)?;
        observer.observe(&format!("{prefix}.input_layernorm"), &h)?;
        let h = match (self.layer_policy, cache) {
            (
                LayerPolicy::SelfAttention(AttentionPolicy::Full),
                Some(LayerCache::FullAttention(cache)),
            ) => self
                .self_attn
                .as_mut()
                .expect("full attention layer")
                .forward_with_observer(
                    FullAttentionInput {
                        x: &h,
                        mask,
                        cache: Some(cache),
                    },
                    stream,
                    &format!("{prefix}.self_attn"),
                    observer,
                )?,
            (LayerPolicy::SelfAttention(AttentionPolicy::Full), None) => self
                .self_attn
                .as_mut()
                .expect("full attention layer")
                .forward_with_observer(
                    FullAttentionInput {
                        x: &h,
                        mask,
                        cache: None,
                    },
                    stream,
                    &format!("{prefix}.self_attn"),
                    observer,
                )?,
            (LayerPolicy::LinearAttention, Some(LayerCache::LinearAttention(cache))) => self
                .linear_attn
                .as_mut()
                .expect("linear attention layer")
                .forward_with_observer(
                    LinearAttentionInput {
                        x: &h,
                        cache: Some(cache),
                    },
                    stream,
                    &format!("{prefix}.linear_attn"),
                    observer,
                )?,
            (LayerPolicy::LinearAttention, None) => self
                .linear_attn
                .as_mut()
                .expect("linear attention layer")
                .forward_with_observer(
                    LinearAttentionInput { x: &h, cache: None },
                    stream,
                    &format!("{prefix}.linear_attn"),
                    observer,
                )?,
            (policy, Some(_)) => {
                return Err(Exception::custom(format!(
                    "Qwen hybrid cache kind does not match layer policy {policy:?}"
                )))
            }
            (LayerPolicy::SelfAttention(AttentionPolicy::Sliding { .. }), _) => {
                return Err(Exception::custom(
                    "Qwen hybrid execution does not support sliding self-attention",
                ))
            }
        };
        observer.observe(&format!("{prefix}.attention_output"), &h)?;
        observer.observe(&format!("{prefix}.residual_delta_attention"), &h)?;
        match self.layer_policy {
            LayerPolicy::SelfAttention(AttentionPolicy::Full) => {
                profile_array(PerfComponent::FullAttention, &h)?
            }
            LayerPolicy::LinearAttention => profile_array(PerfComponent::LinearAttention, &h)?,
            LayerPolicy::SelfAttention(AttentionPolicy::Sliding { .. }) => unreachable!(),
        }
        let h = residual.add(h, stream)?;
        observer.observe(&format!("{prefix}.post_attention_residual"), &h)?;
        observer.observe(&format!("{prefix}.residual_after_attention"), &h)?;

        let feed_forward_name = if self.mlp.is_moe() { "moe" } else { "mlp" };
        observer.observe(&format!("{prefix}.residual_before_{feed_forward_name}"), &h)?;
        let residual = h.clone();
        let post_normed = self.post_attention_layernorm.forward(&h, stream)?;
        observer.observe(&format!("{prefix}.post_attention_layernorm"), &post_normed)?;
        let h = self.mlp.forward_with_observer(
            &post_normed,
            stream,
            &format!("{prefix}.{feed_forward_name}"),
            observer,
        )?;
        observer.observe(&format!("{prefix}.{feed_forward_name}_output"), &h)?;
        observer.observe(&format!("{prefix}.residual_delta_{feed_forward_name}"), &h)?;
        let output = residual.add(h, stream)?;
        let output = observer
            .intervene(&format!("{prefix}.output"), &output)?
            .unwrap_or(output);
        observer.observe(&format!("{prefix}.output"), &output)?;
        observer.observe(
            &format!("{prefix}.residual_after_{feed_forward_name}"),
            &output,
        )?;
        Ok(output)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// Embedded Qwen multi-token-prediction head.
pub(crate) struct MtpModule {
    #[param]
    pub(crate) pre_fc_norm_hidden: Qwen3NextRmsNorm,
    #[param]
    pub(crate) pre_fc_norm_embedding: Qwen3NextRmsNorm,
    #[param]
    pub(crate) fc: QwenLinear,
    #[param]
    pub(crate) layers: Vec<TransformerBlock>,
    #[param]
    pub(crate) norm: Qwen3NextRmsNorm,
}

impl MtpModule {
    pub(crate) fn new_with_format(
        args: &ModelArgs,
        format: QwenWeightFormat,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        if args.mtp_num_hidden_layers < 0 {
            return Err(Exception::custom(
                "Qwen MTP layer count must be non-negative",
            ));
        }
        Ok(Self {
            pre_fc_norm_hidden: Qwen3NextRmsNorm::new(args.hidden_size, args.rms_norm_eps, stream)?,
            pre_fc_norm_embedding: Qwen3NextRmsNorm::new(
                args.hidden_size,
                args.rms_norm_eps,
                stream,
            )?,
            fc: QwenLinear::new(
                args.hidden_size * 2,
                args.hidden_size,
                false,
                // Native Qwen MTP checkpoints keep this fusion projection dense,
                // including when the decoder layer itself is FP8 or affine packed.
                QwenWeightFormat::Dense,
                stream,
            )?,
            layers: (0..args.mtp_num_hidden_layers)
                .map(|index| {
                    TransformerBlock::new_mtp_with_format(args, index as usize, format, stream)
                })
                .collect::<Result<Vec<_>, _>>()?,
            norm: Qwen3NextRmsNorm::new(args.hidden_size, args.rms_norm_eps, stream)?,
        })
    }

    pub(crate) fn forward(
        &mut self,
        hidden: &Array,
        embeddings: &Array,
        cache: &mut [LayerCache],
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if cache.len() != self.layers.len() {
            return Err(Exception::custom(format!(
                "Qwen MTP cache has {} layers, expected {}",
                cache.len(),
                self.layers.len()
            )));
        }
        let layer = self
            .layers
            .first_mut()
            .ok_or_else(|| Exception::custom("Qwen checkpoint does not contain MTP layers"))?;
        let layer_cache = cache
            .first_mut()
            .ok_or_else(|| Exception::custom("Qwen MTP cache is empty"))?;
        let embeddings = self.pre_fc_norm_embedding.forward(embeddings, stream)?;
        let hidden = self.pre_fc_norm_hidden.forward(hidden, stream)?;
        let fused = concatenate_axis(&[&embeddings, &hidden], -1, stream)?;
        let mut fused = self.fc.forward(&fused, stream)?;
        let mask = if fused.dim(1) > 1 {
            let offset = match &*layer_cache {
                LayerCache::FullAttention(cache) => cache.offset(),
                LayerCache::LinearAttention(_) => 0,
            };
            match create_attention_mask(&fused, &offset_cache(offset), Some(true), stream)? {
                Some(AttentionMask::Array(mask)) => Some(mask),
                Some(AttentionMask::Causal) => {
                    return Err(Exception::custom("Qwen MTP requires an array causal mask"));
                }
                None => None,
            }
        } else {
            None
        };
        // Each checkpoint MTP layer corresponds to a speculative step rather
        // than a sequential transformer stack. The generalized runtime currently
        // requests one native Qwen proposal, so execute step zero only.
        fused = layer.forward(
            BlockInput {
                x: &fused,
                mask: mask.as_ref(),
                cache: Some(layer_cache),
            },
            stream,
        )?;
        self.norm.forward(&fused, stream)
    }

    pub(crate) fn forward_tensor_parallel(
        &mut self,
        hidden: &Array,
        embeddings: &Array,
        cache: &mut [LayerCache],
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if cache.len() != self.layers.len() {
            return Err(Exception::custom(format!(
                "Qwen MTP cache has {} layers, expected {}",
                cache.len(),
                self.layers.len()
            )));
        }
        let layer = self
            .layers
            .first_mut()
            .ok_or_else(|| Exception::custom("Qwen checkpoint does not contain MTP layers"))?;
        let layer_cache = cache
            .first_mut()
            .ok_or_else(|| Exception::custom("Qwen MTP cache is empty"))?;
        let embeddings = self.pre_fc_norm_embedding.forward(embeddings, stream)?;
        let hidden = self.pre_fc_norm_hidden.forward(hidden, stream)?;
        let fused = concatenate_axis(&[&embeddings, &hidden], -1, stream)?;
        let fused = self.fc.forward(&fused, stream)?;
        let mask = if fused.dim(1) > 1 {
            let offset = match &*layer_cache {
                LayerCache::FullAttention(cache) => cache.offset(),
                LayerCache::LinearAttention(_) => 0,
            };
            match create_attention_mask(&fused, &offset_cache(offset), Some(true), stream)? {
                Some(AttentionMask::Array(mask)) => Some(mask),
                Some(AttentionMask::Causal) => {
                    return Err(Exception::custom("Qwen MTP requires an array causal mask"));
                }
                None => None,
            }
        } else {
            None
        };
        let fused = layer.forward_tensor_parallel(
            BlockInput {
                x: &fused,
                mask: mask.as_ref(),
                cache: Some(layer_cache),
            },
            group,
            stream,
        )?;
        self.norm.forward(&fused, stream)
    }

    pub(crate) fn forward_with_expert_executor<F>(
        &mut self,
        hidden: &Array,
        embeddings: &Array,
        cache: &mut [LayerCache],
        tensor_group: Option<&safemlx::distributed::Group>,
        execute: F,
        stream: &Stream,
    ) -> Result<Array, Exception>
    where
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        if cache.len() != self.layers.len() {
            return Err(Exception::custom(format!(
                "Qwen MTP cache has {} layers, expected {}",
                cache.len(),
                self.layers.len()
            )));
        }
        let layer = self
            .layers
            .first_mut()
            .ok_or_else(|| Exception::custom("Qwen checkpoint does not contain MTP layers"))?;
        let layer_cache = cache
            .first_mut()
            .ok_or_else(|| Exception::custom("Qwen MTP cache is empty"))?;
        let embeddings = self.pre_fc_norm_embedding.forward(embeddings, stream)?;
        let hidden = self.pre_fc_norm_hidden.forward(hidden, stream)?;
        let fused = concatenate_axis(&[&embeddings, &hidden], -1, stream)?;
        let fused = self.fc.forward(&fused, stream)?;
        let mask = if fused.dim(1) > 1 {
            let offset = match &*layer_cache {
                LayerCache::FullAttention(cache) => cache.offset(),
                LayerCache::LinearAttention(_) => 0,
            };
            match create_attention_mask(&fused, &offset_cache(offset), Some(true), stream)? {
                Some(AttentionMask::Array(mask)) => Some(mask),
                Some(AttentionMask::Causal) => {
                    return Err(Exception::custom("Qwen MTP requires an array causal mask"));
                }
                None => None,
            }
        } else {
            None
        };
        let input = BlockInput {
            x: &fused,
            mask: mask.as_ref(),
            cache: Some(layer_cache),
        };
        let fused = match tensor_group {
            Some(group) => {
                layer.forward_tensor_with_expert_executor(input, group, stream, execute)?
            }
            None => layer.forward_sparse_experts(input, stream, execute)?,
        };
        self.norm.forward(&fused, stream)
    }

    pub(crate) fn len(&self) -> usize {
        self.layers.len()
    }
}

/// Qwen3.5 MoE text transformer body without the language-model head.
#[derive(Debug, Clone, ModuleParameters)]
pub struct Qwen35TextModel {
    /// Token vocabulary size.
    pub vocab_size: i32,
    /// Number of decoder layers.
    pub num_hidden_layers: i32,
    #[param]
    /// Token embedding table.
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[param]
    /// Transformer blocks.
    pub layers: Vec<TransformerBlock>,
    #[param]
    /// Final normalization.
    pub norm: Qwen3NextRmsNorm,
}

impl Qwen35TextModel {
    /// Creates an unloaded Qwen3.5 MoE text transformer body.
    pub fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        Self::new_with_format(args, QwenWeightFormat::for_text(args, None), stream)
    }

    fn new_with_format(
        args: &ModelArgs,
        format: QwenWeightFormat,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let embed_tokens = common::linear::unloaded_maybe_quantized_embedding(
            args.vocab_size,
            args.hidden_size,
            args.weight_format_for("model.embed_tokens.weight", format)
                .affine(),
            stream,
        )?;
        let layers = (0..args.num_hidden_layers)
            .map(|idx| TransformerBlock::new_with_format(args, idx as usize, format, stream))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            vocab_size: args.vocab_size,
            num_hidden_layers: args.num_hidden_layers,
            embed_tokens,
            layers,
            norm: Qwen3NextRmsNorm::new(args.hidden_size, args.rms_norm_eps, stream)?,
        })
    }

    /// Forward pass that reports activations to an observer.
    pub fn forward_with_observer(
        &mut self,
        input: ModelInput<'_>,
        stream: &Stream,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        let ModelInput {
            inputs,
            inputs_embeds,
            mask,
            mut cache,
        } = input;
        let mut h = match inputs_embeds {
            Some(inputs_embeds) => inputs_embeds.clone(),
            None => self.embed_tokens.forward(inputs, stream)?,
        };
        observer.observe("model.embed_tokens", &h)?;
        profile_array(PerfComponent::Embed, &h)?;
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
        if let Some(mask) = mask.as_ref() {
            observer.observe("model.attention_mask", mask)?;
        }

        if let Some(cache) = cache.as_mut() {
            for (i, (layer, layer_cache)) in self
                .layers
                .iter_mut()
                .zip(cache.layers.iter_mut())
                .enumerate()
            {
                h = layer.forward_with_observer(
                    BlockInput {
                        x: &h,
                        mask: mask.as_ref(),
                        cache: Some(layer_cache),
                    },
                    stream,
                    &format!("model.layers.{i}"),
                    observer,
                )?;
            }
        } else {
            for (i, layer) in self.layers.iter_mut().enumerate() {
                h = layer.forward_with_observer(
                    BlockInput {
                        x: &h,
                        mask: mask.as_ref(),
                        cache: None,
                    },
                    stream,
                    &format!("model.layers.{i}"),
                    observer,
                )?;
            }
        }
        let h = self.norm.forward(&h, stream)?;
        observer.observe("model.norm", &h)?;
        profile_array(PerfComponent::FinalNorm, &h)?;
        Ok(h)
    }
}

/// Input for a Qwen3.5 MoE text forward pass.
pub struct ModelInput<'a> {
    /// Token ids with shape `[batch, sequence]`.
    pub inputs: &'a Array,
    /// Optional prepared embeddings with shape `[batch, sequence, hidden]`.
    pub inputs_embeds: Option<&'a Array>,
    /// Optional attention mask.
    pub mask: Option<&'a Array>,
    /// Optional heterogeneous cache.
    pub cache: Option<&'a mut Cache>,
}

impl Qwen35TextModel {
    pub(crate) fn forward_hidden(
        &mut self,
        input: ModelInput<'_>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let ModelInput {
            inputs,
            inputs_embeds,
            mask,
            mut cache,
        } = input;
        let mut h = match inputs_embeds {
            Some(inputs_embeds) => inputs_embeds.clone(),
            None => self.embed_tokens.forward(inputs, stream)?,
        };
        profile_array(PerfComponent::Embed, &h)?;
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
        Ok(h)
    }
}

impl Module<ModelInput<'_>> for Qwen35TextModel {
    type Output = Array;
    type Error = Exception;

    fn forward(
        &mut self,
        input: ModelInput<'_>,
        stream: &Stream,
    ) -> Result<Self::Output, Self::Error> {
        let hidden = self.forward_hidden(input, stream)?;
        let hidden = self.norm.forward(&hidden, stream)?;
        profile_array(PerfComponent::FinalNorm, &hidden)?;
        Ok(hidden)
    }

    fn training_mode(&mut self, mode: bool) {
        self.embed_tokens.training_mode(mode);
        for layer in &mut self.layers {
            layer.training_mode(mode);
        }
        self.norm.training_mode(mode);
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

#[derive(Debug, Clone, ModuleParameters)]
/// Qwen3.5 MoE causal language model.
pub struct Model {
    /// Model configuration.
    pub args: ModelArgs,
    /// Optional vision configuration.
    pub vision_args: Option<VisionConfig>,
    /// Optional image token id rejected by text-only generation.
    pub image_token_id: Option<i32>,
    /// Optional video media token id.
    pub video_token_id: Option<i32>,
    #[param]
    /// Optional Qwen vision encoder.
    pub visual: Option<QwenVisionTransformer>,
    #[param]
    /// Text transformer body.
    pub model: Qwen35TextModel,
    #[param]
    /// Embedded multi-token-prediction head when present in the checkpoint.
    pub(crate) mtp: Option<MtpModule>,
    #[param]
    /// Optional untied language-model head.
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    /// Creates an unloaded Qwen3.5 MoE causal language model.
    pub fn new(
        args: ModelArgs,
        image_token_id: Option<i32>,
        video_token_id: Option<i32>,
        vision_args: Option<VisionConfig>,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Self::new_with_affine(
            args,
            image_token_id,
            video_token_id,
            vision_args,
            None,
            stream,
        )
    }

    fn new_with_affine(
        args: ModelArgs,
        image_token_id: Option<i32>,
        video_token_id: Option<i32>,
        vision_args: Option<VisionConfig>,
        affine: Option<WeightQuantization>,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        validate_text_model_args(&args, "Qwen hybrid")
            .map_err(|error| Exception::custom(error.to_string()))?;
        if args.model_type == "qwen3_next" {
            super::qwen3_next::fused_projection_widths(&args)
                .map_err(|error| Exception::custom(error.to_string()))?;
        }
        let format = QwenWeightFormat::for_text(&args, affine);
        let model = Qwen35TextModel::new_with_format(&args, format, stream)?;
        let mtp = (args.mtp_num_hidden_layers > 0)
            .then(|| MtpModule::new_with_format(&args, format, stream))
            .transpose()?;
        let visual = vision_args
            .clone()
            .map(|vision_args| QwenVisionTransformer::new(vision_args, stream))
            .transpose()?;
        let lm_head = if !args.tie_word_embeddings {
            Some(
                common::linear::build_unloaded_maybe_quantized_lm_head_with_quantization(
                    args.hidden_size,
                    args.vocab_size,
                    args.weight_format_for("lm_head.weight", format).affine(),
                    stream,
                )?,
            )
        } else {
            None
        };
        Ok(Self {
            args,
            vision_args,
            image_token_id,
            video_token_id,
            visual,
            model,
            mtp,
            lm_head,
        })
    }

    /// Creates an empty heterogeneous cache for this model.
    pub fn new_cache(&self) -> Cache {
        Cache::new(&self.args).expect("validated Qwen hybrid layer schedule")
    }

    /// Creates resident hybrid state or pages growing full-attention state
    /// under the same policy used by layerwise and tensor-parallel execution.
    pub fn new_cache_with_options(&self, policy: CacheResidencyPolicy) -> Result<Cache, Exception> {
        match policy {
            CacheResidencyPolicy::Device => Ok(self.new_cache()),
            CacheResidencyPolicy::Paged(options) => Cache::new_paged(&self.args, options, None),
        }
    }

    /// Returns the configured model type.
    pub fn model_type(&self) -> &str {
        &self.args.model_type
    }

    pub(crate) fn save_prompt_cache(
        cache: &mut Cache,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        stream: &Stream,
    ) -> Result<PromptCacheManifest, Exception> {
        let end = i64::try_from(prefix_token_ids.len())
            .map_err(|_| Exception::custom("Qwen hybrid prompt length exceeds i64"))?;
        if cache.offset() as i64 != end {
            return Err(Exception::custom(
                "Qwen hybrid cache offset does not match the persisted prefix",
            ));
        }
        let mut blocks = Vec::new();
        let mut state = Vec::new();
        let paged_manager = cache.layers.iter().find_map(|layer| match layer {
            LayerCache::FullAttention(cache) => cache.manager().cloned(),
            LayerCache::LinearAttention(_) => None,
        });
        if paged_manager.is_some() {
            for layer in &mut cache.layers {
                if let LayerCache::FullAttention(cache) = layer {
                    cache.finalize()?;
                }
            }
        }
        for (layer, cache) in cache.layers.iter().enumerate() {
            match cache {
                LayerCache::FullAttention(cache) => {
                    if paged_manager.is_some() {
                        continue;
                    }
                    let (keys, values) = cache.snapshot_arrays(stream)?.ok_or_else(|| {
                        Exception::custom("Qwen full-attention cache state is missing")
                    })?;
                    blocks.push(PromptCacheSnapshotBlock {
                        global_layer: layer,
                        start: 0,
                        end,
                        rank: None,
                        arrays: CacheBlockArrays::KeyValue { keys, values },
                    });
                }
                LayerCache::LinearAttention(cache) => {
                    state.push(PromptCacheStateArray {
                        owner: StateTensorOwner::Layer(layer),
                        role: StateTensorRole::Convolution { slot: 0 },
                        array: cache.conv_state.as_ref().ok_or_else(|| {
                            Exception::custom("Qwen linear convolution state is missing")
                        })?,
                    });
                    state.push(PromptCacheStateArray {
                        owner: StateTensorOwner::Layer(layer),
                        role: StateTensorRole::Recurrent,
                        array: cache.recurrent_state.as_ref().ok_or_else(|| {
                            Exception::custom("Qwen linear recurrent state is missing")
                        })?,
                    });
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
    pub(crate) fn load_prompt_cache(
        args: &ModelArgs,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        stream: &Stream,
    ) -> Result<(Cache, PromptCacheManifest), Exception> {
        let layer_count = args.num_hidden_layers as usize;
        let identity = PromptCacheModelIdentity {
            model_family: "qwen_hybrid".into(),
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
            &identity,
            stream,
        )
    }

    pub(crate) fn load_prompt_cache_with_identity(
        args: &ModelArgs,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        identity: &PromptCacheModelIdentity,
        stream: &Stream,
    ) -> Result<(Cache, PromptCacheManifest), Exception> {
        let (blocks, state, manifest) =
            open_prompt_cache_snapshot(directory, expected, identity, prefix_token_ids, stream)
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
            .map(|state| ((state.owner, state.role), state.array))
            .collect::<BTreeMap<_, _>>();
        let mut cache = Cache::new(args).map_err(|error| Exception::custom(error.to_string()))?;
        let end = i32::try_from(prefix_token_ids.len())
            .map_err(|_| Exception::custom("Qwen hybrid prompt length exceeds i32"))?;
        for (layer, cache) in cache.layers.iter_mut().enumerate() {
            match cache {
                LayerCache::FullAttention(cache) => {
                    let mut layer_blocks = blocks.remove(&layer).ok_or_else(|| {
                        Exception::custom("Qwen full-attention prompt-cache block is missing")
                    })?;
                    layer_blocks.sort_by_key(|block| block.start);
                    let mut expected_start = 0;
                    let mut keys = Vec::with_capacity(layer_blocks.len());
                    let mut values = Vec::with_capacity(layer_blocks.len());
                    for block in layer_blocks {
                        if block.start != expected_start {
                            return Err(Exception::custom(
                                "Qwen full-attention prompt-cache blocks are not contiguous",
                            ));
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
                            _ => {
                                return Err(Exception::custom("Qwen prompt-cache kind mismatch"));
                            }
                        }
                    }
                    if expected_start != i64::from(end) {
                        return Err(Exception::custom(format!(
                            "Qwen full-attention prompt-cache blocks end at {expected_start}, expected {end}"
                        )));
                    }
                    let keys = concatenate_axis(&keys, -2, stream)?;
                    let values = concatenate_axis(&values, -2, stream)?;
                    cache.restore_resident(keys, values, end)?;
                }
                LayerCache::LinearAttention(cache) => {
                    cache.conv_state = Some(
                        state
                            .remove(&(
                                StateTensorOwner::Layer(layer),
                                StateTensorRole::Convolution { slot: 0 },
                            ))
                            .ok_or_else(|| {
                                Exception::custom("Qwen linear convolution state is missing")
                            })?,
                    );
                    cache.recurrent_state = Some(
                        state
                            .remove(&(StateTensorOwner::Layer(layer), StateTensorRole::Recurrent))
                            .ok_or_else(|| {
                                Exception::custom("Qwen linear recurrent state is missing")
                            })?,
                    );
                    cache.offset = end;
                }
            }
        }
        if !blocks.is_empty() || !state.is_empty() {
            return Err(Exception::custom(
                "Qwen hybrid prompt cache has unexpected state",
            ));
        }
        Ok((cache, manifest))
    }

    fn reject_multimodal_tokens(
        &self,
        inputs: &Array,
        allow_visual: bool,
        stream: &Stream,
    ) -> Result<(), Exception> {
        for (name, token_id) in [
            (
                "image",
                (!allow_visual).then_some(self.image_token_id).flatten(),
            ),
            (
                "video",
                (!allow_visual).then_some(self.video_token_id).flatten(),
            ),
        ] {
            if let Some(token_id) = token_id {
                let contains = inputs
                    .eq(Array::from_int(token_id), stream)?
                    .max(None, stream)?
                    .item::<bool>(&stream);
                if contains {
                    return Err(Exception::custom(format!(
                        "qwen3_5_moe text-generation support does not accept {name} tokens"
                    )));
                }
            }
        }
        Ok(())
    }

    fn project_logits(
        &mut self,
        hidden_states: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let logits = project_logits_maybe_quantized(
            &mut self.lm_head,
            &mut self.model.embed_tokens,
            hidden_states,
            stream,
        )?;
        profile_array(PerfComponent::LmHead, &logits)?;
        Ok(logits)
    }

    pub(crate) fn mtp_len(&self) -> usize {
        self.mtp.as_ref().map_or(0, MtpModule::len)
    }

    pub(crate) fn prefill_mtp(
        &mut self,
        input: runtime_input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<QwenMtpStepOutput, Exception> {
        let tokens = runtime_input::text_token_ids(input, stream)?;
        cache.reset()?;
        self.forward_mtp_tokens(&tokens, cache, stream)
    }

    pub(crate) fn verify_mtp(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<QwenMtpStepOutput, Exception> {
        self.forward_mtp_tokens(tokens, cache, stream)
    }

    fn forward_mtp_tokens(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<QwenMtpStepOutput, Exception> {
        self.reject_multimodal_tokens(tokens, false, stream)?;
        let hidden = self.model.forward_hidden(
            ModelInput {
                inputs: tokens,
                inputs_embeds: None,
                mask: None,
                cache: Some(cache),
            },
            stream,
        )?;
        let normalized = self.model.norm.forward(&hidden, stream)?;
        let logits = self.project_logits(&normalized, stream)?;
        Ok(QwenMtpStepOutput { logits, hidden })
    }

    pub(crate) fn forward_mtp_head(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        cache: &mut [LayerCache],
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let embeddings = self.model.embed_tokens.forward(tokens, stream)?;
        let hidden = self
            .mtp
            .as_mut()
            .ok_or_else(|| Exception::custom("Qwen checkpoint does not contain MTP layers"))?
            .forward(hidden, &embeddings, cache, stream)?;
        self.project_logits(&hidden, stream)
    }

    pub(crate) fn forward_logits(
        &mut self,
        input: ModelInput<'_>,
        last_token_only: bool,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.reject_multimodal_tokens(input.inputs, input.inputs_embeds.is_some(), stream)?;
        let hidden_states = self.model.forward(input, stream)?;
        let hidden_states = if last_token_only {
            hidden_states.try_index_device((.., -1, ..), stream)?
        } else {
            hidden_states
        };
        self.project_logits(&hidden_states, stream)
    }

    /// Forward pass that reports activations to an observer.
    pub fn forward_with_observer(
        &mut self,
        input: ModelInput<'_>,
        stream: &Stream,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        self.reject_multimodal_tokens(input.inputs, input.inputs_embeds.is_some(), stream)?;
        let hidden_states = self.model.forward_with_observer(input, stream, observer)?;
        observer.observe("model.output", &hidden_states)?;
        let logits = self.project_logits(&hidden_states, stream)?;
        observer.observe("lm_head.logits", &logits)?;
        Ok(logits)
    }
}

pub(crate) struct QwenMtpStepOutput {
    pub(crate) logits: Array,
    pub(crate) hidden: Array,
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
        if let Some(visual) = &mut self.visual {
            visual.training_mode(mode);
        }
        if let Some(lm_head) = &mut self.lm_head {
            lm_head.training_mode(mode);
        }
    }
}

pub(crate) struct PreparedQwen35Gguf {
    pub(crate) args: ModelArgs,
    pub(crate) eos_token_ids: Vec<u32>,
    pub(crate) architecture: String,
    pub(crate) modalities: Qwen35Modalities,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct Qwen35Modalities {
    pub(crate) image_token_id: Option<i32>,
    pub(crate) video_token_id: Option<i32>,
    pub(crate) vision_config: Option<VisionConfig>,
}

#[derive(Debug)]
pub(crate) struct Qwen35MmprojGguf {
    pub(crate) metadata: HashMap<String, GgufMetadataValue>,
    pub(crate) checkpoint: GgufCheckpoint,
}

/// Opens and validates the optional sibling Qwen3.5 vision projector.
pub(crate) fn open_sibling_mmproj(gguf_file: &Path) -> Result<Option<Qwen35MmprojGguf>, Error> {
    let Some(path) =
        crate::backend::mlx::runtime::checkpoint::gguf::find_sibling_mmproj(gguf_file, "qwen35")?
    else {
        return Ok(None);
    };
    let checkpoint = GgufCheckpoint::open(path)?;
    let metadata = gguf_metadata(&checkpoint);
    qwen_vl::validate_qwen3_vl_mmproj(&metadata)?;
    Ok(Some(Qwen35MmprojGguf {
        metadata,
        checkpoint,
    }))
}

fn qwen35_gguf_multimodal_geometry(
    checkpoint: &GgufCheckpoint,
    args: &ModelArgs,
    architecture: &str,
    metadata: &HashMap<String, GgufMetadataValue>,
    mmproj: Option<&Qwen35MmprojGguf>,
) -> Result<Qwen35Modalities, Error> {
    let Some(mmproj) = mmproj else {
        return Ok(Qwen35Modalities::default());
    };
    if architecture == "qwen3next" {
        return Err(Error::UnsupportedArchitecture(
            "Qwen3-Next GGUF does not define multimodal projector semantics".into(),
        ));
    }
    crate::composition::mlx::structural::validate_qwen35_projector_gguf(
        checkpoint,
        metadata,
        &mmproj.checkpoint,
        &mmproj.metadata,
    )
    .into_loader_result()?;
    let vision = qwen_vl::qwen_vision_config_from_gguf_catalog(
        &mmproj.checkpoint,
        &mmproj.metadata,
        "Qwen3.5",
    )?;
    if vision.out_hidden_size != args.hidden_size {
        return Err(Error::UnsupportedArchitecture(format!(
            "Qwen3.5 GGUF projector output {} does not match language hidden size {}",
            vision.out_hidden_size, args.hidden_size
        )));
    }
    if vision.deepstack_layer_count() != 0 {
        return Err(Error::UnsupportedArchitecture(format!(
            "Qwen3.5 GGUF projector declares {} DeepStack outputs, but the Qwen3.5 decoder accepts only the primary merger output",
            vision.deepstack_layer_count()
        )));
    }
    let image = qwen_vl::gguf_token_id(metadata, "<|image_pad|>")?;
    let video = qwen_vl::gguf_token_id(metadata, "<|video_pad|>")?;
    Ok(Qwen35Modalities {
        image_token_id: Some(i32::try_from(image).map_err(|_| {
            Error::UnsupportedArchitecture("Qwen3.5 image token exceeds i32".into())
        })?),
        video_token_id: Some(i32::try_from(video).map_err(|_| {
            Error::UnsupportedArchitecture("Qwen3.5 video token exceeds i32".into())
        })?),
        vision_config: Some(vision),
    })
}

pub(crate) fn prepare_qwen35_gguf_checkpoint(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    mmproj: Option<&Qwen35MmprojGguf>,
    weights_stream: &Stream,
) -> Result<PreparedQwen35Gguf, Error> {
    let architecture = qwen35_gguf_string(metadata, "general.architecture")?;
    if !matches!(architecture.as_str(), "qwen35" | "qwen35moe" | "qwen3next") {
        return Err(Error::UnsupportedArchitecture(format!(
            "GGUF architecture {architecture:?}; this loader supports qwen35, qwen35moe, and qwen3next"
        )));
    }
    crate::core::GgufArchitecture::resolve(&architecture)?
        .validate_catalog(checkpoint, metadata)?;
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let block_count = qwen35_gguf_i32(metadata, &key("block_count"), weights_stream)?;
    let nextn_layers =
        qwen35_gguf_optional_i64(metadata, &key("nextn_predict_layers"), weights_stream)?
            .unwrap_or(0);
    let nextn_layers = i32::try_from(nextn_layers).map_err(|_| {
        Error::UnsupportedArchitecture(
            "Qwen3.5 next-token prediction layer count exceeds i32".into(),
        )
    })?;
    if nextn_layers < 0 || nextn_layers >= block_count {
        return Err(Error::UnsupportedArchitecture(format!(
            "Qwen3.5 GGUF has invalid block_count {block_count} and nextn_predict_layers {nextn_layers}"
        )));
    }
    let num_hidden_layers = block_count - nextn_layers;
    let mut args = qwen35_args_from_gguf_catalog(checkpoint, metadata, &architecture)?;
    debug_assert_eq!(args.num_hidden_layers, num_hidden_layers);
    let mut configs = gguf_quantization_configs(checkpoint, qwen35_translate_gguf_weight_name)?;
    if matches!(architecture.as_str(), "qwen35moe" | "qwen3next") {
        for layer in 0..num_hidden_layers {
            let prefix = format!("model.layers.{layer}.mlp.experts");
            if let Some(gate) = configs.remove(&format!("{prefix}.gate_proj")) {
                let up = configs
                    .remove(&format!("{prefix}.up_proj"))
                    .ok_or_else(|| {
                        Error::UnsupportedArchitecture(format!(
                        "Qwen3.5 GGUF layer {layer} is missing routed up-projection affine metadata"
                    ))
                    })?;
                if gate != up {
                    return Err(Error::UnsupportedArchitecture(format!(
                        "Qwen3.5 GGUF layer {layer} routed gate/up affine layouts differ"
                    )));
                }
                configs.insert(format!("{prefix}.gate_up_proj"), gate);
            }
        }
    }
    if architecture == "qwen3next" {
        super::qwen3_next::split_fused_projection_configs(&mut configs)?;
    }
    args.quantized_weight_configs = Some(configs);
    let modalities =
        qwen35_gguf_multimodal_geometry(checkpoint, &args, &architecture, metadata, mmproj)?;
    let eos_token_ids = crate::backend::mlx::gguf_eos_token_ids(metadata)?;
    Ok(PreparedQwen35Gguf {
        args,
        eos_token_ids,
        architecture,
        modalities,
    })
}

pub(crate) fn qwen35_args_from_gguf_catalog(
    arrays: &impl GgufTensorNames,
    metadata: &HashMap<String, GgufMetadataValue>,
    architecture: &str,
) -> Result<ModelArgs, Error> {
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let block_count = qwen35_gguf_catalog_i32(metadata, &key("block_count"))?;
    let nextn_layers =
        qwen35_gguf_catalog_optional_i64(metadata, &key("nextn_predict_layers"))?.unwrap_or(0);
    let nextn_layers = i32::try_from(nextn_layers).map_err(|_| {
        Error::UnsupportedArchitecture(
            "Qwen3.5 next-token prediction layer count exceeds i32".into(),
        )
    })?;
    if nextn_layers < 0 || nextn_layers >= block_count {
        return Err(Error::UnsupportedArchitecture(format!(
            "Qwen3.5 GGUF has invalid block_count {block_count} and nextn_predict_layers {nextn_layers}"
        )));
    }
    let args =
        qwen35_args_from_gguf_geometry(arrays, metadata, architecture, block_count - nextn_layers)?;
    validate_text_model_args(&args, "Qwen3.5 GGUF")?;
    if architecture == "qwen3next" {
        super::qwen3_next::fused_projection_widths(&args)?;
    }
    Ok(args)
}

fn qwen35_args_from_gguf_geometry(
    arrays: &impl GgufTensorNames,
    metadata: &HashMap<String, GgufMetadataValue>,
    architecture: &str,
    num_hidden_layers: i32,
) -> Result<ModelArgs, Error> {
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let is_moe = matches!(architecture, "qwen35moe" | "qwen3next");
    let hidden_size = qwen35_gguf_catalog_i32(metadata, &key("embedding_length"))?;
    let num_attention_heads = qwen35_gguf_catalog_i32(metadata, &key("attention.head_count"))?;
    let num_key_value_heads = qwen35_gguf_catalog_i32(metadata, &key("attention.head_count_kv"))?;
    let head_dim = qwen35_gguf_catalog_optional_i64(metadata, &key("attention.key_length"))?
        .map(i32::try_from)
        .transpose()
        .map_err(|_| Error::UnsupportedArchitecture("Qwen3.5 head size exceeds i32".into()))?
        .unwrap_or(hidden_size / num_attention_heads);
    let rope_dims = qwen35_gguf_catalog_optional_i64(metadata, &key("rope.dimension_count"))?
        .map(i32::try_from)
        .transpose()
        .map_err(|_| Error::UnsupportedArchitecture("Qwen3.5 RoPE dimension exceeds i32".into()))?
        .unwrap_or(head_dim / 4);
    let full_attention_interval =
        qwen35_gguf_catalog_optional_i64(metadata, &key("full_attention_interval"))?.unwrap_or(4);
    let full_attention_interval = usize::try_from(full_attention_interval).map_err(|_| {
        Error::UnsupportedArchitecture("Qwen3.5 full attention interval must be positive".into())
    })?;
    if full_attention_interval == 0 {
        return Err(Error::UnsupportedArchitecture(
            "Qwen3.5 full attention interval must be positive".into(),
        ));
    }
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
        None => qwen35_gguf_catalog_i32(metadata, &key("vocab_size"))?,
    };
    let rope_theta =
        qwen35_gguf_catalog_optional_f32(metadata, &key("rope.freq_base"))?.unwrap_or(10_000_000.0);
    let mut rope_parameters = HashMap::new();
    rope_parameters.insert("rope_theta".into(), serde_json::json!(rope_theta));
    rope_parameters.insert(
        "partial_rotary_factor".into(),
        serde_json::json!(rope_dims as f32 / head_dim as f32),
    );
    let layer_schedule = LayerSchedule::new(
        num_hidden_layers as usize,
        (0..num_hidden_layers as usize)
            .map(|index| {
                if (index + 1) % full_attention_interval == 0 {
                    LayerPolicy::SelfAttention(AttentionPolicy::Full)
                } else {
                    LayerPolicy::LinearAttention
                }
            })
            .collect(),
    )
    .map_err(|error| Error::UnsupportedArchitecture(format!("Qwen hybrid {error}")))?;

    Ok(ModelArgs {
        model_type: if architecture == "qwen3next" {
            "qwen3_next".into()
        } else if is_moe {
            "qwen3_5_moe_text".into()
        } else {
            "qwen3_5_text".into()
        },
        vocab_size,
        hidden_size,
        num_hidden_layers,
        mtp_num_hidden_layers: 0,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        max_position_embeddings: qwen35_gguf_catalog_i32(metadata, &key("context_length"))?,
        rms_norm_eps: qwen35_gguf_catalog_f32(metadata, &key("attention.layer_norm_rms_epsilon"))?,
        tie_word_embeddings: !arrays.contains_gguf_tensor("output.weight"),
        attention_bias: arrays.any_gguf_tensor(|name| {
            name.ends_with("attn_q.bias")
                || name.ends_with("attn_k.bias")
                || name.ends_with("attn_v.bias")
                || name.ends_with("attn_output.bias")
        }),
        hidden_act: "silu".into(),
        linear_conv_kernel_dim: qwen35_gguf_catalog_i32(metadata, &key("ssm.conv_kernel"))?,
        linear_key_head_dim: qwen35_gguf_catalog_i32(metadata, &key("ssm.state_size"))?,
        linear_value_head_dim: qwen35_gguf_catalog_i32(metadata, &key("ssm.state_size"))?,
        linear_num_key_heads: qwen35_gguf_catalog_i32(metadata, &key("ssm.group_count"))?,
        linear_num_value_heads: qwen35_gguf_catalog_i32(metadata, &key("ssm.time_step_rank"))?,
        intermediate_size: if is_moe {
            0
        } else {
            qwen35_gguf_catalog_i32(metadata, &key("feed_forward_length"))?
        },
        moe_intermediate_size: if is_moe {
            qwen35_gguf_catalog_i32(metadata, &key("expert_feed_forward_length"))?
        } else {
            0
        },
        shared_expert_intermediate_size: if is_moe {
            qwen35_gguf_catalog_i32(metadata, &key("expert_shared_feed_forward_length"))?
        } else {
            0
        },
        num_experts_per_tok: if is_moe {
            qwen35_gguf_catalog_i32(metadata, &key("expert_used_count"))?
        } else {
            0
        },
        num_experts: if is_moe {
            qwen35_gguf_catalog_i32(metadata, &key("expert_count"))?
        } else {
            0
        },
        norm_topk_prob: true,
        layer_schedule,
        rope_parameters: Some(rope_parameters),
        rope_scaling: None,
        quantization_config: None,
        quantization: None,
        quantized_weight_configs: None,
    })
}

#[cfg(test)]
fn qwen35_gguf_block_index(name: &str) -> Option<i32> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

#[cfg(test)]
fn qwen35_translate_gguf_weight(
    name: String,
    mut value: Array,
    affine: Option<AffineQuantization>,
    args: &ModelArgs,
    stream: &Stream,
) -> Result<(String, Array), Error> {
    let affine_component = affine
        .map(|affine| qwen35_affine_component(&name, &value, affine))
        .transpose()?;
    let source_weight_name = match affine_component {
        Some(Qwen35AffineComponent::Weight) | None => name.clone(),
        Some(Qwen35AffineComponent::Scales) => name
            .strip_suffix(".scales")
            .map(|prefix| format!("{prefix}.weight"))
            .unwrap_or_else(|| name.clone()),
        Some(Qwen35AffineComponent::Biases) => name
            .strip_suffix(".biases")
            .map(|prefix| format!("{prefix}.weight"))
            .unwrap_or_else(|| name.clone()),
    };
    // llama.cpp only converts Qwen3.5 value heads from grouped-by-key-head
    // order to tiled broadcast order. Qwen3-Next GGUF retains the original
    // grouped layout, so applying the inverse permutation there corrupts every
    // recurrent projection that contains value-head channels.
    let restore_v_head_order = args.model_type != "qwen3_next";
    if affine.is_some()
        && (source_weight_name.ends_with(".ssm_conv1d.weight")
            || source_weight_name.ends_with(".ssm_a.weight"))
    {
        return Err(Error::UnsupportedArchitecture(format!(
            "Qwen3.5 GGUF affine tensor {source_weight_name:?} requires a non-linear or convolutional transform and cannot be preserved as an MLX affine projection"
        )));
    }
    if restore_v_head_order && source_weight_name.ends_with(".attn_qkv.weight") {
        value = qwen35_restore_v_tail(
            value,
            2 * args.linear_num_key_heads * args.linear_key_head_dim,
            0,
            args,
            stream,
        )?;
    } else if source_weight_name.ends_with(".ssm_conv1d.weight") {
        if restore_v_head_order {
            value = qwen35_restore_v_tail(
                value,
                2 * args.linear_num_key_heads * args.linear_key_head_dim,
                0,
                args,
                stream,
            )?;
        }
        value = value.reshape(&[value.dim(0), 1, value.dim(1)], stream)?;
    } else if restore_v_head_order && source_weight_name.ends_with(".attn_gate.weight") {
        value = qwen35_restore_v_head_order(value, 0, args.linear_value_head_dim, args, stream)?;
    } else if restore_v_head_order
        && (source_weight_name.ends_with(".ssm_alpha.weight")
            || source_weight_name.ends_with(".ssm_beta.weight")
            || source_weight_name.ends_with(".ssm_dt.bias"))
    {
        value = qwen35_restore_v_head_order(value, 0, 1, args, stream)?;
    } else if source_weight_name.ends_with(".ssm_a") {
        if restore_v_head_order {
            value = qwen35_restore_v_head_order(value, 0, 1, args, stream)?;
        }
        value = value.multiply(Array::from_f32(-1.0), stream)?.log(stream)?;
    } else if restore_v_head_order && source_weight_name.ends_with(".ssm_out.weight") {
        let head_width = match (affine, affine_component) {
            (Some(affine), Some(Qwen35AffineComponent::Weight)) => {
                qwen35_affine_packed_head_width(args.linear_value_head_dim, affine)?
            }
            (Some(affine), Some(Qwen35AffineComponent::Scales | Qwen35AffineComponent::Biases)) => {
                qwen35_affine_scale_head_width(args.linear_value_head_dim, affine)?
            }
            _ => args.linear_value_head_dim,
        };
        value = qwen35_restore_v_head_order(value, 1, head_width, args, stream)?;
    } else if source_weight_name.ends_with(".ffn_gate_inp_shexp.weight") && value.ndim() == 1 {
        value = value.reshape(&[1, value.dim(0)], stream)?;
    }

    if qwen35_is_offset_norm(&source_weight_name) {
        if affine.is_some() {
            return Err(Error::UnsupportedArchitecture(format!(
                "Qwen3.5 GGUF affine normalization tensor {source_weight_name:?} cannot apply the required offset transform while remaining packed"
            )));
        }
        value = value.subtract(Array::from_f32(1.0), stream)?;
    }
    Ok((qwen35_translate_gguf_weight_name(&name), value))
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[cfg(test)]
enum Qwen35AffineComponent {
    Weight,
    Scales,
    Biases,
}

#[cfg(test)]
fn qwen35_affine_component(
    name: &str,
    value: &Array,
    affine: AffineQuantization,
) -> Result<Qwen35AffineComponent, Error> {
    let (component, expected_dtype) = if name.ends_with(".weight") {
        (Qwen35AffineComponent::Weight, Dtype::Uint32)
    } else if name.ends_with(".scales") {
        (Qwen35AffineComponent::Scales, Dtype::Float16)
    } else if name.ends_with(".biases") {
        (Qwen35AffineComponent::Biases, Dtype::Float16)
    } else {
        return Err(Error::UnsupportedArchitecture(format!(
            "Qwen3.5 GGUF affine component {name:?} has an unsupported name"
        )));
    };
    if value.dtype() != expected_dtype {
        return Err(Error::UnsupportedArchitecture(format!(
            "Qwen3.5 GGUF affine component {name:?} has dtype {:?}; expected {expected_dtype:?} for {}-bit groups of {}",
            value.dtype(),
            affine.bits,
            affine.group_size
        )));
    }
    Ok(component)
}

#[cfg(test)]
fn qwen35_affine_packed_head_width(
    head_dim: i32,
    affine: AffineQuantization,
) -> Result<i32, Error> {
    let packed_bits = head_dim
        .checked_mul(affine.bits)
        .ok_or_else(|| Error::UnsupportedArchitecture("affine head width overflow".into()))?;
    if head_dim <= 0 || packed_bits % 32 != 0 {
        return Err(Error::UnsupportedArchitecture(format!(
            "Qwen3.5 affine value-head width {head_dim} at {} bits is not aligned to Uint32 packing",
            affine.bits
        )));
    }
    Ok(packed_bits / 32)
}

#[cfg(test)]
fn qwen35_affine_scale_head_width(head_dim: i32, affine: AffineQuantization) -> Result<i32, Error> {
    if head_dim <= 0 || head_dim % affine.group_size != 0 {
        return Err(Error::UnsupportedArchitecture(format!(
            "Qwen3.5 affine value-head width {head_dim} is not divisible by group size {}",
            affine.group_size
        )));
    }
    Ok(head_dim / affine.group_size)
}

#[cfg(test)]
fn qwen35_restore_v_tail(
    value: Array,
    prefix: i32,
    axis: usize,
    args: &ModelArgs,
    stream: &Stream,
) -> Result<Array, Error> {
    if axis != 0 || value.ndim() != 2 {
        return Err(Error::UnsupportedArchitecture(
            "Qwen3.5 GGUF value-tail restoration expects a rank-2 row projection".into(),
        ));
    }
    if prefix <= 0 || prefix >= value.dim(0) {
        return Err(Error::UnsupportedArchitecture(format!(
            "Qwen3.5 GGUF value-tail split {prefix} is invalid for shape {:?}",
            value.shape()
        )));
    }
    let leading = value.try_index_device((..prefix, ..), stream)?;
    let tail = value.try_index_device((prefix.., ..), stream)?;
    let tail = qwen35_restore_v_head_order(tail, 0, args.linear_value_head_dim, args, stream)?;
    Ok(concatenate_axis(&[leading, tail], 0, stream)?)
}

#[cfg(test)]
fn qwen35_restore_v_head_order(
    value: Array,
    axis: usize,
    head_dim: i32,
    args: &ModelArgs,
    stream: &Stream,
) -> Result<Array, Error> {
    let num_k = args.linear_num_key_heads;
    let num_v = args.linear_num_value_heads;
    if num_k <= 0 || num_v % num_k != 0 || axis >= value.ndim() {
        return Err(Error::UnsupportedArchitecture(format!(
            "invalid Qwen3.5 GGUF value-head layout: {num_v} value heads, {num_k} key heads"
        )));
    }
    let original_shape = value.shape().to_vec();
    if original_shape[axis] != num_v * head_dim {
        return Err(Error::UnsupportedArchitecture(format!(
            "Qwen3.5 GGUF value-head axis has length {}, expected {} in shape {:?}",
            original_shape[axis],
            num_v * head_dim,
            original_shape
        )));
    }
    let repeats = num_v / num_k;
    let mut expanded_shape = original_shape.clone();
    expanded_shape.splice(axis..=axis, [repeats, num_k, head_dim]);
    let expanded = value.reshape(&expanded_shape, stream)?;
    let mut axes = (0..expanded_shape.len() as i32).collect::<Vec<_>>();
    axes.swap(axis, axis + 1);
    Ok(expanded
        .transpose_axes(&axes, stream)?
        .reshape(&original_shape, stream)?)
}

#[cfg(test)]
fn qwen35_is_offset_norm(name: &str) -> bool {
    name == "output_norm.weight"
        || (name.starts_with("blk.")
            && name.ends_with("_norm.weight")
            && !name.ends_with("ssm_norm.weight"))
}

pub(crate) fn qwen35_translate_gguf_weight_name(name: &str) -> String {
    const ROOTS: [(&str, &str); 3] = [
        ("token_embd", "model.embed_tokens"),
        ("output_norm", "model.norm"),
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
    const PARAMETERS: [(&str, &str); 31] = [
        ("attn_norm", "input_layernorm"),
        ("post_attention_norm", "post_attention_layernorm"),
        ("attn_q_norm", "self_attn.q_norm"),
        ("attn_k_norm", "self_attn.k_norm"),
        ("attn_q", "self_attn.q_proj"),
        ("attn_k", "self_attn.k_proj"),
        ("attn_v", "self_attn.v_proj"),
        ("attn_output", "self_attn.o_proj"),
        ("attn_qkvz", "linear_attn.in_proj_qkvz"),
        ("attn_qkv", "linear_attn.in_proj_qkv"),
        ("attn_gate", "linear_attn.in_proj_z"),
        ("ssm_beta", "linear_attn.in_proj_b"),
        ("ssm_alpha", "linear_attn.in_proj_a"),
        ("ssm_ba", "linear_attn.in_proj_ba"),
        ("ssm_conv1d", "linear_attn.conv1d"),
        ("ssm_dt.bias", "linear_attn.dt_bias"),
        ("ssm_a", "linear_attn.A_log"),
        ("ssm_norm", "linear_attn.norm"),
        ("ssm_out", "linear_attn.out_proj"),
        ("ffn_gate_inp_shexp", "mlp.shared_expert_gate"),
        ("ffn_gate_inp", "mlp.gate"),
        ("ffn_gate_shexp", "mlp.shared_expert.gate_proj"),
        ("ffn_up_shexp", "mlp.shared_expert.up_proj"),
        ("ffn_down_shexp", "mlp.shared_expert.down_proj"),
        ("ffn_gate_exps", "mlp.experts.gate_proj"),
        ("ffn_up_exps", "mlp.experts.up_proj"),
        ("ffn_down_exps", "mlp.experts.down_proj"),
        ("ffn_gate", "mlp.gate_proj"),
        ("ffn_up", "mlp.up_proj"),
        ("ffn_down", "mlp.down_proj"),
        ("rope_freqs", "rope_freqs"),
    ];
    for (source, target) in PARAMETERS {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            let mut suffix = parameter.strip_prefix(source).unwrap_or_default();
            if target.starts_with("mlp.experts.") {
                suffix = match suffix {
                    ".weight" => "",
                    ".scales" => "_scales",
                    ".biases" => "_biases",
                    other => other,
                };
            }
            return format!("model.layers.{layer}.{target}{suffix}");
        }
    }
    name.to_string()
}

#[cfg(test)]
fn qwen35_gguf_affine_quantization(
    weight_shape: &[i32],
    scales_shape: &[i32],
    weight_name: &str,
) -> Result<AffineQuantization, Error> {
    crate::backend::mlx::runtime::checkpoint::quantization::gguf_affine_quantization(
        weight_shape,
        scales_shape,
        weight_name,
    )
}

fn qwen35_gguf_string(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<String, Error> {
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

fn qwen35_gguf_i32(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
    _stream: &Stream,
) -> Result<i32, Error> {
    qwen35_gguf_catalog_i32(metadata, key)
}

fn qwen35_gguf_catalog_i32(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<i32, Error> {
    i32::try_from(qwen35_gguf_catalog_i64(metadata, key)?).map_err(|_| {
        Error::UnsupportedArchitecture(format!("GGUF metadata value {key:?} exceeds i32"))
    })
}

fn qwen35_gguf_catalog_i64(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<i64, Error> {
    qwen35_gguf_catalog_optional_i64(metadata, key)?.ok_or_else(|| {
        Error::UnsupportedArchitecture(format!("GGUF metadata is missing required key {key:?}"))
    })
}

fn qwen35_gguf_optional_i64(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
    _stream: &Stream,
) -> Result<Option<i64>, Error> {
    qwen35_gguf_catalog_optional_i64(metadata, key)
}

fn qwen35_gguf_catalog_optional_i64(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<Option<i64>, Error> {
    match metadata.get(key) {
        Some(value) => value.as_i64().map(Some).ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "GGUF metadata key {key:?} must be an integer scalar"
            ))
        }),
        None => Ok(None),
    }
}

fn qwen35_gguf_catalog_f32(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<f32, Error> {
    qwen35_gguf_catalog_optional_f32(metadata, key)?.ok_or_else(|| {
        Error::UnsupportedArchitecture(format!("GGUF metadata is missing required key {key:?}"))
    })
}

fn qwen35_gguf_catalog_optional_f32(
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

/// Loads `tokenizer.json` from a Qwen3.5 model directory.
pub fn load_qwen3_5_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let file = model_dir.as_ref().join("tokenizer.json");
    Tokenizer::from_file(file).map_err(Into::into)
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum Qwen35Variant {
    Dense,
    Moe,
    Qwen3Next,
}

pub(crate) type ParsedQwen35Config = (ModelArgs, Option<i32>, Option<i32>, Option<VisionConfig>);

impl Qwen35Variant {
    fn text_model_type(self) -> &'static str {
        match self {
            Self::Dense => "qwen3_5_text",
            Self::Moe => "qwen3_5_moe_text",
            Self::Qwen3Next => "qwen3_next",
        }
    }
}

pub(crate) fn parse_qwen3_5_config_value(value: Value) -> Result<ParsedQwen35Config, Error> {
    let mut config: TopLevelConfig = serde_json::from_value(value.clone()).map_err(|error| {
        Error::UnsupportedArchitecture(format!("invalid Qwen3.5 config: {error}"))
    })?;
    let model_type = config.model_type.clone();
    let text_value = if matches!(model_type.as_str(), "qwen3_5" | "qwen3_5_moe") {
        value.get("text_config").unwrap_or(&value)
    } else {
        &value
    };
    let (source, variant) = match model_type.as_str() {
        "qwen3_5" | "qwen3_5_moe" => {
            let variant = if model_type == "qwen3_5" {
                Qwen35Variant::Dense
            } else {
                Qwen35Variant::Moe
            };
            let text_config = config.text_config.take().ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "{model_type} config is missing text_config"
                ))
            })?;
            (text_config, variant)
        }
        "qwen3_5_text" | "qwen3_5_moe_text" => {
            let variant = if model_type == "qwen3_5_text" {
                Qwen35Variant::Dense
            } else {
                Qwen35Variant::Moe
            };
            let source = serde_json::from_value(value.clone()).map_err(|error| {
                Error::UnsupportedArchitecture(format!("invalid {model_type} config: {error}"))
            })?;
            (source, variant)
        }
        "qwen3_next" => {
            let source = serde_json::from_value(value.clone()).map_err(|error| {
                Error::UnsupportedArchitecture(format!("invalid qwen3_next config: {error}"))
            })?;
            (source, Qwen35Variant::Qwen3Next)
        }
        other => return Err(Error::UnsupportedModelType(other.to_string())),
    };

    let layer_count = usize::try_from(source.num_hidden_layers).map_err(|_| {
        Error::UnsupportedArchitecture(format!(
            "{} num_hidden_layers must be positive, got {}",
            variant.text_model_type(),
            source.num_hidden_layers
        ))
    })?;
    let layer_policies = if source.layer_types.is_empty() {
        let interval = match text_value.get("full_attention_interval") {
            Some(Value::Number(number)) => number
                .as_u64()
                .ok_or_else(|| {
                    Error::UnsupportedArchitecture(
                        "Qwen hybrid full_attention_interval must be a positive integer".into(),
                    )
                })?
                .try_into()
                .map_err(|_| {
                    Error::UnsupportedArchitecture(
                        "Qwen hybrid full_attention_interval exceeds usize".into(),
                    )
                })?,
            Some(_) => {
                return Err(Error::UnsupportedArchitecture(
                    "Qwen hybrid full_attention_interval must be an integer".into(),
                ))
            }
            None => 4,
        };
        if interval == 0 {
            return Err(Error::UnsupportedArchitecture(
                "Qwen hybrid full_attention_interval must be positive".into(),
            ));
        }
        (0..layer_count)
            .map(|index| {
                if (index + 1).is_multiple_of(interval) {
                    LayerPolicy::SelfAttention(AttentionPolicy::Full)
                } else {
                    LayerPolicy::LinearAttention
                }
            })
            .collect()
    } else {
        source
            .layer_types
            .iter()
            .copied()
            .map(LayerPolicySource::normalize)
            .collect()
    };
    let layer_schedule = LayerSchedule::new(layer_count, layer_policies)
        .map_err(|error| Error::UnsupportedArchitecture(format!("Qwen hybrid {error}")))?;
    let mut args = source.normalize(layer_schedule);

    args.model_type = variant.text_model_type().to_string();
    if variant == Qwen35Variant::Dense {
        // Dense Qwen3.5 configs omit MoE-only fields. ModelArgs historically
        // supplied MoE defaults for those fields, so normalize them explicitly.
        args.num_experts = 0;
        args.num_experts_per_tok = 0;
    }
    args.quantization_config = config
        .quantization_config
        .or_else(|| args.quantization_config.clone());
    args.quantization = config.quantization.or(args.quantization);
    if let Some(tie_word_embeddings) = config.tie_word_embeddings {
        args.tie_word_embeddings = tie_word_embeddings;
    }
    if variant == Qwen35Variant::Qwen3Next && args.rope_parameters.is_none() {
        let mut rope_parameters = HashMap::new();
        if let Some(rope_theta) = value.get("rope_theta").cloned() {
            rope_parameters.insert("rope_theta".into(), rope_theta);
        }
        if let Some(partial_rotary_factor) = value.get("partial_rotary_factor").cloned() {
            rope_parameters.insert("partial_rotary_factor".into(), partial_rotary_factor);
        }
        if !rope_parameters.is_empty() {
            args.rope_parameters = Some(rope_parameters);
        }
    }
    let vision_config = config
        .vision_config
        .map(VisionConfigSource::normalize_qwen3_5)
        .transpose()?;
    Ok((
        args,
        config.image_token_id,
        config.video_token_id,
        vision_config,
    ))
}

/// Reads and normalizes dense or MoE Qwen3.5 model arguments from `config.json`.
pub fn get_qwen3_5_model_args(model_dir: impl AsRef<Path>) -> Result<ParsedQwen35Config, Error> {
    let file = std::fs::File::open(model_dir.as_ref().join("config.json"))?;
    let value = serde_json::from_reader(file)?;
    model_config_from_value(&value)
}

pub(crate) fn model_config_from_value(config: &Value) -> Result<ParsedQwen35Config, Error> {
    let parsed = parse_qwen3_5_config_value(config.clone())?;
    validate_text_model_args(&parsed.0, "Qwen3.5")?;
    Ok(parsed)
}

/// Normalizes a Qwen3.5 or Qwen3-Next JSON configuration into executable text geometry.
pub fn model_args_from_config_value(config: &Value) -> Result<ModelArgs, Error> {
    let args = model_config_from_value(config)?.0;
    if args.model_type == "qwen3_next" {
        super::qwen3_next::fused_projection_widths(&args)?;
    }
    Ok(args)
}

pub(crate) fn validate_text_model_args(args: &ModelArgs, architecture: &str) -> Result<(), Error> {
    validate_rope_scaling_config(&args.rope_config())?;
    for (name, value) in [
        ("vocab_size", args.vocab_size),
        ("hidden_size", args.hidden_size),
        ("num_hidden_layers", args.num_hidden_layers),
        ("num_attention_heads", args.num_attention_heads),
        ("num_key_value_heads", args.num_key_value_heads),
        ("head_dim", args.head_dim),
        ("max_position_embeddings", args.max_position_embeddings),
        ("linear_conv_kernel_dim", args.linear_conv_kernel_dim),
        ("linear_key_head_dim", args.linear_key_head_dim),
        ("linear_value_head_dim", args.linear_value_head_dim),
        ("linear_num_key_heads", args.linear_num_key_heads),
        ("linear_num_value_heads", args.linear_num_value_heads),
    ] {
        if value <= 0 {
            return Err(Error::UnsupportedArchitecture(format!(
                "{architecture} {name} must be positive, got {value}"
            )));
        }
    }
    if args.mtp_num_hidden_layers < 0 {
        return Err(Error::UnsupportedArchitecture(format!(
            "{architecture} mtp_num_hidden_layers must be non-negative"
        )));
    }
    if args.layer_schedule.len() != args.num_hidden_layers as usize {
        return Err(Error::UnsupportedArchitecture(format!(
            "{architecture} layer schedule has {} entries for {} decoder layers",
            args.layer_schedule.len(),
            args.num_hidden_layers
        )));
    }
    if args.layer_schedule.iter().any(|policy| {
        matches!(
            policy,
            LayerPolicy::SelfAttention(AttentionPolicy::Sliding { .. })
        )
    }) {
        return Err(Error::UnsupportedArchitecture(format!(
            "{architecture} does not support sliding self-attention"
        )));
    }
    if args.hidden_size.checked_mul(2).is_none()
        || args
            .num_attention_heads
            .checked_mul(args.head_dim)
            .and_then(|value| value.checked_mul(2))
            .is_none()
        || args
            .num_key_value_heads
            .checked_mul(args.head_dim)
            .is_none()
        || args
            .linear_num_key_heads
            .checked_mul(args.linear_key_head_dim)
            .and_then(|key| {
                args.linear_num_value_heads
                    .checked_mul(args.linear_value_head_dim)
                    .and_then(|value| key.checked_mul(2)?.checked_add(value))
            })
            .is_none()
    {
        return Err(Error::UnsupportedArchitecture(format!(
            "{architecture} projection geometry exceeds i32"
        )));
    }
    if args.linear_num_value_heads % args.linear_num_key_heads != 0 {
        return Err(Error::UnsupportedArchitecture(format!(
            "{architecture} linear value-head count must be divisible by the key-head count"
        )));
    }
    if args.hidden_act != "silu" {
        return Err(Error::UnsupportedArchitecture(format!(
            "unsupported {architecture} activation {:?}",
            args.hidden_act
        )));
    }
    if args.is_moe() {
        if args.moe_intermediate_size <= 0
            || args.shared_expert_intermediate_size <= 0
            || args.num_experts_per_tok <= 0
            || args.num_experts_per_tok > args.num_experts
        {
            return Err(Error::UnsupportedArchitecture(format!(
                "{architecture} MoE requires positive expert widths and top-k no greater than the expert count"
            )));
        }
    } else if args.intermediate_size <= 0 {
        return Err(Error::UnsupportedArchitecture(format!(
            "dense {architecture} intermediate_size must be positive"
        )));
    }
    if let Some(quantization_config) = &args.quantization_config {
        quantization_config.validate_supported()?;
    }
    if let Some(quantization) = args.quantization {
        quantization.validate()?;
    }
    Ok(())
}

pub(crate) fn validate_model_config_value(config: &Value) -> Result<(), Error> {
    model_config_from_value(config).map(|_| ())
}

#[cfg(test)]
fn quantize_packed_expert_tensor(
    value: &Array,
    quantization: WeightQuantization,
    stream: &Stream,
) -> Result<crate::backend::mlx::runtime::checkpoint::quantization::QuantizedTensor, Error> {
    common::moe::quantize_expert_bank(value, quantization, stream)
}

#[derive(Default)]
#[cfg(test)]
struct Fp8ExpertParts {
    gate: Option<Array>,
    gate_scale: Option<Array>,
    up: Option<Array>,
    up_scale: Option<Array>,
    down: Option<Array>,
    down_scale: Option<Array>,
}

#[cfg(test)]
impl Fp8ExpertParts {
    fn is_complete(&self) -> bool {
        self.gate.is_some()
            && self.gate_scale.is_some()
            && self.up.is_some()
            && self.up_scale.is_some()
            && self.down.is_some()
            && self.down_scale.is_some()
    }
}

/// Strict-loads a sharded Qwen FP8 directory while preserving split expert
/// weights and their inverse-scale companions in packed expert-major banks.
///
/// The transform is applied to every non-packed checkpoint tensor before
/// expert detection, which lets architecture adapters split fused FP8 weights
/// and block-scale tensors without dequantizing them. Expert state spans shard
/// boundaries, and a complete layer is packed immediately to bound residency.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn load_qwen_fp8_safetensors_dir_strict_with_transform<F>(
    model: &mut Model,
    model_dir: impl AsRef<Path>,
    weights_stream: &Stream,
    transform_stream: &Stream,
    config: &StrictLoadConfig,
    report: &mut StrictLoadReport,
    num_experts: i32,
    transform: F,
) -> Result<(), Error>
where
    F: Fn(String, Array) -> Result<Vec<(String, Array)>, Error>,
{
    let mut expert_parts: HashMap<(String, i32), Fp8ExpertParts> = HashMap::new();
    let mut complete_experts = HashMap::<String, i32>::new();
    let mut params = model.parameters_mut().flatten();

    for path in safetensors_files(model_dir)? {
        for_each_safetensor_array(path, weights_stream, |key, value| {
            for (key, value) in transform(key, value)? {
                let expert_part = parse_fp8_expert_projection_key(&key)
                    .map(|(prefix, expert, projection)| (prefix, expert, projection, false))
                    .or_else(|| {
                        parse_fp8_expert_scale_key(&key)
                            .map(|(prefix, expert, projection)| (prefix, expert, projection, true))
                    });
                if let Some((prefix, expert, projection, is_scale)) = expert_part {
                    let parts = expert_parts.entry((prefix.clone(), expert)).or_default();
                    let was_complete = parts.is_complete();
                    set_fp8_expert_part(parts, projection, value, is_scale);
                    if !was_complete && parts.is_complete() {
                        let completed = complete_experts.entry(prefix.clone()).or_default();
                        *completed += 1;
                        if *completed == num_experts {
                            for (key, value) in pack_fp8_expert_prefix(
                                &mut expert_parts,
                                &prefix,
                                num_experts,
                                transform_stream,
                            )? {
                                load_array_strict(&mut params, key, value, config, report);
                            }
                            complete_experts.remove(&prefix);
                        }
                    }
                } else {
                    load_array_strict(&mut params, key, value, config, report);
                }
            }
            Ok(())
        })?;
    }

    if let Some((prefix, _)) = expert_parts.keys().next().cloned() {
        pack_fp8_expert_prefix(&mut expert_parts, &prefix, num_experts, transform_stream)?;
    }

    Ok(())
}

#[cfg(test)]
pub(crate) fn transform_split_qwen_fp8_experts(
    loaded: HashMap<String, Array>,
    num_experts: i32,
    stream: &Stream,
) -> Result<HashMap<String, Array>, Error> {
    let mut transformed = HashMap::with_capacity(loaded.len());
    let mut expert_parts = HashMap::<(String, i32), Fp8ExpertParts>::new();
    for (key, value) in loaded {
        let expert_part = parse_fp8_expert_projection_key(&key)
            .map(|(prefix, expert, projection)| (prefix, expert, projection, false))
            .or_else(|| {
                parse_fp8_expert_scale_key(&key)
                    .map(|(prefix, expert, projection)| (prefix, expert, projection, true))
            });
        if let Some((prefix, expert, projection, is_scale)) = expert_part {
            set_fp8_expert_part(
                expert_parts.entry((prefix, expert)).or_default(),
                projection,
                value,
                is_scale,
            );
        } else {
            transformed.insert(key, value);
        }
    }
    let mut prefixes = expert_parts
        .keys()
        .map(|(prefix, _)| prefix.clone())
        .collect::<Vec<_>>();
    prefixes.sort();
    prefixes.dedup();
    for prefix in prefixes {
        transformed.extend(pack_fp8_expert_prefix(
            &mut expert_parts,
            &prefix,
            num_experts,
            stream,
        )?);
    }
    Ok(transformed)
}

#[cfg(test)]
fn set_fp8_expert_part(
    parts: &mut Fp8ExpertParts,
    projection: Fp8ExpertProjection,
    value: Array,
    is_scale: bool,
) {
    match (projection, is_scale) {
        (Fp8ExpertProjection::Gate, false) => parts.gate = Some(value),
        (Fp8ExpertProjection::Gate, true) => parts.gate_scale = Some(value),
        (Fp8ExpertProjection::Up, false) => parts.up = Some(value),
        (Fp8ExpertProjection::Up, true) => parts.up_scale = Some(value),
        (Fp8ExpertProjection::Down, false) => parts.down = Some(value),
        (Fp8ExpertProjection::Down, true) => parts.down_scale = Some(value),
    }
}

#[cfg(test)]
fn pack_fp8_expert_prefix(
    expert_parts: &mut HashMap<(String, i32), Fp8ExpertParts>,
    prefix: &str,
    num_experts: i32,
    stream: &Stream,
) -> Result<HashMap<String, Array>, Error> {
    if num_experts <= 0 {
        return Err(Error::UnsupportedArchitecture(format!(
            "Qwen FP8 expert prefix '{prefix}' has invalid expert count {num_experts}"
        )));
    }
    let mut gate_up_parts = Vec::with_capacity(2 * num_experts as usize);
    let mut gate_up_scale_parts = Vec::with_capacity(2 * num_experts as usize);
    let mut down = Vec::with_capacity(num_experts as usize);
    let mut down_scale = Vec::with_capacity(num_experts as usize);
    let mut gate_shape = None;
    let mut gate_scale_shape = None;
    let mut up_shape = None;
    let mut up_scale_shape = None;
    let mut down_shape = None;
    let mut down_scale_shape = None;
    for expert in 0..num_experts {
        let parts = expert_parts
            .remove(&(prefix.to_string(), expert))
            .ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "Qwen3.5-MoE FP8 checkpoint is missing expert {expert} for '{prefix}'"
                ))
            })?;
        let gate = parts.gate.ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "Qwen3.5-MoE FP8 checkpoint is missing {prefix}.{expert}.gate_proj.weight"
            ))
        })?;
        let gate_scale = parts.gate_scale.ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "Qwen3.5-MoE FP8 checkpoint is missing {prefix}.{expert}.gate_proj.weight_scale_inv"
            ))
        })?;
        let up = parts.up.ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "Qwen3.5-MoE FP8 checkpoint is missing {prefix}.{expert}.up_proj.weight"
            ))
        })?;
        let up_scale = parts.up_scale.ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "Qwen3.5-MoE FP8 checkpoint is missing {prefix}.{expert}.up_proj.weight_scale_inv"
            ))
        })?;
        let down_proj = parts.down.ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "Qwen3.5-MoE FP8 checkpoint is missing {prefix}.{expert}.down_proj.weight"
            ))
        })?;
        let down_proj_scale = parts.down_scale.ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "Qwen3.5-MoE FP8 checkpoint is missing {prefix}.{expert}.down_proj.weight_scale_inv"
            ))
        })?;
        record_fp8_expert_part_shape(&mut gate_shape, &gate, prefix, expert, "gate_proj.weight")?;
        record_fp8_expert_part_shape(
            &mut gate_scale_shape,
            &gate_scale,
            prefix,
            expert,
            "gate_proj.weight_scale_inv",
        )?;
        record_fp8_expert_part_shape(&mut up_shape, &up, prefix, expert, "up_proj.weight")?;
        record_fp8_expert_part_shape(
            &mut up_scale_shape,
            &up_scale,
            prefix,
            expert,
            "up_proj.weight_scale_inv",
        )?;
        record_fp8_expert_part_shape(
            &mut down_shape,
            &down_proj,
            prefix,
            expert,
            "down_proj.weight",
        )?;
        record_fp8_expert_part_shape(
            &mut down_scale_shape,
            &down_proj_scale,
            prefix,
            expert,
            "down_proj.weight_scale_inv",
        )?;
        gate_up_parts.extend([gate, up]);
        gate_up_scale_parts.extend([gate_scale, up_scale]);
        down.push(down_proj);
        down_scale.push(down_proj_scale);
    }

    let gate_shape = gate_shape.expect("positive expert count records a gate shape");
    let gate_scale_shape =
        gate_scale_shape.expect("positive expert count records a gate scale shape");
    let up_shape = up_shape.expect("positive expert count records an up shape");
    let up_scale_shape = up_scale_shape.expect("positive expert count records an up scale shape");
    let down_shape = down_shape.expect("positive expert count records a down shape");
    let down_scale_shape =
        down_scale_shape.expect("positive expert count records a down scale shape");
    if gate_shape[1] != up_shape[1] || gate_scale_shape[1] != up_scale_shape[1] {
        return Err(Error::UnsupportedArchitecture(format!(
            "Qwen FP8 expert prefix '{prefix}' has incompatible gate/up shapes {:?}/{:?} and scale shapes {:?}/{:?}",
            gate_shape, up_shape, gate_scale_shape, up_scale_shape
        )));
    }

    // Concatenating all source tensors directly into each destination bank avoids
    // materializing one gate/up intermediate per expert before stacking the bank.
    let gate_up_proj = concatenate_axis(&gate_up_parts, 0, stream)?.reshape(
        &[num_experts, gate_shape[0] + up_shape[0], gate_shape[1]],
        stream,
    )?;
    let gate_up_proj_scale = concatenate_axis(&gate_up_scale_parts, 0, stream)?.reshape(
        &[
            num_experts,
            gate_scale_shape[0] + up_scale_shape[0],
            gate_scale_shape[1],
        ],
        stream,
    )?;
    let down_proj = concatenate_axis(&down, 0, stream)?
        .reshape(&[num_experts, down_shape[0], down_shape[1]], stream)?;
    let down_proj_scale = concatenate_axis(&down_scale, 0, stream)?.reshape(
        &[num_experts, down_scale_shape[0], down_scale_shape[1]],
        stream,
    )?;
    eval([
        &gate_up_proj,
        &gate_up_proj_scale,
        &down_proj,
        &down_proj_scale,
    ])?;
    Ok(HashMap::from([
        (format!("{prefix}.gate_up_proj"), gate_up_proj),
        (
            format!("{prefix}.gate_up_proj_scale_inv"),
            gate_up_proj_scale,
        ),
        (format!("{prefix}.down_proj"), down_proj),
        (format!("{prefix}.down_proj_scale_inv"), down_proj_scale),
    ]))
}

#[cfg(test)]
fn record_fp8_expert_part_shape(
    expected: &mut Option<[i32; 2]>,
    value: &Array,
    prefix: &str,
    expert: i32,
    component: &str,
) -> Result<(), Error> {
    if value.ndim() != 2 {
        return Err(Error::UnsupportedArchitecture(format!(
            "Qwen FP8 expert tensor {prefix}.{expert}.{component} has rank {}; expected rank 2",
            value.ndim()
        )));
    }
    let shape = [value.dim(0), value.dim(1)];
    if let Some(expected) = expected {
        if *expected != shape {
            return Err(Error::UnsupportedArchitecture(format!(
                "Qwen FP8 expert tensor {prefix}.{expert}.{component} has shape {:?}; expected {:?}",
                value.shape(), expected
            )));
        }
    } else {
        *expected = Some(shape);
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
#[cfg(test)]
enum Fp8ExpertProjection {
    Gate,
    Up,
    Down,
}

#[cfg(test)]
fn parse_fp8_expert_projection_key(key: &str) -> Option<(String, i32, Fp8ExpertProjection)> {
    let (prefix, rest) = key.split_once(".mlp.experts.")?;
    let mut parts = rest.split('.');
    let expert = parts.next()?.parse().ok()?;
    let projection = match parts.next()? {
        "gate_proj" => Fp8ExpertProjection::Gate,
        "up_proj" => Fp8ExpertProjection::Up,
        "down_proj" => Fp8ExpertProjection::Down,
        _ => return None,
    };
    if parts.next()? != "weight" || parts.next().is_some() {
        return None;
    }
    Some((format!("{prefix}.mlp.experts"), expert, projection))
}

#[cfg(test)]
fn parse_fp8_expert_scale_key(key: &str) -> Option<(String, i32, Fp8ExpertProjection)> {
    let weight_key = key
        .strip_suffix(".weight_scale_inv")
        .map(|prefix| format!("{prefix}.weight"))?;
    parse_fp8_expert_projection_key(&weight_key)
}

#[cfg(test)]
pub(crate) fn qwen3_5_strict_load_config(load_visual: bool) -> StrictLoadConfig {
    let config = StrictLoadConfig::default()
        .rewrite_prefix("model.language_model.", "model.")
        .rewrite_prefix("language_model.", "model.")
        .rewrite_prefix("model.model.", "model.")
        .rewrite_prefix("vision_tower.", "visual.")
        .rewrite_prefix("model.visual.", "visual.")
        .rewrite_prefix("model.vision_tower.", "visual.")
        .rewrite_prefix("visual.merger.mlp.0.", "visual.merger.mlp.fc1.")
        .rewrite_prefix("visual.merger.mlp.2.", "visual.merger.mlp.fc2.")
        .rewrite_prefix("vision_tower.merger.mlp.0.", "visual.merger.mlp.fc1.")
        .rewrite_prefix("vision_tower.merger.mlp.2.", "visual.merger.mlp.fc2.")
        .rewrite_prefix("model.visual.merger.mlp.0.", "visual.merger.mlp.fc1.")
        .rewrite_prefix("model.visual.merger.mlp.2.", "visual.merger.mlp.fc2.")
        .rewrite_prefix("model.vision_tower.merger.mlp.0.", "visual.merger.mlp.fc1.")
        .rewrite_prefix("model.vision_tower.merger.mlp.2.", "visual.merger.mlp.fc2.");
    if load_visual {
        config
    } else {
        config
            .allow_unused_prefix("visual.")
            .allow_unused_prefix("vision_tower.")
            .allow_unused_prefix("model.visual.")
            .allow_unused_prefix("model.vision_tower.")
    }
}

impl Model {
    fn prepare_typed_prefill(
        &mut self,
        input: runtime_input::ModelInput<'_>,
        stream: &Stream,
    ) -> Result<runtime_input::PreparedPrefill, Exception> {
        let modality_tokens = [
            self.image_token_id
                .map(|token_id| runtime_input::ModalityToken {
                    modality: runtime_input::Modality::Image,
                    token_id: token_id as u32,
                }),
            self.video_token_id
                .map(|token_id| runtime_input::ModalityToken {
                    modality: runtime_input::Modality::Video,
                    token_id: token_id as u32,
                }),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        let embed_tokens = &mut self.model.embed_tokens;
        let visual = &mut self.visual;
        let vision_args = self.vision_args.as_ref();
        let prepared = runtime_input::prepare_decoder_prefill(
            input,
            &modality_tokens,
            self.args.hidden_size,
            "qwen3_5_moe",
            stream,
            |tokens, stream| embed_tokens.forward(tokens, stream),
            |part, stream| {
                let embeddings = visual_embeddings_from_payload(visual, part, stream)?;
                if part.modality == runtime_input::Modality::Video {
                    video_embedding_chunks(vision_args, part, &embeddings, stream)
                } else {
                    Ok(vec![embeddings])
                }
            },
        )?;
        Ok(prepared)
    }

    fn forward_prepared_prefill(
        &mut self,
        prepared: runtime_input::PreparedPrefill,
        cache: &mut Cache,
        stream: &Stream,
        forward: impl FnOnce(&mut Self, ModelInput<'_>, &Stream) -> Result<Array, Exception>,
    ) -> Result<Array, Exception> {
        let inputs = prepared.tokens();
        let inputs_embeds = prepared.embeddings();
        forward(
            self,
            ModelInput {
                inputs,
                inputs_embeds,
                mask: None,
                cache: Some(cache),
            },
            stream,
        )
    }

    #[cfg(test)]
    pub(crate) fn prefill_typed_with_observer(
        &mut self,
        input: runtime_input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        let prepared = self.prepare_typed_prefill(input, stream)?;
        let logits =
            self.forward_prepared_prefill(prepared, cache, stream, |model, input, stream| {
                model.forward_with_observer(input, stream, observer)
            })?;
        let logits = logits.try_index_device((.., -1, ..), stream)?;
        self.adjust_prefill_logits(logits, cache, stream)
    }
}

fn visual_embeddings_from_payload(
    visual: &mut Option<QwenVisionTransformer>,
    part: &runtime_input::InputPart<'_>,
    stream: &Stream,
) -> Result<Array, Exception> {
    let grid_thw = part.metadata.qwen_grid_thw.ok_or_else(|| {
        Exception::custom(format!(
            "qwen3_5_moe {} input requires qwen_grid_thw metadata",
            part.modality.as_str()
        ))
    })?;
    match part.payload {
        runtime_input::InputPayload::Embeddings(embeddings) => Ok(embeddings.clone()),
        runtime_input::InputPayload::Tensor(tensor) => visual
            .as_mut()
            .ok_or_else(|| {
                Exception::custom(
                    "qwen3_5_moe visual tensor input requires vision_config and visual weights",
                )
            })?
            .forward(tensor, grid_thw, stream),
        runtime_input::InputPayload::TokenIds(_) => Err(Exception::custom(
            "qwen3_5_moe visual input does not accept token-id payloads",
        )),
    }
}

fn video_embedding_chunks(
    vision_args: Option<&VisionConfig>,
    part: &runtime_input::InputPart<'_>,
    embeddings: &Array,
    stream: &Stream,
) -> Result<Vec<Array>, Exception> {
    let grid_thw = part.metadata.qwen_grid_thw.ok_or_else(|| {
        Exception::custom("qwen3_5_moe video input requires qwen_grid_thw metadata")
    })?;
    let grid = grid_thw_from_array(grid_thw, stream)?;
    if grid.len() != 1 {
        return Err(Exception::custom(format!(
            "qwen3_5_moe each video input part requires one grid entry, got {}",
            grid.len()
        )));
    }
    let (grid_t, grid_h, grid_w) = grid[0];
    let merge = vision_args
        .map(|config| config.spatial_merge_size)
        .ok_or_else(|| Exception::custom("qwen3_5_moe video input requires vision_config"))?;
    let chunk_len = grid_h * grid_w / (merge * merge);
    let expected = grid_t * chunk_len;
    if embeddings.dim(1) != expected {
        return Err(Exception::custom(format!(
            "qwen3_5_moe video grid expects {expected} merged embeddings, got {}",
            embeddings.dim(1)
        )));
    }
    let mut chunks = Vec::with_capacity(grid_t as usize);
    for index in 0..grid_t {
        let start = index * chunk_len;
        chunks.push(embeddings.try_index_device((.., start..start + chunk_len, ..), stream)?);
    }
    Ok(chunks)
}

impl CausalModel<Cache> for Model {
    type Tensor = Array;
    type Input<'a> = runtime_input::ModelInput<'a>;
    type Error = Exception;

    fn prefill_input_logits(
        &mut self,
        input: runtime_input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let prepared = self.prepare_typed_prefill(input, stream)?;
        self.forward_prepared_prefill(prepared, cache, stream, |model, input, stream| {
            model.forward_logits(input, true, stream)
        })
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
                inputs_embeds: None,
                mask: None,
                cache: Some(cache),
            },
            stream,
        )?;
        logits.try_index_device((.., -1, ..), stream)
    }

    fn adjust_prefill_logits(
        &mut self,
        mut logits: Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        // Keep the first sampled token dependent on all prefill cache state while
        // avoiding a prompt-length vocabulary projection.
        if let Some(dependency) = cache.prefill_state_dependency(stream)? {
            profile_array(PerfComponent::PrefillStateDependency, &dependency)?;
            logits = logits.add(dependency, stream)?;
        }
        Ok(logits)
    }
}

/// Qwen3.5 MoE token generation iterator.
pub type Generate<'a, S = crate::backend::mlx::runtime::generation::sampler::DefaultSampler> =
    common::generation::Generate<'a, Model, Cache, S>;
