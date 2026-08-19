//! OpenAI GPT-OSS decoder-only mixture-of-experts implementation.

use eredu_checkpoint::WeightQuantization;
use eredu_nn::RopeValue;
use eredu_runtime::CausalModel;

use std::{collections::HashMap, path::Path};

use safemlx::{
    error::Exception,
    fast::ScaledDotProductAttentionMask,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::{
        arange, clip, gather_grouped_rows, gather_qmm_with_mode,
        indexing::{IntoStrideBy, TryIndexOp},
        sigmoid, topk_route_plan, GgufCheckpoint, GgufMetadataValue, QuantizationMode,
    },
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::core::cache::{
    derive_prompt_cache_architecture_fingerprint, validate_prompt_cache_model_identity,
    PromptCacheDescriptor, PromptCacheManifest, PromptCacheModelIdentity,
};

use crate::{
    backend::mlx::error::Error,
    backend::mlx::nn::{
        self as common,
        tensor::{
            create_causal_mask,
            rope::{initialize_rope, validate_rope_scaling_config, RopeVariant},
        },
    },
    backend::mlx::runtime::cache::residency::{open_prompt_cache, CacheResidencyManager},
    backend::mlx::runtime::cache::{ConcatKeyValueCache, KeyValueCache, PagedKeyValueCache},
    backend::mlx::runtime::checkpoint::load::{gguf_quantization_configs, GgufTensorNames},
    backend::mlx::runtime::media::input,
    composition::mlx_architectures::qwen::dense::gguf_string,
    core::attention::{AttentionPolicy, LayerSchedule},
    core::cache::CacheRankIdentity,
};
use eredu_runtime::{CacheResidencyPolicy, CacheResidencyReport, PagedCacheOptions};

fn default_head_dim() -> i32 {
    64
}

fn default_sliding_window() -> i64 {
    128
}

fn default_rope_theta() -> f32 {
    150_000.0
}

fn default_swiglu_limit() -> f32 {
    7.0
}

/// Builds the explicit mask required by one scheduled GPT-OSS attention layer.
///
/// Sliding caches retain exactly `window - 1` past tokens, so single-token
/// decode needs no explicit mask. Multi-token calls still need both causal and
/// sliding constraints over the cache span returned for the current call.
pub(crate) fn attention_mask(
    policy: &AttentionPolicy,
    sequence_length: i32,
    offset: i32,
    stream: &Stream,
) -> Result<Option<Array>, Exception> {
    if sequence_length <= 1 {
        return Ok(None);
    }
    let max_past = policy.window().map(|window| window.get() as i32 - 1);
    create_causal_mask(
        sequence_length,
        Some(offset.min(max_past.unwrap_or(offset))),
        max_past,
        None,
        stream,
    )
    .map(Some)
}

/// GPT-OSS checkpoint quantization metadata.
#[derive(Debug, Clone, Deserialize)]
pub struct MxFp4Config {
    /// Must be `mxfp4` for the published GPT-OSS checkpoints.
    pub quant_method: String,
}

#[derive(Debug, Clone, Deserialize)]
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
    rope_scaling: Option<HashMap<String, RopeValue>>,
    #[serde(default)]
    layer_types: Vec<String>,
    quantization_config: MxFp4Config,
    #[serde(default)]
    quantization: Option<WeightQuantization>,
    #[serde(default = "default_swiglu_limit")]
    swiglu_limit: f32,
}

/// Validated GPT-OSS decoder geometry normalized from checkpoint metadata.
#[derive(Debug, Clone)]
pub struct ModelArgs {
    /// Architecture identifier.
    pub model_type: String,
    /// Transformer width.
    pub hidden_size: i32,
    /// Expert hidden width.
    pub intermediate_size: i32,
    /// Number of transformer blocks.
    pub num_hidden_layers: i32,
    /// Query attention heads.
    pub num_attention_heads: i32,
    /// Key/value attention heads.
    pub num_key_value_heads: i32,
    /// Attention head width.
    pub head_dim: i32,
    /// Vocabulary size.
    pub vocab_size: i32,
    /// Number of local routed experts.
    pub num_local_experts: i32,
    /// Experts selected for each token.
    pub num_experts_per_tok: i32,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// Maximum configured context length.
    pub max_position_embeddings: i32,
    /// RoPE base.
    pub rope_theta: f32,
    /// YaRN scaling configuration.
    pub rope_scaling: Option<HashMap<String, RopeValue>>,
    /// Authoritative attention behavior in decoder-layer order.
    pub attention_schedule: LayerSchedule<AttentionPolicy>,
    /// Published checkpoint MXFP4 metadata.
    pub quantization_config: MxFp4Config,
    /// Optional encoding used by remaining standard dense matrices.
    pub quantization: Option<WeightQuantization>,
    /// Exact per-weight formats for mixed GGUF checkpoints.
    pub quantized_weight_configs: Option<HashMap<String, WeightQuantization>>,
    /// GPT-OSS clipped SwiGLU limit.
    pub swiglu_limit: f32,
}

impl ModelArgs {
    pub(crate) fn weight_quantization_for(&self, name: &str) -> Option<WeightQuantization> {
        self.quantized_weight_configs
            .as_ref()
            .and_then(|configs| configs.get(name).copied())
            .or(self.quantization)
    }

    fn checkpoint_weight_quantization_for(&self, name: &str) -> Option<WeightQuantization> {
        self.quantized_weight_configs
            .as_ref()
            .and_then(|configs| configs.get(name).copied())
    }
    pub(crate) fn validate(&self) -> Result<(), Error> {
        if self.model_type != "gpt_oss" {
            return Err(Error::UnsupportedArchitecture(format!(
                "GPT-OSS loader requires model_type gpt_oss, got {:?}",
                self.model_type
            )));
        }
        if self.quantization_config.quant_method != "mxfp4" {
            return Err(Error::UnsupportedArchitecture(format!(
                "GPT-OSS expert weights require quant_method mxfp4, got {:?}",
                self.quantization_config.quant_method
            )));
        }
        if self.hidden_size <= 0
            || self.intermediate_size <= 0
            || self.num_hidden_layers <= 0
            || self.num_attention_heads <= 0
            || self.num_key_value_heads <= 0
            || self.head_dim <= 0
            || self.vocab_size <= 0
            || self.num_local_experts <= 0
            || self.max_position_embeddings <= 0
        {
            return Err(Error::UnsupportedArchitecture(
                "GPT-OSS dimensions, layer counts, and cache geometry must be positive".into(),
            ));
        }
        if self.hidden_size % 32 != 0 || self.intermediate_size % 32 != 0 {
            return Err(Error::UnsupportedArchitecture(
                "GPT-OSS MXFP4 projection dimensions must be divisible by 32".into(),
            ));
        }
        if self
            .num_attention_heads
            .checked_mul(self.head_dim)
            .is_none()
            || self
                .num_key_value_heads
                .checked_mul(self.head_dim)
                .is_none()
            || self.num_attention_heads % self.num_key_value_heads != 0
        {
            return Err(Error::UnsupportedArchitecture(
                "GPT-OSS attention head configuration is invalid".into(),
            ));
        }
        if self.num_experts_per_tok <= 0 || self.num_experts_per_tok > self.num_local_experts {
            return Err(Error::UnsupportedArchitecture(
                "GPT-OSS expert routing configuration is invalid".into(),
            ));
        }
        if self.attention_schedule.len() != self.num_hidden_layers as usize {
            return Err(Error::UnsupportedArchitecture(format!(
                "GPT-OSS attention schedule has {} entries for {} layers",
                self.attention_schedule.len(),
                self.num_hidden_layers
            )));
        }
        for window in self.attention_schedule.sliding_windows().keys() {
            if window.get() > i32::MAX as u32 {
                return Err(Error::UnsupportedArchitecture(format!(
                    "GPT-OSS sliding attention window exceeds the executable i32 range: {}",
                    window.get()
                )));
            }
        }
        validate_rope_scaling_config(&self.rope_scaling)?;
        Ok(())
    }
}

fn alternating_attention_schedule(
    layer_count: usize,
    window: u32,
) -> Result<LayerSchedule<AttentionPolicy>, Error> {
    let sliding = AttentionPolicy::sliding(window)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    LayerSchedule::new(
        layer_count,
        (0..layer_count)
            .map(|index| {
                if index % 2 == 0 {
                    sliding
                } else {
                    AttentionPolicy::Full
                }
            })
            .collect(),
    )
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

fn normalize_hf_attention_schedule(
    layer_count: i32,
    window: i64,
    layer_types: &[String],
) -> Result<LayerSchedule<AttentionPolicy>, Error> {
    let layers = usize::try_from(layer_count).map_err(|_| {
        Error::UnsupportedArchitecture(format!(
            "GPT-OSS num_hidden_layers must be positive, got {layer_count}"
        ))
    })?;
    if layers == 0 {
        return Err(Error::UnsupportedArchitecture(
            "GPT-OSS num_hidden_layers must be positive, got 0".into(),
        ));
    }
    if window <= 0 {
        return Err(Error::UnsupportedArchitecture(format!(
            "GPT-OSS sliding_window must be positive, got {window}"
        )));
    }
    let window = u32::try_from(window).map_err(|_| {
        Error::UnsupportedArchitecture(format!(
            "GPT-OSS sliding_window exceeds the executable u32 range: {window}"
        ))
    })?;
    if window > i32::MAX as u32 {
        return Err(Error::UnsupportedArchitecture(format!(
            "GPT-OSS sliding_window exceeds the executable i32 range: {window}"
        )));
    }
    if layer_types.is_empty() {
        return alternating_attention_schedule(layers, window);
    }
    if layer_types.len() != layers {
        return Err(Error::UnsupportedArchitecture(format!(
            "GPT-OSS layer_types has {} entries for {layers} layers",
            layer_types.len()
        )));
    }
    let sliding = AttentionPolicy::sliding(window)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let policies = layer_types
        .iter()
        .enumerate()
        .map(|(index, kind)| match kind.as_str() {
            "sliding_attention" => Ok(sliding),
            "full_attention" => Ok(AttentionPolicy::Full),
            _ => Err(Error::UnsupportedArchitecture(format!(
                "GPT-OSS layer_types[{index}] must be sliding_attention or full_attention, got {kind:?}"
            ))),
        })
        .collect::<Result<Vec<_>, _>>()?;
    LayerSchedule::new(layers, policies)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

pub(crate) fn prompt_cache_architecture_fingerprint(args: &ModelArgs) -> String {
    let rope_scaling = args.rope_scaling.as_ref().map_or_else(
        || "none".to_string(),
        |config| {
            let mut entries = config.iter().collect::<Vec<_>>();
            entries.sort_unstable_by_key(|(key, _)| key.as_str());
            entries
                .into_iter()
                .map(|(key, value)| {
                    let value = match value {
                        RopeValue::Float(value) => format!("f32:{:08x}", value.to_bits()),
                        RopeValue::String(value) => format!("string:{value}"),
                        RopeValue::Bool(value) => format!("bool:{value}"),
                    };
                    format!("{key}={value}")
                })
                .collect::<Vec<_>>()
                .join(";")
        },
    );
    derive_prompt_cache_architecture_fingerprint(
        "gpt_oss",
        [
            ("model_type", args.model_type.clone()),
            ("hidden_size", args.hidden_size.to_string()),
            ("intermediate_size", args.intermediate_size.to_string()),
            ("num_hidden_layers", args.num_hidden_layers.to_string()),
            ("num_attention_heads", args.num_attention_heads.to_string()),
            ("num_key_value_heads", args.num_key_value_heads.to_string()),
            ("head_dim", args.head_dim.to_string()),
            ("vocab_size", args.vocab_size.to_string()),
            ("num_local_experts", args.num_local_experts.to_string()),
            ("num_experts_per_tok", args.num_experts_per_tok.to_string()),
            (
                "rms_norm_eps",
                format!("{:08x}", args.rms_norm_eps.to_bits()),
            ),
            (
                "attention_schedule",
                args.attention_schedule.fingerprint_component(),
            ),
            (
                "max_position_embeddings",
                args.max_position_embeddings.to_string(),
            ),
            ("rope_theta", format!("{:08x}", args.rope_theta.to_bits())),
            ("rope_scaling", rope_scaling),
            (
                "quantization_config",
                args.quantization_config.quant_method.clone(),
            ),
            ("quantization", format!("{:?}", args.quantization)),
            (
                "swiglu_limit",
                format!("{:08x}", args.swiglu_limit.to_bits()),
            ),
        ],
    )
}

/// Validates a parsed GPT-OSS configuration.
pub fn validate_model_config_value(config: &serde_json::Value) -> Result<(), Error> {
    model_args_from_config_value(config).map(|_| ())
}

/// Parses and normalizes a Hugging Face GPT-OSS configuration.
///
/// Explicit `layer_types` entries become the authoritative ordered attention
/// schedule. When the published field is omitted, GPT-OSS alternates sliding
/// and full attention beginning with sliding attention at layer zero.
pub fn model_args_from_config_value(config: &serde_json::Value) -> Result<ModelArgs, Error> {
    let source: ModelArgsSource = serde_json::from_value(config.clone()).map_err(|error| {
        Error::UnsupportedArchitecture(format!("invalid gpt_oss config: {error}"))
    })?;
    let attention_schedule = normalize_hf_attention_schedule(
        source.num_hidden_layers,
        source.sliding_window,
        &source.layer_types,
    )?;
    let args = ModelArgs {
        model_type: source.model_type,
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
        rope_scaling: source.rope_scaling,
        attention_schedule,
        quantization_config: source.quantization_config,
        quantization: source.quantization,
        quantized_weight_configs: None,
        swiglu_limit: source.swiglu_limit,
    };
    args.validate()?;
    Ok(args)
}

/// One attention layer with learned sink logits.
#[derive(Debug, Clone, ModuleParameters)]
pub struct Attention {
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    #[param]
    /// Learned per-query-head attention sink.
    pub sinks: Param<Array>,
    #[param]
    /// Query projection.
    pub q_proj: MaybeQuantized<nn::Linear>,
    #[param]
    /// Key projection.
    pub k_proj: MaybeQuantized<nn::Linear>,
    #[param]
    /// Value projection.
    pub v_proj: MaybeQuantized<nn::Linear>,
    #[param]
    /// Attention output projection.
    pub o_proj: MaybeQuantized<nn::Linear>,
    #[param]
    rope: RopeVariant,
}

impl Attention {
    fn new(args: &ModelArgs, layer: usize, stream: &Stream) -> Result<Self, Exception> {
        let prefix = format!("model.layers.{layer}.self_attn");
        Ok(Self {
            n_heads: args.num_attention_heads,
            n_kv_heads: args.num_key_value_heads,
            head_dim: args.head_dim,
            scale: 1.0 / (args.head_dim as f32).sqrt(),
            sinks: Param::<Array>::unloaded(&[args.num_attention_heads], Dtype::Float32, stream)?,
            q_proj: common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                args.num_attention_heads * args.head_dim,
                true,
                args.weight_quantization_for(&format!("{prefix}.q_proj.weight")),
                stream,
            )?,
            k_proj: common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                args.num_key_value_heads * args.head_dim,
                true,
                args.weight_quantization_for(&format!("{prefix}.k_proj.weight")),
                stream,
            )?,
            v_proj: common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                args.num_key_value_heads * args.head_dim,
                true,
                args.weight_quantization_for(&format!("{prefix}.v_proj.weight")),
                stream,
            )?,
            o_proj: common::linear::unloaded_maybe_quantized_linear(
                args.num_attention_heads * args.head_dim,
                args.hidden_size,
                true,
                args.weight_quantization_for(&format!("{prefix}.o_proj.weight")),
                stream,
            )?,
            rope: initialize_rope(
                args.head_dim,
                args.rope_theta,
                false,
                &args.rope_scaling,
                args.max_position_embeddings,
                stream,
            )?,
        })
    }

    fn forward<C: KeyValueCache>(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut C,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let (batch, length) = (shape[0], shape[1]);
        let project = |projection: Array, heads: i32| {
            projection
                .reshape(&[batch, length, heads, self.head_dim], stream)?
                .transpose_axes(&[0, 2, 1, 3], stream)
        };
        let mut q = project(self.q_proj.forward(x, stream)?, self.n_heads)?;
        let mut k = project(self.k_proj.forward(x, stream)?, self.n_kv_heads)?;
        let v = project(self.v_proj.forward(x, stream)?, self.n_kv_heads)?;
        let offset = cache.offset();
        q = self.rope.forward(nn::RopeInput { x: &q, offset }, stream)?;
        k = self.rope.forward(nn::RopeInput { x: &k, offset }, stream)?;
        let (k, v) = cache.update_and_fetch(k, v, stream)?;
        let attended =
            match cache.paged_attention(&q, self.scale, mask, Some(self.sinks.as_ref()), stream)? {
                Some(output) => output,
                None => safemlx::fast::scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    self.scale,
                    mask.map(ScaledDotProductAttentionMask::Array),
                    self.sinks.as_ref(),
                    stream,
                )?,
            };
        self.o_proj.forward(
            &attended
                .transpose_axes(&[0, 2, 1, 3], stream)?
                .reshape(&[batch, length, -1], stream)?,
            stream,
        )
    }

    fn forward_tensor_parallel<C: KeyValueCache>(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut C,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let (batch, length) = (shape[0], shape[1]);
        let project = |projection: Array, heads: i32| {
            projection
                .reshape(&[batch, length, heads, self.head_dim], stream)?
                .transpose_axes(&[0, 2, 1, 3], stream)
        };
        let mut q = project(self.q_proj.forward(x, stream)?, self.n_heads)?;
        let mut k = project(self.k_proj.forward(x, stream)?, self.n_kv_heads)?;
        let v = project(self.v_proj.forward(x, stream)?, self.n_kv_heads)?;
        let offset = cache.offset();
        q = self.rope.forward(nn::RopeInput { x: &q, offset }, stream)?;
        k = self.rope.forward(nn::RopeInput { x: &k, offset }, stream)?;
        let (k, v) = cache.update_and_fetch(k, v, stream)?;
        let attended =
            match cache.paged_attention(&q, self.scale, mask, Some(self.sinks.as_ref()), stream)? {
                Some(output) => output,
                None => safemlx::fast::scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    self.scale,
                    mask.map(ScaledDotProductAttentionMask::Array),
                    self.sinks.as_ref(),
                    stream,
                )?,
            };
        let output = self.o_proj.forward(
            &attended
                .transpose_axes(&[0, 2, 1, 3], stream)?
                .reshape(&[batch, length, -1], stream)?,
            stream,
        )?;
        let bias = match &self.o_proj {
            MaybeQuantized::Original(linear) => linear.bias.as_ref().as_ref(),
            MaybeQuantized::Quantized(linear) => linear.inner.bias.as_ref().as_ref(),
        };
        let partial = match bias {
            Some(bias) => output.subtract(bias, stream)?,
            None => output,
        };
        let reduced = safemlx::distributed::all_sum(&partial, group, stream)?;
        match bias {
            Some(bias) => reduced.add(bias, stream),
            None => Ok(reduced),
        }
    }
}

/// Checkpoint-native combined MXFP4 expert tensors.
#[derive(Debug, Clone, ModuleParameters)]
pub struct Experts {
    num_experts: i32,
    hidden_size: i32,
    intermediate_size: i32,
    limit: f32,
    #[param]
    /// Combined alternating gate/up packed FP4 blocks.
    pub gate_up_proj_blocks: Param<Array>,
    #[param]
    /// Combined alternating gate/up E8M0 scales.
    pub gate_up_proj_scales: Param<Array>,
    #[param]
    /// Combined alternating gate/up projection bias.
    pub gate_up_proj_bias: Param<Array>,
    #[param]
    /// Packed down-projection FP4 blocks.
    pub down_proj_blocks: Param<Array>,
    #[param]
    /// Down-projection E8M0 scales.
    pub down_proj_scales: Param<Array>,
    #[param]
    /// Down-projection bias.
    pub down_proj_bias: Param<Array>,
}

impl Experts {
    pub(crate) fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            num_experts: args.num_local_experts,
            hidden_size: args.hidden_size,
            intermediate_size: args.intermediate_size,
            limit: args.swiglu_limit,
            gate_up_proj_blocks: Param::<Array>::unloaded(
                &[
                    args.num_local_experts,
                    2 * args.intermediate_size,
                    args.hidden_size / 32,
                    16,
                ],
                Dtype::Uint8,
                stream,
            )?,
            gate_up_proj_scales: Param::<Array>::unloaded(
                &[
                    args.num_local_experts,
                    2 * args.intermediate_size,
                    args.hidden_size / 32,
                ],
                Dtype::Uint8,
                stream,
            )?,
            gate_up_proj_bias: Param::<Array>::unloaded(
                &[args.num_local_experts, 2 * args.intermediate_size],
                Dtype::Float32,
                stream,
            )?,
            down_proj_blocks: Param::<Array>::unloaded(
                &[
                    args.num_local_experts,
                    args.hidden_size,
                    args.intermediate_size / 32,
                    16,
                ],
                Dtype::Uint8,
                stream,
            )?,
            down_proj_scales: Param::<Array>::unloaded(
                &[
                    args.num_local_experts,
                    args.hidden_size,
                    args.intermediate_size / 32,
                ],
                Dtype::Uint8,
                stream,
            )?,
            down_proj_bias: Param::<Array>::unloaded(
                &[args.num_local_experts, args.hidden_size],
                Dtype::Float32,
                stream,
            )?,
        })
    }

    fn mxfp4_linear(
        input: &Array,
        blocks: &Array,
        scales: &Array,
        projection_bias: &Array,
        expert_ids: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let routes = input.dim(0);
        let output_size = blocks.dim(1);
        let packed = blocks
            .view::<u32>(stream)?
            .reshape(&[blocks.dim(0), output_size, -1], stream)?;
        let lhs_indices = arange::<i32, u32>(0, routes, 1, stream)?;
        let output = gather_qmm_with_mode(
            input.reshape(&[routes, 1, input.dim(-1)], stream)?,
            packed,
            scales,
            None,
            Some(&lhs_indices),
            Some(expert_ids),
            true,
            32,
            4,
            true,
            QuantizationMode::MxFp4,
            stream,
        )?
        .reshape(&[routes, output_size], stream)?;
        output.add(projection_bias.take_axis(expert_ids, 0, stream)?, stream)
    }

    pub(crate) fn forward(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let tokens = hidden_states.dim(0);
        let plan = topk_route_plan(top_k_index, self.num_experts, stream)?;
        let routed = gather_grouped_rows(hidden_states, &plan, stream)?;

        let gate_up = Self::mxfp4_linear(
            &routed,
            self.gate_up_proj_blocks.as_ref(),
            self.gate_up_proj_scales.as_ref(),
            self.gate_up_proj_bias.as_ref(),
            &plan.sorted_group_ids,
            stream,
        )?;
        let gate = gate_up.try_index_device((.., (0..).stride_by(2)), stream)?;
        let linear = gate_up.try_index_device((.., (1..).stride_by(2)), stream)?;
        let gate = clip(gate, ((), self.limit), stream)?;
        let linear = clip(linear, (-self.limit, self.limit), stream)?;
        let activated = gate
            .multiply(
                sigmoid(gate.multiply(Array::from_f32(1.702), stream)?, stream)?,
                stream,
            )?
            .multiply(linear.add(Array::from_f32(1.0), stream)?, stream)?;
        debug_assert_eq!(activated.dim(-1), self.intermediate_size);

        let output = Self::mxfp4_linear(
            &activated,
            self.down_proj_blocks.as_ref(),
            self.down_proj_scales.as_ref(),
            self.down_proj_bias.as_ref(),
            &plan.sorted_group_ids,
            stream,
        )?;
        debug_assert_eq!(output.dim(-1), self.hidden_size);
        common::moe::weighted_route_sum(output, top_k_weights, &plan, tokens, stream)
    }

    fn forward_tensor_parallel(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let tokens = hidden_states.dim(0);
        let plan = topk_route_plan(top_k_index, self.num_experts, stream)?;
        let routed = gather_grouped_rows(hidden_states, &plan, stream)?;
        let gate_up = Self::mxfp4_linear(
            &routed,
            self.gate_up_proj_blocks.as_ref(),
            self.gate_up_proj_scales.as_ref(),
            self.gate_up_proj_bias.as_ref(),
            &plan.sorted_group_ids,
            stream,
        )?;
        let gate = gate_up.try_index_device((.., (0..).stride_by(2)), stream)?;
        let linear = gate_up.try_index_device((.., (1..).stride_by(2)), stream)?;
        let gate = clip(gate, ((), self.limit), stream)?;
        let linear = clip(linear, (-self.limit, self.limit), stream)?;
        let activated = gate
            .multiply(
                sigmoid(gate.multiply(Array::from_f32(1.702), stream)?, stream)?,
                stream,
            )?
            .multiply(linear.add(Array::from_f32(1.0), stream)?, stream)?;
        let output = Self::mxfp4_linear(
            &activated,
            self.down_proj_blocks.as_ref(),
            self.down_proj_scales.as_ref(),
            self.down_proj_bias.as_ref(),
            &plan.sorted_group_ids,
            stream,
        )?;
        let routed_bias =
            self.down_proj_bias
                .as_ref()
                .take_axis(&plan.sorted_group_ids, 0, stream)?;
        let partial = output.subtract(&routed_bias, stream)?;
        let partial =
            common::moe::weighted_route_sum(partial, top_k_weights, &plan, tokens, stream)?;
        let reduced = safemlx::distributed::all_sum(&partial, group, stream)?;
        let bias =
            common::moe::weighted_route_sum(routed_bias, top_k_weights, &plan, tokens, stream)?;
        reduced.add(bias, stream)
    }

    fn execute_local_routes_tensor_partial(
        &mut self,
        hidden_states: &Array,
        local_expert_ids: &Array,
        tensor_parallel_size: usize,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let ids = local_expert_ids.reshape(&[-1, 1], stream)?;
        let weights =
            safemlx::ops::ones_dtype(&[hidden_states.dim(0), 1], hidden_states.dtype(), stream)?;
        let output = self.forward(hidden_states, &ids, &weights, stream)?;
        if tensor_parallel_size == 1 {
            return Ok(output);
        }
        let bias = self
            .down_proj_bias
            .as_ref()
            .take_axis(local_expert_ids, 0, stream)?;
        output.subtract(&bias, stream)?.add(
            bias.divide(Array::from_f32(tensor_parallel_size as f32), stream)?,
            stream,
        )
    }
}

/// GPT-OSS sparse MoE block.
#[derive(Debug, Clone, ModuleParameters)]
pub struct Mlp {
    top_k: i32,
    #[param]
    /// Biased expert router, matching the checkpoint's `mlp.router` tree.
    pub router: MaybeQuantized<nn::Linear>,
    #[param]
    /// Checkpoint-native routed expert bank.
    pub experts: Experts,
}

impl Mlp {
    fn new(args: &ModelArgs, layer: usize, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            top_k: args.num_experts_per_tok,
            router: common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                args.num_local_experts,
                true,
                // Preserve the established load-time policy: routers remain dense.
                // Checkpoint-native GGUF quantization is still honored exactly.
                args.checkpoint_weight_quantization_for(&format!(
                    "model.layers.{layer}.mlp.router.weight"
                )),
                stream,
            )?,
            experts: Experts::new(args, stream)?,
        })
    }

    fn forward(&mut self, x: &Array, stream: &Stream) -> Result<Array, Exception> {
        let shape = x.shape();
        let flat = x.reshape(&[-1, shape[2]], stream)?;
        let logits = self.router.forward(&flat, stream)?;
        let (indices, weights) = common::moe::top_k_softmax_routing(&logits, self.top_k, stream)?;
        self.experts
            .forward(&flat, &indices, &weights, stream)?
            .reshape(shape, stream)
    }

    pub(crate) fn forward_expert_parallel(
        &mut self,
        x: &Array,
        assignment: &crate::composition::mlx_architectures::distributed::expert::ExpertAssignment,
        group: &safemlx::distributed::Group,
        statistics: &mut crate::composition::mlx_architectures::distributed::expert::RoutingStatistics,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let flat = x.reshape(&[-1, shape[2]], stream)?;
        let router_started = std::time::Instant::now();
        let logits = self.router.forward(&flat, stream)?;
        let (indices, weights) = common::moe::top_k_softmax_routing(&logits, self.top_k, stream)?;
        statistics.router_time += router_started.elapsed();
        let returned =
            crate::composition::mlx_architectures::distributed::expert::dispatch_replicated(
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
        returned.reduced_output.reshape(shape, stream)
    }

    fn forward_tensor_parallel(
        &mut self,
        x: &Array,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let flat = x.reshape(&[-1, shape[2]], stream)?;
        let logits = self.router.forward(&flat, stream)?;
        let (indices, weights) = common::moe::top_k_softmax_routing(&logits, self.top_k, stream)?;
        self.experts
            .forward_tensor_parallel(&flat, &indices, &weights, group, stream)?
            .reshape(shape, stream)
    }

    fn forward_tensor_expert_parallel(
        &mut self,
        x: &Array,
        assignment: &crate::composition::mlx_architectures::distributed::expert::ExpertAssignment,
        tensor_group: &safemlx::distributed::Group,
        expert_group: &safemlx::distributed::Group,
        statistics: &mut crate::composition::mlx_architectures::distributed::expert::RoutingStatistics,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let flat = x.reshape(&[-1, shape[2]], stream)?;
        let router_started = std::time::Instant::now();
        let logits = self.router.forward(&flat, stream)?;
        let (indices, weights) = common::moe::top_k_softmax_routing(&logits, self.top_k, stream)?;
        statistics.router_time += router_started.elapsed();
        let returned =
            crate::composition::mlx_architectures::distributed::expert::dispatch_replicated_with(
                &flat,
                &indices,
                &weights,
                assignment,
                expert_group,
                stream,
                |routes, stream| {
                    self.experts
                        .execute_local_routes_tensor_partial(
                            &routes.hidden,
                            &routes.local_expert_ids,
                            tensor_group.size(),
                            stream,
                        )
                        .map_err(Error::from)
                },
            )
            .map_err(|error| Exception::custom(error.to_string()))?;
        statistics.accumulate(&returned.statistics);
        safemlx::distributed::all_sum(&returned.reduced_output, tensor_group, stream)?
            .reshape(shape, stream)
    }

    fn forward_with_expert_executor<F>(
        &mut self,
        x: &Array,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let shape = x.shape();
        let flat = x.reshape(&[-1, shape[2]], stream)?;
        let logits = self.router.forward(&flat, stream)?;
        let (indices, weights) = common::moe::top_k_softmax_routing(&logits, self.top_k, stream)?;
        execute(&flat, &indices, &weights, stream)?.reshape(shape, stream)
    }
}

/// One GPT-OSS decoder block.
#[derive(Debug, Clone, ModuleParameters)]
pub struct TransformerBlock {
    #[param]
    /// Self-attention.
    pub self_attn: Attention,
    #[param]
    /// Sparse MoE feed-forward block.
    pub mlp: Mlp,
    #[param]
    /// Pre-attention RMSNorm.
    pub input_layernorm: nn::RmsNorm,
    #[param]
    /// Pre-MoE RMSNorm.
    pub post_attention_layernorm: nn::RmsNorm,
}

impl TransformerBlock {
    pub(crate) fn new(args: &ModelArgs, layer: usize, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            self_attn: Attention::new(args, layer, stream)?,
            mlp: Mlp::new(args, layer, stream)?,
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

    pub(crate) fn forward<C: KeyValueCache>(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut C,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(x, stream)?;
        let hidden = x.add(
            self.self_attn.forward(&normed, mask, cache, stream)?,
            stream,
        )?;
        let normed = self.post_attention_layernorm.forward(&hidden, stream)?;
        hidden.add(self.mlp.forward(&normed, stream)?, stream)
    }

    pub(crate) fn forward_tensor_parallel<C: KeyValueCache>(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut C,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(x, stream)?;
        let hidden = x.add(
            self.self_attn
                .forward_tensor_parallel(&normed, mask, cache, group, stream)?,
            stream,
        )?;
        let normed = self.post_attention_layernorm.forward(&hidden, stream)?;
        hidden.add(
            self.mlp.forward_tensor_parallel(&normed, group, stream)?,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_expert_parallel<C: KeyValueCache>(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut C,
        assignment: &crate::composition::mlx_architectures::distributed::expert::ExpertAssignment,
        group: &safemlx::distributed::Group,
        statistics: &mut crate::composition::mlx_architectures::distributed::expert::RoutingStatistics,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(x, stream)?;
        let hidden = x.add(
            self.self_attn.forward(&normed, mask, cache, stream)?,
            stream,
        )?;
        let normed = self.post_attention_layernorm.forward(&hidden, stream)?;
        hidden.add(
            self.mlp
                .forward_expert_parallel(&normed, assignment, group, statistics, stream)?,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_tensor_expert_parallel<C: KeyValueCache>(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut C,
        assignment: &crate::composition::mlx_architectures::distributed::expert::ExpertAssignment,
        tensor_group: &safemlx::distributed::Group,
        expert_group: &safemlx::distributed::Group,
        statistics: &mut crate::composition::mlx_architectures::distributed::expert::RoutingStatistics,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(x, stream)?;
        let hidden = x.add(
            self.self_attn
                .forward_tensor_parallel(&normed, mask, cache, tensor_group, stream)?,
            stream,
        )?;
        let normed = self.post_attention_layernorm.forward(&hidden, stream)?;
        hidden.add(
            self.mlp.forward_tensor_expert_parallel(
                &normed,
                assignment,
                tensor_group,
                expert_group,
                statistics,
                stream,
            )?,
            stream,
        )
    }

    pub(crate) fn forward_with_expert_executor<C: KeyValueCache, F>(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut C,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let normed = self.input_layernorm.forward(x, stream)?;
        let hidden = x.add(
            self.self_attn.forward(&normed, mask, cache, stream)?,
            stream,
        )?;
        let normed = self.post_attention_layernorm.forward(&hidden, stream)?;
        hidden.add(
            self.mlp
                .forward_with_expert_executor(&normed, stream, execute)?,
            stream,
        )
    }

    pub(crate) fn forward_tensor_with_expert_executor<C: KeyValueCache, F>(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut C,
        tensor_group: &safemlx::distributed::Group,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let normed = self.input_layernorm.forward(x, stream)?;
        let hidden = x.add(
            self.self_attn
                .forward_tensor_parallel(&normed, mask, cache, tensor_group, stream)?,
            stream,
        )?;
        let normed = self.post_attention_layernorm.forward(&hidden, stream)?;
        hidden.add(
            self.mlp
                .forward_with_expert_executor(&normed, stream, execute)?,
            stream,
        )
    }
}

/// Per-layer cache matching the canonical attention schedule.
#[derive(Debug, Clone)]
pub enum LayerCache {
    /// Unbounded full-attention cache.
    Full(ConcatKeyValueCache),
    /// Bounded sliding-attention cache.
    Sliding(ConcatKeyValueCache),
    /// Block-addressable full or sliding state.
    Paged(PagedKeyValueCache),
}

impl LayerCache {
    pub(crate) fn attention_policy(&self) -> Result<AttentionPolicy, Exception> {
        fn sliding(window: i32) -> Result<AttentionPolicy, Exception> {
            let window = u32::try_from(window).map_err(|_| {
                Exception::custom(format!("GPT-OSS cache has invalid sliding window {window}"))
            })?;
            AttentionPolicy::sliding(window).map_err(|error| Exception::custom(error.to_string()))
        }
        match self {
            Self::Full(_) => Ok(AttentionPolicy::Full),
            Self::Sliding(cache) => match cache.max_size() {
                Some(window) => sliding(window),
                None => Err(Exception::custom(
                    "GPT-OSS sliding cache is missing its attention window",
                )),
            },
            Self::Paged(cache) => match cache.max_size() {
                Some(window) => sliding(window),
                None => Ok(AttentionPolicy::Full),
            },
        }
    }
}

impl KeyValueCache for LayerCache {
    fn offset(&self) -> i32 {
        match self {
            Self::Full(cache) => cache.offset(),
            Self::Sliding(cache) => cache.offset(),
            Self::Paged(cache) => cache.offset(),
        }
    }

    fn max_size(&self) -> Option<i32> {
        match self {
            Self::Full(cache) => cache.max_size(),
            Self::Sliding(cache) => cache.max_size(),
            Self::Paged(cache) => cache.max_size(),
        }
    }

    fn retained_arrays(&self) -> Vec<&Array> {
        match self {
            Self::Full(cache) => cache.retained_arrays(),
            Self::Sliding(cache) => cache.retained_arrays(),
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
            Self::Full(_) | Self::Sliding(_) => Ok(None),
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
            Self::Sliding(cache) => cache.update_and_fetch(keys, values, stream),
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
            Self::Sliding(cache) => cache.update_for_attention(keys, values, stream),
            Self::Paged(cache) => cache.update_for_attention(keys, values, stream),
        }
    }
}

/// Heterogeneous generation cache for GPT-OSS.
#[derive(Debug, Clone, Default)]
pub struct Cache {
    /// One cache per decoder block.
    pub layers: Vec<LayerCache>,
}

impl Cache {
    pub(crate) fn new_paged(
        attention_schedule: &LayerSchedule<AttentionPolicy>,
        manager: CacheResidencyManager,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, Exception> {
        let layers = attention_schedule
            .iter()
            .enumerate()
            .map(|(layer, policy)| {
                let window = policy.window().map(|window| {
                    i32::try_from(window.get()).expect("validated GPT-OSS sliding window fits i32")
                });
                PagedKeyValueCache::new_with_layout(manager.clone(), layer, window, 0, rank)
                    .map(LayerCache::Paged)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { layers })
    }

    pub(crate) fn reset(&mut self) -> Result<(), Exception> {
        for layer in &mut self.layers {
            match layer {
                LayerCache::Full(cache) => cache.clear(),
                LayerCache::Sliding(cache) => cache.clear(),
                LayerCache::Paged(cache) => {
                    cache.clear()?;
                }
            }
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn offset(&self) -> i32 {
        self.layers.first().map_or(0, LayerCache::offset)
    }

    /// Returns aggregate paged-cache residency observations.
    pub fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        self.layers
            .iter()
            .find_map(|layer| match layer {
                LayerCache::Paged(cache) => Some(cache),
                LayerCache::Full(_) | LayerCache::Sliding(_) => None,
            })
            .map(|cache| cache.report())
            .transpose()
    }

    /// Finalizes and atomically saves an immutable text prefix.
    pub fn save_prompt_cache(
        &mut self,
        destination: impl AsRef<Path>,
        descriptor: crate::core::cache::PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &crate::core::cache::PromptCacheOptions,
    ) -> Result<crate::core::cache::PromptCacheManifest, Exception> {
        let mut manager = None;
        for layer in &mut self.layers {
            if let LayerCache::Paged(cache) = layer {
                cache.finalize()?;
                manager.get_or_insert_with(|| cache.manager().clone());
            }
        }
        manager
            .ok_or_else(|| {
                Exception::custom(
                    "prompt-cache persistence requires an explicitly configured paged GPT-OSS cache",
                )
            })?
            .save_prompt_cache(destination, descriptor, prefix_token_ids, &[], options)
            .map_err(|error| Exception::custom(error.to_string()))
    }
}

/// GPT-OSS transformer body.
#[derive(Debug, Clone, ModuleParameters)]
pub struct GptOssModel {
    attention_schedule: LayerSchedule<AttentionPolicy>,
    #[param]
    /// Token embedding table.
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[param]
    /// Decoder blocks.
    pub layers: Vec<TransformerBlock>,
    #[param]
    /// Final RMSNorm.
    pub norm: nn::RmsNorm,
}

impl GptOssModel {
    fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            attention_schedule: args.attention_schedule.clone(),
            embed_tokens: common::linear::unloaded_maybe_quantized_embedding(
                args.vocab_size,
                args.hidden_size,
                args.weight_quantization_for("model.embed_tokens.weight"),
                stream,
            )?,
            layers: (0..args.num_hidden_layers)
                .map(|layer| TransformerBlock::new(args, layer as usize, stream))
                .collect::<Result<_, _>>()?,
            norm: nn::RmsNorm::unloaded(
                args.hidden_size,
                args.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?,
        })
    }

    fn new_cache(&self) -> Cache {
        Cache {
            layers: self
                .attention_schedule
                .iter()
                .map(|policy| match policy.window() {
                    Some(window) => {
                        LayerCache::Sliding(ConcatKeyValueCache::new_for_sliding_attention(
                            i32::try_from(window.get())
                                .expect("validated GPT-OSS sliding window fits i32"),
                        ))
                    }
                    None => LayerCache::Full(ConcatKeyValueCache::new()),
                })
                .collect(),
        }
    }

    fn new_cache_with_options(&self, policy: CacheResidencyPolicy) -> Result<Cache, Exception> {
        match policy {
            CacheResidencyPolicy::Device => Ok(self.new_cache()),
            CacheResidencyPolicy::Paged(options) => {
                let manager = CacheResidencyManager::new(options)
                    .map_err(|error| Exception::custom(error.to_string()))?;
                self.new_cache_with_manager(manager, None)
            }
        }
    }

    fn new_cache_with_manager(
        &self,
        manager: CacheResidencyManager,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Cache, Exception> {
        Cache::new_paged(&self.attention_schedule, manager, rank)
    }

    pub(crate) fn forward(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if cache.layers.is_empty() {
            *cache = self.new_cache();
        }
        self.validate_cache(cache)?;
        let mut hidden = self.embed_tokens.forward(inputs, stream)?;
        let length = hidden.dim(1);
        for ((layer, layer_cache), policy) in self
            .layers
            .iter_mut()
            .zip(cache.layers.iter_mut())
            .zip(self.attention_schedule.iter())
        {
            let offset = layer_cache.offset();
            let mask = attention_mask(policy, length, offset, stream)?;
            hidden = layer.forward(&hidden, mask.as_ref(), layer_cache, stream)?;
        }
        self.norm.forward(&hidden, stream)
    }

    fn validate_cache(&self, cache: &Cache) -> Result<(), Exception> {
        if cache.layers.len() != self.attention_schedule.len() {
            return Err(Exception::custom(format!(
                "GPT-OSS cache has {} layers, expected {}",
                cache.layers.len(),
                self.attention_schedule.len()
            )));
        }
        for (index, (cache, policy)) in cache
            .layers
            .iter()
            .zip(self.attention_schedule.iter())
            .enumerate()
        {
            let actual = cache.attention_policy()?;
            if actual != *policy {
                return Err(Exception::custom(format!(
                    "GPT-OSS cache policy mismatch at layer {index}: expected {policy:?}, got {actual:?}"
                )));
            }
        }
        Ok(())
    }
}

/// GPT-OSS causal language model.
#[derive(Debug, Clone, ModuleParameters)]
pub struct Model {
    /// Model configuration.
    pub args: ModelArgs,
    #[param]
    /// Transformer body.
    pub model: GptOssModel,
    #[param]
    /// Untied output projection.
    pub lm_head: MaybeQuantized<nn::Linear>,
}

impl Model {
    /// Creates an unloaded GPT-OSS model with the native checkpoint parameter tree.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            model: GptOssModel::new(&args, stream)?,
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

    /// Returns the model architecture identifier.
    pub fn model_type(&self) -> &str {
        &self.args.model_type
    }

    /// Creates caches matching the canonical per-layer attention schedule.
    pub fn new_cache(&self) -> Cache {
        self.model.new_cache()
    }

    /// Creates scheduled attention caches under an explicit cache policy.
    pub fn new_cache_with_options(&self, policy: CacheResidencyPolicy) -> Result<Cache, Exception> {
        self.model.new_cache_with_options(policy)
    }

    pub(crate) fn new_cache_with_manager(
        &self,
        manager: CacheResidencyManager,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Cache, Exception> {
        self.model.new_cache_with_manager(manager, rank)
    }

    /// Lazily catalogs a compatible persisted scheduled-attention prefix.
    pub fn load_prompt_cache(
        &self,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
    ) -> Result<(Cache, PromptCacheManifest), Exception> {
        let layer_count = usize::try_from(self.args.num_hidden_layers)
            .map_err(|_| Exception::custom("invalid GPT-OSS cache layer count"))?;
        let identity = PromptCacheModelIdentity {
            model_family: "gpt_oss".into(),
            effective_model_type: self.args.model_type.clone(),
            architecture_fingerprint: prompt_cache_architecture_fingerprint(&self.args),
            layer_count,
            global_layer_start: 0,
            global_layer_end: layer_count,
            sink_tokens: 0,
            layer_prefix_offsets: vec![0; layer_count],
            topology: Default::default(),
            layer_layout: PromptCacheModelIdentity::key_value_layouts(
                self.args
                    .attention_schedule
                    .iter()
                    .map(|policy| policy.window().map(|window| window.get() as i32)),
                self.args.num_key_value_heads,
                self.args.head_dim,
            )
            .map_err(|error| Exception::custom(error.to_string()))?,
        };
        validate_prompt_cache_model_identity(expected, &identity)
            .map_err(|error| Exception::custom(error.to_string()))?;
        let (manager, manifest) =
            open_prompt_cache(directory, expected, &identity, prefix_token_ids, options)
                .map_err(|error| Exception::custom(error.to_string()))?;
        Ok((self.new_cache_with_manager(manager, None)?, manifest))
    }

    pub(crate) fn forward(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let hidden = self.model.forward(inputs, cache, stream)?;
        self.lm_head.forward(&hidden, stream)
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
        self.forward(&tokens, cache, stream)?
            .try_index_device((.., -1, ..), stream)
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.forward(input_tokens, cache, stream)?
            .try_index_device((.., -1, ..), stream)
    }
}

/// GPT-OSS token generation iterator.
pub type Generate<'a, S = crate::backend::mlx::runtime::generation::sampler::DefaultSampler> =
    common::generation::Generate<'a, Model, Cache, S>;

pub(crate) struct PreparedGptOssGguf {
    pub(crate) args: ModelArgs,
    pub(crate) eos_token_ids: Vec<u32>,
}

pub(crate) fn prepare_gguf_checkpoint(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    _stream: &Stream,
) -> Result<PreparedGptOssGguf, Error> {
    let architecture = gguf_string(metadata, "general.architecture")?;
    if architecture != "gpt-oss" {
        return Err(Error::UnsupportedArchitecture(format!(
            "GGUF architecture {architecture:?}; this loader supports gpt-oss"
        )));
    }
    crate::composition::mlx::structural::validate_gguf(
        crate::core::GgufArchitecture::GptOss,
        checkpoint,
        metadata,
        crate::backend::mlx::ModelLoadOptions::default(),
    )
    .into_loader_result()?;
    checkpoint
        .catalog()
        .translated_outputs(translate_gguf_weight_name)
        .map_err(safemlx::error::IoError::from)?;
    let mut configs = gguf_quantization_configs(checkpoint, translate_gguf_weight_name)?;
    configs.retain(|name, _| !name.contains(".mlp.experts."));
    let mut args = args_from_gguf_catalog(checkpoint, metadata)?;
    args.quantized_weight_configs = Some(configs);
    args.validate()?;
    Ok(PreparedGptOssGguf {
        args,
        eos_token_ids: crate::backend::mlx::gguf_eos_token_ids(metadata)?,
    })
}

pub(crate) fn args_from_gguf_catalog(
    _arrays: &impl GgufTensorNames,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> Result<ModelArgs, Error> {
    let key = |suffix: &str| format!("gpt-oss.{suffix}");
    let hidden_size = gguf_required_i32(metadata, &key("embedding_length"))?;
    let num_attention_heads = gguf_required_i32(metadata, &key("attention.head_count"))?;
    if num_attention_heads <= 0 {
        return Err(Error::UnsupportedArchitecture(
            "GPT-OSS GGUF attention head count must be positive".into(),
        ));
    }
    let head_dim = gguf_optional_i32(metadata, &key("attention.key_length"))?
        .unwrap_or(hidden_size / num_attention_heads);
    let vocab_size = gguf_vocab_size(metadata, &key("vocab_size"))?;
    let rope_scaling = gguf_rope_scaling(metadata, "gpt-oss")?;
    let num_hidden_layers = gguf_required_i32(metadata, &key("block_count"))?;
    let sliding_window = gguf_required_i32(metadata, &key("attention.sliding_window"))?;
    let layers = usize::try_from(num_hidden_layers).map_err(|_| {
        Error::UnsupportedArchitecture(format!(
            "GPT-OSS GGUF block count must be positive, got {num_hidden_layers}"
        ))
    })?;
    if sliding_window <= 0 {
        return Err(Error::UnsupportedArchitecture(format!(
            "GPT-OSS GGUF sliding window must be positive, got {sliding_window}"
        )));
    }
    let args = ModelArgs {
        model_type: "gpt_oss".into(),
        hidden_size,
        intermediate_size: gguf_required_i32(metadata, &key("expert_feed_forward_length"))?,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads: gguf_required_i32(metadata, &key("attention.head_count_kv"))?,
        head_dim,
        vocab_size,
        num_local_experts: gguf_required_i32(metadata, &key("expert_count"))?,
        num_experts_per_tok: gguf_required_i32(metadata, &key("expert_used_count"))?,
        rms_norm_eps: gguf_f32(metadata, &key("attention.layer_norm_rms_epsilon"))?,
        max_position_embeddings: gguf_required_i32(metadata, &key("context_length"))?,
        rope_theta: gguf_optional_f32(metadata, &key("rope.freq_base"))?
            .unwrap_or_else(default_rope_theta),
        rope_scaling,
        attention_schedule: alternating_attention_schedule(layers, sliding_window as u32)?,
        quantization_config: MxFp4Config {
            quant_method: "mxfp4".into(),
        },
        quantization: None,
        quantized_weight_configs: None,
        swiglu_limit: gguf_optional_f32(metadata, &key("swiglu_clamp_exp"))?
            .unwrap_or_else(default_swiglu_limit),
    };
    args.validate()?;
    Ok(args)
}

fn gguf_vocab_size(
    metadata: &HashMap<String, GgufMetadataValue>,
    fallback: &str,
) -> Result<i32, Error> {
    match metadata
        .get("tokenizer.ggml.tokens")
        .and_then(GgufMetadataValue::as_strings)
    {
        Some(tokens) => i32::try_from(tokens.len()).map_err(|_| {
            Error::UnsupportedArchitecture("GGUF tokenizer vocabulary exceeds i32".into())
        }),
        None if metadata.contains_key("tokenizer.ggml.tokens") => {
            Err(Error::UnsupportedArchitecture(
                "GGUF tokenizer.ggml.tokens metadata has the wrong type".into(),
            ))
        }
        None => gguf_required_i32(metadata, fallback),
    }
}

fn gguf_required_i32(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<i32, Error> {
    gguf_optional_i32(metadata, key)?.ok_or_else(|| {
        Error::UnsupportedArchitecture(format!("GGUF metadata is missing required key {key:?}"))
    })
}

fn gguf_optional_i32(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<Option<i32>, Error> {
    metadata
        .get(key)
        .map(|value| {
            value
                .as_i64()
                .and_then(|value| i32::try_from(value).ok())
                .ok_or_else(|| {
                    Error::UnsupportedArchitecture(format!(
                        "GGUF metadata key {key:?} must be an i32 scalar"
                    ))
                })
        })
        .transpose()
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
    metadata
        .get(key)
        .map(|value| {
            value.as_f32().ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "GGUF metadata key {key:?} must be a numeric scalar"
                ))
            })
        })
        .transpose()
}

fn gguf_optional_string(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<Option<String>, Error> {
    match metadata.get(key) {
        Some(GgufMetadataValue::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(Error::UnsupportedArchitecture(format!(
            "GGUF metadata key {key:?} must be a string"
        ))),
        None => Ok(None),
    }
}

fn gguf_rope_scaling(
    metadata: &HashMap<String, GgufMetadataValue>,
    architecture: &str,
) -> Result<Option<HashMap<String, RopeValue>>, Error> {
    let key = |suffix: &str| format!("{architecture}.rope.scaling.{suffix}");
    let Some(kind) = gguf_optional_string(metadata, &key("type"))? else {
        return Ok(None);
    };
    if matches!(kind.as_str(), "none" | "default") {
        return Ok(None);
    }
    if kind != "yarn" {
        return Err(Error::UnsupportedArchitecture(format!(
            "GPT-OSS GGUF RoPE scaling type {kind:?} is unsupported"
        )));
    }
    Ok(Some(HashMap::from([
        ("rope_type".into(), RopeValue::String("yarn".into())),
        (
            "factor".into(),
            RopeValue::Float(gguf_f32(metadata, &key("factor"))?),
        ),
        (
            "original_max_position_embeddings".into(),
            RopeValue::Float(gguf_f32(metadata, &key("original_context_length"))?),
        ),
        (
            "beta_fast".into(),
            RopeValue::Float(gguf_optional_f32(metadata, &key("yarn_beta_fast"))?.unwrap_or(32.0)),
        ),
        (
            "beta_slow".into(),
            RopeValue::Float(gguf_optional_f32(metadata, &key("yarn_beta_slow"))?.unwrap_or(1.0)),
        ),
        ("truncate".into(), RopeValue::Bool(false)),
    ])))
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
        ("attn_norm", "input_layernorm"),
        ("attn_post_norm", "post_attention_layernorm"),
        ("attn_q", "self_attn.q_proj"),
        ("attn_k", "self_attn.k_proj"),
        ("attn_v", "self_attn.v_proj"),
        ("attn_output", "self_attn.o_proj"),
        ("ffn_gate_inp", "mlp.router"),
        ("ffn_gate_exps", "mlp.experts.gate_proj"),
        ("ffn_up_exps", "mlp.experts.up_proj"),
        ("ffn_down_exps", "mlp.experts.down_proj"),
    ] {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            return format!(
                "model.layers.{layer}.{}",
                parameter.replacen(source, target, 1)
            );
        }
    }
    if parameter == "attn_sinks.weight" || parameter == "attn_sinks" {
        return format!("model.layers.{layer}.self_attn.sinks");
    }
    name.to_string()
}

/// Reads GPT-OSS model arguments from `config.json`.
pub fn get_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let config =
        serde_json::from_reader(std::fs::File::open(model_dir.as_ref().join("config.json"))?)?;
    model_args_from_config_value(&config)
}

/// Loads `tokenizer.json` from a GPT-OSS model directory.
pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    Tokenizer::from_file(model_dir.as_ref().join("tokenizer.json")).map_err(Into::into)
}
