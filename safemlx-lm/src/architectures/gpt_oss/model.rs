//! OpenAI GPT-OSS decoder-only mixture-of-experts implementation.

use std::{collections::HashMap, path::Path};

use safemlx::{
    error::Exception,
    fast::ScaledDotProductAttentionMask,
    macros::ModuleParameters,
    module::{Module, ModuleParametersExt, Param},
    nn,
    ops::{
        arange, clip, gather_grouped_rows, gather_qmm_with_mode,
        indexing::{IntoStrideBy, TryIndexOp},
        sigmoid, stack_axis, topk_route_plan, GgufCheckpoint, GgufMetadataValue, QuantizationMode,
    },
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::{
    api::{common, common::generation::CausalLm, input},
    architectures::qwen::dense::gguf_string,
    error::Error,
    nn::tensor::{
        create_causal_mask,
        rope::{initialize_rope, FloatOrString, RopeVariant},
    },
    runtime::cache::residency::{
        derive_prompt_cache_architecture_fingerprint, open_prompt_cache,
        validate_prompt_cache_model_identity, CacheRankIdentity, CacheResidencyManager,
        CacheResidencyPolicy, CacheResidencyReport, PagedCacheOptions, PromptCacheDescriptor,
        PromptCacheManifest, PromptCacheModelIdentity,
    },
    runtime::checkpoint::load::{
        gguf_metadata, gguf_quantization_configs, load_named_array_strict,
        load_safetensors_dir_quantized_strict, load_safetensors_dir_strict, GgufTensorNames,
        StrictLoadConfig, StrictLoadReport,
    },
    runtime::checkpoint::quantization::WeightQuantization,
    runtime::{
        attention::{AttentionPolicy, LayerSchedule},
        cache::{ConcatKeyValueCache, KeyValueCache, PagedKeyValueCache},
    },
};

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
    rope_scaling: Option<HashMap<String, FloatOrString>>,
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
    pub rope_scaling: Option<HashMap<String, FloatOrString>>,
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
                        FloatOrString::Float(value) => format!("f32:{:08x}", value.to_bits()),
                        FloatOrString::String(value) => format!("string:{value}"),
                        FloatOrString::Bool(value) => format!("bool:{value}"),
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

    fn forward_tensor_parallel(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut LayerCache,
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

    pub(crate) fn forward_tensor_parallel(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut LayerCache,
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

    pub(crate) fn forward_with_expert_executor<F>(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut LayerCache,
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
        descriptor: crate::runtime::cache::residency::PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &crate::runtime::cache::residency::PromptCacheOptions,
    ) -> Result<crate::runtime::cache::residency::PromptCacheManifest, Exception> {
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

    pub(crate) fn forward_with_expert_executor<F>(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Exception>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        if cache.layers.is_empty() {
            *cache = self.new_cache();
        }
        self.validate_cache(cache)?;
        let mut hidden = self.embed_tokens.forward(inputs, stream)?;
        let length = hidden.dim(1);
        for (index, ((layer, layer_cache), policy)) in self
            .layers
            .iter_mut()
            .zip(cache.layers.iter_mut())
            .zip(self.attention_schedule.iter())
            .enumerate()
        {
            let offset = layer_cache.offset();
            let mask = attention_mask(policy, length, offset, stream)?;
            hidden = layer.forward_with_expert_executor(
                &hidden,
                mask.as_ref(),
                layer_cache,
                stream,
                |flat, ids, weights, stream| execute(index, flat, ids, weights, stream),
            )?;
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
        self.lm_head.forward(&hidden, stream)
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
pub type Generate<'a, S = crate::runtime::generation::sampler::DefaultSampler> =
    common::generation::Generate<'a, Model, Cache, S>;

pub(crate) struct LoadedGptOssGguf {
    pub(crate) model: Model,
    pub(crate) eos_token_ids: Vec<u32>,
}

pub(crate) struct PreparedGptOssGguf {
    pub(crate) args: ModelArgs,
    pub(crate) eos_token_ids: Vec<u32>,
}

/// Loads a canonical llama.cpp `gpt-oss` GGUF checkpoint.
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
) -> Result<LoadedGptOssGguf, Error> {
    let mut prepared = prepare_gguf_checkpoint(checkpoint, &metadata, weights_stream)?;
    if let Some(quantization) = quantization {
        if quantization != WeightQuantization::MxFp4 {
            return Err(Error::Quantization(
                "GPT-OSS GGUF load-time quantization only supports MXFP4 dense projections".into(),
            ));
        }
        prepared.args.quantization = Some(quantization);
        prepared.args.quantized_weight_configs = None;
    }
    let mut model = Model::new(prepared.args, stream)?;
    let config = StrictLoadConfig::default();
    let mut report = StrictLoadReport::default();
    let mut materializer = checkpoint.materializer();

    for tensor in checkpoint.catalog().tensors() {
        let physical = &tensor.descriptor().name;
        if physical.contains("ffn_gate_exps")
            || physical.contains("ffn_up_exps")
            || physical.contains("ffn_down_exps")
        {
            continue;
        }
        for (name, value) in materializer.converted_tensor(physical)?.into_arrays() {
            load_named_array_strict(
                &mut model,
                translate_gguf_weight_name(&name),
                value,
                quantization.map(|value| (value, stream)),
                &config,
                &mut report,
            )?;
        }
    }

    for layer in 0..model.args.num_hidden_layers as usize {
        let source = format!("blk.{layer}");
        let target = format!("model.layers.{layer}.mlp.experts");
        let converted = |materializer: &mut safemlx::ops::GgufMaterializer,
                         physical: String|
         -> Result<HashMap<String, Array>, Error> {
            Ok(materializer
                .converted_tensor(&physical)?
                .into_arrays()
                .into_iter()
                .collect())
        };
        let gate_physical = format!("{source}.ffn_gate_exps.weight");
        let up_physical = format!("{source}.ffn_up_exps.weight");
        let down_physical = format!("{source}.ffn_down_exps.weight");
        let gate = converted(&mut materializer, gate_physical.clone())?;
        let up = converted(&mut materializer, up_physical.clone())?;
        let down = converted(&mut materializer, down_physical.clone())?;
        let get = |arrays: &HashMap<String, Array>, name: String| {
            arrays.get(&name).cloned().ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "GPT-OSS GGUF is missing converted tensor {name:?}"
                ))
            })
        };
        let gate_weight = get(&gate, gate_physical.clone())?;
        let up_weight = get(&up, up_physical.clone())?;
        let gate_scales = get(&gate, gate_physical.replace(".weight", ".scales"))?;
        let up_scales = get(&up, up_physical.replace(".weight", ".scales"))?;
        let experts = model.args.num_local_experts;
        let intermediate = model.args.intermediate_size;
        let hidden = model.args.hidden_size;
        let gate_up_weight = stack_axis(&[gate_weight, up_weight], 2, weights_stream)?
            .reshape(&[experts, 2 * intermediate, hidden / 8], weights_stream)?
            .view::<u8>(weights_stream)?
            .reshape(
                &[experts, 2 * intermediate, hidden / 32, 16],
                weights_stream,
            )?;
        let gate_up_scales = stack_axis(&[gate_scales, up_scales], 2, weights_stream)?
            .reshape(&[experts, 2 * intermediate, hidden / 32], weights_stream)?;
        let down_weight = get(&down, down_physical.clone())?
            .view::<u8>(weights_stream)?
            .reshape(&[experts, hidden, intermediate / 32, 16], weights_stream)?;
        let down_scales = get(&down, down_physical.replace(".weight", ".scales"))?;

        for (name, value) in [
            (format!("{target}.gate_up_proj_blocks"), gate_up_weight),
            (format!("{target}.gate_up_proj_scales"), gate_up_scales),
            (format!("{target}.down_proj_blocks"), down_weight),
            (format!("{target}.down_proj_scales"), down_scales),
        ] {
            load_named_array_strict(&mut model, name, value, None, &config, &mut report)?;
        }

        let gate_bias_name = format!("{source}.ffn_gate_exps.bias");
        let up_bias_name = format!("{source}.ffn_up_exps.bias");
        let down_bias_name = format!("{source}.ffn_down_exps.bias");
        let gate_bias = materializer
            .converted_tensor(&gate_bias_name)?
            .into_arrays()
            .into_iter()
            .next()
            .map(|(_, value)| value)
            .ok_or_else(|| Error::UnsupportedArchitecture("empty GPT-OSS gate bias".into()))?;
        let up_bias = materializer
            .converted_tensor(&up_bias_name)?
            .into_arrays()
            .into_iter()
            .next()
            .map(|(_, value)| value)
            .ok_or_else(|| Error::UnsupportedArchitecture("empty GPT-OSS up bias".into()))?;
        let gate_up_bias = stack_axis(&[gate_bias, up_bias], 2, weights_stream)?
            .reshape(&[experts, 2 * intermediate], weights_stream)?;
        let down_bias = materializer
            .converted_tensor(&down_bias_name)?
            .into_arrays()
            .into_iter()
            .next()
            .map(|(_, value)| value)
            .ok_or_else(|| Error::UnsupportedArchitecture("empty GPT-OSS down bias".into()))?;
        for (name, value) in [
            (format!("{target}.gate_up_proj_bias"), gate_up_bias),
            (format!("{target}.down_proj_bias"), down_bias),
        ] {
            load_named_array_strict(&mut model, name, value, None, &config, &mut report)?;
        }
    }
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;
    Ok(LoadedGptOssGguf {
        model,
        eos_token_ids: prepared.eos_token_ids,
    })
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
    crate::api::structural::validate_gguf(
        crate::api::GgufArchitecture::GptOss,
        checkpoint,
        metadata,
        crate::api::ModelLoadOptions::default(),
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
        eos_token_ids: crate::api::gguf_eos_token_ids(metadata)?,
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
) -> Result<Option<HashMap<String, FloatOrString>>, Error> {
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
        ("rope_type".into(), FloatOrString::String("yarn".into())),
        (
            "factor".into(),
            FloatOrString::Float(gguf_f32(metadata, &key("factor"))?),
        ),
        (
            "original_max_position_embeddings".into(),
            FloatOrString::Float(gguf_f32(metadata, &key("original_context_length"))?),
        ),
        (
            "beta_fast".into(),
            FloatOrString::Float(
                gguf_optional_f32(metadata, &key("yarn_beta_fast"))?.unwrap_or(32.0),
            ),
        ),
        (
            "beta_slow".into(),
            FloatOrString::Float(
                gguf_optional_f32(metadata, &key("yarn_beta_slow"))?.unwrap_or(1.0),
            ),
        ),
        ("truncate".into(), FloatOrString::Bool(false)),
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

/// Loads a GPT-OSS safetensors checkpoint strictly, without rewriting keys.
pub fn load_model(
    model_dir: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    crate::api::structural::validate_safetensors_load_path(
        crate::api::ModelKind::GptOss,
        model_dir,
        crate::api::ModelLoadOptions::default(),
    )?;
    let mut model = Model::new(get_model_args(model_dir)?, stream)?;
    let config = StrictLoadConfig::default();
    let mut report = StrictLoadReport::default();
    load_safetensors_dir_strict(&mut model, model_dir, weights_stream, &config, &mut report)?;
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;
    Ok(model)
}

/// Loads GPT-OSS while MXFP4-quantizing eligible dense matrices one at a time.
///
/// Checkpoint-native routed experts are loaded unchanged. The router remains
/// dense; attention projections, token embeddings, and the LM head use the
/// requested MXFP4 encoding.
pub fn load_model_quantized(
    model_dir: impl AsRef<Path>,
    quantization: WeightQuantization,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    if quantization != WeightQuantization::MxFp4 {
        return Err(Error::Quantization(
            "GPT-OSS native MXFP4 experts cannot be implicitly dequantized and requantized to affine"
                .into(),
        ));
    }
    let model_dir = model_dir.as_ref();
    crate::api::structural::validate_safetensors_load_path(
        crate::api::ModelKind::GptOss,
        model_dir,
        crate::api::ModelLoadOptions::with_quantization(quantization),
    )?;
    let mut args = get_model_args(model_dir)?;
    if !crate::runtime::checkpoint::quantization::should_quantize_on_load(
        "GPT-OSS dense matrices",
        args.quantization,
        quantization,
    )? {
        return load_model(model_dir, stream, weights_stream);
    }
    args.quantization = Some(quantization);
    let mut model = Model::new(args, stream)?;
    let config = StrictLoadConfig::default();
    let mut report = StrictLoadReport::default();
    load_safetensors_dir_quantized_strict(
        &mut model,
        model_dir,
        weights_stream,
        stream,
        quantization,
        &config,
        &mut report,
    )?;
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;
    Ok(model)
}

/// Loads `tokenizer.json` from a GPT-OSS model directory.
pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    Tokenizer::from_file(model_dir.as_ref().join("tokenizer.json")).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use safemlx::{
        module::ModuleParameters,
        ops::{ones_dtype, zeros_dtype, GgufMetadataValue},
        Array, Device, DeviceType, ExecutionContext,
    };

    use super::{Cache, Model, ModelArgs, MxFp4Config};
    use crate::{
        nn::rope::FloatOrString,
        runtime::{
            attention::{AttentionPolicy, LayerSchedule},
            cache::KeyValueCache,
        },
    };

    fn tiny_args() -> ModelArgs {
        ModelArgs {
            model_type: "gpt_oss".into(),
            hidden_size: 32,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 32,
            vocab_size: 32,
            num_local_experts: 2,
            num_experts_per_tok: 1,
            rms_norm_eps: 1e-5,
            max_position_embeddings: 128,
            rope_theta: 150_000.0,
            rope_scaling: Some(HashMap::from([
                ("rope_type".into(), FloatOrString::String("yarn".into())),
                ("factor".into(), FloatOrString::Float(2.0)),
                (
                    "original_max_position_embeddings".into(),
                    FloatOrString::Float(64.0),
                ),
                ("beta_fast".into(), FloatOrString::Float(32.0)),
                ("beta_slow".into(), FloatOrString::Float(1.0)),
                ("truncate".into(), FloatOrString::Bool(false)),
            ])),
            attention_schedule: LayerSchedule::new(
                2,
                vec![AttentionPolicy::sliding(8).unwrap(), AttentionPolicy::Full],
            )
            .unwrap(),
            quantization_config: MxFp4Config {
                quant_method: "mxfp4".into(),
            },
            quantization: None,
            quantized_weight_configs: None,
            swiglu_limit: 7.0,
        }
    }

    fn initialize_zero_model(model: &mut Model, stream: &safemlx::Stream) {
        for (name, parameter) in model.parameters_mut().flatten() {
            let shape = parameter.shape().to_vec();
            let dtype = parameter.dtype();
            *parameter = if name.ends_with("_scales") {
                Array::full::<u8>(&shape, Array::from_slice(&[127u8], &[]), stream).unwrap()
            } else if name.ends_with("layernorm.weight") || name.as_ref() == "model.norm.weight" {
                ones_dtype(&shape, dtype, stream).unwrap()
            } else {
                zeros_dtype(&shape, dtype, stream).unwrap()
            };
        }
    }

    #[test]
    fn gpt_oss_fingerprint_preserves_arbitrary_distinct_windows() {
        let mut args = tiny_args();
        args.num_hidden_layers = 4;
        args.attention_schedule = LayerSchedule::new(
            4,
            vec![
                AttentionPolicy::sliding(3).unwrap(),
                AttentionPolicy::Full,
                AttentionPolicy::sliding(9).unwrap(),
                AttentionPolicy::Full,
            ],
        )
        .unwrap();
        args.validate().unwrap();
        assert_eq!(
            args.attention_schedule
                .iter()
                .map(|policy| policy.window().map(|window| window.get() as i32))
                .collect::<Vec<_>>(),
            vec![Some(3), None, Some(9), None]
        );
        let fingerprint = super::prompt_cache_architecture_fingerprint(&args);
        args.attention_schedule = LayerSchedule::new(
            4,
            vec![
                AttentionPolicy::Full,
                AttentionPolicy::sliding(3).unwrap(),
                AttentionPolicy::sliding(9).unwrap(),
                AttentionPolicy::Full,
            ],
        )
        .unwrap();
        assert_ne!(
            fingerprint,
            super::prompt_cache_architecture_fingerprint(&args)
        );
    }

    #[test]
    fn gguf_names_translate_to_native_parameter_tree() {
        assert_eq!(
            super::translate_gguf_weight_name("blk.3.attn_sinks.weight"),
            "model.layers.3.self_attn.sinks"
        );
        assert_eq!(
            super::translate_gguf_weight_name("blk.3.ffn_gate_inp.bias"),
            "model.layers.3.mlp.router.bias"
        );
        assert_eq!(
            super::translate_gguf_weight_name("blk.3.ffn_gate_exps.scales"),
            "model.layers.3.mlp.experts.gate_proj.scales"
        );
    }

    #[test]
    fn canonical_mxfp4_gguf_loads() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let args = tiny_args();
        let mut arrays = HashMap::new();
        let mut insert = |name: &str, shape: &[i32]| {
            arrays.insert(
                name.to_string(),
                Array::zeros::<f32>(shape, stream).unwrap(),
            );
        };
        insert("token_embd.weight", &[args.vocab_size, args.hidden_size]);
        insert("output_norm.weight", &[args.hidden_size]);
        insert("output.weight", &[args.vocab_size, args.hidden_size]);
        for layer in 0..args.num_hidden_layers {
            let prefix = format!("blk.{layer}");
            insert(&format!("{prefix}.attn_norm.weight"), &[args.hidden_size]);
            insert(
                &format!("{prefix}.attn_post_norm.weight"),
                &[args.hidden_size],
            );
            for (name, output) in [
                ("attn_q", args.num_attention_heads * args.head_dim),
                ("attn_k", args.num_key_value_heads * args.head_dim),
                ("attn_v", args.num_key_value_heads * args.head_dim),
            ] {
                insert(
                    &format!("{prefix}.{name}.weight"),
                    &[output, args.hidden_size],
                );
                insert(&format!("{prefix}.{name}.bias"), &[output]);
            }
            insert(
                &format!("{prefix}.attn_output.weight"),
                &[args.hidden_size, args.num_attention_heads * args.head_dim],
            );
            insert(&format!("{prefix}.attn_output.bias"), &[args.hidden_size]);
            insert(
                &format!("{prefix}.attn_sinks.weight"),
                &[args.num_attention_heads],
            );
            insert(
                &format!("{prefix}.ffn_gate_inp.weight"),
                &[args.num_local_experts, args.hidden_size],
            );
            insert(
                &format!("{prefix}.ffn_gate_inp.bias"),
                &[args.num_local_experts],
            );
            for name in ["gate", "up"] {
                insert(
                    &format!("{prefix}.ffn_{name}_exps.weight"),
                    &[
                        args.num_local_experts,
                        args.intermediate_size,
                        args.hidden_size,
                    ],
                );
                insert(
                    &format!("{prefix}.ffn_{name}_exps.bias"),
                    &[args.num_local_experts, args.intermediate_size],
                );
            }
            insert(
                &format!("{prefix}.ffn_down_exps.weight"),
                &[
                    args.num_local_experts,
                    args.hidden_size,
                    args.intermediate_size,
                ],
            );
            insert(
                &format!("{prefix}.ffn_down_exps.bias"),
                &[args.num_local_experts, args.hidden_size],
            );
        }
        let metadata = HashMap::from([
            (
                "general.architecture".into(),
                GgufMetadataValue::String("gpt-oss".into()),
            ),
            (
                "gpt-oss.embedding_length".into(),
                GgufMetadataValue::Uint32(args.hidden_size as u32),
            ),
            (
                "gpt-oss.block_count".into(),
                GgufMetadataValue::Uint32(args.num_hidden_layers as u32),
            ),
            (
                "gpt-oss.expert_feed_forward_length".into(),
                GgufMetadataValue::Uint32(args.intermediate_size as u32),
            ),
            (
                "gpt-oss.attention.head_count".into(),
                GgufMetadataValue::Uint32(args.num_attention_heads as u32),
            ),
            (
                "gpt-oss.attention.head_count_kv".into(),
                GgufMetadataValue::Uint32(args.num_key_value_heads as u32),
            ),
            (
                "gpt-oss.attention.key_length".into(),
                GgufMetadataValue::Uint32(args.head_dim as u32),
            ),
            (
                "gpt-oss.attention.layer_norm_rms_epsilon".into(),
                GgufMetadataValue::Float32(args.rms_norm_eps),
            ),
            (
                "gpt-oss.attention.sliding_window".into(),
                GgufMetadataValue::Uint32(
                    args.attention_schedule
                        .get(0)
                        .and_then(|policy| policy.window())
                        .unwrap()
                        .get(),
                ),
            ),
            (
                "gpt-oss.context_length".into(),
                GgufMetadataValue::Uint32(args.max_position_embeddings as u32),
            ),
            (
                "gpt-oss.rope.freq_base".into(),
                GgufMetadataValue::Float32(args.rope_theta),
            ),
            (
                "gpt-oss.expert_count".into(),
                GgufMetadataValue::Uint32(args.num_local_experts as u32),
            ),
            (
                "gpt-oss.expert_used_count".into(),
                GgufMetadataValue::Uint32(args.num_experts_per_tok as u32),
            ),
            (
                "gpt-oss.vocab_size".into(),
                GgufMetadataValue::Uint32(args.vocab_size as u32),
            ),
        ]);
        let fixture =
            crate::test_utils::SyntheticGguf::with_packed_tensors(&arrays, &metadata, |name, _| {
                name.contains("_exps.weight")
                    .then_some(safemlx_gguf::GgmlType::MxFp4)
            });
        let loaded = super::load_gguf(fixture.path(), stream, stream).unwrap();
        assert_eq!(loaded.args.num_hidden_layers, args.num_hidden_layers);
        assert_eq!(loaded.args.num_local_experts, args.num_local_experts);
        assert_eq!(
            loaded.args.attention_schedule.fingerprint_component(),
            "s8,f"
        );
    }

    #[test]
    fn published_config_shape_is_accepted() {
        let value = serde_json::json!({
            "model_type": "gpt_oss",
            "hidden_size": 2880,
            "intermediate_size": 2880,
            "num_hidden_layers": 24,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "vocab_size": 201088,
            "num_local_experts": 32,
            "num_experts_per_tok": 4,
            "rms_norm_eps": 1e-5,
            "sliding_window": 128,
            "max_position_embeddings": 131072,
            "rope_theta": 150000,
            "rope_scaling": {
                "rope_type": "yarn", "factor": 32.0,
                "original_max_position_embeddings": 4096,
                "beta_fast": 32.0, "beta_slow": 1.0
            },
            "layer_types": std::iter::repeat_n(["sliding_attention", "full_attention"], 12).flatten().collect::<Vec<_>>(),
            "quantization_config": {"quant_method": "mxfp4"}
        });
        let args = super::model_args_from_config_value(&value).unwrap();
        assert_eq!(args.attention_schedule.len(), 24);
        assert_eq!(args.attention_schedule.full_layer_count(), 12);
        assert_eq!(args.attention_schedule.sliding_layer_count(), 12);
    }

    fn schedule_config(layer_types: serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "model_type": "gpt_oss",
            "hidden_size": 32,
            "intermediate_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "vocab_size": 32,
            "num_local_experts": 2,
            "num_experts_per_tok": 1,
            "rms_norm_eps": 1e-5,
            "sliding_window": 8,
            "max_position_embeddings": 128,
            "rope_theta": 150000,
            "rope_scaling": null,
            "layer_types": layer_types,
            "quantization_config": {"quant_method": "mxfp4"}
        })
    }

    #[test]
    fn hf_schedule_normalization_preserves_fallback_and_arbitrary_order() {
        let mut fallback = schedule_config(serde_json::json!([]));
        fallback.as_object_mut().unwrap().remove("layer_types");
        let fallback = super::model_args_from_config_value(&fallback).unwrap();
        assert_eq!(
            fallback.attention_schedule.fingerprint_component(),
            "s8,f,s8,f"
        );

        let explicit = super::model_args_from_config_value(&schedule_config(serde_json::json!([
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention"
        ])))
        .unwrap();
        assert_eq!(
            explicit.attention_schedule.fingerprint_component(),
            "f,s8,s8,f"
        );
    }

    #[test]
    fn hf_schedule_normalization_rejects_invalid_metadata_exactly() {
        let invalid = [
            schedule_config(serde_json::json!(["full_attention"])),
            schedule_config(serde_json::json!([
                "full_attention",
                "invalid_attention",
                "full_attention",
                "full_attention"
            ])),
        ];
        for config in invalid {
            assert!(super::model_args_from_config_value(&config).is_err());
        }
        for window in [
            serde_json::json!(0),
            serde_json::json!(-1),
            serde_json::json!(i64::from(i32::MAX) + 1),
        ] {
            let mut config = schedule_config(serde_json::json!([
                "full_attention",
                "sliding_attention",
                "full_attention",
                "full_attention"
            ]));
            config["sliding_window"] = window;
            assert!(super::model_args_from_config_value(&config).is_err());
        }
    }

    #[test]
    fn fingerprint_includes_the_complete_ordered_attention_schedule() {
        let first = super::model_args_from_config_value(&schedule_config(serde_json::json!([
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention"
        ])))
        .unwrap();
        let second = super::model_args_from_config_value(&schedule_config(serde_json::json!([
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention"
        ])))
        .unwrap();
        assert_ne!(
            super::prompt_cache_architecture_fingerprint(&first),
            super::prompt_cache_architecture_fingerprint(&second)
        );
    }

    #[test]
    fn cache_geometry_supports_arbitrary_order_and_distinct_windows() {
        let ctx = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let mut args = tiny_args();
        args.num_hidden_layers = 4;
        args.attention_schedule = LayerSchedule::new(
            4,
            vec![
                AttentionPolicy::Full,
                AttentionPolicy::sliding(3).unwrap(),
                AttentionPolicy::sliding(5).unwrap(),
                AttentionPolicy::Full,
            ],
        )
        .unwrap();
        let model = Model::new(args, ctx.stream()).unwrap();
        let cache = model.new_cache();
        assert_eq!(
            cache
                .layers
                .iter()
                .map(KeyValueCache::max_size)
                .collect::<Vec<_>>(),
            vec![None, Some(3), Some(5), None]
        );
    }

    #[test]
    fn parameter_tree_matches_native_checkpoint_names() {
        let ctx = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let model = Model::new(tiny_args(), ctx.stream()).unwrap();
        let parameters = model.parameters().flatten();
        for key in [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.sinks",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.bias",
            "model.layers.0.mlp.router.weight",
            "model.layers.0.mlp.router.bias",
            "model.layers.0.mlp.experts.gate_up_proj_blocks",
            "model.layers.0.mlp.experts.gate_up_proj_scales",
            "model.layers.0.mlp.experts.gate_up_proj_bias",
            "model.layers.0.mlp.experts.down_proj_blocks",
            "model.layers.0.mlp.experts.down_proj_scales",
            "model.layers.0.mlp.experts.down_proj_bias",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.norm.weight",
            "lm_head.weight",
        ] {
            assert!(parameters.contains_key(key), "missing parameter key {key}");
        }
    }

    #[test]
    fn zero_weight_forward_exercises_mxfp4_and_mixed_cache() {
        let ctx = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let mut model = Model::new(tiny_args(), stream).unwrap();
        initialize_zero_model(&mut model, stream);
        let tokens = Array::from_slice(&[1i32, 2, 3], &[1, 3]);
        let logits = model
            .forward(&tokens, &mut Cache::default(), stream)
            .unwrap();
        assert_eq!(logits.shape(), &[1, 3, 32]);
        assert_eq!(logits.max(None, stream).unwrap().item::<f32>(stream), 0.0);
    }

    #[test]
    fn arbitrary_schedule_ordinary_and_paged_caches_match_and_retain_exactly() {
        use crate::runtime::cache::residency::{CacheResidencyPolicy, PagedCacheOptions};

        let ctx = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let mut args = tiny_args();
        args.num_hidden_layers = 4;
        args.attention_schedule = LayerSchedule::new(
            4,
            vec![
                AttentionPolicy::Full,
                AttentionPolicy::sliding(4).unwrap(),
                AttentionPolicy::Full,
                AttentionPolicy::sliding(2).unwrap(),
            ],
        )
        .unwrap();
        let mut ordinary = Model::new(args, stream).unwrap();
        initialize_zero_model(&mut ordinary, stream);
        let mut paged = ordinary.clone();
        let mut ordinary_cache = ordinary.new_cache();
        let options = PagedCacheOptions::new(2, 1 << 20, 1 << 20, 1)
            .unwrap()
            .with_full_attention(true);
        let mut paged_cache = paged
            .new_cache_with_options(CacheResidencyPolicy::Paged(options))
            .unwrap();
        for tokens in [
            Array::from_slice(&[1u32, 2, 3], &[1, 3]),
            Array::from_slice(&[4u32, 5], &[1, 2]),
            Array::from_slice(&[6u32], &[1, 1]),
        ] {
            let expected = ordinary
                .forward(&tokens, &mut ordinary_cache, stream)
                .unwrap();
            let actual = paged.forward(&tokens, &mut paged_cache, stream).unwrap();
            assert_eq!(expected.shape(), actual.shape());
            assert_eq!(
                expected.max(None, stream).unwrap().item::<f32>(stream),
                actual.max(None, stream).unwrap().item::<f32>(stream)
            );
        }
        assert_eq!(
            ordinary_cache
                .layers
                .iter()
                .map(|cache| {
                    cache
                        .retained_arrays()
                        .first()
                        .map(|array| array.dim(-2))
                        .unwrap_or(0)
                })
                .collect::<Vec<_>>(),
            vec![6, 3, 6, 1]
        );
    }
}
