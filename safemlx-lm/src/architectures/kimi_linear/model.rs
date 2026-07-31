//! Kimi Linear hybrid KDA/MLA causal language model.

use std::{collections::HashMap, path::Path};

use safemlx::{
    error::Exception,
    macros::ModuleParameters,
    module::{Module, ModuleParameters, ModuleParametersExt, Param},
    nn,
    ops::{
        concatenate_axis, exp, indexing::TryIndexOp, mean_axis, rsqrt, sigmoid, GgufCheckpoint,
        GgufMetadataValue,
    },
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::{
    api::{
        input as runtime_input,
        qwen3_5_moe::{QwenLinear, QwenWeightFormat},
    },
    architectures::deepseek_v3::model::{
        DeepSeekQuantizationConfig, ModelArgs as DeepSeekArgs, MultiHeadLatentAttention,
    },
    error::Error,
    nn::{
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
    runtime::{
        cache::CompressedLatentCache,
        checkpoint::{
            load::{
                gguf_metadata, gguf_quantization_configs, load_named_array_strict,
                load_safetensors_dir_strict_with_split_swiglu_experts_and_transform,
                GgufTensorNames, StrictLoadConfig, StrictLoadReport,
            },
            quantization::WeightQuantization,
        },
        execution::inspection::{ActivationObserver, MoeRoutingObservation},
    },
};

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

/// KDA/full-attention layer layout and recurrent dimensions.
#[derive(Debug, Clone, Deserialize)]
pub struct LinearAttentionConfig {
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

/// Deserialized Kimi Linear text configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
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
    pub linear_attn_config: LinearAttentionConfig,
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

impl ModelArgs {
    /// Returns the layer's configured attention kind.
    pub fn is_kda_layer(&self, layer: usize) -> bool {
        self.linear_attn_config
            .kda_layers
            .contains(&(layer as i32 + 1))
    }

    /// Returns whether this layer uses routed experts.
    pub fn is_moe_layer(&self, layer: usize) -> bool {
        self.num_experts > 0
            && layer as i32 >= self.first_k_dense_replace
            && layer as i32 % self.moe_layer_freq == 0
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
            ("kda.num_heads", self.linear_attn_config.num_heads),
            ("kda.head_dim", self.linear_attn_config.head_dim),
            (
                "kda.short_conv_kernel_size",
                self.linear_attn_config.short_conv_kernel_size,
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
            || self.first_k_dense_replace < 0
            || self.first_k_dense_replace > self.num_hidden_layers
            || self.moe_layer_freq <= 0
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
        let mut seen = vec![false; self.num_hidden_layers as usize];
        for (kind, layers) in [
            ("KDA", &self.linear_attn_config.kda_layers),
            ("MLA", &self.linear_attn_config.full_attn_layers),
        ] {
            for &layer in layers {
                if layer <= 0 || layer > self.num_hidden_layers {
                    return Err(Error::UnsupportedArchitecture(format!(
                        "Kimi Linear {kind} layer index {layer} is outside 1..={}",
                        self.num_hidden_layers
                    )));
                }
                let slot = &mut seen[layer as usize - 1];
                if *slot {
                    return Err(Error::UnsupportedArchitecture(format!(
                        "Kimi Linear layer {layer} occurs more than once"
                    )));
                }
                *slot = true;
            }
        }
        if seen.iter().any(|present| !present) {
            return Err(Error::UnsupportedArchitecture(
                "Kimi Linear KDA and MLA layer lists must cover every decoder layer".into(),
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
            first_k_dense_replace: self.first_k_dense_replace,
            moe_layer_freq: self.moe_layer_freq,
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
    parse_config_value(value.clone()).map(|_| ())
}

pub(crate) fn parse_config_value(value: Value) -> Result<ModelArgs, Error> {
    let args: ModelArgs = serde_json::from_value(value).map_err(|error| {
        Error::UnsupportedArchitecture(format!("invalid Kimi Linear config: {error}"))
    })?;
    args.validate()?;
    Ok(args)
}

/// Loads and validates `config.json`.
pub fn get_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let file = std::fs::File::open(model_dir.as_ref().join("config.json"))?;
    parse_config_value(serde_json::from_reader(file)?)
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
            layers: (0..args.num_hidden_layers as usize)
                .map(|layer| {
                    if args.is_kda_layer(layer) {
                        LayerCache::Kda(KdaCache::default())
                    } else {
                        LayerCache::Mla(CompressedLatentCache::new())
                    }
                })
                .collect(),
        }
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
    pub(super) fn new(args: &ModelArgs, layer: usize, stream: &Stream) -> Result<Self, Exception> {
        let heads = args.linear_attn_config.num_heads;
        let head_dim = args.linear_attn_config.head_dim;
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
                args.linear_attn_config.short_conv_kernel_size,
                false,
                stream,
            )?,
            k_conv1d: DepthwiseConv1d::new(
                projection,
                args.linear_attn_config.short_conv_kernel_size,
                false,
                stream,
            )?,
            v_conv1d: DepthwiseConv1d::new(
                projection,
                args.linear_attn_config.short_conv_kernel_size,
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
enum Attention {
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
struct SparseMoe {
    #[param]
    gate: TopKRouter,
    #[param]
    experts: PackedSwiGluExperts,
    #[param]
    shared_experts: SwiGluMlp,
}

impl SparseMoe {
    fn new(args: &ModelArgs, layer: usize, stream: &Stream) -> Result<Self, Exception> {
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
                args.moe_intermediate_size,
                gate_up_quantization,
                down_quantization,
                stream,
            )?,
            shared_experts: args.unloaded_swiglu(
                &format!("{prefix}.shared_experts"),
                args.moe_intermediate_size * args.num_shared_experts,
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
        assignment: &crate::architectures::distributed::expert::ExpertAssignment,
        group: &safemlx::distributed::Group,
        statistics: &mut crate::architectures::distributed::expert::RoutingStatistics,
        stream: &Stream,
        observer: &mut Option<&mut dyn ActivationObserver>,
        prefix: &str,
    ) -> Result<Array, Exception> {
        let shape = input.shape().to_vec();
        let flat = input.reshape(&[-1, input.dim(-1)], stream)?;
        crate::architectures::distributed::expert::materialize_timing_phase([&flat])?;
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
        crate::architectures::distributed::expert::materialize_timing_phase([&indices, &weights])?;
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
        let shared_started = std::time::Instant::now();
        let shared = self.shared_experts.forward(&flat, stream)?;
        crate::architectures::distributed::expert::materialize_timing_phase([&shared])?;
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
enum FeedForward {
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
    self_attn: Attention,
    #[param]
    mlp: FeedForward,
    #[param]
    input_layernorm: nn::RmsNorm,
    #[param]
    post_attention_layernorm: nn::RmsNorm,
}

impl DecoderLayer {
    pub(super) fn new(args: &ModelArgs, layer: usize, stream: &Stream) -> Result<Self, Exception> {
        let attention = if args.is_kda_layer(layer) {
            Attention::Kda(Box::new(KimiDeltaAttention::new(args, layer, stream)?))
        } else {
            Attention::Mla(Box::new(MultiHeadLatentAttention::new_with_nope(
                &args.deepseek_mla_args(),
                layer as i32,
                true,
                stream,
            )?))
        };
        let mlp = if args.is_moe_layer(layer) {
            FeedForward::Moe(Box::new(SparseMoe::new(args, layer, stream)?))
        } else {
            let prefix = format!("model.layers.{layer}.mlp");
            FeedForward::Dense(Box::new(args.unloaded_swiglu(
                &prefix,
                args.intermediate_size,
                stream,
            )?))
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

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_expert_parallel(
        &mut self,
        input: &Array,
        mask: Option<&Array>,
        cache: Option<&mut LayerCache>,
        assignment: &crate::architectures::distributed::expert::ExpertAssignment,
        group: &safemlx::distributed::Group,
        statistics: &mut crate::architectures::distributed::expert::RoutingStatistics,
        stream: &Stream,
        observer: &mut Option<&mut dyn ActivationObserver>,
        prefix: &str,
    ) -> Result<Array, Exception> {
        let normalized = self.input_layernorm.forward(input, stream)?;
        let attention = match (&mut self.self_attn, cache) {
            (Attention::Kda(attention), Some(LayerCache::Kda(cache))) => attention.forward_impl(
                &normalized,
                Some(cache),
                stream,
                &format!("{prefix}.self_attn"),
                observer,
            )?,
            (Attention::Mla(attention), Some(LayerCache::Mla(cache))) => attention.forward_shared(
                &normalized,
                mask,
                Some(cache),
                stream,
                &format!("{prefix}.self_attn"),
                observer,
            )?,
            _ => {
                return Err(Exception::custom(
                    "Kimi Linear expert-parallel cache layer kind mismatch",
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
                observer,
                &format!("{prefix}.mlp"),
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
            layers: (0..args.num_hidden_layers as usize)
                .map(|layer| DecoderLayer::new(args, layer, stream))
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

    /// Returns the stable model type.
    pub fn model_type(&self) -> &str {
        &self.args.model_type
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
            if cache.layers.len() != self.model.layers.len() {
                return Err(Exception::custom(
                    "Kimi Linear cache layer count does not match model",
                ));
            }
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

    /// Runs pure expert-parallel Kimi inference with replicated nonexpert weights.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_expert_parallel(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Cache,
        assignment: &crate::architectures::distributed::expert::ExpertAssignment,
        group: &safemlx::distributed::Group,
        statistics: &mut crate::architectures::distributed::expert::RoutingStatistics,
        mut observer: Option<&mut dyn ActivationObserver>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let mut hidden = self.model.embed_tokens.forward(inputs, stream)?;
        let offset = cache.offset();
        let generated_mask = if mask.is_none() && hidden.dim(1) > 1 && offset > 0 {
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
        let mask = mask.or(generated_mask.as_ref());
        if cache.layers.len() != self.model.layers.len() {
            return Err(Exception::custom(
                "Kimi Linear expert-parallel cache layer count mismatch",
            ));
        }
        for (index, (layer, layer_cache)) in self
            .model
            .layers
            .iter_mut()
            .zip(&mut cache.layers)
            .enumerate()
        {
            hidden = layer.forward_expert_parallel(
                &hidden,
                mask,
                Some(layer_cache),
                assignment,
                group,
                statistics,
                stream,
                &mut observer,
                &format!("model.layers.{index}"),
            )?;
        }
        let hidden = self.model.norm.forward(&hidden, stream)?;
        project_logits_maybe_quantized(
            &mut self.lm_head,
            &mut self.model.embed_tokens,
            &hidden,
            stream,
        )
    }

    /// Runs Kimi inference with routed experts supplied by a sparse cache.
    pub(crate) fn forward_cached_expert_parallel<F>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Cache,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Exception>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let mut hidden = self.model.embed_tokens.forward(inputs, stream)?;
        let offset = cache.offset();
        let generated_mask = if mask.is_none() && hidden.dim(1) > 1 && offset > 0 {
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
        let mask = mask.or(generated_mask.as_ref());
        if cache.layers.len() != self.model.layers.len() {
            return Err(Exception::custom(
                "Kimi Linear sparse expert-parallel cache layer count mismatch",
            ));
        }
        for (index, (layer, layer_cache)) in self
            .model
            .layers
            .iter_mut()
            .zip(&mut cache.layers)
            .enumerate()
        {
            hidden = layer.forward_sparse_experts(
                &hidden,
                mask,
                Some(layer_cache),
                stream,
                |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
            )?;
        }
        let hidden = self.model.norm.forward(&hidden, stream)?;
        project_logits_maybe_quantized(
            &mut self.lm_head,
            &mut self.model.embed_tokens,
            &hidden,
            stream,
        )
    }

    /// Retains only the routed experts owned by this EP rank.
    pub(crate) fn partition_routed_experts(
        &mut self,
        assignment: &crate::architectures::distributed::expert::ExpertAssignment,
        stream: &Stream,
    ) -> Result<usize, Error> {
        let ids = assignment.local_global_expert_ids();
        if ids.is_empty() {
            return Err(Error::Parallel(
                "Kimi Linear expert assignment owns no local experts".into(),
            ));
        }
        let indices = Array::from_slice(
            &ids.iter().map(|id| *id as i32).collect::<Vec<_>>(),
            &[ids.len() as i32],
        );
        let mut bytes = 0usize;
        for layer in &mut self.model.layers {
            let FeedForward::Moe(moe) = &mut layer.mlp else {
                continue;
            };
            let select = |value: &Array| value.take_axis(&indices, 0, stream);
            let gate_up = select(moe.experts.gate_up_proj.as_ref())?;
            bytes += gate_up.nbytes();
            moe.experts.gate_up_proj = Param::new(gate_up);
            if let Some(value) = moe.experts.gate_up_proj_scales.as_ref() {
                let value = select(value)?;
                bytes += value.nbytes();
                moe.experts.gate_up_proj_scales = Param::new(Some(value));
            }
            if let Some(value) = moe.experts.gate_up_proj_biases.as_ref() {
                let value = select(value)?;
                bytes += value.nbytes();
                moe.experts.gate_up_proj_biases = Param::new(Some(value));
            }
            let down = select(moe.experts.down_proj.as_ref())?;
            bytes += down.nbytes();
            moe.experts.down_proj = Param::new(down);
            if let Some(value) = moe.experts.down_proj_scales.as_ref() {
                let value = select(value)?;
                bytes += value.nbytes();
                moe.experts.down_proj_scales = Param::new(Some(value));
            }
            if let Some(value) = moe.experts.down_proj_biases.as_ref() {
                let value = select(value)?;
                bytes += value.nbytes();
                moe.experts.down_proj_biases = Param::new(Some(value));
            }
            moe.experts.num_experts = ids.len() as i32;
        }
        Ok(bytes)
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
pub type Generate<'a, S = crate::runtime::generation::sampler::DefaultSampler> =
    crate::nn::generation::Generate<'a, Model, Cache, S>;

fn transform_safetensors_weight(
    args: &ModelArgs,
    mut key: String,
    value: Array,
    stream: &Stream,
) -> Result<Vec<(String, Array)>, Error> {
    if key.starts_with("model.mtp.") {
        return Ok(Vec::new());
    }
    key = key.replace(".block_sparse_moe.", ".mlp.");
    if key.ends_with(".q_conv1d.weight")
        || key.ends_with(".k_conv1d.weight")
        || key.ends_with(".v_conv1d.weight")
    {
        let projection = args.linear_attn_config.num_heads * args.linear_attn_config.head_dim;
        let kernel = args.linear_attn_config.short_conv_kernel_size;
        if value.size() as i32 != projection * kernel {
            return Err(Error::UnsupportedArchitecture(format!(
                "Kimi Linear convolution tensor {key} has shape {:?}, expected {projection}x{kernel}",
                value.shape()
            )));
        }
        return Ok(vec![(
            key,
            value.reshape(&[projection, 1, kernel], stream)?,
        )]);
    }
    if key.ends_with(".A_log") {
        let heads = args.linear_attn_config.num_heads;
        if value.size() != heads as usize {
            return Err(Error::UnsupportedArchitecture(format!(
                "Kimi Linear transition tensor {key} has shape {:?}, expected {heads} values",
                value.shape()
            )));
        }
        return Ok(vec![(key, value.reshape(&[1, 1, heads, 1], stream)?)]);
    }
    Ok(vec![(key, value)])
}

fn strict_load_config() -> StrictLoadConfig {
    StrictLoadConfig::default()
}

/// Loads the official sharded safetensors checkpoint.
pub fn load_model(
    model_dir: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    load_model_impl(model_dir.as_ref(), None, stream, weights_stream)
}

fn load_model_impl(
    model_dir: &Path,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let mut args = get_model_args(model_dir)?;
    if let Some(quantization) = quantization {
        quantization.validate()?;
        args.quantization = Some(quantization);
    }
    let mut model = Model::new(args.clone(), stream)?;
    let mut report = StrictLoadReport::default();
    load_safetensors_dir_strict_with_split_swiglu_experts_and_transform(
        &mut model,
        model_dir,
        weights_stream,
        stream,
        quantization,
        &strict_load_config(),
        &mut report,
        args.num_experts,
        |key, value| transform_safetensors_weight(&args, key, value, stream),
    )?;
    report.finish(&model, &strict_load_config())?;
    model.copy_to_stream(stream)?;
    Ok(model)
}

/// Loads a floating-point checkpoint while quantizing eligible weights.
pub fn load_model_quantized(
    model_dir: impl AsRef<Path>,
    quantization: WeightQuantization,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let args = get_model_args(model_dir)?;
    if !crate::runtime::checkpoint::quantization::should_quantize_on_load(
        "Kimi Linear",
        args.quantization,
        quantization,
    )? {
        return load_model(model_dir, stream, weights_stream);
    }
    load_model_impl(model_dir, Some(quantization), stream, weights_stream)
}

/// Kimi Linear model plus architecture-owned GGUF stop IDs.
pub(crate) struct LoadedKimiLinearGguf {
    pub(crate) model: Model,
    pub(crate) eos_token_ids: Vec<u32>,
}

pub(crate) struct PreparedKimiLinearGguf {
    pub(crate) args: ModelArgs,
    pub(crate) eos_token_ids: Vec<u32>,
}

/// Loads a llama.cpp `kimi-linear` GGUF checkpoint.
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
) -> Result<LoadedKimiLinearGguf, Error> {
    let prepared = prepare_gguf_checkpoint(checkpoint, &metadata, quantization, weights_stream)?;
    let args = prepared.args;
    let mut model = Model::new(args.clone(), stream)?;
    let config = StrictLoadConfig::default().allow_unused_prefix("rope_freqs.");
    let mut report = StrictLoadReport::default();
    let mut materializer = checkpoint.materializer();
    for tensor in checkpoint.catalog().tensors() {
        let physical_name = &tensor.descriptor().name;
        if physical_name.contains("ffn_gate_exps") || physical_name.contains("ffn_up_exps") {
            continue;
        }
        let group = materializer.converted_tensor(physical_name)?;
        for (source_name, value) in group.into_arrays() {
            let target_name = translate_gguf_weight_name(&source_name);
            let value = normalize_gguf_weight(&target_name, value, &args, weights_stream)?;
            load_named_array_strict(
                &mut model,
                target_name,
                value,
                quantization.map(|quantization| (quantization, stream)),
                &config,
                &mut report,
            )?;
        }
    }
    for layer in 0..args.num_hidden_layers {
        if !args.is_moe_layer(layer as usize) {
            continue;
        }
        let source_prefix = format!("blk.{layer}");
        let target_prefix = format!("model.layers.{layer}.mlp.experts");
        let gate = materializer
            .converted_tensor(&format!("{source_prefix}.ffn_gate_exps.weight"))?
            .into_arrays()
            .into_iter()
            .collect::<HashMap<_, _>>();
        let up = materializer
            .converted_tensor(&format!("{source_prefix}.ffn_up_exps.weight"))?
            .into_arrays()
            .into_iter()
            .collect::<HashMap<_, _>>();
        for (source_suffix, target_suffix) in
            [("weight", ""), ("scales", "_scales"), ("biases", "_biases")]
        {
            let gate_name = format!("{source_prefix}.ffn_gate_exps.{source_suffix}");
            let up_name = format!("{source_prefix}.ffn_up_exps.{source_suffix}");
            let (gate, up) = match (gate.get(&gate_name), up.get(&up_name)) {
                (Some(gate), Some(up)) => (gate, up),
                (None, None) if source_suffix != "weight" => continue,
                (gate, up) => {
                    return Err(Error::UnsupportedArchitecture(format!(
                        "Kimi Linear GGUF has mismatched expert components {gate_name:?} ({}) and {up_name:?} ({})",
                        if gate.is_some() { "present" } else { "missing" },
                        if up.is_some() { "present" } else { "missing" },
                    )));
                }
            };
            let value = concatenate_axis(&[gate, up], 1, weights_stream)?;
            load_named_array_strict(
                &mut model,
                format!("{target_prefix}.gate_up_proj{target_suffix}"),
                value,
                quantization.map(|quantization| (quantization, stream)),
                &config,
                &mut report,
            )?;
        }
    }
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;
    Ok(LoadedKimiLinearGguf {
        model,
        eos_token_ids: prepared.eos_token_ids,
    })
}

pub(crate) fn prepare_gguf_checkpoint(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    quantization: Option<WeightQuantization>,
    weights_stream: &Stream,
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
    let mut args = args_from_gguf(checkpoint, metadata, weights_stream)?;
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
        eos_token_ids: crate::api::gguf_eos_token_ids(metadata)?,
    })
}

fn combine_expert_gate_up_formats(
    formats: &mut HashMap<String, WeightQuantization>,
    args: &ModelArgs,
) -> Result<(), Error> {
    for layer in 0..args.num_hidden_layers {
        if !args.is_moe_layer(layer as usize) {
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
        let expected = args.linear_attn_config.num_heads * args.linear_attn_config.head_dim;
        let kernel = args.linear_attn_config.short_conv_kernel_size;
        if value.size() != (expected * kernel) as usize {
            return Err(Error::UnsupportedArchitecture(format!(
                "Kimi Linear convolution {name:?} has shape {:?}, expected {expected}x{kernel}",
                value.shape()
            )));
        }
        return Ok(value.reshape(&[expected, 1, kernel], stream)?);
    }
    if name.ends_with(".A_log") {
        let heads = args.linear_attn_config.num_heads;
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

fn args_from_gguf(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    _stream: &Stream,
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
    Ok(ModelArgs {
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
        linear_attn_config: LinearAttentionConfig {
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
    })
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
    crate::runtime::checkpoint::tiktoken::load_kimi_k2(model_dir)
}

#[cfg(test)]
mod tests {
    use safemlx::{module::ModuleParameters, Array, Device, DeviceType, ExecutionContext};
    use serde_json::json;

    use super::{
        load_gguf, load_model, load_tokenizer, normalize_gguf_weight, parse_config_value,
        translate_gguf_weight_name, Model, ModelInput,
    };

    fn config() -> serde_json::Value {
        json!({
            "model_type": "kimi_linear",
            "vocab_size": 64,
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "intermediate_size": 16,
            "head_dim": 4,
            "model_max_length": 128,
            "rms_norm_eps": 0.00001,
            "rope_theta": 10000.0,
            "linear_attn_config": {
                "kda_layers": [1],
                "full_attn_layers": [2],
                "num_heads": 2,
                "head_dim": 4,
                "short_conv_kernel_size": 2
            },
            "num_experts": 4,
            "moe_intermediate_size": 8,
            "kv_lora_rank": 4,
            "q_lora_rank": null,
            "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 2,
            "v_head_dim": 2,
            "mla_use_nope": true,
            "num_experts_per_token": 2,
            "num_shared_experts": 1,
            "moe_router_activation_func": "sigmoid",
            "moe_renormalize": true,
            "routed_scaling_factor": 1.0,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "use_grouped_topk": true,
            "num_expert_group": 1,
            "topk_group": 1,
            "tie_word_embeddings": false,
            "num_nextn_predict_layers": 0
        })
    }

    #[test]
    fn validates_hybrid_layer_partition() {
        let args = parse_config_value(config()).unwrap();
        assert!(args.is_kda_layer(0));
        assert!(!args.is_kda_layer(1));

        let mut duplicate = config();
        duplicate["linear_attn_config"]["full_attn_layers"] = json!([1]);
        assert!(parse_config_value(duplicate).is_err());
    }

    #[test]
    fn tiny_parameter_tree_contains_kda_mla_dense_and_sparse_contracts() {
        let args = parse_config_value(config()).unwrap();
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let model = Model::new(args, context.stream()).unwrap();
        let parameters = model.parameters().flatten();

        for (name, shape) in [
            ("model.layers.0.self_attn.q_conv1d.weight", vec![8, 1, 2]),
            ("model.layers.0.self_attn.A_log", vec![1, 1, 2, 1]),
            (
                "model.layers.1.self_attn.kv_a_proj_with_mqa.weight",
                vec![6, 8],
            ),
            ("model.layers.1.self_attn.kv_b_proj.weight", vec![8, 4]),
            ("model.layers.1.mlp.experts.gate_up_proj", vec![4, 16, 8]),
            ("model.layers.1.mlp.experts.down_proj", vec![4, 8, 8]),
            (
                "model.layers.1.mlp.shared_experts.gate_proj.weight",
                vec![8, 8],
            ),
            ("model.layers.1.mlp.gate.e_score_correction_bias", vec![4]),
        ] {
            assert_eq!(
                parameters
                    .get(name)
                    .unwrap_or_else(|| panic!("{name}"))
                    .shape(),
                shape
            );
        }
    }

    #[test]
    fn translates_modern_and_legacy_gguf_names() {
        assert_eq!(
            translate_gguf_weight_name("blk.3.attn_k_b.weight"),
            "model.layers.3.self_attn.k_b_proj.weight"
        );
        assert_eq!(
            translate_gguf_weight_name("blk.3.attn_kv_b.weight"),
            "model.layers.3.self_attn.kv_b_proj.weight"
        );
        assert_eq!(
            translate_gguf_weight_name("blk.2.ssm_conv1d_q.weight"),
            "model.layers.2.self_attn.q_conv1d.weight"
        );
        assert_eq!(
            translate_gguf_weight_name("blk.2.ssm_a.weight"),
            "model.layers.2.self_attn.A_log"
        );
        assert_eq!(
            translate_gguf_weight_name("blk.1.ffn_gate_shexp.weight"),
            "model.layers.1.mlp.shared_experts.gate_proj.weight"
        );
        assert_eq!(
            translate_gguf_weight_name("blk.1.exp_probs_b.bias"),
            "model.layers.1.mlp.gate.e_score_correction_bias"
        );
    }

    #[test]
    fn normalizes_gguf_transition_rates_and_convolution_rank() {
        let args = parse_config_value(config()).unwrap();
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let transition = normalize_gguf_weight(
            "model.layers.0.self_attn.A_log",
            Array::from_slice(&[-1.0f32, -4.0], &[1, 1, 2, 1]),
            &args,
            stream,
        )
        .unwrap();
        assert_eq!(transition.shape(), &[1, 1, 2, 1]);
        let transition = transition.evaluated().unwrap();
        let values = transition.as_slice::<f32>();
        assert!(values[0].abs() < 1e-6);
        assert!((values[1] - 4.0f32.ln()).abs() < 1e-6);
        assert!(normalize_gguf_weight(
            "model.layers.0.self_attn.A_log",
            Array::from_slice(&[-1.0f32, 0.0], &[2]),
            &args,
            stream,
        )
        .is_err());

        let convolution = normalize_gguf_weight(
            "model.layers.0.self_attn.q_conv1d.weight",
            Array::from_slice(&vec![0.0f32; 2 * 4 * 2], &[1, 8, 1, 2]),
            &args,
            stream,
        )
        .unwrap();
        assert_eq!(convolution.shape(), &[8, 1, 2]);
    }

    #[test]
    #[ignore = "requires KIMI_LINEAR_MODEL_DIR and a Metal device"]
    fn real_safetensors_prefill_decode() {
        let dir = std::env::var_os("KIMI_LINEAR_MODEL_DIR")
            .map(std::path::PathBuf::from)
            .expect("KIMI_LINEAR_MODEL_DIR");
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        let tokenizer = load_tokenizer(&dir).unwrap();
        assert_eq!(tokenizer.token_to_id("<|im_end|>"), Some(163586));
        let encoded = tokenizer
            .encode("Hello 世界! I'm testing Kimi.", false)
            .unwrap();
        assert!(!encoded.get_ids().is_empty());
        let mut model = load_model(&dir, stream, stream).unwrap();
        let mut cache = model.new_cache();
        let logits = model
            .forward_logits(
                ModelInput {
                    inputs: &Array::from_slice(&[163584i32, 42], &[1, 2]),
                    mask: None,
                    cache: Some(&mut cache),
                },
                true,
                stream,
            )
            .unwrap();
        let _ = logits.evaluated().unwrap();
        assert_eq!(logits.shape(), &[1, model.args.vocab_size]);
        assert_eq!(cache.offset(), 2);
    }

    #[test]
    #[ignore = "requires KIMI_LINEAR_GGUF and a Metal device"]
    fn real_gguf_prefill_decode() {
        let path = std::env::var_os("KIMI_LINEAR_GGUF")
            .map(std::path::PathBuf::from)
            .expect("KIMI_LINEAR_GGUF");
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        let mut model = load_gguf(path, stream, stream).unwrap();
        let mut cache = model.new_cache();
        let logits = model
            .forward_logits(
                ModelInput {
                    inputs: &Array::from_slice(&[163584i32, 42], &[1, 2]),
                    mask: None,
                    cache: Some(&mut cache),
                },
                true,
                stream,
            )
            .unwrap();
        let _ = logits.evaluated().unwrap();
        assert_eq!(logits.shape(), &[1, model.args.vocab_size]);
        assert_eq!(cache.offset(), 2);
    }
}
