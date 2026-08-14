//! DeepSeek-V4 configuration and model components.
//!
//! The public configuration deliberately describes the architecture rather
//! than individual repository names.  Flash, Pro, Base, Instruct, MTP, and
//! fused DSpark checkpoints therefore share one validation path.

use std::path::Path;

use serde::Deserialize;
use serde_json::Value;

use safemlx::{
    error::Exception,
    macros::ModuleParameters,
    module::{Module, ModuleParametersExt},
    nn,
    ops::{broadcast_to, concatenate_axis, indexing::NewAxis, indexing::TryIndexOp, mean_axis},
    Array, Dtype, Stream,
};

use crate::{
    api::{
        input as runtime_input,
        qwen3_5::{QwenLinear as Linear, QwenWeightFormat as WeightFormat},
    },
    error::Error,
    nn::{
        generation::CausalLm,
        hyper_connections::{expand, HyperConnection, HyperHead},
    },
    runtime::attention::{AttentionPolicy, LayerSchedule},
    runtime::cache::residency::{derive_prompt_cache_architecture_fingerprint, LayerCachePolicy},
    runtime::checkpoint::load::{
        load_safetensors_dir_strict_with_split_swiglu_experts_and_transform, StrictLoadConfig,
        StrictLoadReport,
    },
};

use super::{
    attention::{Attention, AttentionCache},
    layers::{rms_norm, Moe},
};

fn default_model_type() -> String {
    "deepseek_v4".into()
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}

fn default_rope_theta() -> f32 {
    10_000.0
}

fn default_compress_rope_theta() -> f32 {
    160_000.0
}

fn default_sliding_window() -> i32 {
    128
}

fn default_hc_mult() -> i32 {
    4
}

fn default_hc_iterations() -> i32 {
    20
}

fn default_hc_epsilon() -> f32 {
    1e-6
}

fn default_scoring_function() -> String {
    "sqrtsoftplus".into()
}

fn default_topk_method() -> String {
    "noaux_tc".into()
}

fn default_true() -> bool {
    true
}

fn default_one() -> i32 {
    1
}

fn default_float_one() -> f32 {
    1.0
}

/// DeepSeek YaRN parameters used by V4 checkpoints.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct YarnConfig {
    /// Scaling algorithm; must be `yarn`.
    #[serde(alias = "rope_type")]
    pub r#type: String,
    /// Context extension factor.
    pub factor: f32,
    /// Original trained context length.
    pub original_max_position_embeddings: i32,
    /// Fast correction rotations.
    #[serde(default = "default_beta_fast")]
    pub beta_fast: f32,
    /// Slow correction rotations.
    #[serde(default = "default_beta_slow")]
    pub beta_slow: f32,
}

fn default_beta_fast() -> f32 {
    32.0
}

fn default_beta_slow() -> f32 {
    1.0
}

/// Official block-FP8 metadata.  V4 may store its scales as UE8M0.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct Fp8QuantizationConfig {
    /// Quantization method.
    pub quant_method: String,
    /// FP8 value format.
    pub fmt: String,
    /// Dynamic activation scaling marker.
    pub activation_scheme: String,
    /// Two-dimensional weight block.
    pub weight_block_size: Vec<i32>,
    /// Optional scale encoding (`ue8m0` in native V4 checkpoints).
    #[serde(default)]
    pub scale_fmt: Option<String>,
}

/// Configuration present only in a fused DSpark checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DsparkConfig {
    /// Number of tokens produced by one draft block.
    pub block_size: i32,
    /// Token used to seed unfilled draft positions.
    pub noise_token_id: i32,
    /// Target decoder layers captured by the drafter.
    pub target_layer_ids: Vec<i32>,
    /// Rank of the token-conditioned Markov head.
    pub markov_rank: i32,
}

/// Validated DeepSeek-V4 architecture configuration.
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: i32,
    pub moe_intermediate_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub qk_rope_head_dim: i32,
    pub q_lora_rank: i32,
    pub o_lora_rank: i32,
    pub o_groups: i32,
    pub vocab_size: i32,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: i32,
    pub rope_theta: f32,
    pub compress_rope_theta: f32,
    pub rope_scaling: Option<YarnConfig>,
    pub sliding_window: i32,
    pub compress_ratios: Vec<i32>,
    pub index_n_heads: i32,
    pub index_head_dim: i32,
    pub index_topk: i32,
    pub hc_mult: i32,
    pub hc_sinkhorn_iters: i32,
    pub hc_eps: f32,
    pub n_routed_experts: i32,
    pub n_shared_experts: i32,
    pub num_experts_per_tok: i32,
    pub num_hash_layers: i32,
    pub scoring_func: String,
    pub topk_method: String,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f32,
    pub swiglu_limit: f32,
    pub num_nextn_predict_layers: i32,
    pub expert_dtype: Option<String>,
    pub quantization_config: Option<Fp8QuantizationConfig>,
    pub tie_word_embeddings: bool,
    pub dspark: Option<DsparkConfig>,
}

#[derive(Debug, Clone, Deserialize)]
struct ModelArgsSource {
    #[serde(default = "default_model_type")]
    model_type: String,
    hidden_size: i32,
    moe_intermediate_size: i32,
    num_hidden_layers: i32,
    num_attention_heads: i32,
    #[serde(default = "default_one")]
    num_key_value_heads: i32,
    head_dim: i32,
    qk_rope_head_dim: i32,
    q_lora_rank: i32,
    o_lora_rank: i32,
    o_groups: i32,
    vocab_size: i32,
    #[serde(default = "default_rms_norm_eps")]
    rms_norm_eps: f32,
    max_position_embeddings: i32,
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    #[serde(default = "default_compress_rope_theta")]
    compress_rope_theta: f32,
    #[serde(default)]
    rope_scaling: Option<YarnConfig>,
    #[serde(default = "default_sliding_window")]
    sliding_window: i32,
    #[serde(default)]
    compress_ratios: Vec<i32>,
    index_n_heads: i32,
    index_head_dim: i32,
    index_topk: i32,
    #[serde(default = "default_hc_mult")]
    hc_mult: i32,
    #[serde(default = "default_hc_iterations")]
    hc_sinkhorn_iters: i32,
    #[serde(default = "default_hc_epsilon")]
    hc_eps: f32,
    n_routed_experts: i32,
    #[serde(default = "default_one")]
    n_shared_experts: i32,
    num_experts_per_tok: i32,
    #[serde(default)]
    num_hash_layers: i32,
    #[serde(default = "default_scoring_function")]
    scoring_func: String,
    #[serde(default = "default_topk_method")]
    topk_method: String,
    #[serde(default = "default_true")]
    norm_topk_prob: bool,
    #[serde(default = "default_float_one")]
    routed_scaling_factor: f32,
    #[serde(default)]
    swiglu_limit: f32,
    #[serde(default)]
    num_nextn_predict_layers: i32,
    #[serde(default)]
    expert_dtype: Option<String>,
    #[serde(default)]
    quantization_config: Option<Fp8QuantizationConfig>,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default)]
    dspark_block_size: Option<i32>,
    #[serde(default)]
    dspark_noise_token_id: Option<i32>,
    #[serde(default)]
    dspark_target_layer_ids: Option<Vec<i32>>,
    #[serde(default)]
    dspark_markov_rank: Option<i32>,
}

impl ModelArgs {
    /// Parses and validates one Hugging Face `config.json` value.
    pub fn from_value(value: Value) -> Result<Self, Error> {
        let source: ModelArgsSource = serde_json::from_value(value)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        source.normalize()
    }

    /// Returns the configured compression ratio for a target or prediction layer.
    pub fn compression_ratio(&self, layer: usize) -> Option<i32> {
        self.compress_ratios.get(layer).copied()
    }

    /// Whether this checkpoint contains a fused DSpark drafter.
    pub const fn has_dspark(&self) -> bool {
        self.dspark.is_some()
    }

    pub(crate) fn validate(&self) -> Result<(), Error> {
        if self.model_type != "deepseek_v4" {
            return Err(Error::UnsupportedModelType(self.model_type.clone()));
        }
        for (name, value) in [
            ("hidden_size", self.hidden_size),
            ("moe_intermediate_size", self.moe_intermediate_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("head_dim", self.head_dim),
            ("qk_rope_head_dim", self.qk_rope_head_dim),
            ("q_lora_rank", self.q_lora_rank),
            ("o_lora_rank", self.o_lora_rank),
            ("o_groups", self.o_groups),
            ("vocab_size", self.vocab_size),
            ("max_position_embeddings", self.max_position_embeddings),
            ("sliding_window", self.sliding_window),
            ("index_n_heads", self.index_n_heads),
            ("index_head_dim", self.index_head_dim),
            ("index_topk", self.index_topk),
            ("hc_mult", self.hc_mult),
            ("hc_sinkhorn_iters", self.hc_sinkhorn_iters),
            ("n_routed_experts", self.n_routed_experts),
            ("n_shared_experts", self.n_shared_experts),
            ("num_experts_per_tok", self.num_experts_per_tok),
        ] {
            if value <= 0 {
                return Err(unsupported(format!("{name} must be positive, got {value}")));
            }
        }
        if self.num_key_value_heads != 1 {
            return Err(unsupported("num_key_value_heads must be one"));
        }
        if self.qk_rope_head_dim >= self.head_dim || self.qk_rope_head_dim % 2 != 0 {
            return Err(unsupported(
                "qk_rope_head_dim must be even and smaller than head_dim",
            ));
        }
        if self.num_attention_heads % self.o_groups != 0 || self.index_n_heads % self.o_groups != 0
        {
            return Err(unsupported(
                "attention and index heads must be divisible by o_groups",
            ));
        }
        if self.num_experts_per_tok > self.n_routed_experts
            || self.num_hash_layers < 0
            || self.num_hash_layers > self.num_hidden_layers
        {
            return Err(unsupported("invalid routed/hash expert dimensions"));
        }
        if self.scoring_func != "sqrtsoftplus" || self.topk_method != "noaux_tc" {
            return Err(unsupported(format!(
                "unsupported router scoring/top-k combination {:?}/{:?}",
                self.scoring_func, self.topk_method
            )));
        }
        if !self.norm_topk_prob {
            return Err(unsupported("norm_topk_prob=false is not supported by V4"));
        }
        if !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || !self.hc_eps.is_finite()
            || self.hc_eps <= 0.0
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 0.0
            || !self.compress_rope_theta.is_finite()
            || self.compress_rope_theta <= 0.0
            || !self.routed_scaling_factor.is_finite()
            || self.routed_scaling_factor <= 0.0
        {
            return Err(unsupported(
                "normalization, RoPE, and routing scales must be finite and positive",
            ));
        }
        // Official checkpoints append one compression entry per embedded
        // prediction block.  Those blocks are ordinary V4 decoder layers and
        // therefore use the same attention construction path as the target.
        let required_ratios =
            usize::try_from(self.num_hidden_layers + self.num_nextn_predict_layers)
                .map_err(|_| unsupported("layer count is negative"))?;
        if self.compress_ratios.len() != required_ratios {
            return Err(unsupported(format!(
                "compress_ratios has {} entries but {required_ratios} decoder blocks require entries",
                self.compress_ratios.len()
            )));
        }
        if self
            .compress_ratios
            .iter()
            .any(|ratio| !matches!(ratio, 0 | 4 | 128))
        {
            return Err(unsupported("compress_ratios may contain only 0, 4, or 128"));
        }
        if self.compress_ratios[self.num_hidden_layers as usize..]
            .iter()
            .any(|ratio| *ratio != 0)
        {
            return Err(unsupported(
                "embedded MTP and DSpark decoder blocks require local attention",
            ));
        }
        if let Some(yarn) = &self.rope_scaling {
            if yarn.r#type != "yarn"
                || yarn.factor <= 0.0
                || yarn.original_max_position_embeddings <= 0
            {
                return Err(unsupported("invalid YaRN configuration"));
            }
        }
        if let Some(quantization) = &self.quantization_config {
            if quantization.quant_method != "fp8"
                || quantization.fmt != "e4m3"
                || quantization.activation_scheme != "dynamic"
                || quantization.weight_block_size.as_slice() != [128, 128]
                || quantization
                    .scale_fmt
                    .as_deref()
                    .is_some_and(|format| format != "ue8m0")
            {
                return Err(unsupported(format!(
                    "unsupported native FP8 configuration {quantization:?}"
                )));
            }
        }
        if self
            .expert_dtype
            .as_deref()
            .is_some_and(|dtype| dtype != "fp4" && dtype != "fp8")
        {
            return Err(unsupported(format!(
                "unsupported expert_dtype {:?}",
                self.expert_dtype
            )));
        }
        if let Some(dspark) = &self.dspark {
            if dspark.block_size <= 0
                || dspark.noise_token_id < 0
                || dspark.noise_token_id >= self.vocab_size
                || dspark.markov_rank <= 0
                || dspark.target_layer_ids.is_empty()
                || dspark
                    .target_layer_ids
                    .iter()
                    .any(|layer| *layer < 0 || *layer >= self.num_hidden_layers)
            {
                return Err(unsupported("invalid fused DSpark configuration"));
            }
        }
        Ok(())
    }
}

impl ModelArgsSource {
    fn normalize(mut self) -> Result<ModelArgs, Error> {
        if self.compress_ratios.is_empty() {
            let layers = self.num_hidden_layers.max(0) as usize;
            self.compress_ratios = (0..layers)
                .map(|layer| {
                    if layer == 0 || layer + 1 == layers {
                        0
                    } else if layer % 2 == 1 {
                        4
                    } else {
                        128
                    }
                })
                .collect();
            self.compress_ratios.extend(std::iter::repeat_n(
                0,
                self.num_nextn_predict_layers.max(0) as usize,
            ));
        }
        let dspark_fields = [
            self.dspark_block_size.is_some(),
            self.dspark_noise_token_id.is_some(),
            self.dspark_target_layer_ids.is_some(),
            self.dspark_markov_rank.is_some(),
        ];
        let dspark = if dspark_fields.iter().all(|present| *present) {
            Some(DsparkConfig {
                block_size: self.dspark_block_size.expect("checked"),
                noise_token_id: self.dspark_noise_token_id.expect("checked"),
                target_layer_ids: self.dspark_target_layer_ids.expect("checked"),
                markov_rank: self.dspark_markov_rank.expect("checked"),
            })
        } else if dspark_fields.iter().any(|present| *present) {
            return Err(unsupported(
                "fused DSpark checkpoints must provide all dspark_* fields",
            ));
        } else {
            None
        };
        let args = ModelArgs {
            model_type: self.model_type,
            hidden_size: self.hidden_size,
            moe_intermediate_size: self.moe_intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            qk_rope_head_dim: self.qk_rope_head_dim,
            q_lora_rank: self.q_lora_rank,
            o_lora_rank: self.o_lora_rank,
            o_groups: self.o_groups,
            vocab_size: self.vocab_size,
            rms_norm_eps: self.rms_norm_eps,
            max_position_embeddings: self.max_position_embeddings,
            rope_theta: self.rope_theta,
            compress_rope_theta: self.compress_rope_theta,
            rope_scaling: self.rope_scaling,
            sliding_window: self.sliding_window,
            compress_ratios: self.compress_ratios,
            index_n_heads: self.index_n_heads,
            index_head_dim: self.index_head_dim,
            index_topk: self.index_topk,
            hc_mult: self.hc_mult,
            hc_sinkhorn_iters: self.hc_sinkhorn_iters,
            hc_eps: self.hc_eps,
            n_routed_experts: self.n_routed_experts,
            n_shared_experts: self.n_shared_experts,
            num_experts_per_tok: self.num_experts_per_tok,
            num_hash_layers: self.num_hash_layers,
            scoring_func: self.scoring_func,
            topk_method: self.topk_method,
            norm_topk_prob: self.norm_topk_prob,
            routed_scaling_factor: self.routed_scaling_factor,
            swiglu_limit: self.swiglu_limit,
            num_nextn_predict_layers: self.num_nextn_predict_layers,
            expert_dtype: self.expert_dtype,
            quantization_config: self.quantization_config,
            tie_word_embeddings: self.tie_word_embeddings,
            dspark,
        };
        args.validate()?;
        Ok(args)
    }
}

fn unsupported(message: impl Into<String>) -> Error {
    Error::UnsupportedArchitecture(format!("DeepSeek-V4 {}", message.into()))
}

/// Validates a model configuration and returns its normalized arguments.
pub fn validate_model_config_value(value: &Value) -> Result<(), Error> {
    ModelArgs::from_value(value.clone()).map(|_| ())
}

/// Per-layer V4 cache state.
#[derive(Debug, Clone)]
pub struct Cache {
    pub(crate) layers: Vec<AttentionCache>,
    pub(crate) mtp_layers: Vec<AttentionCache>,
}

/// Embedded draft cache shared by resident, layerwise, and distributed execution.
pub(crate) type DraftCache = Vec<AttentionCache>;

impl Cache {
    /// Current target-decoder sequence offset.
    pub fn offset(&self) -> i32 {
        self.layers.first().map_or(0, AttentionCache::offset)
    }

    pub(crate) fn reset(&mut self) {
        for cache in self.layers.iter_mut().chain(&mut self.mtp_layers) {
            cache.clear();
        }
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// One V4 decoder layer with mHC attention and MoE residual cycles.
pub struct DecoderLayer {
    #[param]
    pub(crate) attn: Attention,
    #[param]
    pub(crate) ffn: Moe,
    #[param]
    pub(crate) attn_norm: nn::RmsNorm,
    #[param]
    pub(crate) ffn_norm: nn::RmsNorm,
    #[param]
    pub(crate) attn_hc: HyperConnection,
    #[param]
    pub(crate) ffn_hc: HyperConnection,
    norm_epsilon: f32,
}

impl DecoderLayer {
    pub(crate) fn new(args: &ModelArgs, layer: usize, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            attn: Attention::new(args, layer, stream)?,
            ffn: Moe::new(args, layer as i32, stream)?,
            attn_norm: rms_norm(args.hidden_size, args.rms_norm_eps, stream)?,
            ffn_norm: rms_norm(args.hidden_size, args.rms_norm_eps, stream)?,
            attn_hc: HyperConnection::unloaded(
                args.hc_mult,
                args.hidden_size,
                args.hc_sinkhorn_iters as usize,
                args.hc_eps,
                stream,
            )?,
            ffn_hc: HyperConnection::unloaded(
                args.hc_mult,
                args.hidden_size,
                args.hc_sinkhorn_iters as usize,
                args.hc_eps,
                stream,
            )?,
            norm_epsilon: args.rms_norm_eps,
        })
    }

    pub(crate) fn new_parallel(
        args: &ModelArgs,
        layer: usize,
        local_heads: i32,
        head_widths: Vec<usize>,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let mut block = Self::new(args, layer, stream)?;
        block.attn = Attention::new_parallel(args, layer, local_heads, head_widths, stream)?;
        Ok(block)
    }

    pub(crate) fn forward(
        &mut self,
        hidden: &Array,
        mask: Option<&Array>,
        cache: Option<&mut AttentionCache>,
        input_ids: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let residual = hidden;
        let (collapsed, post, combination) =
            self.attn_hc.collapse(hidden, self.norm_epsilon, stream)?;
        let normalized = self.attn_norm.forward(&collapsed, stream)?;
        let attention = self.attn.forward(&normalized, mask, cache, stream)?;
        let hidden = expand(&attention, residual, &post, &combination, stream)?;

        let residual = &hidden;
        let (collapsed, post, combination) =
            self.ffn_hc.collapse(&hidden, self.norm_epsilon, stream)?;
        let normalized = self.ffn_norm.forward(&collapsed, stream)?;
        let feed_forward = self.ffn.forward(&normalized, input_ids, stream)?;
        expand(&feed_forward, residual, &post, &combination, stream)
    }

    pub(crate) fn forward_with_expert_executor(
        &mut self,
        hidden: &Array,
        mask: Option<&Array>,
        cache: Option<&mut AttentionCache>,
        input_ids: &Array,
        stream: &Stream,
        execute: impl FnMut(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    ) -> Result<Array, Exception> {
        let residual = hidden;
        let (collapsed, post, combination) =
            self.attn_hc.collapse(hidden, self.norm_epsilon, stream)?;
        let normalized = self.attn_norm.forward(&collapsed, stream)?;
        let attention = self.attn.forward(&normalized, mask, cache, stream)?;
        let hidden = expand(&attention, residual, &post, &combination, stream)?;

        let residual = &hidden;
        let (collapsed, post, combination) =
            self.ffn_hc.collapse(&hidden, self.norm_epsilon, stream)?;
        let normalized = self.ffn_norm.forward(&collapsed, stream)?;
        let feed_forward =
            self.ffn
                .forward_with_expert_executor(&normalized, input_ids, stream, execute)?;
        expand(&feed_forward, residual, &post, &combination, stream)
    }

    pub(crate) fn forward_tensor_parallel(
        &mut self,
        hidden: &Array,
        mask: Option<&Array>,
        cache: Option<&mut AttentionCache>,
        input_ids: &Array,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let residual = hidden;
        let (collapsed, post, combination) =
            self.attn_hc.collapse(hidden, self.norm_epsilon, stream)?;
        let normalized = self.attn_norm.forward(&collapsed, stream)?;
        let attention =
            self.attn
                .forward_tensor_parallel(&normalized, mask, cache, group, stream)?;
        let hidden = expand(&attention, residual, &post, &combination, stream)?;

        let residual = &hidden;
        let (collapsed, post, combination) =
            self.ffn_hc.collapse(&hidden, self.norm_epsilon, stream)?;
        let normalized = self.ffn_norm.forward(&collapsed, stream)?;
        let feed_forward = self.ffn.forward(&normalized, input_ids, stream)?;
        expand(&feed_forward, residual, &post, &combination, stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_tensor_with_expert_executor(
        &mut self,
        hidden: &Array,
        mask: Option<&Array>,
        cache: Option<&mut AttentionCache>,
        input_ids: &Array,
        group: &safemlx::distributed::Group,
        stream: &Stream,
        execute: impl FnMut(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    ) -> Result<Array, Exception> {
        let residual = hidden;
        let (collapsed, post, combination) =
            self.attn_hc.collapse(hidden, self.norm_epsilon, stream)?;
        let normalized = self.attn_norm.forward(&collapsed, stream)?;
        let attention =
            self.attn
                .forward_tensor_parallel(&normalized, mask, cache, group, stream)?;
        let hidden = expand(&attention, residual, &post, &combination, stream)?;

        let residual = &hidden;
        let (collapsed, post, combination) =
            self.ffn_hc.collapse(&hidden, self.norm_epsilon, stream)?;
        let normalized = self.ffn_norm.forward(&collapsed, stream)?;
        let feed_forward =
            self.ffn
                .forward_with_expert_executor(&normalized, input_ids, stream, execute)?;
        expand(&feed_forward, residual, &post, &combination, stream)
    }

    fn prefill_attention_cache(
        &mut self,
        hidden: &Array,
        cache: &mut AttentionCache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let (collapsed, _, _) = self.attn_hc.collapse(hidden, self.norm_epsilon, stream)?;
        let normalized = self.attn_norm.forward(&collapsed, stream)?;
        let _ = self.attn.forward(&normalized, None, Some(cache), stream)?;
        Ok(())
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// DeepSeek-V4 target decoder.
pub struct TextModel {
    #[param]
    pub(crate) embed_tokens: nn::Embedding,
    #[param]
    pub(crate) layers: Vec<DecoderLayer>,
    #[param]
    pub(crate) norm: nn::RmsNorm,
    #[param]
    pub(crate) hc_head: HyperHead,
    args: ModelArgs,
}

impl TextModel {
    fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            embed_tokens: nn::Embedding::unloaded(
                args.vocab_size,
                args.hidden_size,
                Dtype::Float32,
                stream,
            )?,
            layers: (0..args.num_hidden_layers as usize)
                .map(|layer| DecoderLayer::new(args, layer, stream))
                .collect::<Result<_, _>>()?,
            norm: rms_norm(args.hidden_size, args.rms_norm_eps, stream)?,
            hc_head: HyperHead::unloaded(
                args.hc_mult,
                args.hidden_size,
                args.rms_norm_eps,
                args.hc_eps,
                stream,
            )?,
            args: args.clone(),
        })
    }

    fn forward_streams_and_captures(
        &mut self,
        input_ids: &Array,
        mut cache: Option<&mut Cache>,
        capture_layers: &[i32],
        stream: &Stream,
    ) -> Result<(Array, Option<Array>), Exception> {
        let embedded = self.embed_tokens.forward(input_ids, stream)?;
        let batch = embedded.dim(0);
        let tokens = embedded.dim(1);
        let hidden = embedded.try_index_device((.., .., NewAxis, ..), stream)?;
        let mut hidden = broadcast_to(
            &hidden,
            &[batch, tokens, self.args.hc_mult, self.args.hidden_size],
            stream,
        )?;
        let mut captures = Vec::with_capacity(capture_layers.len());
        for (layer_index, layer) in self.layers.iter_mut().enumerate() {
            let layer_cache = cache
                .as_deref_mut()
                .map(|cache| &mut cache.layers[layer_index]);
            hidden = layer.forward(&hidden, None, layer_cache, input_ids, stream)?;
            if let Some(position) = capture_layers
                .iter()
                .position(|wanted| *wanted == layer_index as i32)
            {
                captures.push((position, mean_axis(&hidden, 2, false, stream)?));
            }
        }
        captures.sort_by_key(|(position, _)| *position);
        let captures = if captures.is_empty() {
            None
        } else {
            Some(concatenate_axis(
                &captures.iter().map(|(_, value)| value).collect::<Vec<_>>(),
                -1,
                stream,
            )?)
        };
        Ok((hidden, captures))
    }

    fn forward_streams(
        &mut self,
        input_ids: &Array,
        cache: Option<&mut Cache>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.forward_streams_and_captures(input_ids, cache, &[], stream)
            .map(|(hidden, _)| hidden)
    }

    fn collapse(&mut self, hidden: &Array, stream: &Stream) -> Result<Array, Exception> {
        let hidden = self.hc_head.forward(hidden, stream)?;
        self.norm.forward(&hidden, stream)
    }

    fn forward(
        &mut self,
        input_ids: &Array,
        cache: Option<&mut Cache>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let hidden = self.forward_streams(input_ids, cache, stream)?;
        self.collapse(&hidden, stream)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
struct MtpLayer {
    #[param]
    e_proj: Linear,
    #[param]
    h_proj: Linear,
    #[param]
    enorm: nn::RmsNorm,
    #[param]
    hnorm: nn::RmsNorm,
    #[param]
    decoder: DecoderLayer,
    #[param]
    norm: nn::RmsNorm,
    #[param]
    hc_head: HyperHead,
}

impl MtpLayer {
    fn new(args: &ModelArgs, depth: usize, stream: &Stream) -> Result<Self, Exception> {
        let format = super::layers::projection_format(args);
        let layer = args.num_hidden_layers as usize + depth;
        Ok(Self {
            e_proj: Linear::new(args.hidden_size, args.hidden_size, false, format, stream)?,
            h_proj: Linear::new(args.hidden_size, args.hidden_size, false, format, stream)?,
            enorm: rms_norm(args.hidden_size, args.rms_norm_eps, stream)?,
            hnorm: rms_norm(args.hidden_size, args.rms_norm_eps, stream)?,
            decoder: DecoderLayer::new(args, layer, stream)?,
            norm: rms_norm(args.hidden_size, args.rms_norm_eps, stream)?,
            hc_head: HyperHead::unloaded(
                args.hc_mult,
                args.hidden_size,
                args.rms_norm_eps,
                args.hc_eps,
                stream,
            )?,
        })
    }

    fn forward(
        &mut self,
        hidden: &Array,
        embedded: &Array,
        tokens: &Array,
        cache: &mut AttentionCache,
        head: &mut Linear,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        let embedded = self.enorm.forward(embedded, stream)?;
        let hidden = self.hnorm.forward(hidden, stream)?;
        let embedded = self
            .e_proj
            .forward(&embedded, stream)?
            .try_index_device((.., .., NewAxis, ..), stream)?;
        let hidden = self.h_proj.forward(&hidden, stream)?;
        let fused = embedded.add(hidden, stream)?;
        let hidden = self
            .decoder
            .forward(&fused, None, Some(cache), tokens, stream)?;
        let collapsed = self.hc_head.forward(&hidden, stream)?;
        let normalized = self.norm.forward(&collapsed, stream)?;
        Ok((head.forward(&normalized, stream)?, hidden))
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct MtpModule {
    #[param]
    layers: Vec<MtpLayer>,
}

impl MtpModule {
    fn new(args: &ModelArgs, stream: &Stream) -> Result<Option<Self>, Exception> {
        let count = args.num_nextn_predict_layers.max(0) as usize;
        if count == 0 || args.dspark.is_some() {
            return Ok(None);
        }
        Ok(Some(Self {
            layers: (0..count)
                .map(|depth| MtpLayer::new(args, depth, stream))
                .collect::<Result<_, _>>()?,
        }))
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct DsparkModule {
    #[param]
    layers: Vec<DecoderLayer>,
    #[param]
    main_proj: Linear,
    #[param]
    main_norm: nn::RmsNorm,
    #[param]
    norm: nn::RmsNorm,
    #[param]
    hc_head: HyperHead,
    #[param]
    markov_w1: nn::Embedding,
    #[param]
    markov_w2: Linear,
    #[param]
    confidence_head: Linear,
    sliding_window: i32,
}

impl DsparkModule {
    fn new(args: &ModelArgs, stream: &Stream) -> Result<Option<Self>, Exception> {
        let Some(config) = &args.dspark else {
            return Ok(None);
        };
        let count = args.num_nextn_predict_layers.max(0) as usize;
        if count == 0 {
            return Err(Exception::custom(
                "DSpark requires at least one draft stage",
            ));
        }
        let format = super::layers::projection_format(args);
        Ok(Some(Self {
            layers: (0..count)
                .map(|depth| {
                    DecoderLayer::new(args, args.num_hidden_layers as usize + depth, stream)
                })
                .collect::<Result<_, _>>()?,
            main_proj: Linear::new(
                args.hidden_size * config.target_layer_ids.len() as i32,
                args.hidden_size,
                false,
                format,
                stream,
            )?,
            main_norm: rms_norm(args.hidden_size, args.rms_norm_eps, stream)?,
            norm: rms_norm(args.hidden_size, args.rms_norm_eps, stream)?,
            hc_head: HyperHead::unloaded(
                args.hc_mult,
                args.hidden_size,
                args.rms_norm_eps,
                args.hc_eps,
                stream,
            )?,
            markov_w1: nn::Embedding::unloaded(
                args.vocab_size,
                config.markov_rank,
                Dtype::Float32,
                stream,
            )?,
            markov_w2: Linear::new(
                config.markov_rank,
                args.vocab_size,
                false,
                WeightFormat::Dense,
                stream,
            )?,
            confidence_head: Linear::new(
                args.hidden_size + config.markov_rank,
                1,
                false,
                WeightFormat::Dense,
                stream,
            )?,
            sliding_window: args.sliding_window,
        }))
    }

    fn prefill_context(
        &mut self,
        captures: &Array,
        caches: &mut [AttentionCache],
        hc_mult: i32,
        hidden_size: i32,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let main = self
            .main_norm
            .forward(&self.main_proj.forward(captures, stream)?, stream)?;
        let batch = main.dim(0);
        let tokens = main.dim(1);
        let hidden = broadcast_to(
            &main.try_index_device((.., .., NewAxis, ..), stream)?,
            &[batch, tokens, hc_mult, hidden_size],
            stream,
        )?;
        for (layer, cache) in self.layers.iter_mut().zip(caches.iter_mut()) {
            layer.prefill_attention_cache(&hidden, cache, stream)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn draft_block(
        &mut self,
        anchor: u32,
        capacity: usize,
        caches: &mut [AttentionCache],
        embedding: &mut nn::Embedding,
        head: &mut Linear,
        config: &DsparkConfig,
        hc_mult: i32,
        hidden_size: i32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if capacity == 0 {
            return Err(Exception::custom(
                "DSpark proposal capacity must be positive",
            ));
        }
        let mut ids = vec![config.noise_token_id as u32; capacity];
        ids[0] = anchor;
        let tokens = Array::from_slice(&ids, &[1, ids.len() as i32]);
        let embedded = embedding.forward(&tokens, stream)?;
        let mut hidden = broadcast_to(
            &embedded.try_index_device((.., .., NewAxis, ..), stream)?,
            &[1, ids.len() as i32, hc_mult, hidden_size],
            stream,
        )?;
        for (layer, cache) in self.layers.iter_mut().zip(caches.iter_mut()) {
            let keys = (cache.offset() + ids.len() as i32).min(self.sliding_window);
            let block_mask = Array::ones::<bool>(&[ids.len() as i32, keys], stream)?;
            hidden = layer.forward(&hidden, Some(&block_mask), Some(cache), &tokens, stream)?;
        }
        let hidden = self.hc_head.forward(&hidden, stream)?;
        let hidden = self.norm.forward(&hidden, stream)?;
        head.forward(&hidden, stream)
    }

    fn adjust_logits(
        &mut self,
        logits: Array,
        last_token: u32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let token = Array::from_slice(&[last_token], &[1, 1]);
        let markov = self.markov_w1.forward(&token, stream)?;
        let adjustment = self
            .markov_w2
            .forward(&markov, stream)?
            .try_index_device((.., 0, ..), stream)?;
        logits.add(adjustment, stream)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// DeepSeek-V4 causal language model.
pub struct Model {
    /// Validated model arguments.
    pub args: ModelArgs,
    #[param]
    /// Target decoder.
    pub model: TextModel,
    #[param]
    /// Language-model output projection.
    pub lm_head: Linear,
    #[param]
    pub(crate) mtp: Option<MtpModule>,
    #[param]
    pub(crate) dspark: Option<DsparkModule>,
}

impl Model {
    /// Creates an unloaded target model.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        args.validate()
            .map_err(|error| Exception::custom(error.to_string()))?;
        let head_format = if args.quantization_config.is_some() {
            WeightFormat::Fp8E8M0
        } else {
            WeightFormat::Dense
        };
        Ok(Self {
            model: TextModel::new(&args, stream)?,
            lm_head: Linear::new(
                args.hidden_size,
                args.vocab_size,
                false,
                head_format,
                stream,
            )?,
            mtp: MtpModule::new(&args, stream)?,
            dspark: DsparkModule::new(&args, stream)?,
            args,
        })
    }

    /// Allocates cache state for every local/compressed attention layer.
    pub fn new_cache(&self) -> Result<Cache, Exception> {
        Ok(Cache {
            layers: self
                .model
                .layers
                .iter()
                .map(|layer| layer.attn.new_cache(self.args.sliding_window))
                .collect::<Result<_, _>>()?,
            mtp_layers: if let Some(mtp) = &self.mtp {
                mtp.layers
                    .iter()
                    .map(|layer| layer.decoder.attn.new_cache(self.args.sliding_window))
                    .collect::<Result<_, _>>()?
            } else if let Some(dspark) = &self.dspark {
                dspark
                    .layers
                    .iter()
                    .map(|layer| layer.attn.new_cache(self.args.sliding_window))
                    .collect::<Result<_, _>>()?
            } else {
                Vec::new()
            },
        })
    }

    pub(crate) fn mtp_len(&self) -> usize {
        self.dspark.as_ref().map_or_else(
            || self.mtp.as_ref().map_or(0, |mtp| mtp.layers.len()),
            |_| self.args.dspark.as_ref().unwrap().block_size as usize,
        )
    }

    fn forward_mtp_target(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<crate::runtime::generation::embedded_mtp::EmbeddedMtpOutput, Exception> {
        let capture_layers = self
            .args
            .dspark
            .as_ref()
            .map_or(&[][..], |config| config.target_layer_ids.as_slice());
        let (hidden, captures) =
            self.model
                .forward_streams_and_captures(tokens, Some(cache), capture_layers, stream)?;
        let collapsed = self.model.collapse(&hidden, stream)?;
        let logits = self.lm_head.forward(&collapsed, stream)?;
        Ok(
            crate::runtime::generation::embedded_mtp::EmbeddedMtpOutput {
                logits,
                hidden: captures.unwrap_or(hidden),
                tokens: tokens.clone(),
            },
        )
    }

    pub(crate) fn forward_mtp_draft(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        depth: usize,
        cache: &mut [AttentionCache],
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        let embedded = self.model.embed_tokens.forward(tokens, stream)?;
        let mtp = self
            .mtp
            .as_mut()
            .ok_or_else(|| Exception::custom("DeepSeek-V4 checkpoint has no embedded MTP"))?;
        let count = mtp.layers.len();
        let index = depth % count;
        mtp.layers[index].forward(
            hidden,
            &embedded,
            tokens,
            &mut cache[index],
            &mut self.lm_head,
            stream,
        )
    }

    /// Returns the cache layout when it can be represented by the generic
    /// prompt-cache schema. Compressed sparse V4 layers carry pooled and
    /// indexer state and therefore require the richer V4 cache schema.
    pub(crate) fn prompt_cache_layer_layout(
        &self,
    ) -> Result<LayerSchedule<LayerCachePolicy>, Error> {
        if self.args.compress_ratios.iter().any(|ratio| *ratio != 0) {
            return Err(Error::UnsupportedArchitecture(
                "DeepSeek-V4 compressed sparse prompt caches require pooled/indexer state".into(),
            ));
        }
        let policy = LayerCachePolicy::key_value(
            AttentionPolicy::sliding(self.args.sliding_window as u32)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
            self.args.num_attention_heads,
            self.args.head_dim,
        )
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        LayerSchedule::new(
            self.args.num_hidden_layers as usize,
            vec![policy; self.args.num_hidden_layers as usize],
        )
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }

    /// Runs target decoding and returns vocabulary logits.
    pub fn forward(
        &mut self,
        input_ids: &Array,
        cache: Option<&mut Cache>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let hidden = self.model.forward(input_ids, cache, stream)?;
        self.lm_head.forward(&hidden, stream)
    }
}

/// Stable identity for all cache-relevant V4 geometry.
pub(crate) fn prompt_cache_architecture_fingerprint(args: &ModelArgs) -> String {
    derive_prompt_cache_architecture_fingerprint(
        "deepseek_v4",
        [
            ("model_type", args.model_type.clone()),
            ("hidden_size", args.hidden_size.to_string()),
            ("num_hidden_layers", args.num_hidden_layers.to_string()),
            ("num_attention_heads", args.num_attention_heads.to_string()),
            ("head_dim", args.head_dim.to_string()),
            ("qk_rope_head_dim", args.qk_rope_head_dim.to_string()),
            ("q_lora_rank", args.q_lora_rank.to_string()),
            ("sliding_window", args.sliding_window.to_string()),
            ("compress_ratios", format!("{:?}", args.compress_ratios)),
            ("index_n_heads", args.index_n_heads.to_string()),
            ("index_head_dim", args.index_head_dim.to_string()),
            ("index_topk", args.index_topk.to_string()),
            ("hc_mult", args.hc_mult.to_string()),
        ],
    )
}

/// Reads and validates V4 arguments from a model directory.
pub fn get_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let value: Value =
        serde_json::from_slice(&std::fs::read(model_dir.as_ref().join("config.json"))?)?;
    ModelArgs::from_value(value)
}

/// Loads an official dense or mixed FP4/FP8 V4 SafeTensors checkpoint.
pub fn load_model(
    model_dir: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let args = get_model_args(model_dir)?;
    crate::api::structural::validate_safetensors_load_path(
        crate::api::ModelKind::DeepSeekV4,
        model_dir,
        crate::api::ModelLoadOptions::default(),
    )?;
    let mut model = Model::new(args.clone(), stream)?;
    let config = StrictLoadConfig::default();
    let mut report = StrictLoadReport::default();
    load_safetensors_dir_strict_with_split_swiglu_experts_and_transform(
        &mut model,
        model_dir,
        weights_stream,
        stream,
        None,
        &config,
        &mut report,
        args.n_routed_experts,
        |key, value| transform_checkpoint_tensor(&args, key, value, stream),
    )?;
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;
    Ok(model)
}

fn transform_checkpoint_tensor(
    args: &ModelArgs,
    mut key: String,
    mut value: Array,
    stream: &Stream,
) -> Result<Vec<(String, Array)>, Error> {
    if let Some(rest) = key.strip_prefix("mtp.").map(str::to_owned) {
        let (depth, field) = rest.split_once('.').ok_or_else(|| {
            Error::UnsupportedArchitecture(format!("invalid DeepSeek-V4 MTP tensor {key:?}"))
        })?;
        if args.dspark.is_some() {
            key = match field {
                "main_proj.weight" | "main_proj.scale" => {
                    format!("dspark.main_proj.{}", field.rsplit_once('.').unwrap().1)
                }
                "main_norm.weight" => "dspark.main_norm.weight".into(),
                "norm.weight" => "dspark.norm.weight".into(),
                "markov_head.markov_w1.weight" => "dspark.markov_w1.weight".into(),
                "markov_head.markov_w2.weight" => "dspark.markov_w2.weight".into(),
                "confidence_head.proj.weight" => "dspark.confidence_head.weight".into(),
                "hc_head_fn" => "dspark.hc_head.function".into(),
                "hc_head_base" => "dspark.hc_head.base".into(),
                "hc_head_scale" => "dspark.hc_head.scale".into(),
                other => format!("dspark.layers.{depth}.{other}"),
            };
        } else {
            let direct = matches!(
                field,
                "e_proj.weight"
                    | "e_proj.scale"
                    | "h_proj.weight"
                    | "h_proj.scale"
                    | "enorm.weight"
                    | "hnorm.weight"
                    | "norm.weight"
                    | "hc_head_fn"
                    | "hc_head_base"
                    | "hc_head_scale"
            );
            key = match field {
                "hc_head_fn" => format!("mtp.layers.{depth}.hc_head.function"),
                "hc_head_base" => format!("mtp.layers.{depth}.hc_head.base"),
                "hc_head_scale" => format!("mtp.layers.{depth}.hc_head.scale"),
                other if direct => format!("mtp.layers.{depth}.{other}"),
                other => format!("mtp.layers.{depth}.decoder.{other}"),
            };
        }
    }
    key = match key.as_str() {
        "embed.weight" => "model.embed_tokens.weight".into(),
        "norm.weight" => "model.norm.weight".into(),
        "head.weight" => "lm_head.weight".into(),
        "hc_head_fn" => "model.hc_head.function".into(),
        "hc_head_base" => "model.hc_head.base".into(),
        "hc_head_scale" => "model.hc_head.scale".into(),
        _ if key.starts_with("layers.") => format!("model.{key}"),
        _ => key,
    };
    for sublayer in ["attn", "ffn"] {
        for (raw, runtime) in [("fn", "function"), ("base", "base"), ("scale", "scale")] {
            key = key.replace(
                &format!(".hc_{sublayer}_{raw}"),
                &format!(".{sublayer}_hc.{runtime}"),
            );
        }
    }
    for (raw, runtime) in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")] {
        key = key.replace(
            &format!(".ffn.shared_experts.{raw}."),
            &format!(".ffn.shared_experts.{runtime}."),
        );
    }
    key = key
        .replace(".ffn.gate.weight", ".ffn.gate.router.weight")
        .replace(".ffn.gate.bias", ".ffn.gate.router.e_score_correction_bias")
        .replace(".ffn.experts.", ".ffn.switch_mlp.");

    // Official wo_a is one 2-D projection. The runtime keeps the group axis
    // explicit so TP can partition groups without a bespoke attention path.
    if let Some(prefix) = key.strip_suffix(".attn.wo_a.weight") {
        let rows = args.o_lora_rank;
        let mut output = Vec::with_capacity(args.o_groups as usize);
        for group in 0..args.o_groups {
            output.push((
                format!("{prefix}.attn.wo_a.projections.{group}.weight"),
                value.try_index_device((group * rows..(group + 1) * rows, ..), stream)?,
            ));
        }
        return Ok(output);
    }
    if let Some(prefix) = key.strip_suffix(".attn.wo_a.scale") {
        let scale_rows = (args.o_lora_rank + 127) / 128;
        let mut output = Vec::with_capacity(args.o_groups as usize);
        for group in 0..args.o_groups {
            output.push((
                format!("{prefix}.attn.wo_a.projections.{group}.weight_scale_inv"),
                value
                    .try_index_device((group * scale_rows..(group + 1) * scale_rows, ..), stream)?,
            ));
        }
        return Ok(output);
    }

    let expert_component = key.contains(".ffn.switch_mlp.");
    if expert_component && key.ends_with(".weight") && args.expert_dtype.as_deref() == Some("fp4") {
        value = value.view::<u32>(stream)?;
    }
    if key.ends_with(".scale") {
        key.truncate(key.len() - ".scale".len());
        key.push_str(if expert_component {
            ".scales"
        } else {
            ".weight_scale_inv"
        });
    }
    Ok(vec![(key, value)])
}

impl CausalLm<Cache> for Model {
    fn prefill_input_logits(
        &mut self,
        input: runtime_input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let tokens = runtime_input::text_token_ids(input, stream)?;
        self.forward(&tokens, Some(cache), stream)?
            .try_index_device((.., -1, ..), stream)
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.forward(input_tokens, Some(cache), stream)?
            .try_index_device((.., -1, ..), stream)
    }
}

impl crate::runtime::generation::embedded_mtp::EmbeddedMtpTarget for Model {
    type Cache = Cache;
    type DraftCache = Vec<AttentionCache>;

    fn prefill_target(
        &mut self,
        input: runtime_input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<crate::runtime::generation::embedded_mtp::EmbeddedMtpOutput, Exception> {
        let tokens = runtime_input::text_token_ids(input, stream)?;
        *cache = self.new_cache()?;
        self.forward_mtp_target(&tokens, cache, stream)
    }

    fn verify_target(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<crate::runtime::generation::embedded_mtp::EmbeddedMtpOutput, Exception> {
        self.forward_mtp_target(tokens, cache, stream)
    }

    fn prefill_draft_cache(
        &mut self,
        output: &crate::runtime::generation::embedded_mtp::EmbeddedMtpOutput,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let sequence = tokens.dim(1);
        if let Some(dspark) = &mut self.dspark {
            return dspark.prefill_context(
                &output.hidden,
                &mut cache.mtp_layers,
                self.args.hc_mult,
                self.args.hidden_size,
                stream,
            );
        }
        if sequence <= 1 {
            return Ok(());
        }
        let hidden = output
            .hidden
            .try_index_device((.., ..sequence - 1, .., ..), stream)?;
        let next = tokens.try_index_device((.., 1..), stream)?;
        for depth in 0..cache.mtp_layers.len() {
            let _ = self.forward_mtp_draft(&hidden, &next, depth, &mut cache.mtp_layers, stream)?;
        }
        Ok(())
    }

    fn draft_cache(cache: &Cache) -> Self::DraftCache {
        cache.mtp_layers.clone()
    }

    fn commit_draft_cache(cache: &mut Cache, draft: &Self::DraftCache) {
        cache.mtp_layers.clone_from(draft);
    }

    fn draft_logits(
        &mut self,
        hidden: &Array,
        last_token: u32,
        draft_index: usize,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        if self.dspark.is_some() {
            return Err(Exception::custom(
                "DSpark uses fused block execution, not sequential MTP layers",
            ));
        }
        let token = Array::from_slice(&[last_token], &[1, 1]);
        self.forward_mtp_draft(hidden, &token, draft_index, cache, stream)
    }

    fn fused_draft_logits(
        &mut self,
        _hidden: &Array,
        last_token: u32,
        proposal_capacity: usize,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<Option<Array>, Exception> {
        let Some(dspark) = self.dspark.as_mut() else {
            return Ok(None);
        };
        // Proposal-block self attention is transactional. Only accepted target
        // captures advance the canonical DSpark context cache during commit.
        let mut proposal_cache = cache.clone();
        let logits = dspark.draft_block(
            last_token,
            proposal_capacity,
            &mut proposal_cache,
            &mut self.model.embed_tokens,
            &mut self.lm_head,
            self.args.dspark.as_ref().unwrap(),
            self.args.hc_mult,
            self.args.hidden_size,
            stream,
        )?;
        Ok(Some(logits))
    }

    fn adjust_fused_draft_logits(
        &mut self,
        logits: Array,
        last_token: u32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.dspark
            .as_mut()
            .ok_or_else(|| Exception::custom("DeepSeek-V4 has no fused DSpark module"))?
            .adjust_logits(logits, last_token, stream)
    }

    fn advance_draft_cache(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        if let Some(dspark) = &mut self.dspark {
            return dspark.prefill_context(
                hidden,
                cache,
                self.args.hc_mult,
                self.args.hidden_size,
                stream,
            );
        }
        for depth in 0..cache.len() {
            let _ = self.forward_mtp_draft(hidden, tokens, depth, cache, stream)?;
        }
        Ok(())
    }

    fn max_draft_tokens(&self) -> usize {
        self.mtp_len()
    }
}

/// DeepSeek-V4 token generation iterator.
pub type Generate<'a, S = crate::runtime::generation::sampler::DefaultSampler> =
    crate::nn::generation::Generate<'a, Model, Cache, S>;

#[cfg(test)]
mod tests {
    use super::ModelArgs;
    use serde_json::json;

    fn minimal_config() -> serde_json::Value {
        json!({
            "model_type": "deepseek_v4",
            "hidden_size": 16,
            "moe_intermediate_size": 8,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "qk_rope_head_dim": 4,
            "q_lora_rank": 8,
            "o_lora_rank": 8,
            "o_groups": 2,
            "vocab_size": 32,
            "max_position_embeddings": 4096,
            "compress_ratios": [0, 4, 128, 0],
            "index_n_heads": 4,
            "index_head_dim": 4,
            "index_topk": 2,
            "n_routed_experts": 8,
            "num_experts_per_tok": 2,
            "num_nextn_predict_layers": 1
        })
    }

    #[test]
    fn mtp_extends_decoder_compression_schedule() {
        let args = ModelArgs::from_value(minimal_config()).unwrap();
        assert_eq!(args.compress_ratios, [0, 4, 128, 0]);
        assert_eq!(args.num_nextn_predict_layers, 1);
    }

    #[test]
    fn rejects_unknown_compression_ratios() {
        let mut value = minimal_config();
        value["compress_ratios"] = json!([0, 8, 128, 0]);
        assert!(ModelArgs::from_value(value).is_err());
    }

    #[test]
    fn defaults_append_local_attention_for_prediction_blocks() {
        let mut value = minimal_config();
        value["compress_ratios"] = json!([]);
        assert_eq!(
            ModelArgs::from_value(value).unwrap().compress_ratios,
            [0, 4, 0, 0]
        );
    }

    #[test]
    fn prediction_blocks_reject_compressed_attention() {
        let mut value = minimal_config();
        value["compress_ratios"] = json!([0, 4, 128, 4]);
        assert!(ModelArgs::from_value(value).is_err());
    }

    #[test]
    fn fused_dspark_metadata_is_atomic() {
        let mut value = minimal_config();
        value["dspark_block_size"] = json!(4);
        assert!(ModelArgs::from_value(value).is_err());

        let mut value = minimal_config();
        value["dspark_block_size"] = json!(4);
        value["dspark_noise_token_id"] = json!(0);
        value["dspark_target_layer_ids"] = json!([0, 2]);
        value["dspark_markov_rank"] = json!(8);
        assert!(ModelArgs::from_value(value).unwrap().has_dspark());
    }
}
