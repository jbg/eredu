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
    ops::{broadcast_to, indexing::NewAxis, indexing::TryIndexOp},
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
        // Compression scheduling belongs to the target stack. Built-in MTP
        // reuses target-layer captures and does not append phantom entries to
        // this list (matching the official configuration schema).
        let required_ratios = usize::try_from(self.num_hidden_layers)
            .map_err(|_| unsupported("layer count is negative"))?;
        if self.compress_ratios.len() != required_ratios {
            return Err(unsupported(format!(
                "compress_ratios has {} entries but {required_ratios} target layers require entries",
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
    layers: Vec<AttentionCache>,
}

impl Cache {
    /// Current target-decoder sequence offset.
    pub fn offset(&self) -> i32 {
        self.layers.first().map_or(0, AttentionCache::offset)
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
    fn new(args: &ModelArgs, layer: usize, stream: &Stream) -> Result<Self, Exception> {
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

    fn forward(
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

    fn forward(
        &mut self,
        input_ids: &Array,
        mut cache: Option<&mut Cache>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let embedded = self.embed_tokens.forward(input_ids, stream)?;
        let batch = embedded.dim(0);
        let tokens = embedded.dim(1);
        let hidden = embedded.try_index_device((.., .., NewAxis, ..), stream)?;
        let mut hidden = broadcast_to(
            &hidden,
            &[batch, tokens, self.args.hc_mult, self.args.hidden_size],
            stream,
        )?;
        for (layer_index, layer) in self.layers.iter_mut().enumerate() {
            let layer_cache = cache
                .as_deref_mut()
                .map(|cache| &mut cache.layers[layer_index]);
            hidden = layer.forward(&hidden, None, layer_cache, input_ids, stream)?;
        }
        let hidden = self.hc_head.forward(&hidden, stream)?;
        self.norm.forward(&hidden, stream)
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
        })
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
    let mut config = StrictLoadConfig::default().allow_unused_prefix("mtp.");
    for layer in args.num_hidden_layers..args.num_hidden_layers + args.num_nextn_predict_layers {
        config = config.allow_unused_prefix(format!("layers.{layer}."));
    }
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
            "compress_ratios": [0, 4, 128],
            "index_n_heads": 4,
            "index_head_dim": 4,
            "index_topk": 2,
            "n_routed_experts": 8,
            "num_experts_per_tok": 2,
            "num_nextn_predict_layers": 1
        })
    }

    #[test]
    fn mtp_does_not_extend_target_compression_schedule() {
        let args = ModelArgs::from_value(minimal_config()).unwrap();
        assert_eq!(args.compress_ratios, [0, 4, 128]);
        assert_eq!(args.num_nextn_predict_layers, 1);
    }

    #[test]
    fn rejects_unknown_compression_ratios() {
        let mut value = minimal_config();
        value["compress_ratios"] = json!([0, 8, 128]);
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
