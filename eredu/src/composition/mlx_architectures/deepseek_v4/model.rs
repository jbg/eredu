//! DeepSeek-V4 configuration and model components.

//!
//! The public configuration deliberately describes the architecture rather
//! than individual repository names.  Flash, Pro, Base, Instruct, MTP, and
//! fused DSpark checkpoints therefore share one validation path.

use eredu_checkpoint::WeightQuantization;
use eredu_runtime::CausalModel;

use std::{
    collections::{BTreeMap, HashMap},
    num::NonZeroU32,
    path::Path,
};

use serde::Deserialize;
use serde_json::Value;

use safemlx::{
    error::Exception,
    macros::ModuleParameters,
    module::Module,
    nn,
    ops::{
        broadcast_to, concatenate_axis, indexing::NewAxis, indexing::TryIndexOp, mean_axis,
        GgufCheckpoint, GgufMetadataArray, GgufMetadataValue,
    },
    Array, Dtype, Stream,
};

use crate::core::cache::{
    derive_prompt_cache_architecture_fingerprint, validate_prompt_cache_model_identity,
    PromptCacheDescriptor, PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
    PromptCacheTopology,
};

use crate::{
    backend::mlx::error::Error,
    backend::mlx::nn::hyper_connections::{expand, HyperConnection, HyperHead},
    backend::mlx::runtime::cache::residency::{
        load_prompt_cache_state_tensors, open_prompt_cache, save_prompt_cache_snapshot,
        CacheBlockArrays, CacheResidencyManager, CacheResidencyPolicy, CacheResidencyReport,
        PagedCacheOptions, PromptCacheSnapshotBlock,
    },
    backend::mlx::runtime::checkpoint::load::gguf_quantization_configs,
    backend::mlx::runtime::media::input as runtime_input,
    composition::mlx_architectures::qwen::hybrid::qwen3_5::{
        QwenLinear as Linear, QwenWeightFormat as WeightFormat,
    },
    core::attention::{AttentionPolicy, LayerSchedule},
    core::cache::{
        CacheRankIdentity, LayerCachePolicy, MutableStateResidency, PoolingStateComponent,
        StateTensorDimension, StateTensorDtype, StateTensorPolicy, StateTensorRole,
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
    /// Uniform runtime format selected by load-time transformation.
    pub quantization: Option<WeightQuantization>,
    /// Exact per-weight formats reconstructed from a mixed GGUF catalog.
    pub quantized_weight_configs: Option<HashMap<String, WeightQuantization>>,
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

    pub(crate) fn weight_format_for(&self, weight_name: &str) -> WeightFormat {
        if self.quantization_config.is_some() {
            WeightFormat::Fp8E8M0
        } else if let Some(format) = self
            .quantized_weight_configs
            .as_ref()
            .and_then(|formats| formats.get(weight_name))
        {
            match *format {
                iq @ WeightQuantization::GgufIQuant { .. } => WeightFormat::IQuant(iq),
                affine => WeightFormat::Affine(affine),
            }
        } else if let Some(quantization) = self.quantization {
            match quantization {
                iq @ WeightQuantization::GgufIQuant { .. } => WeightFormat::IQuant(iq),
                affine => WeightFormat::Affine(affine),
            }
        } else {
            WeightFormat::Dense
        }
    }

    pub(crate) fn has_checkpoint_native_quantization(&self) -> bool {
        self.quantization_config.is_some()
            || self.expert_dtype.is_some()
            || self
                .quantized_weight_configs
                .as_ref()
                .is_some_and(|formats| !formats.is_empty())
    }

    pub(crate) fn resolve_load_time_quantization(
        &self,
        context: &str,
        requested: Option<WeightQuantization>,
    ) -> Result<Option<WeightQuantization>, Error> {
        let Some(requested) = requested else {
            return Ok(None);
        };
        requested.validate()?;
        if self.has_checkpoint_native_quantization() {
            return Err(Error::Quantization(format!(
                "{context} checkpoint already contains checkpoint-native mixed or FP8 weights; implicit dequantization and requantization is unsupported"
            )));
        }
        crate::backend::mlx::runtime::checkpoint::quantization::should_quantize_on_load(
            context,
            self.quantization,
            requested,
        )
        .map(|required| required.then_some(requested))
    }

    pub(crate) fn with_load_time_quantization(
        &self,
        quantization: WeightQuantization,
    ) -> Result<Self, Error> {
        quantization.validate()?;
        let mut target = self.clone();
        target.expert_dtype = None;
        target.quantization_config = None;
        target.quantized_weight_configs = None;
        target.quantization = Some(quantization);
        target.validate()?;
        Ok(target)
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
        if self.quantization_config.is_some() && self.quantized_weight_configs.is_some() {
            return Err(unsupported(
                "native FP8 metadata cannot be combined with GGUF per-weight formats",
            ));
        }
        if self.quantization.is_some()
            && (self.quantization_config.is_some()
                || self.expert_dtype.is_some()
                || self.quantized_weight_configs.is_some())
        {
            return Err(unsupported(
                "load-time quantization cannot be combined with checkpoint-native formats",
            ));
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
            quantization: None,
            quantized_weight_configs: None,
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

    pub(crate) fn reset(&mut self) -> Result<(), Exception> {
        if let Some(manager) = self.residency_manager().cloned() {
            manager
                .clear()
                .map_err(|error| Exception::custom(error.to_string()))?;
        }
        for cache in self.layers.iter_mut().chain(&mut self.mtp_layers) {
            cache.reset_local_after_manager_clear();
        }
        Ok(())
    }

    fn residency_manager(&self) -> Option<&CacheResidencyManager> {
        self.layers
            .iter()
            .chain(&self.mtp_layers)
            .find_map(AttentionCache::residency_manager)
    }

    /// Returns aggregate paging observations for target and embedded draft state.
    pub fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        self.residency_manager()
            .map(|manager| {
                manager
                    .report()
                    .map_err(|error| Exception::custom(error.to_string()))
            })
            .transpose()
    }

    pub(crate) fn restore_target_checkpoint(
        &mut self,
        checkpoint: &Self,
        stream: &Stream,
    ) -> Result<(), Exception> {
        if self.layers.len() != checkpoint.layers.len() {
            return Err(Exception::custom(
                "DeepSeek V4 target cache checkpoint has a different layer count",
            ));
        }
        for (cache, previous) in self.layers.iter_mut().zip(&checkpoint.layers) {
            cache.restore_checkpoint(previous, stream)?;
        }
        Ok(())
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
        let layer = args.num_hidden_layers as usize + depth;
        let root = format!("mtp.{depth}");
        Ok(Self {
            e_proj: Linear::new(
                args.hidden_size,
                args.hidden_size,
                false,
                super::layers::projection_format(args, &format!("{root}.e_proj.weight")),
                stream,
            )?,
            h_proj: Linear::new(
                args.hidden_size,
                args.hidden_size,
                false,
                super::layers::projection_format(args, &format!("{root}.h_proj.weight")),
                stream,
            )?,
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
                super::layers::projection_format(args, "mtp.0.main_proj.weight"),
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
                super::layers::projection_format(
                    args,
                    &format!("mtp.{}.markov_head.markov_w2.weight", count - 1),
                ),
                stream,
            )?,
            confidence_head: Linear::new(
                args.hidden_size + config.markov_rank,
                1,
                false,
                super::layers::projection_format(
                    args,
                    &format!("mtp.{}.confidence_head.proj.weight", count - 1),
                ),
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
        let head_format = args.weight_format_for("head.weight");
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
        empty_cache_for_args(&self.args).map_err(|error| Exception::custom(error.to_string()))
    }

    /// Allocates resident or explicitly bounded key-only attention state.
    pub fn new_cache_with_options(&self, policy: CacheResidencyPolicy) -> Result<Cache, Exception> {
        match policy {
            CacheResidencyPolicy::Device => self.new_cache(),
            CacheResidencyPolicy::Paged(options) => {
                let manager = CacheResidencyManager::new(options)
                    .map_err(|error| Exception::custom(error.to_string()))?;
                paged_cache_for_args(&self.args, manager, 0, None)
                    .map_err(|error| Exception::custom(error.to_string()))
            }
        }
    }

    pub(crate) fn new_cache_with_manager(
        &self,
        manager: CacheResidencyManager,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Cache, Exception> {
        paged_cache_for_args(&self.args, manager, 0, rank)
            .map_err(|error| Exception::custom(error.to_string()))
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
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
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
            crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput {
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

    /// Returns the exact generic prompt-cache layout, including compressed
    /// pooling and sparse-indexer state.
    pub(crate) fn prompt_cache_layer_layout(
        &self,
    ) -> Result<LayerSchedule<LayerCachePolicy>, Error> {
        prompt_cache_layer_layout(&self.args, 0..prompt_cache_layer_count(&self.args))
    }

    /// Atomically persists target and speculative attention state plus
    /// compressed pooling/indexer state.
    pub fn save_prompt_cache(
        &self,
        cache: &mut Cache,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        stream: &Stream,
    ) -> Result<PromptCacheManifest, Error> {
        validate_prompt_cache_model_identity(
            &descriptor,
            &prompt_cache_model_identity(&self.args, PromptCacheTopology::default())?,
        )
        .map_err(|error| Exception::custom(error.to_string()))?;
        save_prompt_cache_with_rank(
            cache,
            destination,
            descriptor,
            prefix_token_ids,
            options,
            None,
            stream,
        )
    }

    /// Restores target and speculative attention state plus compressed
    /// pooling/indexer state exactly.
    pub fn load_prompt_cache(
        &self,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(Cache, PromptCacheManifest), Error> {
        load_prompt_cache_with_identity(
            &self.args,
            directory,
            expected,
            prefix_token_ids,
            prompt_cache_model_identity(&self.args, PromptCacheTopology::default())?,
            options,
            stream,
        )
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

fn pooling_role(stream: u32, component: PoolingStateComponent) -> StateTensorRole {
    StateTensorRole::Pooling { stream, component }
}

fn pooling_state_policies(
    stream: u32,
    ratio: i32,
    width: i32,
    overlapping: bool,
) -> Result<Vec<StateTensorPolicy>, Error> {
    let divisor = NonZeroU32::new(ratio as u32).ok_or_else(|| {
        Error::UnsupportedArchitecture(format!("invalid DeepSeek V4 compression ratio {ratio}"))
    })?;
    let fixed = |value| {
        StateTensorDimension::fixed(value)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    };
    let policy = |component, shape| {
        StateTensorPolicy::new(
            pooling_role(stream, component),
            shape,
            StateTensorDtype::Floating,
            MutableStateResidency::AlwaysDeviceMutable,
        )
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    };
    let source_width = if overlapping { width * 2 } else { width };
    let mut policies = vec![
        policy(
            PoolingStateComponent::PendingValues,
            vec![
                StateTensorDimension::Batch,
                StateTensorDimension::PrefixTokensRem(divisor),
                fixed(source_width)?,
            ],
        )?
        .when_prefix_remainder_nonzero(divisor),
        policy(
            PoolingStateComponent::PendingGates,
            vec![
                StateTensorDimension::Batch,
                StateTensorDimension::PrefixTokensRem(divisor),
                fixed(source_width)?,
            ],
        )?
        .when_prefix_remainder_nonzero(divisor),
        policy(
            PoolingStateComponent::Pooled,
            vec![
                StateTensorDimension::Batch,
                StateTensorDimension::PrefixTokensDiv(divisor),
                fixed(width)?,
            ],
        )?
        .when_prefix_at_least(divisor),
    ];
    if overlapping {
        for component in [
            PoolingStateComponent::OverlapValues,
            PoolingStateComponent::OverlapGates,
        ] {
            policies.push(
                policy(
                    component,
                    vec![StateTensorDimension::Batch, fixed(ratio)?, fixed(width)?],
                )?
                .when_prefix_at_least(divisor),
            );
        }
    }
    Ok(policies)
}

pub(crate) fn prompt_cache_layer_layout(
    args: &ModelArgs,
    range: std::ops::Range<usize>,
) -> Result<LayerSchedule<LayerCachePolicy>, Error> {
    let attention = AttentionPolicy::sliding(args.sliding_window as u32)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let mut policies = Vec::with_capacity(range.len());
    for layer in range.clone() {
        let ratio = args.compress_ratios[layer];
        let policy = if ratio == 0 {
            LayerCachePolicy::key_only(attention, args.num_key_value_heads, args.head_dim)
        } else {
            let mut tensors = pooling_state_policies(0, ratio, args.head_dim, ratio == 4)?;
            if ratio == 4 {
                tensors.extend(pooling_state_policies(1, ratio, args.index_head_dim, true)?);
            }
            LayerCachePolicy::key_only_with_fixed_state(
                attention,
                args.num_key_value_heads,
                args.head_dim,
                tensors,
            )
        }
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        policies.push(policy);
    }
    LayerSchedule::new(range.len(), policies)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

pub(crate) fn prompt_cache_model_identity(
    args: &ModelArgs,
    topology: PromptCacheTopology,
) -> Result<PromptCacheModelIdentity, Error> {
    let layer_count = prompt_cache_layer_count(args);
    prompt_cache_model_identity_for_range(args, topology, 0..layer_count)
}

pub(crate) fn prompt_cache_model_identity_for_range(
    args: &ModelArgs,
    topology: PromptCacheTopology,
    range: std::ops::Range<usize>,
) -> Result<PromptCacheModelIdentity, Error> {
    let layer_count = prompt_cache_layer_count(args);
    Ok(PromptCacheModelIdentity {
        model_family: "deepseek_v4".into(),
        effective_model_type: args.model_type.clone(),
        architecture_fingerprint: prompt_cache_architecture_fingerprint(args),
        layer_count,
        global_layer_start: range.start,
        global_layer_end: range.end,
        sink_tokens: 0,
        layer_prefix_offsets: prompt_cache_layer_prefix_offsets(args, range.clone()),
        topology,
        layer_layout: prompt_cache_layer_layout(args, range)?,
    })
}

pub(crate) fn prompt_cache_layer_count(args: &ModelArgs) -> usize {
    args.num_hidden_layers as usize + args.num_nextn_predict_layers.max(0) as usize
}

pub(crate) fn prompt_cache_layer_prefix_offsets(
    args: &ModelArgs,
    range: std::ops::Range<usize>,
) -> Vec<i32> {
    let target_count = args.num_hidden_layers as usize;
    range
        .map(|layer| {
            if layer >= target_count && args.dspark.is_none() {
                -1
            } else {
                0
            }
        })
        .collect()
}

pub(crate) fn save_prompt_cache_with_rank(
    cache: &mut Cache,
    destination: impl AsRef<Path>,
    descriptor: PromptCacheDescriptor,
    prefix_token_ids: &[u32],
    options: &PromptCacheOptions,
    rank: Option<CacheRankIdentity>,
    stream: &Stream,
) -> Result<PromptCacheManifest, Error> {
    if let Some(manager) = cache.residency_manager().cloned() {
        for layer in cache.layers.iter_mut().chain(&mut cache.mtp_layers) {
            layer.finalize()?;
        }
        let state = cache
            .layers
            .iter()
            .chain(&cache.mtp_layers)
            .enumerate()
            .flat_map(|(index, cache)| {
                cache.prompt_cache_state_arrays(descriptor.global_layer_start + index)
            })
            .collect::<Vec<_>>();
        return manager
            .save_prompt_cache(destination, descriptor, prefix_token_ids, &state, options)
            .map_err(|error| Exception::custom(error.to_string()).into());
    }
    let caches = cache
        .layers
        .iter()
        .chain(&cache.mtp_layers)
        .cloned()
        .collect::<Vec<_>>();
    save_attention_prompt_cache(
        &caches,
        destination,
        descriptor,
        prefix_token_ids,
        options,
        rank,
        stream,
    )
}

pub(crate) fn save_attention_prompt_cache(
    caches: &[AttentionCache],
    destination: impl AsRef<Path>,
    descriptor: PromptCacheDescriptor,
    prefix_token_ids: &[u32],
    options: &PromptCacheOptions,
    rank: Option<CacheRankIdentity>,
    stream: &Stream,
) -> Result<PromptCacheManifest, Error> {
    let prefix_end = i32::try_from(prefix_token_ids.len())
        .map_err(|_| Error::UnsupportedArchitecture("V4 prompt length exceeds i32".into()))?;
    if caches.len() != descriptor.layer_layout.len() {
        return Err(Error::UnsupportedArchitecture(format!(
            "V4 cache contains {} layers, descriptor contains {}",
            caches.len(),
            descriptor.layer_layout.len()
        )));
    }
    let mut blocks = Vec::with_capacity(caches.len());
    for (index, layer_cache) in caches.iter().enumerate() {
        let end = prefix_end
            .checked_add(descriptor.layer_prefix_offsets[index])
            .ok_or_else(|| Error::UnsupportedArchitecture("V4 layer offset overflow".into()))?;
        if end < 0 || layer_cache.offset() != end {
            return Err(Error::UnsupportedArchitecture(format!(
                "V4 layer {index} cache offset {} does not match prefix {end}",
                layer_cache.offset()
            )));
        }
        let Some((keys, values)) = layer_cache.local_snapshot(stream)? else {
            if end == 0 {
                continue;
            }
            return Err(Error::UnsupportedArchitecture(format!(
                "V4 layer {index} local KV state is missing"
            )));
        };
        if values.dim(-1) != 0 {
            return Err(Error::UnsupportedArchitecture(format!(
                "V4 layer {index} local cache unexpectedly contains value channels"
            )));
        }
        // Safetensors deliberately rejects zero-element arrays. Key-only
        // blocks therefore persist one zero sentinel channel per token; the
        // canonical KeyOnly policy validates it and restore drops it.
        let mut sentinel_shape = keys.shape().to_vec();
        *sentinel_shape
            .last_mut()
            .expect("V4 key cache is rank four") = 1;
        let values = safemlx::ops::zeros_dtype(&sentinel_shape, keys.dtype(), stream)?;
        let retained = i64::from(keys.dim(-2));
        blocks.push(PromptCacheSnapshotBlock {
            global_layer: descriptor.global_layer_start + index,
            start: i64::from(end) - retained,
            end: i64::from(end),
            rank,
            arrays: CacheBlockArrays::KeyValue { keys, values },
        });
    }
    let state = caches
        .iter()
        .enumerate()
        .flat_map(|(index, cache)| {
            cache.prompt_cache_state_arrays(descriptor.global_layer_start + index)
        })
        .collect::<Vec<_>>();
    save_prompt_cache_snapshot(
        destination,
        descriptor,
        prefix_token_ids,
        blocks,
        &state,
        options,
    )
    .map_err(|error| Exception::custom(error.to_string()).into())
}

fn empty_cache_for_args(args: &ModelArgs) -> Result<Cache, Error> {
    let target_count = args.num_hidden_layers as usize;
    let draft_count = args.num_nextn_predict_layers.max(0) as usize;
    let build = |layer: usize| {
        AttentionCache::new_for_ratio(args.compress_ratios[layer], args.sliding_window)
            .map_err(Error::from)
    };
    Ok(Cache {
        layers: (0..target_count).map(&build).collect::<Result<_, _>>()?,
        mtp_layers: (target_count..target_count + draft_count)
            .map(build)
            .collect::<Result<_, _>>()?,
    })
}

fn paged_cache_for_args(
    args: &ModelArgs,
    manager: CacheResidencyManager,
    prefix_tokens: i32,
    rank: Option<CacheRankIdentity>,
) -> Result<Cache, Error> {
    let target_count = args.num_hidden_layers as usize;
    let draft_count = args.num_nextn_predict_layers.max(0) as usize;
    let build = |layer: usize| {
        AttentionCache::new_paged_for_ratio(
            args.compress_ratios[layer],
            args.sliding_window,
            manager.clone(),
            layer,
            prefix_tokens,
            rank,
        )
        .map_err(Error::from)
    };
    Ok(Cache {
        layers: (0..target_count).map(&build).collect::<Result<_, _>>()?,
        mtp_layers: (target_count..target_count + draft_count)
            .map(build)
            .collect::<Result<_, _>>()?,
    })
}

pub(crate) fn load_prompt_cache_with_identity(
    args: &ModelArgs,
    directory: impl AsRef<Path>,
    expected: &PromptCacheDescriptor,
    prefix_token_ids: &[u32],
    identity: PromptCacheModelIdentity,
    options: PagedCacheOptions,
    stream: &Stream,
) -> Result<(Cache, PromptCacheManifest), Error> {
    let (manager, manifest) = open_prompt_cache(
        directory.as_ref(),
        expected,
        &identity,
        prefix_token_ids,
        options,
    )
    .map_err(|error| Exception::custom(error.to_string()))?;
    let mut state = load_prompt_cache_state_tensors(directory, &manifest, stream)
        .map_err(|error| Exception::custom(error.to_string()))?
        .into_iter()
        .map(|tensor| ((tensor.owner, tensor.role), tensor.array))
        .collect::<BTreeMap<_, _>>();
    let prefix_tokens = i32::try_from(prefix_token_ids.len())
        .map_err(|_| Error::UnsupportedArchitecture("V4 prompt length exceeds i32".into()))?;
    let mut cache = paged_cache_for_args(
        args,
        manager,
        expected.sink_tokens as i32,
        identity.topology.cache_rank_identity(),
    )?;
    for (index, layer) in cache
        .layers
        .iter_mut()
        .chain(&mut cache.mtp_layers)
        .enumerate()
    {
        let processed_tokens = prefix_tokens
            .checked_add(identity.layer_prefix_offsets[index])
            .ok_or_else(|| Error::UnsupportedArchitecture("V4 layer offset overflow".into()))?;
        layer.restore_prompt_cache_state(
            identity.global_layer_start + index,
            &mut state,
            processed_tokens,
        )?;
    }
    if !state.is_empty() {
        return Err(Error::UnsupportedArchitecture(
            "V4 prompt cache contains unexpected layer state".into(),
        ));
    }
    Ok((cache, manifest))
}

/// Stable identity for all cache-relevant V4 geometry.
pub(crate) fn prompt_cache_architecture_fingerprint(args: &ModelArgs) -> String {
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
            (
                "num_nextn_predict_layers",
                args.num_nextn_predict_layers.to_string(),
            ),
            ("dspark", format!("{:?}", args.dspark)),
            (
                "native_quantization",
                format!("{:?}", args.quantization_config),
            ),
            ("load_time_quantization", format!("{:?}", args.quantization)),
            (
                "quantized_weight_configs",
                quantized_weight_configs.join(";"),
            ),
        ],
    )
}

/// Reads and validates V4 arguments from a model directory.
pub fn get_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let value: Value =
        serde_json::from_slice(&std::fs::read(model_dir.as_ref().join("config.json"))?)?;
    ModelArgs::from_value(value)
}

pub(crate) struct PreparedDeepSeekV4Gguf {
    pub(crate) args: ModelArgs,
    pub(crate) eos_token_ids: Vec<u32>,
}

/// Parses the canonical llama.cpp `deepseek4` metadata and logical catalog.
pub(crate) fn model_args_from_gguf_catalog(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> Result<ModelArgs, Error> {
    let architecture = gguf_string(metadata, "general.architecture")?;
    if architecture != "deepseek4" {
        return Err(unsupported(format!(
            "GGUF architecture is {architecture:?}, expected \"deepseek4\""
        )));
    }
    checkpoint
        .catalog()
        .translated_outputs(translate_gguf_weight_name)
        .map_err(safemlx::error::IoError::from)?;

    let key = |suffix: &str| format!("deepseek4.{suffix}");
    let total_layers = gguf_i32(metadata, &key("block_count"))?;
    let nextn = gguf_optional_i64(metadata, &key("nextn_predict_layers"))?
        .map(i32::try_from)
        .transpose()
        .map_err(|_| unsupported("GGUF nextn_predict_layers exceeds i32"))?
        .unwrap_or(0);
    let main_layers = total_layers
        .checked_sub(nextn)
        .ok_or_else(|| unsupported("GGUF NextN count exceeds block_count"))?;
    let ratios = gguf_i32_array(metadata, &key("attention.compress_ratios"))?;
    let swiglu_exp = gguf_uniform_f32_array(metadata, &key("swiglu_clamp_exp"), total_layers)?;
    let swiglu_shared = gguf_uniform_f32_array(metadata, &key("swiglu_clamp_shexp"), total_layers)?;
    if swiglu_exp.to_bits() != swiglu_shared.to_bits() {
        return Err(unsupported(format!(
            "GGUF routed/shared SwiGLU clamps disagree: {swiglu_exp} versus {swiglu_shared}"
        )));
    }
    let gating = gguf_i32(metadata, &key("expert_gating_func"))?;
    if gating != 3 {
        return Err(unsupported(format!(
            "GGUF expert_gating_func {gating} is not sqrtsoftplus (3)"
        )));
    }
    let vocab_size = match metadata
        .get("tokenizer.ggml.tokens")
        .and_then(GgufMetadataValue::as_strings)
    {
        Some(tokens) => i32::try_from(tokens.len())
            .map_err(|_| unsupported("GGUF tokenizer vocabulary exceeds i32"))?,
        None if metadata.contains_key("tokenizer.ggml.tokens") => {
            return Err(unsupported(
                "GGUF tokenizer.ggml.tokens metadata has the wrong type",
            ))
        }
        None => gguf_i32(metadata, &key("vocab_size"))?,
    };
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
        }),
        Some(scaling) => {
            return Err(unsupported(format!(
                "GGUF RoPE scaling {scaling:?} is unsupported"
            )))
        }
    };

    let hidden_size = gguf_i32(metadata, &key("embedding_length"))?;
    let hc_mult = gguf_i32(metadata, &key("hyper_connection.count"))?;
    let embedding_length_out = gguf_i32(metadata, &key("embedding_length_out"))?;
    if embedding_length_out != hidden_size * hc_mult {
        return Err(unsupported(format!(
            "GGUF embedding_length_out {embedding_length_out} does not equal hidden_size * hyper_connection.count ({})",
            hidden_size * hc_mult
        )));
    }
    let mut args = ModelArgsSource {
        model_type: "deepseek_v4".into(),
        hidden_size,
        moe_intermediate_size: gguf_i32(metadata, &key("expert_feed_forward_length"))?,
        num_hidden_layers: main_layers,
        num_attention_heads: gguf_i32(metadata, &key("attention.head_count"))?,
        num_key_value_heads: gguf_optional_i64(metadata, &key("attention.head_count_kv"))?
            .map(i32::try_from)
            .transpose()
            .map_err(|_| unsupported("GGUF key/value head count exceeds i32"))?
            .unwrap_or(1),
        head_dim: gguf_i32(metadata, &key("attention.key_length"))?,
        qk_rope_head_dim: gguf_i32(metadata, &key("rope.dimension_count"))?,
        q_lora_rank: gguf_i32(metadata, &key("attention.q_lora_rank"))?,
        o_lora_rank: gguf_i32(metadata, &key("attention.output_lora_rank"))?,
        o_groups: gguf_i32(metadata, &key("attention.output_group_count"))?,
        vocab_size,
        rms_norm_eps: gguf_f32(metadata, &key("attention.layer_norm_rms_epsilon"))?,
        max_position_embeddings: gguf_i32(metadata, &key("context_length"))?,
        rope_theta: gguf_optional_f32(metadata, &key("rope.freq_base"))?
            .unwrap_or_else(default_rope_theta),
        compress_rope_theta: gguf_f32(metadata, &key("attention.compress_rope_freq_base"))?,
        rope_scaling,
        sliding_window: gguf_i32(metadata, &key("attention.sliding_window"))?,
        compress_ratios: ratios,
        index_n_heads: gguf_i32(metadata, &key("attention.indexer.head_count"))?,
        index_head_dim: gguf_i32(metadata, &key("attention.indexer.key_length"))?,
        index_topk: gguf_i32(metadata, &key("attention.indexer.top_k"))?,
        hc_mult,
        hc_sinkhorn_iters: gguf_i32(metadata, &key("hyper_connection.sinkhorn_iterations"))?,
        hc_eps: gguf_f32(metadata, &key("hyper_connection.epsilon"))?,
        n_routed_experts: gguf_i32(metadata, &key("expert_count"))?,
        n_shared_experts: gguf_i32(metadata, &key("expert_shared_count"))?,
        num_experts_per_tok: gguf_i32(metadata, &key("expert_used_count"))?,
        num_hash_layers: gguf_i32(metadata, &key("hash_layer_count"))?,
        scoring_func: "sqrtsoftplus".into(),
        topk_method: "noaux_tc".into(),
        norm_topk_prob: gguf_bool(metadata, &key("expert_weights_norm"))?,
        routed_scaling_factor: gguf_f32(metadata, &key("expert_weights_scale"))?,
        swiglu_limit: swiglu_exp,
        num_nextn_predict_layers: nextn,
        expert_dtype: Some("fp4".into()),
        quantization_config: None,
        tie_word_embeddings: false,
        dspark_block_size: None,
        dspark_noise_token_id: None,
        dspark_target_layer_ids: None,
        dspark_markov_rank: None,
    }
    .normalize()?;
    args.quantized_weight_configs = Some(gguf_quantization_configs(
        checkpoint,
        translate_gguf_weight_name,
    )?);
    args.validate()?;
    Ok(args)
}

pub(crate) fn prepare_gguf_checkpoint(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> Result<PreparedDeepSeekV4Gguf, Error> {
    Ok(PreparedDeepSeekV4Gguf {
        args: model_args_from_gguf_catalog(checkpoint, metadata)?,
        eos_token_ids: crate::backend::mlx::gguf_eos_token_ids(metadata)?,
    })
}

/// Canonical llama.cpp tensor names translated to the native V4 source contract.
pub(crate) fn translate_gguf_weight_name(name: &str) -> String {
    for (source, target, strip_weight) in [
        ("token_embd", "embed", false),
        ("output_norm", "norm", false),
        ("output", "head", false),
        ("output_hc_fn", "hc_head_fn", true),
        ("output_hc_base", "hc_head_base", true),
        ("output_hc_scale", "hc_head_scale", true),
    ] {
        if name == source || name.starts_with(&format!("{source}.")) {
            let mut translated = name.replacen(source, target, 1);
            if strip_weight && translated.ends_with(".weight") {
                translated.truncate(translated.len() - ".weight".len());
            }
            return translated;
        }
    }
    let Some(rest) = name.strip_prefix("blk.") else {
        return name.to_string();
    };
    let Some((layer, parameter)) = rest.split_once('.') else {
        return name.to_string();
    };
    let root = format!("layers.{layer}");
    for (source, target, strip_weight) in [
        ("attn_norm", "attn_norm", false),
        ("attn_sinks", "attn.attn_sink", true),
        ("attn_q_a", "attn.wq_a", false),
        ("attn_q_a_norm", "attn.q_norm", false),
        ("attn_q_b", "attn.wq_b", false),
        ("attn_kv", "attn.wkv", false),
        ("attn_kv_a_norm", "attn.kv_norm", false),
        ("attn_output_a", "attn.wo_a", false),
        ("attn_output_b", "attn.wo_b", false),
        ("hc_attn_fn", "hc_attn_fn", true),
        ("hc_attn_base", "hc_attn_base", true),
        ("hc_attn_scale", "hc_attn_scale", true),
        ("hc_ffn_fn", "hc_ffn_fn", true),
        ("hc_ffn_base", "hc_ffn_base", true),
        ("hc_ffn_scale", "hc_ffn_scale", true),
        ("attn_compressor_kv", "attn.compressor.wkv", false),
        ("attn_compressor_gate", "attn.compressor.wgate", false),
        ("attn_compressor_ape", "attn.compressor.ape", true),
        ("attn_compressor_norm", "attn.compressor.norm", false),
        ("indexer.proj", "attn.indexer.weights_proj", false),
        ("indexer.attn_q_b", "attn.indexer.wq_b", false),
        (
            "indexer_compressor_kv",
            "attn.indexer.compressor.wkv",
            false,
        ),
        (
            "indexer_compressor_gate",
            "attn.indexer.compressor.wgate",
            false,
        ),
        (
            "indexer_compressor_ape",
            "attn.indexer.compressor.ape",
            true,
        ),
        (
            "indexer_compressor_norm",
            "attn.indexer.compressor.norm",
            false,
        ),
        ("ffn_norm", "ffn_norm", false),
        ("ffn_gate_inp", "ffn.gate", false),
        ("exp_probs_b", "ffn.gate", false),
        ("ffn_gate_tid2eid", "ffn.gate.tid2eid", true),
        ("ffn_gate_shexp", "ffn.shared_experts.w1", false),
        ("ffn_down_shexp", "ffn.shared_experts.w2", false),
        ("ffn_up_shexp", "ffn.shared_experts.w3", false),
        ("ffn_gate_exps", "ffn.expert_banks.w1", false),
        ("ffn_down_exps", "ffn.expert_banks.w2", false),
        ("ffn_up_exps", "ffn.expert_banks.w3", false),
    ] {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            let mut translated = format!("{root}.{}", parameter.replacen(source, target, 1));
            if target.contains("expert_banks") && translated.ends_with(".scales") {
                translated.truncate(translated.len() - ".scales".len());
                translated.push_str(".scale");
            }
            if strip_weight && translated.ends_with(".weight") {
                translated.truncate(translated.len() - ".weight".len());
            }
            return translated;
        }
    }
    name.to_string()
}

fn gguf_string(metadata: &HashMap<String, GgufMetadataValue>, key: &str) -> Result<String, Error> {
    gguf_optional_string(metadata, key)?
        .ok_or_else(|| unsupported(format!("GGUF metadata is missing required key {key:?}")))
}

fn gguf_optional_string(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<Option<String>, Error> {
    match metadata.get(key) {
        Some(GgufMetadataValue::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(unsupported(format!(
            "GGUF metadata key {key:?} has the wrong type"
        ))),
        None => Ok(None),
    }
}

fn gguf_i32(metadata: &HashMap<String, GgufMetadataValue>, key: &str) -> Result<i32, Error> {
    let value = gguf_optional_i64(metadata, key)?
        .ok_or_else(|| unsupported(format!("GGUF metadata is missing required key {key:?}")))?;
    i32::try_from(value).map_err(|_| unsupported(format!("GGUF value {key:?} exceeds i32")))
}

fn gguf_optional_i64(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<Option<i64>, Error> {
    match metadata.get(key) {
        Some(value) => value
            .as_i64()
            .map(Some)
            .ok_or_else(|| unsupported(format!("GGUF metadata key {key:?} has the wrong type"))),
        None => Ok(None),
    }
}

fn gguf_f32(metadata: &HashMap<String, GgufMetadataValue>, key: &str) -> Result<f32, Error> {
    gguf_optional_f32(metadata, key)?
        .ok_or_else(|| unsupported(format!("GGUF metadata is missing required key {key:?}")))
}

fn gguf_optional_f32(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<Option<f32>, Error> {
    match metadata.get(key) {
        Some(value) => value
            .as_f32()
            .map(Some)
            .ok_or_else(|| unsupported(format!("GGUF metadata key {key:?} has the wrong type"))),
        None => Ok(None),
    }
}

fn gguf_bool(metadata: &HashMap<String, GgufMetadataValue>, key: &str) -> Result<bool, Error> {
    match metadata.get(key) {
        Some(GgufMetadataValue::Bool(value)) => Ok(*value),
        Some(value) => value
            .as_i64()
            .map(|value| value != 0)
            .ok_or_else(|| unsupported(format!("GGUF metadata key {key:?} has the wrong type"))),
        None => Err(unsupported(format!(
            "GGUF metadata is missing required key {key:?}"
        ))),
    }
}

fn gguf_i32_array(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<Vec<i32>, Error> {
    let values = metadata
        .get(key)
        .and_then(GgufMetadataValue::to_i64_vec)
        .ok_or_else(|| {
            unsupported(format!(
                "GGUF metadata key {key:?} must be an integer array"
            ))
        })?;
    values
        .into_iter()
        .map(|value| {
            i32::try_from(value)
                .map_err(|_| unsupported(format!("GGUF array value {key:?} exceeds i32")))
        })
        .collect()
}

fn gguf_uniform_f32_array(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
    expected: i32,
) -> Result<f32, Error> {
    let values = match metadata.get(key) {
        Some(GgufMetadataValue::Array(GgufMetadataArray::Float32(values))) => values.clone(),
        Some(GgufMetadataValue::Array(GgufMetadataArray::Float64(values))) => {
            values.iter().map(|value| *value as f32).collect()
        }
        _ => {
            return Err(unsupported(format!(
                "GGUF metadata key {key:?} must be a float array"
            )))
        }
    };
    if values.len() != expected as usize || values.is_empty() {
        return Err(unsupported(format!(
            "GGUF metadata key {key:?} has {} values, expected {expected}",
            values.len()
        )));
    }
    let first = values[0];
    if values
        .iter()
        .any(|value| value.to_bits() != first.to_bits())
    {
        return Err(unsupported(format!(
            "GGUF metadata key {key:?} must be uniform"
        )));
    }
    Ok(first)
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

impl crate::composition::mlx::speculative::embedded::EmbeddedMtpTarget for Model {
    type Cache = Cache;
    type DraftCache = Vec<AttentionCache>;

    fn prefill_target(
        &mut self,
        input: runtime_input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        let tokens = runtime_input::text_token_ids(input, stream)?;
        cache.reset()?;
        self.forward_mtp_target(&tokens, cache, stream)
    }

    fn verify_target(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        self.forward_mtp_target(tokens, cache, stream)
    }

    fn prefill_draft_cache(
        &mut self,
        output: &crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput,
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

    fn restore_target_checkpoint(
        cache: &mut Self::Cache,
        checkpoint: &Self::Cache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        cache.restore_target_checkpoint(checkpoint, stream)
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
pub type Generate<'a, S = crate::backend::mlx::runtime::generation::sampler::DefaultSampler> =
    crate::backend::mlx::nn::generation::Generate<'a, Model, Cache, S>;
