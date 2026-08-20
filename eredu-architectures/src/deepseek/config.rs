use std::{
    collections::{BTreeMap, HashMap},
    io::Read,
};

use eredu_checkpoint::{BlockFp8Format, BlockFp8ScaleEncoding, LinearFormat, WeightQuantization};
use eredu_core::LayerSchedule;
use eredu_gguf::{MetadataArray, MetadataValue};
use eredu_nn::RopeValue;
use serde::Deserialize;
use serde_json::Value;

/// Invalid or unsupported DeepSeek configuration.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// Configuration JSON could not be decoded.
    #[error("invalid DeepSeek configuration: {0}")]
    Json(#[from] serde_json::Error),
    /// Configuration changes unsupported execution semantics or has invalid geometry.
    #[error("{0}")]
    Invalid(String),
}

fn invalid(message: impl Into<String>) -> ConfigError {
    ConfigError::Invalid(format!("DeepSeek-V3 {}", message.into()))
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
fn default_float_one() -> f32 {
    1.0
}
fn default_beta_fast() -> f32 {
    32.0
}
fn default_beta_slow() -> f32 {
    1.0
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

impl YarnConfig {
    /// Converts normalized YaRN values to the general rotary contract.
    pub fn rope_scaling(&self) -> HashMap<String, RopeValue> {
        HashMap::from([
            ("type".into(), RopeValue::String(self.r#type.clone())),
            ("factor".into(), RopeValue::Float(self.factor)),
            (
                "original_max_position_embeddings".into(),
                RopeValue::Float(self.original_max_position_embeddings as f32),
            ),
            ("beta_fast".into(), RopeValue::Float(self.beta_fast)),
            ("beta_slow".into(), RopeValue::Float(self.beta_slow)),
            ("mscale".into(), RopeValue::Float(self.mscale)),
            (
                "mscale_all_dim".into(),
                RopeValue::Float(self.mscale_all_dim),
            ),
        ])
    }

    /// Returns DeepSeek's YaRN attention-score multiplier.
    pub fn attention_multiplier(&self) -> f32 {
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
    /// Optional exponent-only inverse-scale encoding.
    #[serde(default)]
    pub scale_fmt: Option<String>,
}

impl Fp8QuantizationConfig {
    fn linear_format(&self) -> Result<LinearFormat, ConfigError> {
        if self.quant_method != "fp8"
            || self.fmt != "e4m3"
            || self.activation_scheme != "dynamic"
            || self.weight_block_size.as_slice() != [128, 128]
        {
            return Err(invalid(format!(
                "supports only dynamic E4M3 block-FP8 with weight_block_size [128, 128], got {self:?}"
            )));
        }
        let scale_encoding = match self.scale_fmt.as_deref() {
            None => BlockFp8ScaleEncoding::FloatingPoint,
            Some("ue8m0") => BlockFp8ScaleEncoding::Ue8m0,
            Some(format) => {
                return Err(invalid(format!("unsupported FP8 scale format {format:?}")))
            }
        };
        Ok(LinearFormat::E4M3BlockFp8(
            BlockFp8Format::new(128, 128, scale_encoding)
                .map_err(|error| invalid(error.to_string()))?,
        ))
    }
}

/// Quantization metadata accepted under Hugging Face's `quantization_config` key.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum DeepSeekQuantizationConfig {
    /// Official DeepSeek block-FP8 metadata.
    Fp8(Fp8QuantizationConfig),
    /// General affine, MXFP4, or GGUF quantization metadata.
    Packed(WeightQuantization),
}

/// Feed-forward operator used by one DeepSeek-V3/R1 decoder layer.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum LayerPolicy {
    /// Dense SwiGLU feed-forward block.
    DenseMlp,
    /// Routed and shared expert feed-forward block.
    SparseMoe,
}

#[derive(Debug, Clone, Deserialize)]
struct V3Source {
    #[serde(default = "default_model_type")]
    model_type: String,
    hidden_size: i32,
    intermediate_size: i32,
    moe_intermediate_size: i32,
    num_hidden_layers: i32,
    num_attention_heads: i32,
    vocab_size: i32,
    #[serde(default = "default_rms_norm_eps")]
    rms_norm_eps: f32,
    max_position_embeddings: i32,
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    #[serde(default)]
    rope_scaling: Option<YarnConfig>,
    #[serde(default)]
    q_lora_rank: Option<i32>,
    kv_lora_rank: i32,
    qk_nope_head_dim: i32,
    qk_rope_head_dim: i32,
    v_head_dim: i32,
    first_k_dense_replace: i32,
    #[serde(default = "default_moe_layer_freq")]
    moe_layer_freq: i32,
    n_routed_experts: i32,
    #[serde(default = "default_one")]
    n_shared_experts: i32,
    num_experts_per_tok: i32,
    n_group: i32,
    topk_group: i32,
    #[serde(default = "default_topk_method")]
    topk_method: String,
    #[serde(default = "default_scoring_func")]
    scoring_func: String,
    #[serde(default = "default_true")]
    norm_topk_prob: bool,
    #[serde(default = "default_float_one")]
    routed_scaling_factor: f32,
    #[serde(default)]
    num_nextn_predict_layers: i32,
    #[serde(default)]
    quantization_config: Option<DeepSeekQuantizationConfig>,
    #[serde(default)]
    quantization: Option<WeightQuantization>,
    #[serde(default)]
    tie_word_embeddings: bool,
}

/// Validated backend-neutral DeepSeek-V3/R1 text configuration.
#[derive(Debug, Clone)]
pub struct V3Args {
    /// Canonical model type.
    pub model_type: String,
    /// Model hidden width.
    pub hidden_size: i32,
    /// Dense SwiGLU width.
    pub intermediate_size: i32,
    /// Routed-expert SwiGLU width.
    pub moe_intermediate_size: i32,
    /// Target decoder layer count.
    pub num_hidden_layers: i32,
    /// MLA query head count.
    pub num_attention_heads: i32,
    /// Token vocabulary size.
    pub vocab_size: i32,
    /// RMS normalization epsilon.
    pub rms_norm_eps: f32,
    /// Maximum configured positions.
    pub max_position_embeddings: i32,
    /// Rotary base frequency.
    pub rope_theta: f32,
    /// Optional YaRN extension.
    pub rope_scaling: Option<YarnConfig>,
    /// Optional query LoRA rank.
    pub q_lora_rank: Option<i32>,
    /// Compressed latent width.
    pub kv_lora_rank: i32,
    /// Non-rotary query/key width per head.
    pub qk_nope_head_dim: i32,
    /// Rotary query/key width per head.
    pub qk_rope_head_dim: i32,
    /// Value width per head.
    pub v_head_dim: i32,
    /// Exact dense/MoE layer schedule.
    pub layer_schedule: LayerSchedule<LayerPolicy>,
    /// Routed expert count.
    pub n_routed_experts: i32,
    /// Shared expert count.
    pub n_shared_experts: i32,
    /// Selected experts per token.
    pub num_experts_per_tok: i32,
    /// Router group count.
    pub n_group: i32,
    /// Selected router groups.
    pub topk_group: i32,
    /// Whether selected routes are normalized.
    pub norm_topk_prob: bool,
    /// Final routed contribution multiplier.
    pub routed_scaling_factor: f32,
    /// Embedded MTP layer count.
    pub num_nextn_predict_layers: i32,
    /// Model-wide physical linear encoding.
    pub linear_format: LinearFormat,
    /// Canonical per-parameter overrides used by mixed physical checkpoints.
    pub linear_formats: BTreeMap<String, LinearFormat>,
    /// Whether embedding and output weights are tied.
    pub tie_word_embeddings: bool,
}

impl V3Source {
    fn normalize(self) -> Result<V3Args, ConfigError> {
        let layer_count = usize::try_from(self.num_hidden_layers)
            .map_err(|_| invalid("num_hidden_layers must be positive"))?;
        if layer_count == 0
            || self.first_k_dense_replace < 0
            || self.first_k_dense_replace > self.num_hidden_layers
            || self.moe_layer_freq <= 0
        {
            return Err(invalid("invalid dense/MoE layer schedule"));
        }
        let layer_schedule = LayerSchedule::new(
            layer_count,
            (0..layer_count)
                .map(|layer| {
                    if layer as i32 >= self.first_k_dense_replace
                        && layer as i32 % self.moe_layer_freq == 0
                    {
                        LayerPolicy::SparseMoe
                    } else {
                        LayerPolicy::DenseMlp
                    }
                })
                .collect(),
        )
        .map_err(|error| invalid(error.to_string()))?;
        let linear_format = match (&self.quantization_config, self.quantization) {
            (Some(DeepSeekQuantizationConfig::Fp8(config)), None) => {
                if config.scale_fmt.is_some() {
                    return Err(invalid(
                        "V3 block-FP8 scales must use floating-point storage",
                    ));
                }
                config.linear_format()?
            }
            (Some(DeepSeekQuantizationConfig::Packed(left)), Some(right)) if *left != right => {
                return Err(invalid("quantization and quantization_config disagree"));
            }
            (Some(DeepSeekQuantizationConfig::Packed(format)), _) => LinearFormat::from(*format),
            (None, Some(format)) => LinearFormat::from(format),
            (Some(DeepSeekQuantizationConfig::Fp8(_)), Some(_)) => {
                return Err(invalid(
                    "native block-FP8 cannot be combined with packed quantization metadata",
                ));
            }
            (None, None) => LinearFormat::Dense,
        };
        let args = V3Args {
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
            norm_topk_prob: self.norm_topk_prob,
            routed_scaling_factor: self.routed_scaling_factor,
            num_nextn_predict_layers: self.num_nextn_predict_layers,
            linear_format,
            linear_formats: BTreeMap::new(),
            tie_word_embeddings: self.tie_word_embeddings,
        };
        args.validate()?;
        if self.topk_method != "noaux_tc" || self.scoring_func != "sigmoid" {
            return Err(invalid(
                "requires topk_method=noaux_tc and scoring_func=sigmoid",
            ));
        }
        Ok(args)
    }
}

impl V3Args {
    /// Resolves one canonical matrix's physical encoding.
    pub fn linear_format_for(&self, name: &str) -> LinearFormat {
        self.linear_formats
            .get(name)
            .copied()
            .unwrap_or(self.linear_format)
    }

    /// Validates all derived geometry and execution policies.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.model_type != "deepseek_v3" {
            return Err(invalid(format!(
                "unsupported model_type {:?}",
                self.model_type
            )));
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
                return Err(invalid(format!("{name} must be positive, got {value}")));
            }
        }
        if self.qk_rope_head_dim % 2 != 0
            || self.q_lora_rank.is_some_and(|rank| rank <= 0)
            || !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 0.0
            || !self.routed_scaling_factor.is_finite()
            || self.routed_scaling_factor <= 0.0
        {
            return Err(invalid(
                "invalid rotary, normalization, or low-rank geometry",
            ));
        }
        let group_capacity = self.n_routed_experts.checked_div(self.n_group.max(1));
        if self.n_routed_experts <= 0
            || self.n_shared_experts <= 0
            || self.num_experts_per_tok <= 0
            || self.num_experts_per_tok > self.n_routed_experts
            || self.n_group <= 0
            || self.n_routed_experts % self.n_group != 0
            || self.topk_group <= 0
            || self.topk_group > self.n_group
            || group_capacity
                .is_none_or(|capacity| self.num_experts_per_tok > self.topk_group * capacity)
        {
            return Err(invalid("invalid routed-expert group geometry"));
        }
        if self.tie_word_embeddings || self.num_nextn_predict_layers < 0 {
            return Err(invalid(
                "published V3/R1 requires untied embeddings and nonnegative MTP layers",
            ));
        }
        if let Some(yarn) = &self.rope_scaling {
            if yarn.r#type != "yarn"
                || yarn.factor <= 0.0
                || yarn.original_max_position_embeddings <= 0
            {
                return Err(invalid("invalid YaRN configuration"));
            }
        }
        self.linear_format
            .validate()
            .map_err(|error| invalid(error.to_string()))
    }
}

/// Parses a strict DeepSeek-V3/R1 configuration value.
pub fn parse_v3_config(value: &Value) -> Result<V3Args, ConfigError> {
    let source: V3Source = serde_json::from_value(value.clone())?;
    let args = source.normalize()?;
    if value
        .get("architectures")
        .and_then(Value::as_array)
        .is_some_and(|architectures| {
            !architectures
                .iter()
                .any(|name| name.as_str() == Some("DeepseekV3ForCausalLM"))
        })
        || value
            .get("attention_bias")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        || value
            .get("attention_dropout")
            .and_then(Value::as_f64)
            .unwrap_or(0.0)
            != 0.0
        || value
            .get("hidden_act")
            .and_then(Value::as_str)
            .is_some_and(|activation| activation != "silu")
        || value.get("ep_size").and_then(Value::as_i64).unwrap_or(1) != 1
        || value
            .get("num_key_value_heads")
            .and_then(Value::as_i64)
            .is_some_and(|heads| heads != args.num_attention_heads as i64)
    {
        return Err(invalid("unsupported declared execution semantics"));
    }
    Ok(args)
}

/// Parses a strict DeepSeek-V3/R1 configuration reader.
pub fn parse_v3_reader(reader: impl Read) -> Result<V3Args, ConfigError> {
    parse_v3_config(&serde_json::from_reader(reader)?)
}

fn invalid_v4(message: impl Into<String>) -> ConfigError {
    ConfigError::Invalid(format!("DeepSeek-V4 {}", message.into()))
}

fn default_v4_model_type() -> String {
    "deepseek_v4".into()
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
fn default_v4_scoring() -> String {
    "sqrtsoftplus".into()
}

/// Per-layer V4 attention policy after compression metadata normalization.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum V4AttentionPolicy {
    /// Bounded local key-only attention.
    Local,
    /// Indexed attention over pooled state at the exact compression ratio.
    Compressed {
        /// Number of source tokens represented by one pooled position.
        ratio: i32,
    },
}

/// Checkpoint-native expert-bank encoding.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum ExpertFormat {
    /// Ordinary floating-point experts.
    Dense,
    /// Microscaling FP4 experts.
    MxFp4,
    /// E4M3 block-FP8 experts.
    BlockFp8,
}

/// Configuration present only in a fused DSpark checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DsparkConfig {
    /// Number of tokens emitted by one draft block.
    pub block_size: i32,
    /// Token used to seed unfilled draft positions.
    pub noise_token_id: i32,
    /// Target decoder layers captured by the drafter.
    pub target_layer_ids: Vec<i32>,
    /// Rank of the token-conditioned Markov head.
    pub markov_rank: i32,
}

#[derive(Debug, Clone, Deserialize)]
struct V4Source {
    #[serde(default = "default_v4_model_type")]
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
    #[serde(default = "default_v4_scoring")]
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

/// Validated backend-neutral DeepSeek-V4 architecture configuration.
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub struct V4Args {
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
    pub attention_schedule: LayerSchedule<V4AttentionPolicy>,
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
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f32,
    pub swiglu_limit: Option<eredu_nn::SwiGluLimit>,
    pub num_nextn_predict_layers: i32,
    pub expert_format: ExpertFormat,
    pub linear_format: LinearFormat,
    /// Canonical per-parameter overrides used by mixed physical checkpoints.
    pub linear_formats: BTreeMap<String, LinearFormat>,
    pub tie_word_embeddings: bool,
    pub dspark: Option<DsparkConfig>,
}

impl V4Source {
    fn normalize(mut self) -> Result<V4Args, ConfigError> {
        if self.compress_ratios.is_empty() {
            let layers = usize::try_from(self.num_hidden_layers)
                .map_err(|_| invalid_v4("num_hidden_layers must be positive"))?;
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
        let draft_fields = [
            self.dspark_block_size.is_some(),
            self.dspark_noise_token_id.is_some(),
            self.dspark_target_layer_ids.is_some(),
            self.dspark_markov_rank.is_some(),
        ];
        let dspark = if draft_fields.iter().all(|present| *present) {
            Some(DsparkConfig {
                block_size: self.dspark_block_size.unwrap(),
                noise_token_id: self.dspark_noise_token_id.unwrap(),
                target_layer_ids: self.dspark_target_layer_ids.unwrap(),
                markov_rank: self.dspark_markov_rank.unwrap(),
            })
        } else if draft_fields.iter().any(|present| *present) {
            return Err(invalid_v4("fused DSpark requires all dspark_* fields"));
        } else {
            None
        };
        let attention_schedule = LayerSchedule::new(
            self.compress_ratios.len(),
            self.compress_ratios
                .iter()
                .map(|ratio| match *ratio {
                    0 => Ok(V4AttentionPolicy::Local),
                    4 | 128 => Ok(V4AttentionPolicy::Compressed { ratio: *ratio }),
                    ratio => Err(invalid_v4(format!(
                        "unsupported compression ratio {ratio}; expected 0, 4, or 128"
                    ))),
                })
                .collect::<Result<Vec<_>, _>>()?,
        )
        .map_err(|error| invalid_v4(error.to_string()))?;
        let expert_format = match self.expert_dtype.as_deref() {
            None => ExpertFormat::Dense,
            Some("fp4") => ExpertFormat::MxFp4,
            Some("fp8") => ExpertFormat::BlockFp8,
            Some(format) => return Err(invalid_v4(format!("unsupported expert_dtype {format:?}"))),
        };
        let linear_format = self
            .quantization_config
            .as_ref()
            .map(Fp8QuantizationConfig::linear_format)
            .transpose()?
            .unwrap_or(LinearFormat::Dense);
        let swiglu_limit = (self.swiglu_limit > 0.0)
            .then(|| eredu_nn::SwiGluLimit::new(self.swiglu_limit))
            .transpose()
            .map_err(|error| invalid_v4(error.to_string()))?;
        let args = V4Args {
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
            attention_schedule,
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
            norm_topk_prob: self.norm_topk_prob,
            routed_scaling_factor: self.routed_scaling_factor,
            swiglu_limit,
            num_nextn_predict_layers: self.num_nextn_predict_layers,
            expert_format,
            linear_format,
            linear_formats: BTreeMap::new(),
            tie_word_embeddings: self.tie_word_embeddings,
            dspark,
        };
        args.validate()?;
        if self.scoring_func != "sqrtsoftplus" || self.topk_method != "noaux_tc" {
            return Err(invalid_v4(
                "requires scoring_func=sqrtsoftplus and topk_method=noaux_tc",
            ));
        }
        Ok(args)
    }
}

impl V4Args {
    /// Resolves one canonical matrix's physical encoding.
    pub fn linear_format_for(&self, name: &str) -> LinearFormat {
        self.linear_formats
            .get(name)
            .copied()
            .unwrap_or(self.linear_format)
    }

    /// Validates derived geometry and exact V4 execution policy.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.model_type != "deepseek_v4" {
            return Err(invalid_v4(format!(
                "unsupported model_type {:?}",
                self.model_type
            )));
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
                return Err(invalid_v4(format!("{name} must be positive, got {value}")));
            }
        }
        if self.num_key_value_heads != 1
            || self.qk_rope_head_dim >= self.head_dim
            || self.qk_rope_head_dim % 2 != 0
            || self.num_attention_heads % self.o_groups != 0
            || self.index_n_heads % self.o_groups != 0
            || self.num_experts_per_tok > self.n_routed_experts
            || self.num_hash_layers < 0
            || self.num_hash_layers > self.num_hidden_layers
            || !self.norm_topk_prob
        {
            return Err(invalid_v4("invalid attention, routing, or hash geometry"));
        }
        if ![
            self.rms_norm_eps,
            self.hc_eps,
            self.rope_theta,
            self.compress_rope_theta,
            self.routed_scaling_factor,
        ]
        .iter()
        .all(|value| value.is_finite() && *value > 0.0)
        {
            return Err(invalid_v4(
                "normalization and scaling values must be positive",
            ));
        }
        let expected = usize::try_from(self.num_hidden_layers + self.num_nextn_predict_layers)
            .map_err(|_| invalid_v4("layer count overflow"))?;
        if self.attention_schedule.len() != expected
            || self
                .attention_schedule
                .iter()
                .skip(self.num_hidden_layers as usize)
                .any(|policy| *policy != V4AttentionPolicy::Local)
        {
            return Err(invalid_v4(
                "compression schedule must cover target and local-only prediction layers",
            ));
        }
        if let Some(yarn) = &self.rope_scaling {
            if yarn.r#type != "yarn"
                || yarn.factor <= 0.0
                || yarn.original_max_position_embeddings <= 0
            {
                return Err(invalid_v4("invalid YaRN configuration"));
            }
        }
        if let Some(dspark) = &self.dspark {
            if self.num_nextn_predict_layers <= 0
                || dspark.block_size <= 0
                || dspark.noise_token_id < 0
                || dspark.noise_token_id >= self.vocab_size
                || dspark.markov_rank <= 0
                || dspark.target_layer_ids.is_empty()
                || dspark
                    .target_layer_ids
                    .iter()
                    .any(|layer| *layer < 0 || *layer >= self.num_hidden_layers)
            {
                return Err(invalid_v4("invalid DSpark configuration"));
            }
        }
        self.linear_format
            .validate()
            .map_err(|error| invalid_v4(error.to_string()))
    }

    /// Returns the normalized policy for a target or embedded prediction layer.
    pub fn attention_policy(&self, layer: usize) -> Option<V4AttentionPolicy> {
        self.attention_schedule.get(layer).copied()
    }
}

/// Parses and validates one DeepSeek-V4 configuration value.
pub fn parse_v4_config(value: &Value) -> Result<V4Args, ConfigError> {
    serde_json::from_value::<V4Source>(value.clone())?.normalize()
}

/// Parses and validates one DeepSeek-V4 configuration reader.
pub fn parse_v4_reader(reader: impl Read) -> Result<V4Args, ConfigError> {
    parse_v4_config(&serde_json::from_reader(reader)?)
}

/// Minimal pure tensor catalog required while normalizing DeepSeek GGUF
/// metadata. Tensor payloads and backend arrays are intentionally absent.
pub trait GgufTensorCatalog {
    /// Whether one exact physical GGUF tensor is present.
    fn contains(&self, name: &str) -> bool;
}

/// Returns whether a DeepSeek2 catalog stores the MLA B projection as
/// separate key and value tensors rather than one fused tensor.
pub fn v3_uses_split_kv(catalog: &impl GgufTensorCatalog) -> bool {
    catalog.contains("blk.0.attn_k_b.weight")
}

/// Parses strict DeepSeek2 GGUF metadata into neutral V3/R1 arguments.
pub fn parse_v3_gguf(
    catalog: &impl GgufTensorCatalog,
    metadata: &HashMap<String, MetadataValue>,
) -> Result<V3Args, ConfigError> {
    let architecture = gguf_string(metadata, "general.architecture")?;
    if architecture != "deepseek2" {
        return Err(invalid(format!(
            "GGUF architecture is {architecture:?}, expected \"deepseek2\""
        )));
    }
    let key = |suffix: &str| format!("deepseek2.{suffix}");
    let rope = gguf_i32(metadata, &key("rope.dimension_count"))?;
    let key_width = gguf_i32(metadata, &key("attention.key_length_mla"))?;
    let nope = key_width
        .checked_sub(rope)
        .ok_or_else(|| invalid("GGUF MLA key length is smaller than its rotary length"))?;
    let query_rank = gguf_optional_i64(metadata, &key("attention.q_lora_rank"))?
        .map(i32::try_from)
        .transpose()
        .map_err(|_| invalid("GGUF query LoRA rank exceeds i32"))?
        .filter(|rank| *rank > 0);
    let rope_scaling = parse_gguf_yarn(metadata, "deepseek2", true)?;
    let gating = gguf_optional_i64(metadata, &key("expert_gating_func"))?.unwrap_or(2);
    if gating != 2 {
        return Err(invalid(format!(
            "GGUF expert_gating_func {gating} is not sigmoid (2)"
        )));
    }
    let args = V3Source {
        model_type: "deepseek_v3".into(),
        hidden_size: gguf_i32(metadata, &key("embedding_length"))?,
        intermediate_size: gguf_i32(metadata, &key("feed_forward_length"))?,
        moe_intermediate_size: gguf_i32(metadata, &key("expert_feed_forward_length"))?,
        num_hidden_layers: gguf_i32(metadata, &key("block_count"))?,
        num_attention_heads: gguf_i32(metadata, &key("attention.head_count"))?,
        vocab_size: gguf_vocab_size(metadata, "deepseek2")?,
        rms_norm_eps: gguf_f32(metadata, &key("attention.layer_norm_rms_epsilon"))?,
        max_position_embeddings: gguf_i32(metadata, &key("context_length"))?,
        rope_theta: gguf_optional_f32(metadata, &key("rope.freq_base"))?
            .unwrap_or_else(default_rope_theta),
        rope_scaling,
        q_lora_rank: query_rank,
        kv_lora_rank: gguf_i32(metadata, &key("attention.kv_lora_rank"))?,
        qk_nope_head_dim: nope,
        qk_rope_head_dim: rope,
        v_head_dim: gguf_i32(metadata, &key("attention.value_length_mla"))?,
        first_k_dense_replace: gguf_i32(metadata, &key("leading_dense_block_count"))?,
        moe_layer_freq: 1,
        n_routed_experts: gguf_i32(metadata, &key("expert_count"))?,
        n_shared_experts: gguf_optional_i64(metadata, &key("expert_shared_count"))?
            .map(i32::try_from)
            .transpose()
            .map_err(|_| invalid("GGUF shared expert count exceeds i32"))?
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
        tie_word_embeddings: false,
    }
    .normalize()?;
    let split = v3_uses_split_kv(catalog);
    let fused = catalog.contains("blk.0.attn_kv_b.weight");
    if split == fused {
        return Err(invalid(
            "GGUF catalog must contain exactly one fused or split MLA KV-B layout",
        ));
    }
    Ok(args)
}

/// Parses strict llama.cpp `deepseek4` GGUF metadata into neutral V4
/// arguments. Embedded MTP GGUFs remain companion artifacts and are rejected
/// by the base-model validator.
pub fn parse_v4_gguf(metadata: &HashMap<String, MetadataValue>) -> Result<V4Args, ConfigError> {
    let architecture = gguf_string(metadata, "general.architecture")?;
    if architecture != "deepseek4" {
        return Err(invalid_v4(format!(
            "GGUF architecture is {architecture:?}, expected \"deepseek4\""
        )));
    }
    let key = |suffix: &str| format!("deepseek4.{suffix}");
    let total_layers = gguf_i32(metadata, &key("block_count"))?;
    let prediction_layers = gguf_optional_i64(metadata, &key("nextn_predict_layers"))?
        .map(i32::try_from)
        .transpose()
        .map_err(|_| invalid_v4("GGUF prediction layer count exceeds i32"))?
        .unwrap_or(0);
    let target_layers = total_layers
        .checked_sub(prediction_layers)
        .ok_or_else(|| invalid_v4("GGUF prediction layer count exceeds block count"))?;
    let routed_limit = gguf_uniform_f32_array(metadata, &key("swiglu_clamp_exp"), total_layers)?;
    let shared_limit = gguf_uniform_f32_array(metadata, &key("swiglu_clamp_shexp"), total_layers)?;
    if routed_limit.to_bits() != shared_limit.to_bits() {
        return Err(invalid_v4(
            "GGUF routed and shared expert SwiGLU limits disagree",
        ));
    }
    let gating = gguf_i32(metadata, &key("expert_gating_func"))?;
    if gating != 3 {
        return Err(invalid_v4(format!(
            "GGUF expert_gating_func {gating} is not sqrt-softplus (3)"
        )));
    }
    let hidden = gguf_i32(metadata, &key("embedding_length"))?;
    let streams = gguf_i32(metadata, &key("hyper_connection.count"))?;
    let expected_output = hidden
        .checked_mul(streams)
        .ok_or_else(|| invalid_v4("GGUF hyper-connection width overflows i32"))?;
    let actual_output = gguf_i32(metadata, &key("embedding_length_out"))?;
    if actual_output != expected_output {
        return Err(invalid_v4(format!(
            "GGUF embedding_length_out {actual_output} does not equal {expected_output}"
        )));
    }
    V4Source {
        model_type: "deepseek_v4".into(),
        hidden_size: hidden,
        moe_intermediate_size: gguf_i32(metadata, &key("expert_feed_forward_length"))?,
        num_hidden_layers: target_layers,
        num_attention_heads: gguf_i32(metadata, &key("attention.head_count"))?,
        num_key_value_heads: gguf_optional_i64(metadata, &key("attention.head_count_kv"))?
            .map(i32::try_from)
            .transpose()
            .map_err(|_| invalid_v4("GGUF key/value head count exceeds i32"))?
            .unwrap_or(1),
        head_dim: gguf_i32(metadata, &key("attention.key_length"))?,
        qk_rope_head_dim: gguf_i32(metadata, &key("rope.dimension_count"))?,
        q_lora_rank: gguf_i32(metadata, &key("attention.q_lora_rank"))?,
        o_lora_rank: gguf_i32(metadata, &key("attention.output_lora_rank"))?,
        o_groups: gguf_i32(metadata, &key("attention.output_group_count"))?,
        vocab_size: gguf_vocab_size(metadata, "deepseek4")?,
        rms_norm_eps: gguf_f32(metadata, &key("attention.layer_norm_rms_epsilon"))?,
        max_position_embeddings: gguf_i32(metadata, &key("context_length"))?,
        rope_theta: gguf_optional_f32(metadata, &key("rope.freq_base"))?
            .unwrap_or_else(default_rope_theta),
        compress_rope_theta: gguf_f32(metadata, &key("attention.compress_rope_freq_base"))?,
        rope_scaling: parse_gguf_yarn(metadata, "deepseek4", false)?,
        sliding_window: gguf_i32(metadata, &key("attention.sliding_window"))?,
        compress_ratios: gguf_i32_array(metadata, &key("attention.compress_ratios"))?,
        index_n_heads: gguf_i32(metadata, &key("attention.indexer.head_count"))?,
        index_head_dim: gguf_i32(metadata, &key("attention.indexer.key_length"))?,
        index_topk: gguf_i32(metadata, &key("attention.indexer.top_k"))?,
        hc_mult: streams,
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
        swiglu_limit: routed_limit,
        num_nextn_predict_layers: prediction_layers,
        expert_dtype: Some("fp4".into()),
        quantization_config: None,
        tie_word_embeddings: false,
        dspark_block_size: None,
        dspark_noise_token_id: None,
        dspark_target_layer_ids: None,
        dspark_markov_rank: None,
    }
    .normalize()
}

fn parse_gguf_yarn(
    metadata: &HashMap<String, MetadataValue>,
    architecture: &str,
    deepseek2_log_multiplier: bool,
) -> Result<Option<YarnConfig>, ConfigError> {
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    match gguf_optional_string(metadata, &key("rope.scaling.type"))? {
        None => Ok(None),
        Some(kind) if matches!(kind.as_str(), "none" | "default") => Ok(None),
        Some(kind) if kind == "yarn" => Ok(Some(YarnConfig {
            r#type: kind,
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
            mscale_all_dim: if deepseek2_log_multiplier {
                gguf_optional_f32(metadata, &key("rope.scaling.yarn_log_multiplier"))?
                    .map(|value| value / 0.1)
                    .unwrap_or(1.0)
            } else {
                0.0
            },
        })),
        Some(kind) => Err(ConfigError::Invalid(format!(
            "DeepSeek GGUF RoPE scaling {kind:?} is unsupported"
        ))),
    }
}

fn gguf_vocab_size(
    metadata: &HashMap<String, MetadataValue>,
    architecture: &str,
) -> Result<i32, ConfigError> {
    match metadata
        .get("tokenizer.ggml.tokens")
        .and_then(MetadataValue::as_strings)
    {
        Some(tokens) => i32::try_from(tokens.len())
            .map_err(|_| ConfigError::Invalid("GGUF vocabulary exceeds i32".into())),
        None if metadata.contains_key("tokenizer.ggml.tokens") => Err(ConfigError::Invalid(
            "GGUF tokenizer.ggml.tokens has the wrong type".into(),
        )),
        None => gguf_i32(metadata, &format!("{architecture}.vocab_size")),
    }
}

fn gguf_string(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<String, ConfigError> {
    gguf_optional_string(metadata, key)?.ok_or_else(|| gguf_missing(key))
}

fn gguf_optional_string(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<String>, ConfigError> {
    match metadata.get(key) {
        Some(MetadataValue::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(gguf_wrong_type(key)),
        None => Ok(None),
    }
}

fn gguf_i32(metadata: &HashMap<String, MetadataValue>, key: &str) -> Result<i32, ConfigError> {
    let value = gguf_optional_i64(metadata, key)?.ok_or_else(|| gguf_missing(key))?;
    i32::try_from(value)
        .map_err(|_| ConfigError::Invalid(format!("GGUF metadata value {key:?} exceeds i32")))
}

fn gguf_optional_i64(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<i64>, ConfigError> {
    match metadata.get(key) {
        Some(value) => value.as_i64().map(Some).ok_or_else(|| gguf_wrong_type(key)),
        None => Ok(None),
    }
}

fn gguf_f32(metadata: &HashMap<String, MetadataValue>, key: &str) -> Result<f32, ConfigError> {
    gguf_optional_f32(metadata, key)?.ok_or_else(|| gguf_missing(key))
}

fn gguf_optional_f32(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<f32>, ConfigError> {
    match metadata.get(key) {
        Some(value) => value.as_f32().map(Some).ok_or_else(|| gguf_wrong_type(key)),
        None => Ok(None),
    }
}

fn gguf_optional_bool(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<bool>, ConfigError> {
    match metadata.get(key) {
        Some(MetadataValue::Bool(value)) => Ok(Some(*value)),
        Some(value) => value
            .as_i64()
            .map(|value| Some(value != 0))
            .ok_or_else(|| gguf_wrong_type(key)),
        None => Ok(None),
    }
}

fn gguf_bool(metadata: &HashMap<String, MetadataValue>, key: &str) -> Result<bool, ConfigError> {
    gguf_optional_bool(metadata, key)?.ok_or_else(|| gguf_missing(key))
}

fn gguf_i32_array(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Vec<i32>, ConfigError> {
    metadata
        .get(key)
        .and_then(MetadataValue::to_i64_vec)
        .ok_or_else(|| gguf_wrong_type(key))?
        .into_iter()
        .map(|value| {
            i32::try_from(value).map_err(|_| {
                ConfigError::Invalid(format!("GGUF metadata array {key:?} exceeds i32"))
            })
        })
        .collect()
}

fn gguf_uniform_f32_array(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
    expected: i32,
) -> Result<f32, ConfigError> {
    let values = match metadata.get(key) {
        Some(MetadataValue::Array(MetadataArray::Float32(values))) => values.clone(),
        Some(MetadataValue::Array(MetadataArray::Float64(values))) => {
            values.iter().map(|value| *value as f32).collect()
        }
        _ => return Err(gguf_wrong_type(key)),
    };
    let expected = usize::try_from(expected)
        .map_err(|_| ConfigError::Invalid("GGUF array length is negative".into()))?;
    if values.len() != expected || values.is_empty() {
        return Err(ConfigError::Invalid(format!(
            "GGUF metadata array {key:?} has {} values, expected {expected}",
            values.len()
        )));
    }
    let first = values[0];
    if values
        .iter()
        .any(|value| value.to_bits() != first.to_bits())
    {
        return Err(ConfigError::Invalid(format!(
            "GGUF metadata array {key:?} must be uniform"
        )));
    }
    Ok(first)
}

fn gguf_missing(key: &str) -> ConfigError {
    ConfigError::Invalid(format!("GGUF metadata is missing required key {key:?}"))
}

fn gguf_wrong_type(key: &str) -> ConfigError {
    ConfigError::Invalid(format!("GGUF metadata key {key:?} has the wrong type"))
}

/// Derives the canonical V3/R1 cache-relevant architecture fingerprint.
pub fn v3_architecture_fingerprint(args: &V3Args) -> String {
    eredu_core::cache::derive_prompt_cache_architecture_fingerprint(
        "deepseek_v3",
        [
            ("model_type", args.model_type.clone()),
            ("hidden_size", args.hidden_size.to_string()),
            ("num_hidden_layers", args.num_hidden_layers.to_string()),
            ("num_attention_heads", args.num_attention_heads.to_string()),
            ("kv_lora_rank", args.kv_lora_rank.to_string()),
            ("qk_rope_head_dim", args.qk_rope_head_dim.to_string()),
            ("layer_schedule", format!("{:?}", args.layer_schedule)),
            ("linear_format", format!("{:?}", args.linear_format)),
        ],
    )
}

/// Derives the canonical V4 cache-relevant architecture fingerprint.
pub fn v4_architecture_fingerprint(args: &V4Args) -> String {
    eredu_core::cache::derive_prompt_cache_architecture_fingerprint(
        "deepseek_v4",
        [
            ("model_type", args.model_type.clone()),
            ("hidden_size", args.hidden_size.to_string()),
            ("num_hidden_layers", args.num_hidden_layers.to_string()),
            ("num_attention_heads", args.num_attention_heads.to_string()),
            ("head_dim", args.head_dim.to_string()),
            (
                "attention_schedule",
                format!("{:?}", args.attention_schedule),
            ),
            ("hc_mult", args.hc_mult.to_string()),
            ("dspark", format!("{:?}", args.dspark)),
            ("linear_format", format!("{:?}", args.linear_format)),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Catalog(Vec<String>);

    impl GgufTensorCatalog for Catalog {
        fn contains(&self, name: &str) -> bool {
            self.0.iter().any(|candidate| candidate == name)
        }
    }

    fn fixture() -> Value {
        serde_json::json!({
            "architectures": ["DeepseekV3ForCausalLM"],
            "model_type": "deepseek_v3",
            "hidden_size": 16,
            "intermediate_size": 32,
            "moe_intermediate_size": 8,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "vocab_size": 128,
            "max_position_embeddings": 4096,
            "q_lora_rank": 4,
            "kv_lora_rank": 4,
            "qk_nope_head_dim": 6,
            "qk_rope_head_dim": 2,
            "v_head_dim": 8,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 2,
            "n_routed_experts": 8,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "n_group": 2,
            "topk_group": 1,
            "topk_method": "noaux_tc",
            "scoring_func": "sigmoid",
            "norm_topk_prob": true,
            "routed_scaling_factor": 1.0,
            "tie_word_embeddings": false,
            "attention_dropout": 0.0,
            "hidden_act": "silu"
        })
    }

    #[test]
    fn normalizes_v3_layer_schedule_without_backend_types() {
        let args = parse_v3_config(&fixture()).unwrap();
        assert_eq!(
            args.layer_schedule.iter().copied().collect::<Vec<_>>(),
            [
                LayerPolicy::DenseMlp,
                LayerPolicy::DenseMlp,
                LayerPolicy::SparseMoe,
                LayerPolicy::DenseMlp,
            ]
        );
        assert_eq!(args.linear_format, LinearFormat::Dense);
    }

    #[test]
    fn maps_official_fp8_metadata_to_general_linear_format() {
        let mut fixture = fixture();
        fixture["quantization_config"] = serde_json::json!({
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128]
        });
        assert!(matches!(
            parse_v3_config(&fixture).unwrap().linear_format,
            LinearFormat::E4M3BlockFp8(_)
        ));
    }

    #[test]
    fn rejects_v3_route_and_attention_semantic_drift() {
        let mut route_fixture = fixture();
        route_fixture["scoring_func"] = Value::String("softmax".into());
        assert!(parse_v3_config(&route_fixture).is_err());
        let mut attention_fixture = fixture();
        attention_fixture["attention_dropout"] = Value::from(0.1);
        assert!(parse_v3_config(&attention_fixture).is_err());
    }

    fn v4_fixture() -> Value {
        serde_json::json!({
            "model_type": "deepseek_v4",
            "hidden_size": 16,
            "moe_intermediate_size": 8,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "qk_rope_head_dim": 2,
            "q_lora_rank": 4,
            "o_lora_rank": 4,
            "o_groups": 2,
            "vocab_size": 128,
            "max_position_embeddings": 4096,
            "sliding_window": 8,
            "compress_ratios": [0, 4, 0, 0],
            "index_n_heads": 4,
            "index_head_dim": 4,
            "index_topk": 2,
            "hc_mult": 2,
            "hc_sinkhorn_iters": 4,
            "n_routed_experts": 8,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "num_hash_layers": 1,
            "scoring_func": "sqrtsoftplus",
            "topk_method": "noaux_tc",
            "norm_topk_prob": true,
            "routed_scaling_factor": 1.0,
            "swiglu_limit": 7.0,
            "num_nextn_predict_layers": 1
        })
    }

    #[test]
    fn normalizes_v4_attention_and_limit_policies() {
        let args = parse_v4_config(&v4_fixture()).unwrap();
        assert_eq!(args.attention_policy(0), Some(V4AttentionPolicy::Local));
        assert_eq!(
            args.attention_policy(1),
            Some(V4AttentionPolicy::Compressed { ratio: 4 })
        );
        assert_eq!(args.attention_policy(3), Some(V4AttentionPolicy::Local));
        assert_eq!(args.swiglu_limit.unwrap().get(), 7.0);
    }

    #[test]
    fn maps_v4_ue8m0_fp8_and_validates_dspark_atomically() {
        let mut fixture = v4_fixture();
        fixture["quantization_config"] = serde_json::json!({
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "scale_fmt": "ue8m0"
        });
        assert!(matches!(
            parse_v4_config(&fixture).unwrap().linear_format,
            LinearFormat::E4M3BlockFp8(BlockFp8Format {
                scale_encoding: BlockFp8ScaleEncoding::Ue8m0,
                ..
            })
        ));
        fixture["dspark_block_size"] = Value::from(4);
        assert!(parse_v4_config(&fixture).is_err());
    }

    #[test]
    fn deepseek_state_and_moe_policies_are_neutral_and_scheduled() {
        let v3 = parse_v3_config(&fixture()).unwrap();
        let v3_layout = crate::deepseek::v3::state_layout(&v3).unwrap();
        assert_eq!(v3_layout.len(), 4);
        assert_eq!(
            v3_layout.components(0).unwrap()[0].role.stable_name(),
            "attention.compressed_latent"
        );
        let v3_moe = crate::deepseek::v3::moe_policy(&v3, 2).unwrap();
        assert_eq!(v3_moe.expert_groups, 2);
        assert!(v3_moe.correction_bias.is_some());

        let v4 = parse_v4_config(&v4_fixture()).unwrap();
        let v4_layout = crate::deepseek::v4::state_layout(&v4).unwrap();
        assert_eq!(v4_layout.len(), 4);
        assert_eq!(v4_layout.components(0).unwrap().len(), 1);
        let compressed = v4_layout.components(1).unwrap();
        assert_eq!(compressed.len(), 11);
        assert!(compressed.iter().any(|component| {
            component.role.stable_name() == "state.pooling.1.pooled"
                && component.residency == eredu_core::cache::StateResidencyClass::SealablePaged
        }));
        assert!(crate::deepseek::v4::moe_policy(&v4, 0)
            .unwrap()
            .correction_bias
            .is_none());
        assert!(crate::deepseek::v4::moe_policy(&v4, 1)
            .unwrap()
            .correction_bias
            .is_some());
    }

    #[test]
    fn parses_v3_gguf_without_backend_metadata_types() {
        let metadata = HashMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String("deepseek2".into()),
            ),
            (
                "deepseek2.embedding_length".into(),
                MetadataValue::Uint32(128),
            ),
            (
                "deepseek2.feed_forward_length".into(),
                MetadataValue::Uint32(256),
            ),
            (
                "deepseek2.expert_feed_forward_length".into(),
                MetadataValue::Uint32(64),
            ),
            ("deepseek2.block_count".into(), MetadataValue::Uint32(2)),
            (
                "deepseek2.attention.head_count".into(),
                MetadataValue::Uint32(4),
            ),
            ("deepseek2.vocab_size".into(), MetadataValue::Uint32(128)),
            (
                "deepseek2.attention.layer_norm_rms_epsilon".into(),
                MetadataValue::Float32(1e-6),
            ),
            (
                "deepseek2.context_length".into(),
                MetadataValue::Uint32(4096),
            ),
            (
                "deepseek2.rope.dimension_count".into(),
                MetadataValue::Uint32(8),
            ),
            (
                "deepseek2.attention.key_length_mla".into(),
                MetadataValue::Uint32(32),
            ),
            (
                "deepseek2.attention.kv_lora_rank".into(),
                MetadataValue::Uint32(32),
            ),
            (
                "deepseek2.attention.value_length_mla".into(),
                MetadataValue::Uint32(24),
            ),
            (
                "deepseek2.leading_dense_block_count".into(),
                MetadataValue::Uint32(1),
            ),
            ("deepseek2.expert_count".into(), MetadataValue::Uint32(4)),
            (
                "deepseek2.expert_shared_count".into(),
                MetadataValue::Uint32(1),
            ),
            (
                "deepseek2.expert_used_count".into(),
                MetadataValue::Uint32(2),
            ),
            (
                "deepseek2.expert_group_count".into(),
                MetadataValue::Uint32(2),
            ),
            (
                "deepseek2.expert_group_used_count".into(),
                MetadataValue::Uint32(1),
            ),
        ]);
        let catalog = Catalog(vec!["blk.0.attn_k_b.weight".into()]);
        let args = parse_v3_gguf(&catalog, &metadata).unwrap();
        assert_eq!(args.qk_nope_head_dim, 24);
        assert!(v3_uses_split_kv(&catalog));
    }

    #[test]
    fn parses_v4_uniform_arrays_and_compression_schedule_from_gguf() {
        let metadata = HashMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String("deepseek4".into()),
            ),
            ("deepseek4.block_count".into(), MetadataValue::Uint32(3)),
            (
                "deepseek4.attention.compress_ratios".into(),
                MetadataValue::Array(MetadataArray::Int32(vec![0, 4, 0])),
            ),
            (
                "deepseek4.swiglu_clamp_exp".into(),
                MetadataValue::Array(MetadataArray::Float32(vec![7.0; 3])),
            ),
            (
                "deepseek4.swiglu_clamp_shexp".into(),
                MetadataValue::Array(MetadataArray::Float32(vec![7.0; 3])),
            ),
            (
                "deepseek4.expert_gating_func".into(),
                MetadataValue::Uint32(3),
            ),
            (
                "deepseek4.embedding_length".into(),
                MetadataValue::Uint32(128),
            ),
            (
                "deepseek4.hyper_connection.count".into(),
                MetadataValue::Uint32(2),
            ),
            (
                "deepseek4.embedding_length_out".into(),
                MetadataValue::Uint32(256),
            ),
            (
                "deepseek4.expert_feed_forward_length".into(),
                MetadataValue::Uint32(64),
            ),
            (
                "deepseek4.attention.head_count".into(),
                MetadataValue::Uint32(4),
            ),
            (
                "deepseek4.attention.head_count_kv".into(),
                MetadataValue::Uint32(1),
            ),
            (
                "deepseek4.attention.key_length".into(),
                MetadataValue::Uint32(32),
            ),
            (
                "deepseek4.rope.dimension_count".into(),
                MetadataValue::Uint32(8),
            ),
            (
                "deepseek4.attention.q_lora_rank".into(),
                MetadataValue::Uint32(32),
            ),
            (
                "deepseek4.attention.output_lora_rank".into(),
                MetadataValue::Uint32(16),
            ),
            (
                "deepseek4.attention.output_group_count".into(),
                MetadataValue::Uint32(2),
            ),
            ("deepseek4.vocab_size".into(), MetadataValue::Uint32(128)),
            (
                "deepseek4.attention.layer_norm_rms_epsilon".into(),
                MetadataValue::Float32(1e-6),
            ),
            (
                "deepseek4.context_length".into(),
                MetadataValue::Uint32(4096),
            ),
            (
                "deepseek4.attention.compress_rope_freq_base".into(),
                MetadataValue::Float32(160_000.0),
            ),
            (
                "deepseek4.attention.sliding_window".into(),
                MetadataValue::Uint32(128),
            ),
            (
                "deepseek4.attention.indexer.head_count".into(),
                MetadataValue::Uint32(4),
            ),
            (
                "deepseek4.attention.indexer.key_length".into(),
                MetadataValue::Uint32(16),
            ),
            (
                "deepseek4.attention.indexer.top_k".into(),
                MetadataValue::Uint32(2),
            ),
            (
                "deepseek4.hyper_connection.sinkhorn_iterations".into(),
                MetadataValue::Uint32(4),
            ),
            (
                "deepseek4.hyper_connection.epsilon".into(),
                MetadataValue::Float32(1e-6),
            ),
            ("deepseek4.expert_count".into(), MetadataValue::Uint32(4)),
            (
                "deepseek4.expert_shared_count".into(),
                MetadataValue::Uint32(1),
            ),
            (
                "deepseek4.expert_used_count".into(),
                MetadataValue::Uint32(2),
            ),
            (
                "deepseek4.hash_layer_count".into(),
                MetadataValue::Uint32(1),
            ),
            (
                "deepseek4.expert_weights_norm".into(),
                MetadataValue::Bool(true),
            ),
            (
                "deepseek4.expert_weights_scale".into(),
                MetadataValue::Float32(1.0),
            ),
        ]);
        let args = parse_v4_gguf(&metadata).unwrap();
        assert_eq!(
            args.attention_policy(1),
            Some(V4AttentionPolicy::Compressed { ratio: 4 })
        );
        assert_eq!(args.expert_format, ExpertFormat::MxFp4);
        assert_eq!(args.swiglu_limit.unwrap().get(), 7.0);
    }
}
