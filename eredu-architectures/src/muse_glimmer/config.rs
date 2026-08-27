//! Strict Muse-Glimmer text and vision configuration normalization.

use std::collections::{HashMap, HashSet};

use crate::rotary::RopeValue;
use eredu_checkpoint::{LinearFormat, WeightQuantization};
use eredu_core::{AttentionPolicy, LayerSchedule};
use eredu_gguf::{MetadataArray, MetadataValue};
use serde::Deserialize;
use serde_json::Value;

use crate::GgufTensorCatalog;

/// Invalid or unsupported Muse-Glimmer configuration.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// JSON decoding failed.
    #[error("invalid Muse-Glimmer configuration: {0}")]
    Json(#[from] serde_json::Error),
    /// Normalized geometry or policy is unsupported.
    #[error("{0}")]
    Invalid(String),
}

fn invalid(message: impl Into<String>) -> ConfigError {
    ConfigError::Invalid(message.into())
}

/// Checkpoint-specific norm and rotary convention.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum WeightConvention {
    /// Centered norms and rotate-half RoPE.
    HuggingFace,
    /// Shifted norms and permuted traditional RoPE.
    Gguf,
}

impl WeightConvention {
    /// Returns whether execution uses traditional interleaved RoPE pairs.
    pub const fn uses_traditional_rope(self) -> bool {
        matches!(self, Self::Gguf)
    }
}

/// Windowed or full vision self-attention.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum VisionAttentionPolicy {
    /// Window-partitioned attention.
    Window,
    /// Full packed-sequence attention.
    Full,
}

/// Validated Muse-Glimmer vision tower geometry.
#[derive(Debug, Clone)]
pub struct VisionConfig {
    /// Vision hidden width.
    pub hidden_size: i32,
    /// Vision MLP width.
    pub intermediate_size: i32,
    /// Attention head count.
    pub num_heads: i32,
    /// Spatial patch edge.
    pub patch_size: i32,
    /// Temporal patch depth.
    pub temporal_patch_size: i32,
    /// Spatial merge edge.
    pub merge_size: i32,
    /// Learned position-table height.
    pub position_height: i32,
    /// Learned position-table width.
    pub position_width: i32,
    /// Layer-normalization epsilon.
    pub layer_norm_eps: f32,
    /// Two-axis rotary base.
    pub rope_theta: f32,
    /// Ordered attention policy.
    pub schedule: Vec<VisionAttentionPolicy>,
    /// Per-weight mixed encodings.
    pub quantized_weight_configs: HashMap<String, WeightQuantization>,
    /// Optional uniform physical encoding selected by load-time policy.
    pub weight_quantization: Option<WeightQuantization>,
    /// Language decoder output width.
    pub language_hidden_size: i32,
    /// Hidden width of the two-layer language adapter.
    pub projector_hidden_size: i32,
}

impl VisionConfig {
    /// Returns the exact number of vision blocks.
    pub fn layer_count(&self) -> usize {
        self.schedule.len()
    }

    /// Resolves one vision weight's physical encoding.
    pub fn weight_quantization_for(&self, name: &str) -> Option<WeightQuantization> {
        self.quantized_weight_configs
            .get(name)
            .copied()
            .or(self.weight_quantization)
    }

    /// Resolves one text projection's complete physical format.
    pub fn linear_format_for(&self, name: &str) -> LinearFormat {
        self.weight_quantization_for(name)
            .map(Into::into)
            .unwrap_or(LinearFormat::Dense)
    }
}

#[derive(Debug, Deserialize)]
struct VisionSource {
    model_type: String,
    hidden_size: i32,
    intermediate_size: i32,
    num_attention_heads: i32,
    num_hidden_layers: i32,
    patch_size: i32,
    patch_temporal: i32,
    merge_size: i32,
    pos_emb_height: i32,
    pos_emb_width: i32,
    max_position_embeddings: i32,
    layer_norm_eps: f32,
    hidden_act: String,
    layer_types: Vec<String>,
    rope_parameters: HashMap<String, Value>,
}

/// Validated text, media-token, and projector configuration.
#[derive(Debug, Clone)]
pub struct DecoderConfig {
    /// Family identity of the text submodel.
    pub model_type: String,
    /// Decoder hidden width.
    pub hidden_size: i32,
    /// Decoder depth.
    pub num_hidden_layers: i32,
    /// Dense SwiGLU width.
    pub intermediate_size: i32,
    /// Query head count.
    pub num_attention_heads: i32,
    /// Input/final RMS epsilon.
    pub rms_norm_eps: f32,
    /// Post-branch RMS epsilon.
    pub post_norm_eps: f32,
    /// Stored vocabulary size.
    pub vocab_size: i32,
    /// Key/value head count.
    pub num_key_value_heads: i32,
    /// Maximum configured context.
    pub max_position_embeddings: i32,
    /// Text rotary base.
    pub rope_theta: f32,
    /// Per-layer rotary enablement; NoPE layers are false.
    pub layer_uses_rope: Vec<bool>,
    /// Per-head width.
    pub head_dim: i32,
    /// Tied output-head policy.
    pub tie_word_embeddings: bool,
    /// Optional text rotary scaling.
    pub rope_scaling: Option<HashMap<String, RopeValue>>,
    /// Dense activation.
    pub hidden_act: String,
    /// Inference attention dropout.
    pub attention_dropout: f32,
    /// Optional attention-bias declaration.
    pub attention_bias: Option<bool>,
    /// Optional MLP-bias declaration.
    pub mlp_bias: Option<bool>,
    /// Per-expert SwiGLU intermediate width; zero for dense checkpoints.
    pub moe_intermediate_size: i32,
    /// Routed expert count; zero for dense checkpoints.
    pub num_experts: i32,
    /// Experts selected per token; zero for dense checkpoints.
    pub num_experts_per_tok: i32,
    /// Whether selected softmax probabilities are renormalized.
    pub norm_topk_prob: bool,
    /// Ordered full/sliding schedule.
    pub attention_schedule: LayerSchedule<AttentionPolicy>,
    /// Scale after weightless Q/K normalization.
    pub qk_scale_factor: f32,
    /// Raw output-logit multiplier.
    pub output_multiplier: f32,
    /// Positive tanh logit cap.
    pub final_logit_softcapping: f32,
    /// Source-specific parameter convention.
    pub weight_convention: WeightConvention,
    /// Canonical uniform quantization.
    pub quantization: Option<WeightQuantization>,
    /// Exact names quantized under a mixed artifact.
    pub quantized_weights: Option<HashSet<String>>,
    /// Per-weight mixed encodings.
    pub quantized_weight_configs: Option<HashMap<String, WeightQuantization>>,
    /// Optional native vision tower supplied by the main artifact or a GGUF projector.
    pub vision_config: Option<VisionConfig>,
    /// Image placeholder token.
    pub image_token_id: u32,
    /// Video placeholder token.
    pub video_token_id: u32,
    /// Pixel-shuffle width entering the adapter.
    pub vision_out_hidden_size: i32,
    /// Hidden width of the vision adapter.
    pub projector_hidden_size: i32,
}

#[derive(Debug, Deserialize)]
struct TextSource {
    model_type: String,
    hidden_size: i32,
    num_hidden_layers: i32,
    intermediate_size: i32,
    num_attention_heads: i32,
    rms_norm_eps: f32,
    post_norm_eps: f32,
    vocab_size: i32,
    num_key_value_heads: i32,
    max_position_embeddings: i32,
    #[serde(default)]
    rope_theta: Option<f32>,
    #[serde(default)]
    rope_parameters: HashMap<String, RopeValue>,
    layer_types: Vec<String>,
    layer_rope_theta: Vec<f32>,
    sliding_window: i32,
    #[serde(default)]
    head_dim: i32,
    tie_word_embeddings: bool,
    rope_scaling: Option<HashMap<String, RopeValue>>,
    #[serde(default = "default_silu", alias = "hidden_activation")]
    hidden_act: String,
    #[serde(default)]
    attention_dropout: f32,
    #[serde(default)]
    attention_bias: Option<bool>,
    #[serde(default)]
    mlp_bias: Option<bool>,
    #[serde(default)]
    moe_intermediate_size: i32,
    #[serde(default)]
    num_experts: i32,
    #[serde(default)]
    num_experts_per_tok: i32,
    #[serde(default)]
    norm_topk_prob: bool,
    #[serde(default)]
    quantization: Option<WeightQuantization>,
    #[serde(default)]
    quantization_config: Option<WeightQuantization>,
    qk_scale_factor: f32,
    output_multiplier: f32,
    final_logit_softcapping: f32,
}

fn default_silu() -> String {
    "silu".into()
}

impl DecoderConfig {
    /// Stable schedule, geometry, and modality identity for persisted state.
    pub fn architecture_fingerprint(&self) -> String {
        eredu_core::cache::derive_prompt_cache_architecture_fingerprint(
            "muse-glimmer",
            [
                ("model_type", self.model_type.clone()),
                ("hidden", self.hidden_size.to_string()),
                ("vocab", self.vocab_size.to_string()),
                (
                    "schedule",
                    self.attention_schedule
                        .iter()
                        .map(|policy| format!("{policy:?}"))
                        .collect::<Vec<_>>()
                        .join(","),
                ),
                (
                    "vision",
                    self.vision_config.as_ref().map_or_else(
                        || "none".into(),
                        |vision| {
                            format!(
                                "{}:{}:{}:{}:{}",
                                vision.hidden_size,
                                vision.layer_count(),
                                vision.patch_size,
                                vision.temporal_patch_size,
                                vision.merge_size
                            )
                        },
                    ),
                ),
                (
                    "tokens",
                    format!("{}:{}", self.image_token_id, self.video_token_id),
                ),
                ("quantization", format!("{:?}", self.quantization)),
                (
                    "quantized_weights",
                    crate::cache_identity::string_set(self.quantized_weights.as_ref()),
                ),
                (
                    "quantized_weight_configs",
                    crate::cache_identity::debug_map(self.quantized_weight_configs.as_ref()),
                ),
                (
                    "vision_quantization",
                    self.vision_config.as_ref().map_or_else(
                        || "none".into(),
                        |vision| {
                            format!(
                                "default={:?};overrides={}",
                                vision.weight_quantization,
                                crate::cache_identity::debug_map(Some(
                                    &vision.quantized_weight_configs
                                ))
                            )
                        },
                    ),
                ),
            ],
        )
    }

    /// Parses and strictly validates one released Hugging Face family config.
    pub fn from_hf_json(bytes: &[u8]) -> Result<Self, ConfigError> {
        let value: Value = serde_json::from_slice(bytes)?;
        Self::from_hf_value(&value)
    }

    /// Parses an already-decoded Hugging Face family config.
    pub fn from_hf_value(value: &Value) -> Result<Self, ConfigError> {
        if value.get("model_type").and_then(Value::as_str) != Some("muse_glimmer") {
            return Err(invalid("Muse-Glimmer model_type must be muse_glimmer"));
        }
        if let Some(architectures) = value.get("architectures") {
            let architectures = architectures
                .as_array()
                .ok_or_else(|| invalid("architectures must be an array"))?;
            if architectures.len() != 1
                || architectures[0].as_str() != Some("MuseGlimmerForConditionalGeneration")
            {
                return Err(invalid(
                    "Muse-Glimmer requires MuseGlimmerForConditionalGeneration",
                ));
            }
        }
        let text_value = value
            .get("text_config")
            .ok_or_else(|| invalid("Muse-Glimmer config is missing text_config"))?;
        reject_execution_overrides(text_value)?;
        let mut source: TextSource = serde_json::from_value(text_value.clone())?;
        if source.head_dim == 0
            && source.num_attention_heads > 0
            && source.hidden_size % source.num_attention_heads == 0
        {
            source.head_dim = source.hidden_size / source.num_attention_heads;
        }
        let schedule = text_schedule(&source)?;
        let rope_theta = source
            .rope_theta
            .or_else(|| match source.rope_parameters.get("rope_theta") {
                Some(RopeValue::Float(value)) => Some(*value),
                _ => None,
            })
            .unwrap_or(500_000.0);
        let projector_hidden_size = positive_i32(value, "projector_hidden_size")?;
        let vision = vision_config(
            value
                .get("vision_config")
                .ok_or_else(|| invalid("Muse-Glimmer config is missing vision_config"))?,
            source.hidden_size,
            projector_hidden_size,
        )?;
        let quantization = match (source.quantization, source.quantization_config) {
            (Some(first), Some(second)) if first != second => {
                return Err(invalid(
                    "Muse-Glimmer quantization and quantization_config disagree",
                ));
            }
            (Some(value), _) | (_, Some(value)) => Some(value),
            (None, None) => None,
        };
        let config = Self {
            model_type: source.model_type,
            hidden_size: source.hidden_size,
            num_hidden_layers: source.num_hidden_layers,
            intermediate_size: source.intermediate_size,
            num_attention_heads: source.num_attention_heads,
            rms_norm_eps: source.rms_norm_eps,
            post_norm_eps: source.post_norm_eps,
            vocab_size: source.vocab_size,
            num_key_value_heads: source.num_key_value_heads,
            max_position_embeddings: source.max_position_embeddings,
            rope_theta,
            layer_uses_rope: source
                .layer_rope_theta
                .iter()
                .map(|theta| *theta != 0.0)
                .collect(),
            head_dim: source.head_dim,
            tie_word_embeddings: source.tie_word_embeddings,
            rope_scaling: source.rope_scaling,
            hidden_act: source.hidden_act,
            attention_dropout: source.attention_dropout,
            attention_bias: source.attention_bias,
            mlp_bias: source.mlp_bias,
            moe_intermediate_size: source.moe_intermediate_size,
            num_experts: source.num_experts,
            num_experts_per_tok: source.num_experts_per_tok,
            norm_topk_prob: source.norm_topk_prob,
            attention_schedule: schedule,
            qk_scale_factor: source.qk_scale_factor,
            output_multiplier: source.output_multiplier,
            final_logit_softcapping: source.final_logit_softcapping,
            weight_convention: WeightConvention::HuggingFace,
            quantization,
            quantized_weights: None,
            quantized_weight_configs: None,
            image_token_id: required_u32(value, "image_token_id")?,
            video_token_id: required_u32(value, "video_token_id")?,
            vision_out_hidden_size: positive_i32(value, "out_hidden_size")?,
            projector_hidden_size,
            vision_config: Some(vision),
        };
        config.validate()?;
        Ok(config)
    }

    /// Parses the released text GGUF metadata and architecture-owned physical
    /// tensor semantics without depending on a backend checkpoint type.
    /// Projector metadata is admitted independently.
    pub fn from_gguf_catalog<C: GgufTensorCatalog + ?Sized>(
        catalog: &C,
        metadata: &HashMap<String, MetadataValue>,
    ) -> Result<Self, ConfigError> {
        let architecture = gguf_string(metadata, "general.architecture")?;
        if architecture != "muse-glimmer" {
            return Err(invalid(format!(
                "Muse-Glimmer GGUF architecture must be muse-glimmer, got {architecture:?}"
            )));
        }
        let key = |suffix: &str| format!("muse-glimmer.{suffix}");
        let layers = gguf_i32(metadata, &key("block_count"))?;
        let hidden_size = gguf_i32(metadata, &key("embedding_length"))?;
        let heads = gguf_i32(metadata, &key("attention.head_count"))?;
        if layers <= 0 || hidden_size <= 0 || heads <= 0 {
            return Err(invalid(
                "Muse-Glimmer GGUF block, embedding, and head counts must be positive",
            ));
        }
        let key_value_heads =
            gguf_optional_i32(metadata, &key("attention.head_count_kv"))?.unwrap_or(heads);
        let head_dim = gguf_optional_i32(metadata, &key("attention.key_length"))?
            .unwrap_or(hidden_size / heads);
        if let Some(value_dim) = gguf_optional_i32(metadata, &key("attention.value_length"))? {
            if value_dim != head_dim {
                return Err(invalid(format!(
                    "Muse-Glimmer GGUF value width {value_dim} does not match head width {head_dim}"
                )));
            }
        }
        if let Some(rotary_dim) = gguf_optional_i32(metadata, &key("rope.dimension_count"))? {
            if rotary_dim != head_dim {
                return Err(invalid(format!(
                    "Muse-Glimmer GGUF rotary width {rotary_dim} does not match head width {head_dim}"
                )));
            }
        }
        if gguf_optional_bool(metadata, &key("attention.causal"))? == Some(false) {
            return Err(invalid("Muse-Glimmer GGUF attention must be causal"));
        }
        if let Some(activation) = gguf_optional_string(metadata, &key("hidden_activation"))? {
            if activation != "silu" {
                return Err(invalid(format!(
                    "Muse-Glimmer GGUF activation must be silu, got {activation:?}"
                )));
            }
        }
        let schedule = gguf_attention_schedule(metadata, layers)?;
        let layer_uses_rope = schedule
            .iter()
            .map(|policy| policy.window().is_some())
            .collect();
        let experts = gguf_optional_i32(metadata, &key("expert_count"))?.unwrap_or(0);
        let is_moe = experts > 0;
        let config = Self {
            model_type: "muse_glimmer_text".into(),
            hidden_size,
            num_hidden_layers: layers,
            intermediate_size: if is_moe {
                gguf_optional_i32(metadata, &key("feed_forward_length"))?.unwrap_or(0)
            } else {
                gguf_i32(metadata, &key("feed_forward_length"))?
            },
            num_attention_heads: heads,
            rms_norm_eps: gguf_f32(metadata, &key("attention.layer_norm_rms_epsilon"))?,
            post_norm_eps: gguf_optional_f32(metadata, &key("attention.post_norm_rms_epsilon"))?
                .unwrap_or(1e-8),
            vocab_size: gguf_vocab_size(metadata, &key("vocab_size"))?,
            num_key_value_heads: key_value_heads,
            max_position_embeddings: gguf_i32(metadata, &key("context_length"))?,
            rope_theta: gguf_optional_f32(metadata, &key("rope.freq_base"))?.unwrap_or(1_000_000.0),
            layer_uses_rope,
            head_dim,
            tie_word_embeddings: !catalog.contains("output.weight"),
            rope_scaling: gguf_rope_scaling(metadata)?,
            hidden_act: "silu".into(),
            attention_dropout: 0.0,
            attention_bias: Some(false),
            mlp_bias: Some(false),
            moe_intermediate_size: if is_moe {
                gguf_i32(metadata, &key("expert_feed_forward_length"))?
            } else {
                0
            },
            num_experts: experts,
            num_experts_per_tok: if is_moe {
                gguf_i32(metadata, &key("expert_used_count"))?
            } else {
                0
            },
            norm_topk_prob: is_moe,
            attention_schedule: schedule,
            qk_scale_factor: 3.87,
            output_multiplier: gguf_f32(metadata, &key("logit_scale"))?,
            final_logit_softcapping: gguf_f32(metadata, &key("final_logit_softcapping"))?,
            weight_convention: WeightConvention::Gguf,
            quantization: None,
            quantized_weights: None,
            quantized_weight_configs: None,
            vision_config: None,
            image_token_id: gguf_optional_i32(metadata, &key("image_token_id"))?
                .map(|value| u32::try_from(value).map_err(|_| invalid("negative image token id")))
                .transpose()?
                .unwrap_or(200_092),
            video_token_id: gguf_optional_i32(metadata, &key("video_token_id"))?
                .map(|value| u32::try_from(value).map_err(|_| invalid("negative video token id")))
                .transpose()?
                .unwrap_or(200_091),
            vision_out_hidden_size: 6_144,
            projector_hidden_size: 4_096,
        };
        config.validate()?;
        Ok(config)
    }

    /// Validates and applies the official image-only projector GGUF geometry.
    pub fn with_gguf_projector_metadata(
        mut self,
        metadata: &HashMap<String, MetadataValue>,
        quantized_weight_configs: HashMap<String, WeightQuantization>,
    ) -> Result<Self, ConfigError> {
        for (key, expected) in [
            ("general.architecture", "clip"),
            ("general.type", "mmproj"),
            ("clip.projector_type", "muse-glimmer"),
        ] {
            let value = gguf_string(metadata, key)?;
            if value != expected {
                return Err(invalid(format!(
                    "Muse-Glimmer projector {key:?} must be {expected:?}, got {value:?}"
                )));
            }
        }
        if gguf_optional_bool(metadata, "clip.has_vision_encoder")? != Some(true) {
            return Err(invalid(
                "Muse-Glimmer projector must contain its vision encoder",
            ));
        }
        let expected = [
            ("clip.vision.embedding_length", 1_536),
            ("clip.vision.feed_forward_length", 8_960),
            ("clip.vision.block_count", 50),
            ("clip.vision.attention.head_count", 16),
            ("clip.vision.patch_size", 14),
            ("clip.vision.spatial_merge_size", 2),
            ("clip.vision.image_size", 896),
            ("clip.vision.projection_dim", self.hidden_size),
        ];
        for (key, expected) in expected {
            let actual = gguf_i32(metadata, key)?;
            if actual != expected {
                return Err(invalid(format!(
                    "Muse-Glimmer projector {key:?} is {actual}, expected {expected}"
                )));
            }
        }
        let epsilon = gguf_f32(metadata, "clip.vision.attention.layer_norm_epsilon")?;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(invalid(
                "Muse-Glimmer projector layer norm epsilon must be positive",
            ));
        }
        let mut vision = official_gguf_vision(self.hidden_size);
        vision.layer_norm_eps = epsilon;
        vision.quantized_weight_configs = quantized_weight_configs;
        self.vision_config = Some(vision);
        self.validate()?;
        Ok(self)
    }

    /// Resolves one text weight's physical encoding.
    pub fn weight_quantization_for(&self, name: &str) -> Option<WeightQuantization> {
        if let Some(format) = self
            .quantized_weight_configs
            .as_ref()
            .and_then(|formats| formats.get(name))
        {
            return Some(*format);
        }
        let format = self.quantization?;
        match &self.quantized_weights {
            Some(names) if !names.contains(name) => None,
            _ => Some(format),
        }
    }

    /// Resolves one text projection's complete physical format.
    pub fn linear_format_for(&self, name: &str) -> LinearFormat {
        self.weight_quantization_for(name)
            .map(Into::into)
            .unwrap_or(LinearFormat::Dense)
    }

    /// Returns whether decoder blocks use the routed gated-product bank.
    pub const fn is_moe(&self) -> bool {
        self.num_experts > 0
    }

    pub(crate) fn validate(&self) -> Result<(), ConfigError> {
        if self.model_type != "muse_glimmer_text" {
            return Err(invalid("text model_type must be muse_glimmer_text"));
        }
        for (name, value) in [
            ("hidden_size", self.hidden_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("head_dim", self.head_dim),
            ("vocab_size", self.vocab_size),
            ("max_position_embeddings", self.max_position_embeddings),
        ] {
            if value <= 0 {
                return Err(invalid(format!("{name} must be positive, got {value}")));
            }
        }
        if self.is_moe() {
            if self.moe_intermediate_size <= 0
                || self.num_experts_per_tok <= 0
                || self.num_experts_per_tok > self.num_experts
                || self.intermediate_size < 0
            {
                return Err(invalid("Muse-Glimmer MoE geometry is invalid"));
            }
        } else if self.intermediate_size <= 0
            || self.moe_intermediate_size != 0
            || self.num_experts_per_tok != 0
            || self.norm_topk_prob
        {
            return Err(invalid(
                "Muse-Glimmer dense feed-forward geometry is invalid",
            ));
        }
        if self.num_attention_heads % self.num_key_value_heads != 0
            || self.hidden_act != "silu"
            || self.attention_dropout != 0.0
            || self.attention_bias == Some(true)
            || self.mlp_bias == Some(true)
        {
            return Err(invalid(
                "Muse-Glimmer text attention, activation, bias, or dropout policy is unsupported",
            ));
        }
        for (name, value) in [
            ("rms_norm_eps", self.rms_norm_eps),
            ("post_norm_eps", self.post_norm_eps),
            ("rope_theta", self.rope_theta),
            ("qk_scale_factor", self.qk_scale_factor),
            ("output_multiplier", self.output_multiplier),
            ("final_logit_softcapping", self.final_logit_softcapping),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(invalid(format!("{name} must be finite and positive")));
            }
        }
        if let Some(vision) = &self.vision_config {
            if self.vision_out_hidden_size != vision.hidden_size * 4 {
                return Err(invalid(
                    "out_hidden_size must be four times the vision hidden width",
                ));
            }
        }
        if self.image_token_id >= self.vocab_size as u32
            || self.video_token_id >= self.vocab_size as u32
            || self.image_token_id == self.video_token_id
        {
            return Err(invalid("image/video placeholder tokens are invalid"));
        }
        validate_rope_scaling(self.rope_scaling.as_ref())
    }
}

fn text_schedule(source: &TextSource) -> Result<LayerSchedule<AttentionPolicy>, ConfigError> {
    let layers = usize::try_from(source.num_hidden_layers)
        .ok()
        .filter(|layers| *layers > 0)
        .ok_or_else(|| invalid("num_hidden_layers must be positive"))?;
    if source.layer_types.len() != layers || source.layer_rope_theta.len() != layers {
        return Err(invalid(
            "layer_types and layer_rope_theta must match decoder depth",
        ));
    }
    let sliding = AttentionPolicy::from_sliding_window(Some(source.sliding_window))
        .map_err(|error| invalid(error.to_string()))?;
    let policies = source
        .layer_types
        .iter()
        .zip(&source.layer_rope_theta)
        .enumerate()
        .map(
            |(layer, (kind, theta))| match (kind.as_str(), *theta == 0.0) {
                ("sliding_attention", false) => Ok(sliding),
                ("full_attention", true) => Ok(AttentionPolicy::Full),
                _ => Err(invalid(format!(
                    "layer {layer} has incompatible attention kind and RoPE theta"
                ))),
            },
        )
        .collect::<Result<Vec<_>, _>>()?;
    LayerSchedule::new(layers, policies).map_err(|error| invalid(error.to_string()))
}

fn vision_config(
    value: &Value,
    language_hidden_size: i32,
    projector_hidden_size: i32,
) -> Result<VisionConfig, ConfigError> {
    let source: VisionSource = serde_json::from_value(value.clone())?;
    if source.model_type != "muse_glimmer_vision"
        || source.hidden_act != "gelu"
        || source.hidden_size <= 0
        || source.intermediate_size <= 0
        || source.num_attention_heads <= 0
        || source.hidden_size % source.num_attention_heads != 0
        || source.patch_size <= 0
        || source.patch_temporal <= 0
        || source.merge_size <= 0
        || source.pos_emb_height <= 0
        || source.pos_emb_width <= 0
        || source.pos_emb_height * source.pos_emb_width != source.max_position_embeddings
        || !source.layer_norm_eps.is_finite()
        || source.layer_norm_eps <= 0.0
    {
        return Err(invalid("Muse-Glimmer vision geometry is invalid"));
    }
    if source.layer_types.len() != source.num_hidden_layers as usize {
        return Err(invalid("vision layer_types must match depth"));
    }
    let schedule = source
        .layer_types
        .iter()
        .map(|kind| match kind.as_str() {
            "window_attention" => Ok(VisionAttentionPolicy::Window),
            "full_attention" => Ok(VisionAttentionPolicy::Full),
            _ => Err(invalid(format!("unsupported vision layer type {kind:?}"))),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let rope_theta = source
        .rope_parameters
        .get("rope_theta")
        .and_then(Value::as_f64)
        .map(|value| value as f32)
        .filter(|value| value.is_finite() && *value > 0.0)
        .ok_or_else(|| invalid("vision rope_theta must be positive"))?;
    if source
        .rope_parameters
        .get("rope_type")
        .and_then(Value::as_str)
        != Some("default")
    {
        return Err(invalid("vision supports only default two-axis RoPE"));
    }
    Ok(VisionConfig {
        hidden_size: source.hidden_size,
        intermediate_size: source.intermediate_size,
        num_heads: source.num_attention_heads,
        patch_size: source.patch_size,
        temporal_patch_size: source.patch_temporal,
        merge_size: source.merge_size,
        position_height: source.pos_emb_height,
        position_width: source.pos_emb_width,
        layer_norm_eps: source.layer_norm_eps,
        rope_theta,
        schedule,
        quantized_weight_configs: HashMap::new(),
        weight_quantization: None,
        language_hidden_size,
        projector_hidden_size,
    })
}

fn official_gguf_vision(language_hidden_size: i32) -> VisionConfig {
    VisionConfig {
        hidden_size: 1_536,
        intermediate_size: 8_960,
        num_heads: 16,
        patch_size: 14,
        temporal_patch_size: 1,
        merge_size: 2,
        position_height: 32,
        position_width: 32,
        layer_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        schedule: (0..50)
            .map(|layer| {
                if layer == 49 || layer % 4 == 3 {
                    VisionAttentionPolicy::Full
                } else {
                    VisionAttentionPolicy::Window
                }
            })
            .collect(),
        quantized_weight_configs: HashMap::new(),
        weight_quantization: None,
        language_hidden_size,
        projector_hidden_size: 4_096,
    }
}

fn gguf_attention_schedule(
    metadata: &HashMap<String, MetadataValue>,
    layers: i32,
) -> Result<LayerSchedule<AttentionPolicy>, ConfigError> {
    let count = usize::try_from(layers)
        .ok()
        .filter(|count| *count > 0)
        .ok_or_else(|| invalid("Muse-Glimmer GGUF block count must be positive"))?;
    let pattern_key = "muse-glimmer.attention.sliding_window_pattern";
    let pattern = match metadata.get(pattern_key) {
        Some(MetadataValue::Array(MetadataArray::Bool(values))) if values.len() == count => {
            Some(values.clone())
        }
        Some(MetadataValue::Array(MetadataArray::Bool(values))) => {
            return Err(invalid(format!(
                "Muse-Glimmer GGUF sliding pattern has {} entries for {count} layers",
                values.len()
            )))
        }
        Some(_) => return Err(invalid("Muse-Glimmer GGUF sliding pattern must be boolean")),
        None => None,
    };
    let window = gguf_optional_i32(metadata, "muse-glimmer.attention.sliding_window")?;
    match (window, pattern) {
        (None, None) => LayerSchedule::all_full(count).map_err(|error| invalid(error.to_string())),
        (None, Some(pattern)) if pattern.iter().all(|local| !local) => {
            LayerSchedule::all_full(count).map_err(|error| invalid(error.to_string()))
        }
        (None, Some(_)) => Err(invalid(
            "Muse-Glimmer GGUF enables sliding layers without a sliding window",
        )),
        (Some(window), pattern) => {
            let window = u32::try_from(window)
                .ok()
                .filter(|window| *window > 0)
                .ok_or_else(|| invalid("Muse-Glimmer GGUF sliding window must be positive"))?;
            let pattern = pattern.unwrap_or_else(|| vec![true; count]);
            LayerSchedule::from_sliding_pattern(count, &pattern, Some(window))
                .map_err(|error| invalid(error.to_string()))
        }
    }
}

fn gguf_rope_scaling(
    metadata: &HashMap<String, MetadataValue>,
) -> Result<Option<HashMap<String, RopeValue>>, ConfigError> {
    let Some(kind) = gguf_optional_string(metadata, "muse-glimmer.rope.scaling.type")? else {
        return Ok(None);
    };
    match kind.as_str() {
        "none" | "default" => Ok(None),
        "linear" => Ok(Some(HashMap::from([
            ("rope_type".into(), RopeValue::String("linear".into())),
            (
                "factor".into(),
                RopeValue::Float(gguf_f32(metadata, "muse-glimmer.rope.scaling.factor")?),
            ),
        ]))),
        "yarn" => {
            for suffix in ["yarn_ext_factor", "yarn_attn_factor", "yarn_log_multiplier"] {
                if metadata.contains_key(&format!("muse-glimmer.rope.scaling.{suffix}")) {
                    return Err(invalid(format!(
                        "Muse-Glimmer GGUF YaRN field {suffix:?} changes unsupported attention semantics"
                    )));
                }
            }
            Ok(Some(HashMap::from([
                ("rope_type".into(), RopeValue::String("yarn".into())),
                (
                    "factor".into(),
                    RopeValue::Float(gguf_f32(metadata, "muse-glimmer.rope.scaling.factor")?),
                ),
                (
                    "original_max_position_embeddings".into(),
                    RopeValue::Float(gguf_f32(
                        metadata,
                        "muse-glimmer.rope.scaling.original_context_length",
                    )?),
                ),
                (
                    "beta_fast".into(),
                    RopeValue::Float(
                        gguf_optional_f32(metadata, "muse-glimmer.rope.scaling.yarn_beta_fast")?
                            .unwrap_or(32.0),
                    ),
                ),
                (
                    "beta_slow".into(),
                    RopeValue::Float(
                        gguf_optional_f32(metadata, "muse-glimmer.rope.scaling.yarn_beta_slow")?
                            .unwrap_or(1.0),
                    ),
                ),
                ("truncate".into(), RopeValue::Bool(false)),
            ])))
        }
        _ => Err(invalid(format!(
            "unsupported Muse-Glimmer GGUF RoPE scaling type {kind:?}"
        ))),
    }
}

fn gguf_vocab_size(
    metadata: &HashMap<String, MetadataValue>,
    fallback: &str,
) -> Result<i32, ConfigError> {
    gguf_i32(metadata, fallback)
}

fn gguf_string(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<String, ConfigError> {
    gguf_optional_string(metadata, key)?
        .ok_or_else(|| invalid(format!("Muse-Glimmer GGUF is missing {key:?}")))
}

fn gguf_optional_string(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<String>, ConfigError> {
    metadata
        .get(key)
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| invalid(format!("Muse-Glimmer GGUF {key:?} must be a string")))
        })
        .transpose()
}

fn gguf_i32(metadata: &HashMap<String, MetadataValue>, key: &str) -> Result<i32, ConfigError> {
    gguf_optional_i32(metadata, key)?
        .ok_or_else(|| invalid(format!("Muse-Glimmer GGUF is missing {key:?}")))
}

fn gguf_optional_i32(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<i32>, ConfigError> {
    metadata
        .get(key)
        .map(|value| {
            value
                .as_i64()
                .and_then(|value| i32::try_from(value).ok())
                .ok_or_else(|| invalid(format!("Muse-Glimmer GGUF {key:?} must be an i32")))
        })
        .transpose()
}

fn gguf_f32(metadata: &HashMap<String, MetadataValue>, key: &str) -> Result<f32, ConfigError> {
    gguf_optional_f32(metadata, key)?
        .ok_or_else(|| invalid(format!("Muse-Glimmer GGUF is missing {key:?}")))
}

fn gguf_optional_f32(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<f32>, ConfigError> {
    metadata
        .get(key)
        .map(|value| {
            value
                .as_f32()
                .ok_or_else(|| invalid(format!("Muse-Glimmer GGUF {key:?} must be numeric")))
        })
        .transpose()
}

fn gguf_optional_bool(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<bool>, ConfigError> {
    metadata
        .get(key)
        .map(|value| {
            value
                .as_bool()
                .ok_or_else(|| invalid(format!("Muse-Glimmer GGUF {key:?} must be boolean")))
        })
        .transpose()
}

fn reject_execution_overrides(value: &Value) -> Result<(), ConfigError> {
    for field in [
        "sliding_window_pattern",
        "attention_type",
        "use_qk_norm",
        "qk_norm",
        "attention_chunk_size",
        "value_head_dim",
        "attention_output_bias",
    ] {
        if value.get(field).is_some() {
            return Err(invalid(format!("unsupported execution field {field:?}")));
        }
    }
    if value
        .get("partial_rotary_factor")
        .is_some_and(|value| value.as_f64() != Some(1.0))
    {
        return Err(invalid("partial_rotary_factor must equal one"));
    }
    if value
        .get("rope_interleaved")
        .is_some_and(|value| value.as_bool() != Some(false))
    {
        return Err(invalid("rope_interleaved must be false"));
    }
    Ok(())
}

fn validate_rope_scaling(scaling: Option<&HashMap<String, RopeValue>>) -> Result<(), ConfigError> {
    let Some(scaling) = scaling else {
        return Ok(());
    };
    let kind = scaling
        .get("rope_type")
        .or_else(|| scaling.get("type"))
        .and_then(|value| match value {
            RopeValue::String(value) => Some(value.as_str()),
            _ => None,
        })
        .ok_or_else(|| invalid("rope_scaling requires a string type"))?;
    let allowed: &[&str] = match kind {
        "default" => &["type", "rope_type"],
        "linear" => &["type", "rope_type", "factor"],
        "yarn" => &[
            "type",
            "rope_type",
            "factor",
            "original_max_position_embeddings",
            "beta_fast",
            "beta_slow",
            "mscale",
            "mscale_all_dim",
            "truncate",
        ],
        _ => return Err(invalid(format!("unsupported RoPE scaling type {kind:?}"))),
    };
    if let Some(key) = scaling.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(invalid(format!(
            "RoPE scaling field {key:?} affects unsupported execution semantics"
        )));
    }
    let numeric = |key: &str| {
        scaling.get(key).and_then(|value| match value {
            RopeValue::Float(value) if value.is_finite() => Some(*value),
            RopeValue::String(value) => value.parse::<f32>().ok().filter(|value| value.is_finite()),
            _ => None,
        })
    };
    if matches!(kind, "linear" | "yarn") && numeric("factor").is_none_or(|factor| factor <= 0.0) {
        return Err(invalid("scaled RoPE requires a finite positive factor"));
    }
    if kind == "yarn"
        && numeric("original_max_position_embeddings").is_none_or(|value| value <= 0.0)
    {
        return Err(invalid(
            "YaRN requires positive original_max_position_embeddings",
        ));
    }
    crate::rotary::normalize_algorithm(Some(scaling))
        .map(|_| ())
        .map_err(invalid)
}

fn positive_i32(value: &Value, name: &str) -> Result<i32, ConfigError> {
    value
        .get(name)
        .and_then(Value::as_i64)
        .and_then(|value| i32::try_from(value).ok())
        .filter(|value| *value > 0)
        .ok_or_else(|| invalid(format!("{name} must be a positive integer")))
}

fn required_u32(value: &Value, name: &str) -> Result<u32, ConfigError> {
    value
        .get(name)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| invalid(format!("{name} must be an unsigned integer")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Value {
        serde_json::json!({
            "architectures":["MuseGlimmerForConditionalGeneration"],"model_type":"muse_glimmer",
            "image_token_id":22,"video_token_id":23,"out_hidden_size":32,"projector_hidden_size":16,
            "text_config":{"model_type":"muse_glimmer_text","hidden_size":16,"num_hidden_layers":2,
              "intermediate_size":24,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":4,
              "rms_norm_eps":0.00001,"post_norm_eps":0.00001,"vocab_size":24,"max_position_embeddings":64,
              "rope_theta":10000.0,"layer_types":["sliding_attention","full_attention"],
              "layer_rope_theta":[10000.0,0.0],"sliding_window":8,"tie_word_embeddings":false,
              "hidden_act":"silu","attention_dropout":0.0,"qk_scale_factor":1.0,
              "output_multiplier":1.0,"final_logit_softcapping":30.0},
            "vision_config":{"model_type":"muse_glimmer_vision","hidden_size":8,"intermediate_size":12,
              "num_attention_heads":2,"num_hidden_layers":1,"patch_size":2,"patch_temporal":1,
              "merge_size":2,"pos_emb_height":2,"pos_emb_width":2,"max_position_embeddings":4,
              "layer_norm_eps":0.00001,"hidden_act":"gelu","layer_types":["full_attention"],
              "rope_parameters":{"rope_theta":10000.0,"rope_type":"default"}}
        })
    }

    #[test]
    fn freezes_nope_full_and_rotary_sliding_schedule() {
        let config = DecoderConfig::from_hf_value(&config()).unwrap();
        assert!(config.attention_schedule.get(0).unwrap().window().is_some());
        assert_eq!(
            *config.attention_schedule.get(1).unwrap(),
            AttentionPolicy::Full
        );
        assert_eq!(config.layer_uses_rope, [true, false]);
        assert_eq!(
            config.vision_config.as_ref().unwrap().schedule,
            [VisionAttentionPolicy::Full]
        );
    }

    #[test]
    fn prompt_cache_fingerprint_includes_component_quantization() {
        let args = DecoderConfig::from_hf_value(&config()).unwrap();
        let dense = args.architecture_fingerprint();
        let format = eredu_checkpoint::AffineQuantization::new(16, 4)
            .unwrap()
            .into();

        let mut text_quantized = args.clone();
        text_quantized.quantization = Some(format);
        assert_ne!(dense, text_quantized.architecture_fingerprint());

        let mut vision_quantized = args;
        vision_quantized
            .vision_config
            .as_mut()
            .unwrap()
            .weight_quantization = Some(format);
        assert_ne!(dense, vision_quantized.architecture_fingerprint());
    }

    #[test]
    fn canonicalizes_quantization_alias_and_rejects_conflicts() {
        let mut value = config();
        value["text_config"]["quantization_config"] =
            serde_json::json!({"group_size": 32, "bits": 4});
        let args = DecoderConfig::from_hf_value(&value).unwrap();
        assert_eq!(
            args.quantization,
            Some(WeightQuantization::Affine(
                eredu_checkpoint::AffineQuantization::new(32, 4).unwrap()
            ))
        );

        value["text_config"]["quantization"] = serde_json::json!({"group_size": 64, "bits": 4});
        let error = DecoderConfig::from_hf_value(&value).unwrap_err();
        assert_eq!(
            error.to_string(),
            "Muse-Glimmer quantization and quantization_config disagree"
        );
    }

    #[test]
    fn family_owns_prompt_cache_identity() {
        let args = DecoderConfig::from_hf_value(&config()).unwrap();
        let layout = crate::muse_glimmer::state_layout(&args).unwrap();
        let identity = crate::muse_glimmer::state_identity(
            &args,
            &layout,
            0,
            eredu_core::cache::PromptCacheTopology::default(),
        )
        .unwrap()
        .prompt_cache_identity(&layout)
        .unwrap();

        assert_eq!(identity.model_family, "muse_glimmer");
        assert_eq!(
            identity.architecture_fingerprint,
            args.architecture_fingerprint()
        );
        assert_eq!(identity.layer_prefix_offsets, vec![0; layout.len()]);
    }

    #[test]
    fn rejects_attention_rope_mismatch() {
        let mut value = config();
        value["text_config"]["layer_rope_theta"][1] = Value::from(10000.0);
        assert!(DecoderConfig::from_hf_value(&value).is_err());
    }

    #[test]
    fn normalizes_softmax_top_k_moe_geometry() {
        let mut value = config();
        value["text_config"]["intermediate_size"] = Value::from(0);
        value["text_config"]["moe_intermediate_size"] = Value::from(12);
        value["text_config"]["num_experts"] = Value::from(8);
        value["text_config"]["num_experts_per_tok"] = Value::from(2);
        value["text_config"]["norm_topk_prob"] = Value::from(true);
        let config = DecoderConfig::from_hf_value(&value).unwrap();
        assert!(config.is_moe());
        assert_eq!(config.num_experts_per_tok, 2);

        value["text_config"]["num_experts_per_tok"] = Value::from(9);
        assert!(DecoderConfig::from_hf_value(&value).is_err());
    }

    #[test]
    fn rope_scaling_rejects_missing_geometry_and_semantic_extensions() {
        let mut value = config();
        value["text_config"]["rope_scaling"] =
            serde_json::json!({"rope_type":"linear","factor":0.0});
        assert!(DecoderConfig::from_hf_value(&value).is_err());
        value["text_config"]["rope_scaling"] =
            serde_json::json!({"rope_type":"linear","factor":2.0,"attention_factor":1.0});
        assert!(DecoderConfig::from_hf_value(&value).is_err());
    }

    #[test]
    fn portable_gguf_parser_freezes_text_schedule_and_projector_identity() {
        let mut metadata = HashMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String("muse-glimmer".into()),
            ),
            ("muse-glimmer.block_count".into(), MetadataValue::Uint32(2)),
            (
                "muse-glimmer.embedding_length".into(),
                MetadataValue::Uint32(32),
            ),
            (
                "muse-glimmer.attention.head_count".into(),
                MetadataValue::Uint32(4),
            ),
            (
                "muse-glimmer.attention.head_count_kv".into(),
                MetadataValue::Uint32(2),
            ),
            (
                "muse-glimmer.attention.key_length".into(),
                MetadataValue::Uint32(8),
            ),
            (
                "muse-glimmer.feed_forward_length".into(),
                MetadataValue::Uint32(64),
            ),
            (
                "muse-glimmer.vocab_size".into(),
                MetadataValue::Uint32(200_100),
            ),
            (
                "muse-glimmer.context_length".into(),
                MetadataValue::Uint32(128),
            ),
            (
                "muse-glimmer.attention.layer_norm_rms_epsilon".into(),
                MetadataValue::Float32(1e-5),
            ),
            (
                "muse-glimmer.attention.sliding_window".into(),
                MetadataValue::Uint32(16),
            ),
            (
                "muse-glimmer.logit_scale".into(),
                MetadataValue::Float32(0.25),
            ),
            (
                "muse-glimmer.final_logit_softcapping".into(),
                MetadataValue::Float32(20.0),
            ),
        ]);
        metadata.insert(
            "muse-glimmer.attention.sliding_window_pattern".into(),
            MetadataValue::Array(MetadataArray::Bool(vec![true, false])),
        );
        struct Catalog(bool);

        impl GgufTensorCatalog for Catalog {
            fn contains(&self, name: &str) -> bool {
                self.0 && name == "output.weight"
            }

            fn any(&self, mut predicate: impl FnMut(&str) -> bool) -> bool {
                self.0 && predicate("output.weight")
            }
        }

        let tied = DecoderConfig::from_gguf_catalog(&Catalog(false), &metadata).unwrap();
        assert!(tied.tie_word_embeddings);
        let config = DecoderConfig::from_gguf_catalog(&Catalog(true), &metadata).unwrap();
        assert_eq!(config.weight_convention, WeightConvention::Gguf);
        assert!(config.attention_schedule.get(0).unwrap().window().is_some());
        assert_eq!(
            *config.attention_schedule.get(1).unwrap(),
            AttentionPolicy::Full
        );
        assert_eq!(config.layer_uses_rope, [true, false]);
        assert!(!config.tie_word_embeddings);
        assert!(config.vision_config.is_none());

        let projector = HashMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String("clip".into()),
            ),
            (
                "general.type".into(),
                MetadataValue::String("mmproj".into()),
            ),
            (
                "clip.projector_type".into(),
                MetadataValue::String("muse-glimmer".into()),
            ),
            ("clip.has_vision_encoder".into(), MetadataValue::Bool(true)),
            (
                "clip.vision.embedding_length".into(),
                MetadataValue::Uint32(1_536),
            ),
            (
                "clip.vision.feed_forward_length".into(),
                MetadataValue::Uint32(8_960),
            ),
            ("clip.vision.block_count".into(), MetadataValue::Uint32(50)),
            (
                "clip.vision.attention.head_count".into(),
                MetadataValue::Uint32(16),
            ),
            ("clip.vision.patch_size".into(), MetadataValue::Uint32(14)),
            (
                "clip.vision.spatial_merge_size".into(),
                MetadataValue::Uint32(2),
            ),
            ("clip.vision.image_size".into(), MetadataValue::Uint32(896)),
            (
                "clip.vision.projection_dim".into(),
                MetadataValue::Uint32(32),
            ),
            (
                "clip.vision.attention.layer_norm_epsilon".into(),
                MetadataValue::Float32(1e-5),
            ),
        ]);
        let config = config
            .with_gguf_projector_metadata(&projector, HashMap::new())
            .unwrap();
        let vision = config.vision_config.as_ref().unwrap();
        assert_eq!(vision.layer_count(), 50);
        assert_eq!(vision.temporal_patch_size, 1);

        let mut invalid = projector;
        invalid.insert("clip.vision.image_size".into(), MetadataValue::Uint32(448));
        assert!(config
            .with_gguf_projector_metadata(&invalid, HashMap::new())
            .is_err());
    }
}
