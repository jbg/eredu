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

/// Distinct released identities sharing Muse construction and execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MuseIdentity {
    /// Muse Glimmer checkpoints.
    Glimmer,
    /// Muse Spark Open checkpoints.
    Spark,
}

impl MuseIdentity {
    /// Serialized outer-model identity.
    pub const fn model_type(self) -> &'static str {
        match self {
            Self::Glimmer => "muse_glimmer",
            Self::Spark => "muse_spark",
        }
    }
    /// Serialized decoder identity.
    pub const fn text_model_type(self) -> &'static str {
        match self {
            Self::Glimmer => "muse_glimmer_text",
            Self::Spark => "muse_spark_text",
        }
    }
    fn architecture(self) -> &'static str {
        match self {
            Self::Glimmer => "MuseGlimmerForConditionalGeneration",
            Self::Spark => "MuseSparkForConditionalGeneration",
        }
    }
    fn vision_model_type(self) -> &'static str {
        match self {
            Self::Glimmer => "muse_glimmer_vision",
            Self::Spark => "muse_spark_vision",
        }
    }
}

/// Weightless normalization after the media adapter's final projection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PerceptionNormalization {
    /// Established backend RMS arithmetic.
    Backend {
        /// Positive RMS epsilon.
        epsilon: f32,
    },
    /// Normalize in FP32, then round back to the activation dtype.
    Float32 {
        /// Positive RMS epsilon.
        epsilon: f32,
    },
}

/// Precision boundary at a host-media tower's first projection.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum MediaInputPrecision {
    /// Retain the caller's floating input representation.
    Preserve,
    /// Round host input to the bound projection's floating parameter precision.
    Parameter,
}

/// Precision of projected perception features entering the text decoder.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum MediaEmbeddingPrecision {
    /// Retain the adapter's output representation.
    Preserve,
    /// Round perception features to the bound vocabulary lookup's output type.
    Decoder,
}

/// Coupled precision boundaries in the vision patch and attention equations.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum VisionArithmetic {
    /// Established patch, rotary and fused-attention arithmetic.
    Native,
    /// Parameter-precision patches and positions, FP32 rotary, and explicitly
    /// rounded attention scores and probabilities.
    Rounded,
}

/// Traversal of the resampled learned vision position table.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum VisionPositionLayout {
    /// Resample in the declared height/width raster order.
    Raster,
    /// Resample a width/height grid, retaining its flattened row order.
    TransposedRaster,
}

impl VisionArithmetic {
    pub(crate) fn input_precision(self) -> MediaInputPrecision {
        match self {
            Self::Native => MediaInputPrecision::Preserve,
            Self::Rounded => MediaInputPrecision::Parameter,
        }
    }

    pub(crate) fn attention(self) -> eredu_nn::AttentionArithmetic {
        match self {
            Self::Native => eredu_nn::AttentionArithmetic::Fused,
            Self::Rounded => eredu_nn::AttentionArithmetic::InputScores,
        }
    }
}

impl MediaInputPrecision {
    pub(crate) fn prepare<T: eredu_nn::Tensor, L: eredu_nn::LinearOperator<T>>(
        self,
        input: &T,
        projection: &L,
        context: &T::Context,
    ) -> Result<T, eredu_nn::Error> {
        match self {
            Self::Preserve => Ok(input.clone()),
            Self::Parameter => {
                let element = projection.parameter_element_type().ok_or_else(|| {
                    eredu_nn::Error::backend(
                        "Muse media projection has no declared floating parameter precision",
                    )
                })?;
                if input.element_type() == Some(element) {
                    Ok(input.clone())
                } else {
                    input.cast_float(element, context)
                }
            }
        }
    }
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

/// Arithmetic around scheduled gated attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionEquation {
    /// Apply the query gain inside normalization; retain attended head magnitudes.
    QueryScaled,
    /// Round Q/K normalization before query scaling and normalize attended heads.
    NormalizedHeads,
    /// Normalize Q/K and attended heads, applying the query multiplier to the
    /// attention score scale after rotary application.
    NormalizedHeadsScoreScaled,
}

/// Block normalization order and arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockNormalization {
    /// Learned output-scaled pre- and post-branch norms.
    OutputScaled,
    /// FP32 centered pre-norms, weightless attention post-norm and input-scaled FF post-norm.
    InputScaledFeedForward,
}

/// Precision and stage boundaries of vocabulary normalization and output transforms.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum VocabularyArithmetic {
    /// Existing backend arithmetic and result representation.
    Backend,
    /// Normalize in FP32 and preserve input precision at every output-transform stage.
    StageRounded,
    /// Retain FP32 decoder ingress and separately scaled FP32 softcap stages.
    Float32,
}

/// Residual combination declared independently of source naming.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResidualEquation {
    /// Ordinary residual addition.
    Add,
    /// Learned FP32 combination with explicit temperature and variance floor.
    Learned {
        /// Positive divisor of gate logits.
        temperature: f32,
        /// Positive floor before the residual coefficient's square root.
        minimum_variance: f32,
    },
    /// Checkpoint stores the already resolved stream and branch coefficients.
    StoredCoefficients,
}

/// Routing and expert composition, independent of the checkpoint namespace.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RoutedFeedForwardEquation {
    /// Decoder-width experts with softmax selection.
    DirectSoftmax,
    /// Route from decoder width, execute normalized latent-width experts,
    /// then project the mixture back to decoder width.
    CompressedSigmoid {
        /// Width on both sides of each expert bank.
        expert_width: i32,
        /// Positive multiplier before sigmoid scoring.
        logit_scale: f32,
        /// Positive multiplier after selected-score normalization.
        coefficient_scale: f32,
    },
}

/// Windowed or full vision self-attention.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum VisionAttentionPolicy {
    /// Window-partitioned attention.
    Window,
    /// Full packed-sequence attention.
    Full,
}

/// Media accepted by the selected vision artifact and its processor contract.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum VisionInputPolicy {
    /// Released projector consumes still images only.
    Images,
    /// The source consumes still images and timestamped video groups.
    ImagesAndVideo,
}

/// Validated Muse vision tower geometry and source policy.
#[derive(Debug, Clone)]
pub struct VisionConfig {
    /// Accepted inputs of this component, independently of decoder encoding.
    pub input_policy: VisionInputPolicy,
    /// Patch, rotary and attention precision boundaries.
    pub arithmetic: VisionArithmetic,
    /// Source-specific learned-position resampling coordinates.
    pub position_layout: VisionPositionLayout,
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
    /// Exact perception normalization after language projection.
    pub perception_normalization: PerceptionNormalization,
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
    /// Precision boundary between media adapters and text embeddings.
    pub media_embedding_precision: MediaEmbeddingPrecision,
    /// Checkpoint identity, separate from shared equation policies.
    pub identity: MuseIdentity,
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
    /// Q/K and attended-head normalization arithmetic.
    pub attention_equation: AttentionEquation,
    /// Ordered block normalization equations.
    pub block_normalization: BlockNormalization,
    /// Residual arithmetic and learned-gate topology.
    pub residual_equation: ResidualEquation,
    /// Expert input/output geometry, routing and mixture equation.
    pub routed_feed_forward: RoutedFeedForwardEquation,
    /// Canonical uniform quantization.
    pub quantization: Option<WeightQuantization>,
    /// Exact names quantized under a mixed artifact.
    pub quantized_weights: Option<HashSet<String>>,
    /// Per-weight mixed encodings.
    pub quantized_weight_configs: Option<HashMap<String, WeightQuantization>>,
    /// Optional native vision tower supplied by the main artifact or a GGUF projector.
    pub vision_config: Option<VisionConfig>,
    /// Declared audio tower; source coverage is checked independently at admission.
    pub audio_config: Option<super::audio::AudioConfig>,
    /// Audio placeholder identity, including text artifacts requiring a companion.
    pub audio_token_id: Option<u32>,
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
    #[serde(default, alias = "num_local_experts")]
    num_experts: i32,
    #[serde(default)]
    moe_hidden_size: Option<i32>,
    #[serde(default)]
    residual_gate_temperature: Option<f32>,
    #[serde(default)]
    routed_scaling_factor: Option<f32>,
    #[serde(default)]
    num_shared_experts: i32,
    #[serde(default)]
    mlp_layer_types: Option<Vec<String>>,
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
    /// Rotary products follow the selected attention and physical source equations.
    pub const fn rotary_arithmetic(&self) -> eredu_nn::RotaryArithmetic {
        match (self.attention_equation, self.weight_convention) {
            (AttentionEquation::NormalizedHeads, WeightConvention::HuggingFace) => {
                eredu_nn::RotaryArithmetic::InputProducts
            }
            (AttentionEquation::NormalizedHeadsScoreScaled, _) => {
                eredu_nn::RotaryArithmetic::InputTableFloat32Products
            }
            _ => eredu_nn::RotaryArithmetic::Native,
        }
    }
    /// Score and probability rounding follow the released attention equation.
    pub const fn attention_arithmetic(&self) -> eredu_nn::AttentionArithmetic {
        match (self.attention_equation, self.weight_convention) {
            (AttentionEquation::NormalizedHeadsScoreScaled, _) => {
                eredu_nn::AttentionArithmetic::Float32
            }
            (AttentionEquation::NormalizedHeads, WeightConvention::HuggingFace) => {
                eredu_nn::AttentionArithmetic::InputScores
            }
            _ => eredu_nn::AttentionArithmetic::Fused,
        }
    }
    /// Vocabulary normalization implied by the normalized equation/source policies.
    pub const fn vocabulary_arithmetic(&self) -> VocabularyArithmetic {
        match (self.block_normalization, self.weight_convention) {
            (BlockNormalization::OutputScaled, _) => VocabularyArithmetic::Backend,
            (BlockNormalization::InputScaledFeedForward, WeightConvention::HuggingFace) => {
                VocabularyArithmetic::StageRounded
            }
            (BlockNormalization::InputScaledFeedForward, WeightConvention::Gguf) => {
                VocabularyArithmetic::Float32
            }
        }
    }
    /// Stable schedule, geometry, and modality identity for persisted state.
    pub fn architecture_fingerprint(&self) -> String {
        eredu_core::cache::derive_prompt_cache_architecture_fingerprint(
            self.identity.model_type(),
            [
                ("model_type", self.identity.text_model_type().into()),
                (
                    "attention_equation",
                    format!("{:?}", self.attention_equation),
                ),
                (
                    "block_normalization",
                    format!("{:?}", self.block_normalization),
                ),
                (
                    "vocabulary_arithmetic",
                    format!("{:?}", self.vocabulary_arithmetic()),
                ),
                (
                    "rotary_arithmetic",
                    format!("{:?}", self.rotary_arithmetic()),
                ),
                (
                    "attention_arithmetic",
                    format!("{:?}", self.attention_arithmetic()),
                ),
                ("residual_equation", format!("{:?}", self.residual_equation)),
                (
                    "media_embedding_precision",
                    format!("{:?}", self.media_embedding_precision),
                ),
                (
                    "routed_feed_forward",
                    format!("{:?}", self.routed_feed_forward),
                ),
                ("hidden", self.hidden_size.to_string()),
                (
                    "text_geometry",
                    format!(
                        "{}:{}:{}:{}:{}:{}:{}:{}:{}",
                        self.num_hidden_layers,
                        self.intermediate_size,
                        self.num_attention_heads,
                        self.num_key_value_heads,
                        self.head_dim,
                        self.max_position_embeddings,
                        self.num_experts,
                        self.num_experts_per_tok,
                        self.moe_intermediate_size
                    ),
                ),
                (
                    "text_arithmetic",
                    format!(
                        "{:?}:{}:{}:{}:{}:{}:{}",
                        self.weight_convention,
                        self.rms_norm_eps,
                        self.post_norm_eps,
                        self.qk_scale_factor,
                        self.output_multiplier,
                        self.final_logit_softcapping,
                        self.tie_word_embeddings
                    ),
                ),
                (
                    "rotary",
                    format!(
                        "{}:{:?}:{}",
                        self.rope_theta,
                        self.layer_uses_rope,
                        crate::cache_identity::debug_map(self.rope_scaling.as_ref())
                    ),
                ),
                (
                    "audio",
                    self.audio_config.as_ref().map_or_else(
                        || "none".into(),
                        |audio| {
                            format!(
                                "{:?}:{:?}:{:?}:{}:{}:{}:{}:{}:{:?}",
                                audio.hidden_sizes,
                                audio.depths,
                                audio.patch_sizes,
                                audio.conv_kernel_size,
                                audio.mlp_ratio,
                                audio.rms_norm_eps,
                                audio.out_hidden_size,
                                crate::cache_identity::debug_map(Some(&audio.linear_formats)),
                                audio.input_precision,
                            )
                        },
                    ),
                ),
                ("audio_token", format!("{:?}", self.audio_token_id)),
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
                                "{}:{}:{}:{}:{}:{}:{}:{}:{}:{:?}:{}:{}:{}:{:?}:{:?}:{:?}:{:?}",
                                vision.hidden_size,
                                vision.layer_count(),
                                vision.patch_size,
                                vision.temporal_patch_size,
                                vision.merge_size,
                                vision.position_height,
                                vision.position_width,
                                vision.layer_norm_eps,
                                vision.rope_theta,
                                vision.perception_normalization,
                                vision.intermediate_size,
                                vision.num_heads,
                                vision.projector_hidden_size,
                                vision.schedule,
                                vision.arithmetic,
                                vision.position_layout,
                                vision.input_policy,
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
        let identity = match value.get("model_type").and_then(Value::as_str) {
            Some("muse_glimmer") => MuseIdentity::Glimmer,
            Some("muse_spark") => MuseIdentity::Spark,
            _ => return Err(invalid("expected a Muse Glimmer or Spark model identity")),
        };
        if let Some(architectures) = value.get("architectures") {
            let architectures = architectures
                .as_array()
                .ok_or_else(|| invalid("architectures must be an array"))?;
            if architectures.len() != 1
                || architectures[0].as_str() != Some(identity.architecture())
            {
                return Err(invalid(
                    "Muse architecture declaration differs from model identity",
                ));
            }
        }
        let text_value = value
            .get("text_config")
            .ok_or_else(|| invalid("Muse-Glimmer config is missing text_config"))?;
        reject_execution_overrides(text_value)?;
        let mut source: TextSource = serde_json::from_value(text_value.clone())?;
        if source.model_type != identity.text_model_type() {
            return Err(invalid(
                "Muse decoder identity differs from outer model identity",
            ));
        }
        let (attention_equation, block_normalization, residual_equation, routed_feed_forward) =
            match identity {
                MuseIdentity::Glimmer => (
                    AttentionEquation::QueryScaled,
                    BlockNormalization::OutputScaled,
                    ResidualEquation::Add,
                    RoutedFeedForwardEquation::DirectSoftmax,
                ),
                MuseIdentity::Spark => {
                    let kinds = source
                        .mlp_layer_types
                        .as_ref()
                        .ok_or_else(|| invalid("Spark requires an explicit MLP layer schedule"))?;
                    if kinds.len() != source.num_hidden_layers as usize
                        || kinds.iter().any(|kind| kind != "sparse")
                        || source.num_shared_experts != 0
                        || source.num_experts <= 0
                    {
                        return Err(invalid("Spark checkpoint requires its declared sparse-expert schedule without shared experts"));
                    }
                    (
                        AttentionEquation::NormalizedHeads,
                        BlockNormalization::InputScaledFeedForward,
                        ResidualEquation::Learned {
                            temperature: source.residual_gate_temperature.ok_or_else(|| {
                                invalid("Spark residual gate temperature is missing")
                            })?,
                            minimum_variance: 1e-3,
                        },
                        RoutedFeedForwardEquation::CompressedSigmoid {
                            expert_width: source.moe_hidden_size.ok_or_else(|| {
                                invalid("Spark compressed expert width is missing")
                            })?,
                            logit_scale: source.output_multiplier,
                            coefficient_scale: source
                                .routed_scaling_factor
                                .ok_or_else(|| invalid("Spark routed scaling factor is missing"))?,
                        },
                    )
                }
            };
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
        let vision = match value.get("vision_config").filter(|value| !value.is_null()) {
            Some(vision) => Some(vision_config(
                vision,
                source.hidden_size,
                projector_hidden_size,
                identity,
                match identity {
                    MuseIdentity::Glimmer => PerceptionNormalization::Backend { epsilon: 1e-5 },
                    MuseIdentity::Spark => PerceptionNormalization::Float32 {
                        epsilon: source.rms_norm_eps,
                    },
                },
            )?),
            None if identity == MuseIdentity::Glimmer => {
                return Err(invalid("Muse Glimmer config is missing vision_config"))
            }
            None => None,
        };
        let audio_config = match (
            identity,
            value.get("audio_config").filter(|value| !value.is_null()),
        ) {
            (MuseIdentity::Spark, Some(audio)) => Some(
                super::audio::AudioConfig::from_hf_value(audio, source.hidden_size)
                    .map_err(|error| invalid(error.to_string()))?,
            ),
            _ => None,
        };
        let audio_token_id = match identity {
            MuseIdentity::Spark => Some(required_u32(value, "audio_token_id")?),
            MuseIdentity::Glimmer => None,
        };
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
            identity,
            media_embedding_precision: match identity {
                MuseIdentity::Glimmer => MediaEmbeddingPrecision::Preserve,
                MuseIdentity::Spark => MediaEmbeddingPrecision::Decoder,
            },
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
            norm_topk_prob: source.norm_topk_prob || identity == MuseIdentity::Spark,
            attention_schedule: schedule,
            qk_scale_factor: source.qk_scale_factor,
            output_multiplier: source.output_multiplier,
            final_logit_softcapping: source.final_logit_softcapping,
            weight_convention: WeightConvention::HuggingFace,
            attention_equation,
            block_normalization,
            residual_equation,
            routed_feed_forward,
            quantization,
            quantized_weights: None,
            quantized_weight_configs: None,
            image_token_id: required_u32(value, "image_token_id")?,
            video_token_id: required_u32(value, "video_token_id")?,
            vision_out_hidden_size: positive_i32(value, "out_hidden_size")?,
            projector_hidden_size,
            vision_config: vision,
            audio_config,
            audio_token_id,
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
        let identity = match architecture.as_str() {
            "muse-glimmer" => MuseIdentity::Glimmer,
            "muse-spark" => MuseIdentity::Spark,
            _ => {
                return Err(invalid(format!(
                    "unsupported Muse GGUF architecture {architecture:?}"
                )))
            }
        };
        let key = |suffix: &str| format!("{architecture}.{suffix}");
        let layers = gguf_i32(metadata, &key("block_count"))?;
        let hidden_size = gguf_i32(metadata, &key("embedding_length"))?;
        let heads = gguf_i32(metadata, &key("attention.head_count"))?;
        if layers <= 0 || hidden_size <= 0 || heads <= 0 {
            return Err(invalid(
                "Muse GGUF block, embedding, and head counts must be positive",
            ));
        }
        let key_value_heads =
            gguf_optional_i32(metadata, &key("attention.head_count_kv"))?.unwrap_or(heads);
        let head_dim = gguf_optional_i32(metadata, &key("attention.key_length"))?
            .unwrap_or(hidden_size / heads);
        if let Some(value_dim) = gguf_optional_i32(metadata, &key("attention.value_length"))? {
            if value_dim != head_dim {
                return Err(invalid(format!(
                    "Muse GGUF value width {value_dim} does not match head width {head_dim}"
                )));
            }
        }
        if let Some(rotary_dim) = gguf_optional_i32(metadata, &key("rope.dimension_count"))? {
            if rotary_dim != head_dim {
                return Err(invalid(format!(
                    "Muse GGUF rotary width {rotary_dim} does not match head width {head_dim}"
                )));
            }
        }
        if gguf_optional_bool(metadata, &key("attention.causal"))? == Some(false) {
            return Err(invalid("Muse GGUF attention must be causal"));
        }
        if let Some(activation) = gguf_optional_string(metadata, &key("hidden_activation"))? {
            if activation != "silu" {
                return Err(invalid(format!(
                    "Muse GGUF activation must be silu, got {activation:?}"
                )));
            }
        }
        let schedule = gguf_attention_schedule(metadata, layers, &architecture)?;
        let layer_uses_rope = schedule
            .iter()
            .map(|policy| policy.window().is_some())
            .collect();
        let experts = gguf_optional_i32(metadata, &key("expert_count"))?.unwrap_or(0);
        let is_moe = experts > 0;
        let (
            post_norm_eps,
            attention_equation,
            block_normalization,
            residual_equation,
            routed_feed_forward,
        ) = match identity {
            MuseIdentity::Glimmer => (
                gguf_optional_f32(metadata, &key("attention.post_norm_rms_epsilon"))?
                    .unwrap_or(1e-8),
                AttentionEquation::QueryScaled,
                BlockNormalization::OutputScaled,
                ResidualEquation::Add,
                RoutedFeedForwardEquation::DirectSoftmax,
            ),
            MuseIdentity::Spark => {
                if !metadata.contains_key(&key("attention.sliding_window_pattern")) {
                    return Err(invalid(
                        "Spark GGUF requires its explicit full/sliding attention schedule",
                    ));
                }
                if gguf_i32(metadata, &key("expert_gating_func"))? != 2
                    || gguf_optional_bool(metadata, &key("expert_weights_norm"))? != Some(true)
                {
                    return Err(invalid(
                        "Spark GGUF requires normalized sigmoid expert routing",
                    ));
                }
                (
                    gguf_f32(metadata, &key("post_norm_epsilon"))?,
                    AttentionEquation::NormalizedHeads,
                    BlockNormalization::InputScaledFeedForward,
                    ResidualEquation::StoredCoefficients,
                    RoutedFeedForwardEquation::CompressedSigmoid {
                        expert_width: gguf_i32(metadata, &key("moe_latent_size"))?,
                        // The released converter folds this multiplier into the router weight.
                        logit_scale: 1.0,
                        coefficient_scale: gguf_f32(metadata, &key("expert_weights_scale"))?,
                    },
                )
            }
        };
        let config = Self {
            identity,
            media_embedding_precision: match identity {
                MuseIdentity::Glimmer => MediaEmbeddingPrecision::Preserve,
                MuseIdentity::Spark => MediaEmbeddingPrecision::Decoder,
            },
            hidden_size,
            num_hidden_layers: layers,
            intermediate_size: if is_moe {
                gguf_optional_i32(metadata, &key("feed_forward_length"))?.unwrap_or(0)
            } else {
                gguf_i32(metadata, &key("feed_forward_length"))?
            },
            num_attention_heads: heads,
            rms_norm_eps: gguf_f32(metadata, &key("attention.layer_norm_rms_epsilon"))?,
            post_norm_eps,
            vocab_size: gguf_vocab_size(metadata, &key("vocab_size"))?,
            num_key_value_heads: key_value_heads,
            max_position_embeddings: gguf_i32(metadata, &key("context_length"))?,
            rope_theta: match identity {
                MuseIdentity::Glimmer => {
                    gguf_optional_f32(metadata, &key("rope.freq_base"))?.unwrap_or(1_000_000.0)
                }
                MuseIdentity::Spark => gguf_f32(metadata, &key("rope.freq_base_swa"))?,
            },
            layer_uses_rope,
            head_dim,
            tie_word_embeddings: !catalog.contains("output.weight"),
            rope_scaling: gguf_rope_scaling(metadata, &architecture)?,
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
            qk_scale_factor: match identity {
                MuseIdentity::Glimmer => 3.87,
                MuseIdentity::Spark => 1.0,
            },
            output_multiplier: gguf_f32(metadata, &key("logit_scale"))?,
            final_logit_softcapping: gguf_f32(metadata, &key("final_logit_softcapping"))?,
            weight_convention: WeightConvention::Gguf,
            attention_equation,
            block_normalization,
            residual_equation,
            routed_feed_forward,
            quantization: None,
            quantized_weights: None,
            quantized_weight_configs: None,
            vision_config: None,
            audio_config: None,
            audio_token_id: match identity {
                MuseIdentity::Glimmer => None,
                MuseIdentity::Spark => Some(gguf_token_id(metadata, "<|audio|>")?),
            },
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
        if identity == MuseIdentity::Spark {
            for (actual, token) in [
                (config.image_token_id, "<|patch|>"),
                (config.video_token_id, "<|video|>"),
            ] {
                if actual != gguf_token_id(metadata, token)? {
                    return Err(invalid(
                        "Spark GGUF media token declarations disagree with the tokenizer",
                    ));
                }
            }
        }
        config.validate()?;
        Ok(config)
    }

    /// Validates and applies the selected GGUF vision and waveform companion geometry.
    pub fn with_gguf_projector_metadata(
        mut self,
        metadata: &HashMap<String, MetadataValue>,
        quantized_weight_configs: HashMap<String, WeightQuantization>,
    ) -> Result<Self, ConfigError> {
        for (key, expected) in [("general.architecture", "clip"), ("general.type", "mmproj")] {
            let value = gguf_string(metadata, key)?;
            if value != expected {
                return Err(invalid(format!(
                    "Muse-Glimmer projector {key:?} must be {expected:?}, got {value:?}"
                )));
            }
        }
        let legacy = gguf_optional_string(metadata, "clip.projector_type")?;
        let vision_type = gguf_optional_string(metadata, "clip.vision.projector_type")?;
        if legacy
            .as_ref()
            .zip(vision_type.as_ref())
            .is_some_and(|(first, second)| first != second)
            || legacy.as_deref().or(vision_type.as_deref()) != Some("muse-glimmer")
        {
            return Err(invalid(
                "Muse projector vision type is absent or inconsistent",
            ));
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
        if self.identity == MuseIdentity::Spark {
            vision.perception_normalization = PerceptionNormalization::Float32 {
                epsilon: self.rms_norm_eps,
            };
        }
        let mut audio = if gguf_optional_bool(metadata, "clip.has_audio_encoder")? == Some(true) {
            if self.identity != MuseIdentity::Spark {
                return Err(invalid(
                    "selected Muse identity has no released waveform tower",
                ));
            }
            Some(
                super::AudioConfig::from_gguf_metadata(metadata, self.hidden_size)
                    .map_err(|error| invalid(error.to_string()))?,
            )
        } else {
            None
        };
        for (name, format) in quantized_weight_configs {
            if name.starts_with("model.audio_tower.") {
                audio
                    .as_mut()
                    .ok_or_else(|| invalid("audio format declared without a waveform tower"))?
                    .linear_formats
                    .insert(name, format.into());
            } else if name.starts_with("model.vision_") {
                vision.quantized_weight_configs.insert(name, format);
            } else {
                return Err(invalid(
                    "projector format refers outside the media components",
                ));
            }
        }
        self.vision_config = Some(vision);
        self.audio_config = audio;
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
        if let RoutedFeedForwardEquation::CompressedSigmoid {
            expert_width,
            logit_scale,
            coefficient_scale,
        } = self.routed_feed_forward
        {
            if !self.is_moe()
                || expert_width <= 0
                || !logit_scale.is_finite()
                || logit_scale <= 0.0
                || !coefficient_scale.is_finite()
                || coefficient_scale <= 0.0
            {
                return Err(invalid("Muse compressed experts require positive routed geometry and finite positive scales"));
            }
        }
        if let ResidualEquation::Learned {
            temperature,
            minimum_variance,
        } = self.residual_equation
        {
            if !temperature.is_finite()
                || temperature <= 0.0
                || !minimum_variance.is_finite()
                || minimum_variance <= 0.0
            {
                return Err(invalid(
                    "Muse residual temperature and variance floor must be positive and finite",
                ));
            }
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
            if vision.language_hidden_size != self.hidden_size
                || vision.projector_hidden_size != self.projector_hidden_size
            {
                return Err(invalid(
                    "Muse vision adapter geometry differs from decoder declaration",
                ));
            }
            let epsilon = match vision.perception_normalization {
                PerceptionNormalization::Backend { epsilon }
                | PerceptionNormalization::Float32 { epsilon } => epsilon,
            };
            if !epsilon.is_finite() || epsilon <= 0.0 {
                return Err(invalid(
                    "Muse perception epsilon must be finite and positive",
                ));
            }
            if self.vision_out_hidden_size != vision.hidden_size * 4 {
                return Err(invalid(
                    "out_hidden_size must be four times the vision hidden width",
                ));
            }
        }
        if let Some(audio) = &self.audio_config {
            audio
                .validate()
                .map_err(|error| invalid(error.to_string()))?;
            if audio.out_hidden_size != self.hidden_size || self.audio_token_id.is_none() {
                return Err(invalid(
                    "Muse audio geometry or placeholder identity is incompatible with decoder",
                ));
            }
        }
        if self.audio_token_id.is_some_and(|token| {
            token >= self.vocab_size as u32
                || token == self.image_token_id
                || token == self.video_token_id
        }) {
            return Err(invalid("Muse audio placeholder identity is invalid"));
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
    identity: MuseIdentity,
    perception_normalization: PerceptionNormalization,
) -> Result<VisionConfig, ConfigError> {
    let source: VisionSource = serde_json::from_value(value.clone())?;
    if source.model_type != identity.vision_model_type()
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
        != Some(match identity {
            MuseIdentity::Glimmer => "default",
            MuseIdentity::Spark => "axial",
        })
    {
        return Err(invalid(
            "vision rotary declaration differs from its Muse identity",
        ));
    }
    Ok(VisionConfig {
        input_policy: VisionInputPolicy::ImagesAndVideo,
        arithmetic: match identity {
            MuseIdentity::Glimmer => VisionArithmetic::Native,
            MuseIdentity::Spark => VisionArithmetic::Rounded,
        },
        position_layout: match identity {
            MuseIdentity::Glimmer => VisionPositionLayout::Raster,
            MuseIdentity::Spark => VisionPositionLayout::TransposedRaster,
        },
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
        perception_normalization,
    })
}

fn official_gguf_vision(language_hidden_size: i32) -> VisionConfig {
    VisionConfig {
        input_policy: VisionInputPolicy::Images,
        arithmetic: VisionArithmetic::Native,
        position_layout: VisionPositionLayout::Raster,
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
        perception_normalization: PerceptionNormalization::Backend { epsilon: 1e-5 },
    }
}

fn gguf_attention_schedule(
    metadata: &HashMap<String, MetadataValue>,
    layers: i32,
    architecture: &str,
) -> Result<LayerSchedule<AttentionPolicy>, ConfigError> {
    let count = usize::try_from(layers)
        .ok()
        .filter(|count| *count > 0)
        .ok_or_else(|| invalid("Muse GGUF block count must be positive"))?;
    let pattern_key = format!("{architecture}.attention.sliding_window_pattern");
    let pattern = match metadata.get(&pattern_key) {
        Some(MetadataValue::Array(MetadataArray::Bool(values))) if values.len() == count => {
            Some(values.clone())
        }
        Some(MetadataValue::Array(MetadataArray::Bool(values))) => {
            return Err(invalid(format!(
                "Muse GGUF sliding pattern has {} entries for {count} layers",
                values.len()
            )))
        }
        Some(_) => return Err(invalid("Muse GGUF sliding pattern must be boolean")),
        None => None,
    };
    let window = gguf_optional_i32(
        metadata,
        &format!("{architecture}.attention.sliding_window"),
    )?;
    match (window, pattern) {
        (None, None) => LayerSchedule::all_full(count).map_err(|error| invalid(error.to_string())),
        (None, Some(pattern)) if pattern.iter().all(|local| !local) => {
            LayerSchedule::all_full(count).map_err(|error| invalid(error.to_string()))
        }
        (None, Some(_)) => Err(invalid(
            "Muse GGUF enables sliding layers without a sliding window",
        )),
        (Some(window), pattern) => {
            let window = u32::try_from(window)
                .ok()
                .filter(|window| *window > 0)
                .ok_or_else(|| invalid("Muse GGUF sliding window must be positive"))?;
            let pattern = pattern.unwrap_or_else(|| vec![true; count]);
            LayerSchedule::from_sliding_pattern(count, &pattern, Some(window))
                .map_err(|error| invalid(error.to_string()))
        }
    }
}

fn gguf_rope_scaling(
    metadata: &HashMap<String, MetadataValue>,
    architecture: &str,
) -> Result<Option<HashMap<String, RopeValue>>, ConfigError> {
    let key = |suffix: &str| format!("{architecture}.rope.scaling.{suffix}");
    let Some(kind) = gguf_optional_string(metadata, &key("type"))? else {
        return Ok(None);
    };
    match kind.as_str() {
        "none" | "default" => Ok(None),
        "linear" => Ok(Some(HashMap::from([
            ("rope_type".into(), RopeValue::String("linear".into())),
            (
                "factor".into(),
                RopeValue::Float(gguf_f32(metadata, &key("factor"))?),
            ),
        ]))),
        "yarn" => {
            for suffix in ["yarn_ext_factor", "yarn_attn_factor", "yarn_log_multiplier"] {
                if metadata.contains_key(&key(suffix)) {
                    return Err(invalid(format!(
                        "Muse GGUF YaRN field {suffix:?} changes unsupported attention semantics"
                    )));
                }
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
                    RopeValue::Float(
                        gguf_optional_f32(metadata, &key("yarn_beta_fast"))?.unwrap_or(32.0),
                    ),
                ),
                (
                    "beta_slow".into(),
                    RopeValue::Float(
                        gguf_optional_f32(metadata, &key("yarn_beta_slow"))?.unwrap_or(1.0),
                    ),
                ),
                ("truncate".into(), RopeValue::Bool(false)),
            ])))
        }
        _ => Err(invalid(format!(
            "unsupported Muse GGUF RoPE scaling type {kind:?}"
        ))),
    }
}

fn gguf_token_id(
    metadata: &HashMap<String, MetadataValue>,
    token: &str,
) -> Result<u32, ConfigError> {
    match metadata.get("tokenizer.ggml.tokens") {
        Some(MetadataValue::Array(MetadataArray::String(tokens))) => tokens
            .iter()
            .position(|value| value == token)
            .and_then(|index| u32::try_from(index).ok())
            .ok_or_else(|| invalid(format!("Muse GGUF tokenizer is missing {token:?}"))),
        _ => Err(invalid("Muse GGUF tokenizer token identities are absent")),
    }
}

fn gguf_vocab_size(
    metadata: &HashMap<String, MetadataValue>,
    fallback: &str,
) -> Result<i32, ConfigError> {
    if let Some(value) = gguf_optional_i32(metadata, fallback)? {
        return Ok(value);
    }
    match metadata.get("tokenizer.ggml.tokens") {
        Some(MetadataValue::Array(MetadataArray::String(tokens))) => i32::try_from(tokens.len())
            .ok()
            .filter(|count| *count > 0)
            .ok_or_else(|| invalid("Muse GGUF tokenizer vocabulary size is invalid")),
        _ => Err(invalid(format!(
            "Muse GGUF is missing {fallback:?} and tokenizer tokens"
        ))),
    }
}

fn gguf_string(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<String, ConfigError> {
    gguf_optional_string(metadata, key)?
        .ok_or_else(|| invalid(format!("Muse GGUF is missing {key:?}")))
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
                .ok_or_else(|| invalid(format!("Muse GGUF {key:?} must be a string")))
        })
        .transpose()
}

fn gguf_i32(metadata: &HashMap<String, MetadataValue>, key: &str) -> Result<i32, ConfigError> {
    gguf_optional_i32(metadata, key)?
        .ok_or_else(|| invalid(format!("Muse GGUF is missing {key:?}")))
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
                .ok_or_else(|| invalid(format!("Muse GGUF {key:?} must be an i32")))
        })
        .transpose()
}

fn gguf_f32(metadata: &HashMap<String, MetadataValue>, key: &str) -> Result<f32, ConfigError> {
    gguf_optional_f32(metadata, key)?
        .ok_or_else(|| invalid(format!("Muse GGUF is missing {key:?}")))
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
                .ok_or_else(|| invalid(format!("Muse GGUF {key:?} must be numeric")))
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
                .ok_or_else(|| invalid(format!("Muse GGUF {key:?} must be boolean")))
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
