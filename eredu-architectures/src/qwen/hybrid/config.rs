//! Strict hybrid text and conditional-generation configuration.

use std::collections::{BTreeMap, HashMap};

use crate::rotary::RopeValue;
use eredu_checkpoint::{BlockFp8Format, BlockFp8ScaleEncoding, LinearFormat, WeightQuantization};
use eredu_core::{
    attention::{AttentionPolicy, LayerSchedule},
    cache::{
        derive_prompt_cache_architecture_fingerprint, LayerCachePolicy, MutableStateResidency,
        StateTensorDimension, StateTensorDtype, StateTensorPolicy, StateTensorRole,
    },
};
use eredu_gguf::MetadataValue;
use eredu_runtime::{StateLayout, StateSegmentLifetime, StateSegmentSpec};
use serde::{Deserialize, Deserializer};
use serde_json::Value;

use crate::qwen::vision::{VisionConfig, VisionConfigSource};
use crate::GgufTensorCatalog;

/// Stable segment identity for target decoder state.
pub const TARGET_STATE_SEGMENT: &str = "target";
/// Stable segment identity for checkpoint-embedded prediction state.
pub const PREDICTION_STATE_SEGMENT: &str = "prediction";

/// Stateful operator policy for one hybrid decoder layer.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum HybridLayerPolicy {
    /// Recurrent gated-delta attention.
    LinearAttention,
    /// Ordinary full self-attention.
    SelfAttention(AttentionPolicy),
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum LayerPolicySource {
    LinearAttention,
    FullAttention,
}

impl LayerPolicySource {
    const fn normalize(self) -> HybridLayerPolicy {
        match self {
            Self::LinearAttention => HybridLayerPolicy::LinearAttention,
            Self::FullAttention => HybridLayerPolicy::SelfAttention(AttentionPolicy::Full),
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

/// Validated model variant over the one hybrid decoder implementation.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum HybridVariant {
    /// Dense Qwen3.5 text.
    Qwen35Dense,
    /// Routed/shared-expert Qwen3.5 text.
    Qwen35Moe,
    /// Qwen3-Next dense or routed text selected by expert geometry.
    Qwen3Next,
}

impl HybridVariant {
    /// Canonical architecture family published by the model registry.
    pub const fn model_kind(self) -> crate::ModelKind {
        match self {
            Self::Qwen35Dense | Self::Qwen35Moe => crate::ModelKind::Qwen35,
            Self::Qwen3Next => crate::ModelKind::Qwen3Next,
        }
    }

    /// Stable effective text model type.
    pub const fn model_type(self) -> &'static str {
        match self {
            Self::Qwen35Dense => "qwen3_5_text",
            Self::Qwen35Moe => "qwen3_5_moe_text",
            Self::Qwen3Next => "qwen3_next",
        }
    }

    fn from_qwen35_model_type(model_type: &str) -> Option<Self> {
        match model_type {
            "qwen3_5" | "qwen3_5_text" => Some(Self::Qwen35Dense),
            "qwen3_5_moe" | "qwen3_5_moe_text" => Some(Self::Qwen35Moe),
            _ => None,
        }
    }
}

/// Normalized Qwen3-Next/Qwen3.5 text configuration.
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// Validated variant policy.
    pub variant: HybridVariant,
    /// Effective text model type.
    pub model_type: String,
    /// Token vocabulary size.
    pub vocab_size: i32,
    /// Transformer hidden size.
    pub hidden_size: i32,
    /// Number of decoder layers.
    pub num_hidden_layers: i32,
    /// Actual configured embedded prediction depth.
    pub mtp_num_hidden_layers: i32,
    /// Full-attention query heads.
    pub num_attention_heads: i32,
    /// Full-attention key/value heads.
    pub num_key_value_heads: i32,
    /// Full-attention head width.
    pub head_dim: i32,
    /// Maximum configured sequence length.
    pub max_position_embeddings: i32,
    /// RMS normalization epsilon.
    pub rms_norm_eps: f32,
    /// Whether logits reuse the embedding table.
    pub tie_word_embeddings: bool,
    /// Whether full-attention projections carry bias.
    pub attention_bias: bool,
    /// Dense/expert activation.
    pub hidden_act: String,
    /// Causal convolution kernel width.
    pub linear_conv_kernel_dim: i32,
    /// Key head width for recurrent layers.
    pub linear_key_head_dim: i32,
    /// Value head width for recurrent layers.
    pub linear_value_head_dim: i32,
    /// Key heads for recurrent layers.
    pub linear_num_key_heads: i32,
    /// Value heads for recurrent layers.
    pub linear_num_value_heads: i32,
    /// Dense SwiGLU intermediate width.
    pub intermediate_size: i32,
    /// Routed-expert intermediate width.
    pub moe_intermediate_size: i32,
    /// Always-on shared-expert intermediate width.
    pub shared_expert_intermediate_size: i32,
    /// Experts selected per token.
    pub num_experts_per_tok: i32,
    /// Routed expert count.
    pub num_experts: i32,
    /// Whether selected routing probabilities are renormalized.
    pub norm_topk_prob: bool,
    /// Authoritative recurrent/full-attention layer order.
    pub layer_schedule: LayerSchedule<HybridLayerPolicy>,
    /// Raw family RoPE policy retained for stable identity.
    pub rope_parameters: Option<HashMap<String, Value>>,
    /// Raw alternative RoPE scaling policy.
    pub rope_scaling: Option<HashMap<String, Value>>,
    /// Native block-FP8 policy.
    pub fp8: Option<QwenFp8QuantizationConfig>,
    /// Default affine/MXFP4/GGUF weight policy.
    pub quantization: Option<WeightQuantization>,
    /// Exact canonical parameter format overrides.
    pub linear_formats: HashMap<String, LinearFormat>,
}

/// Rank-local mutable-state geometry for one scheduled hybrid layer.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum HybridStateGeometry {
    /// Ordinary append-only key/value state.
    FullAttention {
        /// Rank-local key/value heads.
        key_value_heads: i32,
    },
    /// Fixed convolution history plus FP32 recurrent matrix.
    LinearAttention {
        /// Rank-local key heads.
        key_heads: i32,
        /// Rank-local value heads.
        value_heads: i32,
    },
}

impl HybridConfig {
    /// Whether this policy selects routed plus shared experts.
    pub const fn is_moe(&self) -> bool {
        self.num_experts > 0
    }

    /// Complete checkpoint format for a canonical linear weight.
    pub fn linear_format(&self, weight: &str) -> LinearFormat {
        if let Some(format) = self.linear_formats.get(weight) {
            return *format;
        }
        if weight.ends_with(".mlp.gate.weight")
            || weight.ends_with(".mlp.shared_expert_gate.weight")
            || (self.variant == HybridVariant::Qwen3Next
                && (weight.ends_with(".linear_attn.in_proj_b.weight")
                    || weight.ends_with(".linear_attn.in_proj_a.weight")
                    || weight.ends_with(".linear_attn.in_proj_ba.weight")))
        {
            return LinearFormat::Dense;
        }
        if let Some(quantization) = self.quantization {
            return quantization.into();
        }
        if let Some(fp8) = &self.fp8 {
            if !fp8.excludes(weight) {
                return LinearFormat::E4M3BlockFp8(
                    BlockFp8Format::new(128, 128, BlockFp8ScaleEncoding::FloatingPoint)
                        .expect("fixed FP8 geometry is valid"),
                );
            }
        }
        LinearFormat::Dense
    }

    /// Base rotary frequency.
    pub fn rope_theta(&self) -> f32 {
        float_config_value(&self.rope_parameters, "rope_theta")
            .or_else(|| float_config_value(&self.rope_scaling, "rope_theta"))
            .unwrap_or(1_000_000.0)
    }

    /// Fraction of each full-attention head rotated by the default policy.
    pub fn partial_rotary_factor(&self) -> f32 {
        float_config_value(&self.rope_parameters, "partial_rotary_factor")
            .or_else(|| float_config_value(&self.rope_scaling, "partial_rotary_factor"))
            .unwrap_or(0.25)
    }

    /// Exact rotated width after proportional/partial policy.
    pub fn rope_dimensions(&self) -> i32 {
        let rope_type = string_config_value(&self.rope_parameters, "rope_type")
            .or_else(|| string_config_value(&self.rope_scaling, "rope_type"))
            .unwrap_or("default");
        if rope_type == "proportional" {
            self.head_dim
        } else {
            ((self.head_dim as f32 * self.partial_rotary_factor()).round() as i32)
                .max(2)
                .min(self.head_dim)
        }
    }

    /// Converts the selected external RoPE policy into architecture-owned scalar values.
    pub fn rope_config(&self) -> Option<HashMap<String, RopeValue>> {
        rope_config_value(
            self.rope_parameters
                .clone()
                .or_else(|| self.rope_scaling.clone()),
        )
    }

    /// Validates all hybrid geometry and selected physical formats.
    pub fn validate(&self) -> Result<(), HybridConfigError> {
        let rope_config = self.rope_config();
        crate::rotary::normalize_algorithm(rope_config.as_ref()).map_err(invalid)?;
        for (name, value) in [
            ("vocab_size", self.vocab_size),
            ("hidden_size", self.hidden_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("head_dim", self.head_dim),
            ("max_position_embeddings", self.max_position_embeddings),
            ("linear_conv_kernel_dim", self.linear_conv_kernel_dim),
            ("linear_key_head_dim", self.linear_key_head_dim),
            ("linear_value_head_dim", self.linear_value_head_dim),
            ("linear_num_key_heads", self.linear_num_key_heads),
            ("linear_num_value_heads", self.linear_num_value_heads),
        ] {
            if value <= 0 {
                return Err(invalid(format!(
                    "hybrid {name} must be positive, got {value}"
                )));
            }
        }
        if self.head_dim < 2 {
            return Err(invalid(format!(
                "hybrid head_dim must be at least 2, got {}",
                self.head_dim
            )));
        }
        if self.mtp_num_hidden_layers < 0 {
            return Err(invalid("mtp_num_hidden_layers must be non-negative"));
        }
        if self.layer_schedule.len() != self.num_hidden_layers as usize {
            return Err(invalid(format!(
                "hybrid layer schedule has {} entries for {} layers",
                self.layer_schedule.len(),
                self.num_hidden_layers
            )));
        }
        if self.layer_schedule.iter().any(|policy| {
            matches!(
                policy,
                HybridLayerPolicy::SelfAttention(AttentionPolicy::Sliding { .. })
            )
        }) {
            return Err(invalid(
                "hybrid decoder does not admit sliding self-attention",
            ));
        }
        if self.linear_num_value_heads % self.linear_num_key_heads != 0 {
            return Err(invalid(
                "linear value-head count must be divisible by key-head count",
            ));
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(invalid(
                "attention query-head count must be divisible by key/value heads",
            ));
        }
        self.num_attention_heads
            .checked_mul(self.head_dim)
            .and_then(|width| width.checked_mul(2))
            .ok_or_else(|| invalid("full-attention projection width overflowed"))?;
        let key_width = self
            .linear_num_key_heads
            .checked_mul(self.linear_key_head_dim)
            .ok_or_else(|| invalid("linear key width overflowed"))?;
        let value_width = self
            .linear_num_value_heads
            .checked_mul(self.linear_value_head_dim)
            .ok_or_else(|| invalid("linear value width overflowed"))?;
        key_width
            .checked_mul(2)
            .and_then(|width| width.checked_add(value_width))
            .ok_or_else(|| invalid("linear fused projection width overflowed"))?;
        if self.hidden_act != "silu" {
            return Err(invalid(format!(
                "unsupported hybrid activation {:?}",
                self.hidden_act
            )));
        }
        if self.is_moe() {
            if self.moe_intermediate_size <= 0
                || self.shared_expert_intermediate_size <= 0
                || self.num_experts_per_tok <= 0
                || self.num_experts_per_tok > self.num_experts
            {
                return Err(invalid(
                    "MoE requires positive routed/shared widths and valid top-k",
                ));
            }
        } else if self.intermediate_size <= 0 {
            return Err(invalid("dense hybrid intermediate_size must be positive"));
        }
        if let Some(fp8) = &self.fp8 {
            fp8.validate()?;
        }
        if let Some(quantization) = self.quantization {
            quantization
                .validate()
                .map_err(|error| invalid(error.to_string()))?;
        }
        for (name, format) in &self.linear_formats {
            if name.trim().is_empty() {
                return Err(invalid("linear-format identity must not be empty"));
            }
            format
                .validate()
                .map_err(|error| invalid(error.to_string()))?;
        }
        validate_rope_policy(self.rope_parameters.as_ref().or(self.rope_scaling.as_ref()))?;
        if self.variant == HybridVariant::Qwen3Next {
            fused_projection_widths(self)?;
        }
        Ok(())
    }
}

/// Native FP8 configuration supported by the shared hybrid decoder.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct QwenFp8QuantizationConfig {
    /// Quantization method; must be `fp8`.
    pub quant_method: String,
    /// FP8 scalar format; must be `e4m3`.
    pub fmt: String,
    /// Activation policy; must be `dynamic`.
    pub activation_scheme: String,
    /// Weight block geometry; must be `[128, 128]`.
    #[serde(default)]
    pub weight_block_size: Option<Vec<i32>>,
    /// Canonical module patterns intentionally retained in dense storage.
    #[serde(default)]
    pub modules_to_not_convert: Vec<String>,
}

impl QwenFp8QuantizationConfig {
    fn validate(&self) -> Result<(), HybridConfigError> {
        if self.quant_method != "fp8"
            || self.fmt != "e4m3"
            || self.activation_scheme != "dynamic"
            || self.weight_block_size.as_deref() != Some(&[128, 128])
        {
            return Err(invalid(format!("unsupported hybrid FP8 policy {self:?}")));
        }
        if self
            .modules_to_not_convert
            .iter()
            .any(|name| name.trim().is_empty())
        {
            return Err(invalid("FP8 exclusion names must not be empty"));
        }
        Ok(())
    }

    fn excludes(&self, weight: &str) -> bool {
        self.modules_to_not_convert
            .iter()
            .any(|module| weight == module || weight.starts_with(&format!("{module}.")))
    }
}

#[derive(Debug, Clone, Deserialize)]
struct HybridConfigSource {
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

impl HybridConfigSource {
    fn normalize(
        self,
        variant: HybridVariant,
        layer_schedule: LayerSchedule<HybridLayerPolicy>,
    ) -> HybridConfig {
        HybridConfig {
            variant,
            model_type: variant.model_type().into(),
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
            fp8: self.quantization_config,
            quantization: self.quantization,
            linear_formats: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct TopLevelConfig {
    model_type: String,
    #[serde(default)]
    text_config: Option<HybridConfigSource>,
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

/// Normalized hybrid text plus optional conditional-generation policy.
#[derive(Debug, Clone)]
pub struct ParsedHybridConfig {
    /// The shared text decoder policy.
    pub text: HybridConfig,
    /// Image placeholder token for conditional-generation configs.
    pub image_token_id: Option<i32>,
    /// Video placeholder token for conditional-generation configs.
    pub video_token_id: Option<i32>,
    /// Optional shared vision encoder policy.
    pub vision: Option<VisionConfig>,
}

/// Parses a shared projector GGUF with Qwen3.5 window-scheduled semantics.
pub fn vision_config_from_gguf_catalog(
    catalog: &impl crate::qwen::vision::VisionGgufCatalog,
    metadata: &HashMap<String, MetadataValue>,
) -> Result<VisionConfig, HybridConfigError> {
    crate::qwen::vision::config_from_gguf_catalog(
        catalog,
        metadata,
        crate::qwen::vision::VisionMode::WindowScheduled,
    )
    .map_err(|error| invalid(error.to_string()))
}

/// Strictly normalizes every admitted hybrid family model type.
pub fn model_args_from_config_value(
    value: &Value,
) -> Result<ParsedHybridConfig, HybridConfigError> {
    let mut top: TopLevelConfig = serde_json::from_value(value.clone())
        .map_err(|error| invalid(format!("invalid Qwen hybrid config: {error}")))?;
    let top_model_type = top.model_type.clone();
    let text_value = if matches!(top_model_type.as_str(), "qwen3_5" | "qwen3_5_moe") {
        value.get("text_config").unwrap_or(value)
    } else {
        value
    };
    let (source, variant) = match top_model_type.as_str() {
        "qwen3_5" | "qwen3_5_moe" => {
            let source = top.text_config.take().ok_or_else(|| {
                invalid(format!("{top_model_type} config is missing text_config"))
            })?;
            let variant =
                HybridVariant::from_qwen35_model_type(&source.model_type).ok_or_else(|| {
                    invalid(format!(
                        "{top_model_type} has unsupported nested text model type {:?}",
                        source.model_type
                    ))
                })?;
            (source, variant)
        }
        "qwen3_5_text" | "qwen3_5_moe_text" => {
            let variant = if top_model_type == "qwen3_5_text" {
                HybridVariant::Qwen35Dense
            } else {
                HybridVariant::Qwen35Moe
            };
            let source = serde_json::from_value(value.clone())
                .map_err(|error| invalid(format!("invalid {top_model_type} config: {error}")))?;
            (source, variant)
        }
        "qwen3_next" => {
            let source = serde_json::from_value(value.clone())
                .map_err(|error| invalid(format!("invalid qwen3_next config: {error}")))?;
            (source, HybridVariant::Qwen3Next)
        }
        other => return Err(HybridConfigError::UnsupportedModelType(other.into())),
    };
    let layer_count = usize::try_from(source.num_hidden_layers)
        .ok()
        .filter(|count| *count > 0)
        .ok_or_else(|| invalid("num_hidden_layers must be positive"))?;
    let matching_source_type = if variant == HybridVariant::Qwen3Next {
        source.model_type == variant.model_type()
    } else {
        HybridVariant::from_qwen35_model_type(&source.model_type) == Some(variant)
    };
    if !matching_source_type {
        return Err(invalid(format!(
            "{} requires nested text model type {:?}, got {:?}",
            top_model_type,
            variant.model_type(),
            source.model_type
        )));
    }
    let policies = if source.layer_types.is_empty() {
        let interval = match text_value.get("full_attention_interval") {
            Some(Value::Number(number)) => number
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .filter(|value| *value > 0)
                .ok_or_else(|| invalid("full_attention_interval must be a positive integer"))?,
            Some(_) => return Err(invalid("full_attention_interval must be an integer")),
            None => 4,
        };
        (0..layer_count)
            .map(|index| {
                if (index + 1).is_multiple_of(interval) {
                    HybridLayerPolicy::SelfAttention(AttentionPolicy::Full)
                } else {
                    HybridLayerPolicy::LinearAttention
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
    let schedule = LayerSchedule::new(layer_count, policies)
        .map_err(|error| invalid(format!("hybrid {error}")))?;
    let mut text = source.normalize(variant, schedule);
    if variant == HybridVariant::Qwen35Dense {
        text.num_experts = 0;
        text.num_experts_per_tok = 0;
    }
    text.fp8 = top.quantization_config.or(text.fp8);
    text.quantization = top.quantization.or(text.quantization);
    if let Some(tied) = top.tie_word_embeddings {
        text.tie_word_embeddings = tied;
    }
    if variant == HybridVariant::Qwen3Next && text.rope_parameters.is_none() {
        let mut parameters = HashMap::new();
        for key in ["rope_theta", "partial_rotary_factor"] {
            if let Some(value) = value.get(key).cloned() {
                parameters.insert(key.into(), value);
            }
        }
        if !parameters.is_empty() {
            text.rope_parameters = Some(parameters);
        }
    }
    text.validate()?;
    let vision = top
        .vision_config
        .map(VisionConfigSource::normalize_qwen3_5)
        .transpose()
        .map_err(|error| invalid(error.to_string()))?;
    if variant == HybridVariant::Qwen3Next
        && (top.image_token_id.is_some() || top.video_token_id.is_some() || vision.is_some())
    {
        return Err(invalid("qwen3_next is a text-only architecture"));
    }
    match (&vision, top.image_token_id, top.video_token_id) {
        (Some(_), Some(image), Some(video)) if image != video => {}
        (None, None, None) => {}
        (Some(_), _, _) => {
            return Err(invalid(
                "conditional Qwen3.5 requires distinct image_token_id and video_token_id",
            ))
        }
        (None, Some(_), _) | (None, _, Some(_))
            if !matches!(top_model_type.as_str(), "qwen3_5" | "qwen3_5_moe") =>
        {
            return Err(invalid(
                "media placeholder IDs require a conditional vision_config",
            ))
        }
        (None, _, _) => {}
    }
    Ok(ParsedHybridConfig {
        text,
        image_token_id: top.image_token_id,
        video_token_id: top.video_token_id,
        vision,
    })
}

/// Strictly normalizes a Qwen3-Next or Qwen3.5 GGUF catalog without opening
/// device tensors. GGUF counts next-token prediction blocks in `block_count`,
/// but the admitted llama.cpp artifact contract exposes those blocks as
/// opaque extras rather than the complete embedded-MTP module. The returned
/// schedule therefore excludes them and leaves embedded MTP disabled.
pub fn model_args_from_gguf_catalog(
    arrays: &impl GgufTensorCatalog,
    metadata: &HashMap<String, MetadataValue>,
) -> Result<ParsedHybridConfig, HybridConfigError> {
    let architecture = gguf_string(metadata, "general.architecture")?;
    let variant = match architecture.as_str() {
        "qwen35" => HybridVariant::Qwen35Dense,
        "qwen35moe" => HybridVariant::Qwen35Moe,
        "qwen3next" => HybridVariant::Qwen3Next,
        other => {
            return Err(invalid(format!(
                "unsupported Qwen hybrid GGUF architecture {other:?}"
            )))
        }
    };
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let block_count = gguf_i32(metadata, &key("block_count"))?;
    let gguf_nextn_layers: i32 = gguf_optional_i64(metadata, &key("nextn_predict_layers"))?
        .unwrap_or(0)
        .try_into()
        .map_err(|_| invalid("GGUF next-token prediction layer count exceeds i32"))?;
    if gguf_nextn_layers < 0 || gguf_nextn_layers >= block_count {
        return Err(invalid(format!(
            "GGUF has invalid block_count {block_count} and nextn_predict_layers {gguf_nextn_layers}"
        )));
    }
    let num_hidden_layers = block_count - gguf_nextn_layers;
    let hidden_size = gguf_i32(metadata, &key("embedding_length"))?;
    let num_attention_heads = gguf_i32(metadata, &key("attention.head_count"))?;
    let num_key_value_heads = gguf_i32(metadata, &key("attention.head_count_kv"))?;
    if num_attention_heads <= 0 {
        return Err(invalid("GGUF attention head count must be positive"));
    }
    let head_dim = gguf_optional_i64(metadata, &key("attention.key_length"))?
        .map(i32::try_from)
        .transpose()
        .map_err(|_| invalid("GGUF attention key length exceeds i32"))?
        .unwrap_or(hidden_size / num_attention_heads);
    let rope_dimensions = gguf_optional_i64(metadata, &key("rope.dimension_count"))?
        .map(i32::try_from)
        .transpose()
        .map_err(|_| invalid("GGUF rotary dimension exceeds i32"))?
        .unwrap_or(head_dim / 4);
    let interval: usize = gguf_optional_i64(metadata, &key("full_attention_interval"))?
        .unwrap_or(4)
        .try_into()
        .map_err(|_| invalid("GGUF full-attention interval must be positive"))?;
    if interval == 0 {
        return Err(invalid("GGUF full-attention interval must be positive"));
    }
    let layer_schedule = LayerSchedule::new(
        num_hidden_layers as usize,
        (0..num_hidden_layers as usize)
            .map(|index| {
                if (index + 1).is_multiple_of(interval) {
                    HybridLayerPolicy::SelfAttention(AttentionPolicy::Full)
                } else {
                    HybridLayerPolicy::LinearAttention
                }
            })
            .collect(),
    )
    .map_err(|error| invalid(format!("hybrid {error}")))?;
    let vocab_size = gguf_i32(metadata, &key("vocab_size"))?;
    let declared_experts = gguf_optional_i64(metadata, &key("expert_count"))?
        .map(i32::try_from)
        .transpose()
        .map_err(|_| invalid("GGUF expert count exceeds i32"))?;
    let is_moe = variant == HybridVariant::Qwen35Moe || declared_experts.is_some_and(|n| n > 0);
    let rope_theta = gguf_optional_f32(metadata, &key("rope.freq_base"))?.unwrap_or(10_000_000.0);
    let mut rope_parameters = HashMap::new();
    rope_parameters.insert("rope_theta".into(), serde_json::json!(rope_theta));
    rope_parameters.insert(
        "partial_rotary_factor".into(),
        serde_json::json!(rope_dimensions as f32 / head_dim as f32),
    );
    let attention_bias = (0..num_hidden_layers).any(|layer| {
        [
            "attn_q.bias",
            "attn_k.bias",
            "attn_v.bias",
            "attn_output.bias",
        ]
        .iter()
        .any(|suffix| arrays.contains(&format!("blk.{layer}.{suffix}")))
    });
    let text = HybridConfig {
        variant,
        model_type: variant.model_type().into(),
        vocab_size,
        hidden_size,
        num_hidden_layers,
        mtp_num_hidden_layers: 0,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        max_position_embeddings: gguf_i32(metadata, &key("context_length"))?,
        rms_norm_eps: gguf_f32(metadata, &key("attention.layer_norm_rms_epsilon"))?,
        tie_word_embeddings: !arrays.contains("output.weight"),
        attention_bias,
        hidden_act: "silu".into(),
        linear_conv_kernel_dim: gguf_i32(metadata, &key("ssm.conv_kernel"))?,
        linear_key_head_dim: gguf_i32(metadata, &key("ssm.state_size"))?,
        linear_value_head_dim: gguf_i32(metadata, &key("ssm.state_size"))?,
        linear_num_key_heads: gguf_i32(metadata, &key("ssm.group_count"))?,
        linear_num_value_heads: gguf_i32(metadata, &key("ssm.time_step_rank"))?,
        intermediate_size: if is_moe {
            0
        } else {
            gguf_i32(metadata, &key("feed_forward_length"))?
        },
        moe_intermediate_size: if is_moe {
            gguf_i32(metadata, &key("expert_feed_forward_length"))?
        } else {
            0
        },
        shared_expert_intermediate_size: if is_moe {
            gguf_i32(metadata, &key("expert_shared_feed_forward_length"))?
        } else {
            0
        },
        num_experts_per_tok: if is_moe {
            gguf_i32(metadata, &key("expert_used_count"))?
        } else {
            0
        },
        num_experts: if is_moe {
            declared_experts.ok_or_else(|| invalid("GGUF routed model is missing expert_count"))?
        } else {
            0
        },
        norm_topk_prob: true,
        layer_schedule,
        rope_parameters: Some(rope_parameters),
        rope_scaling: None,
        fp8: None,
        quantization: None,
        linear_formats: HashMap::new(),
    };
    text.validate()?;
    Ok(ParsedHybridConfig {
        text,
        image_token_id: None,
        video_token_id: None,
        vision: None,
    })
}

/// Attaches an independently validated shared-vision projector to a Qwen3.5
/// text GGUF policy without interpreting facade-owned tokenizer metadata.
pub fn with_gguf_vision_projector(
    mut parsed: ParsedHybridConfig,
    vision: VisionConfig,
) -> Result<ParsedHybridConfig, HybridConfigError> {
    if parsed.text.variant == HybridVariant::Qwen3Next {
        return Err(invalid("Qwen3-Next GGUF cannot attach a vision projector"));
    }
    vision
        .validate_for(crate::qwen::vision::VisionMode::WindowScheduled)
        .map_err(|error| invalid(error.to_string()))?;
    if vision.out_hidden_size != parsed.text.hidden_size {
        return Err(invalid(format!(
            "Qwen3.5 projector output {} does not match text hidden size {}",
            vision.out_hidden_size, parsed.text.hidden_size
        )));
    }
    parsed.vision = Some(vision);
    Ok(parsed)
}

/// Binds facade-resolved media placeholders to admitted Qwen3.5 GGUF geometry.
pub fn with_media_token_ids(
    mut parsed: ParsedHybridConfig,
    image_token_id: u32,
    video_token_id: u32,
) -> Result<ParsedHybridConfig, HybridConfigError> {
    if parsed.vision.is_none() {
        return Err(invalid(
            "Qwen3.5 media token IDs require admitted vision geometry",
        ));
    }
    let image =
        i32::try_from(image_token_id).map_err(|_| invalid("Qwen3.5 image token id exceeds i32"))?;
    let video =
        i32::try_from(video_token_id).map_err(|_| invalid("Qwen3.5 video token id exceeds i32"))?;
    if image == video {
        return Err(invalid("Qwen3.5 image and video placeholders must differ"));
    }
    if image >= parsed.text.vocab_size || video >= parsed.text.vocab_size {
        return Err(invalid(format!(
            "Qwen3.5 media token ids {image} and {video} must fit structural vocabulary {}",
            parsed.text.vocab_size
        )));
    }
    parsed.image_token_id = Some(image);
    parsed.video_token_id = Some(video);
    Ok(parsed)
}

fn gguf_string(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<String, HybridConfigError> {
    match metadata.get(key) {
        Some(MetadataValue::String(value)) => Ok(value.clone()),
        Some(_) => Err(invalid(format!(
            "GGUF metadata key {key:?} has the wrong type"
        ))),
        None => Err(invalid(format!(
            "GGUF metadata is missing required key {key:?}"
        ))),
    }
}

fn gguf_i32(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<i32, HybridConfigError> {
    i32::try_from(
        gguf_optional_i64(metadata, key)?
            .ok_or_else(|| invalid(format!("GGUF metadata is missing required key {key:?}")))?,
    )
    .map_err(|_| invalid(format!("GGUF metadata value {key:?} exceeds i32")))
}

fn gguf_optional_i64(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<i64>, HybridConfigError> {
    match metadata.get(key) {
        Some(value) => value.as_i64().map(Some).ok_or_else(|| {
            invalid(format!(
                "GGUF metadata key {key:?} must be an integer scalar"
            ))
        }),
        None => Ok(None),
    }
}

fn gguf_f32(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<f32, HybridConfigError> {
    gguf_optional_f32(metadata, key)?
        .ok_or_else(|| invalid(format!("GGUF metadata is missing required key {key:?}")))
}

fn gguf_optional_f32(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<f32>, HybridConfigError> {
    match metadata.get(key) {
        Some(value) => value.as_f32().map(Some).ok_or_else(|| {
            invalid(format!(
                "GGUF metadata key {key:?} must be a numeric scalar"
            ))
        }),
        None => Ok(None),
    }
}

/// Grouped physical widths of Q, K, V, gate and beta/decay projections.
pub fn fused_projection_widths(
    config: &HybridConfig,
) -> Result<([i32; 4], i32), HybridConfigError> {
    if config.linear_num_key_heads <= 0
        || config.linear_num_value_heads <= 0
        || config.linear_value_head_dim <= 0
        || config.linear_num_value_heads % config.linear_num_key_heads != 0
    {
        return Err(invalid("invalid grouped fused projection dimensions"));
    }
    let value_dim = config
        .linear_num_value_heads
        .checked_mul(config.linear_value_head_dim)
        .ok_or_else(|| invalid("fused projection dimension overflow"))?;
    if value_dim % config.linear_num_key_heads != 0 {
        return Err(invalid("invalid grouped fused projection dimensions"));
    }
    let value_per_key = value_dim / config.linear_num_key_heads;
    Ok((
        [
            config.linear_key_head_dim,
            config.linear_key_head_dim,
            value_per_key,
            value_per_key,
        ],
        config.linear_num_value_heads / config.linear_num_key_heads,
    ))
}

/// Converts physical row widths to native FP8 scale-block widths.
pub fn fp8_block_row_widths(widths: &[i32]) -> Result<Vec<i32>, HybridConfigError> {
    widths
        .iter()
        .map(|width| {
            if *width <= 0 || *width % 128 != 0 {
                return Err(invalid(format!(
                    "FP8 fused projection width {width} is not divisible by 128"
                )));
            }
            Ok(*width / 128)
        })
        .collect()
}

/// Stable prompt-cache identity for global hybrid architecture semantics.
pub fn prompt_cache_architecture_fingerprint(config: &HybridConfig) -> String {
    let fp8 = config.fp8.as_ref().map_or_else(
        || "none".into(),
        |fp8| {
            let mut exclusions = fp8.modules_to_not_convert.clone();
            exclusions.sort_unstable();
            format!(
                "{}:{}:{}:{:?}:{}",
                fp8.quant_method,
                fp8.fmt,
                fp8.activation_scheme,
                fp8.weight_block_size,
                exclusions.join(";")
            )
        },
    );
    derive_prompt_cache_architecture_fingerprint(
        config.variant.model_kind().canonical_name(),
        [
            ("model_type", config.model_type.clone()),
            ("layers", config.num_hidden_layers.to_string()),
            ("mtp_layers", config.mtp_num_hidden_layers.to_string()),
            (
                "layer_types",
                config
                    .layer_schedule
                    .iter()
                    .map(|policy| format!("{policy:?}"))
                    .collect::<Vec<_>>()
                    .join(","),
            ),
            ("kv_heads", config.num_key_value_heads.to_string()),
            ("head_dim", config.head_dim.to_string()),
            ("linear_conv", config.linear_conv_kernel_dim.to_string()),
            ("linear_key_dim", config.linear_key_head_dim.to_string()),
            ("linear_value_dim", config.linear_value_head_dim.to_string()),
            ("linear_key_heads", config.linear_num_key_heads.to_string()),
            (
                "linear_value_heads",
                config.linear_num_value_heads.to_string(),
            ),
            ("max_positions", config.max_position_embeddings.to_string()),
            (
                "rope_parameters",
                canonical_config_map(&config.rope_parameters),
            ),
            ("rope_scaling", canonical_config_map(&config.rope_scaling)),
            ("fp8", fp8),
            ("quantization", format!("{:?}", config.quantization)),
            (
                "linear_formats",
                crate::cache_identity::debug_map(Some(&config.linear_formats)),
            ),
        ],
    )
}

/// Declares global mutable state for the exact ordered hybrid schedule.
pub fn state_layout(config: &HybridConfig) -> Result<StateLayout, HybridConfigError> {
    let geometry = config
        .layer_schedule
        .iter()
        .map(|policy| match policy {
            HybridLayerPolicy::SelfAttention(_) => HybridStateGeometry::FullAttention {
                key_value_heads: config.num_key_value_heads,
            },
            HybridLayerPolicy::LinearAttention => HybridStateGeometry::LinearAttention {
                key_heads: config.linear_num_key_heads,
                value_heads: config.linear_num_value_heads,
            },
        })
        .collect::<Vec<_>>();
    state_layout_with_geometry(config, &geometry)
}

/// Declares rank-local mutable state while retaining global schedule identity.
pub fn state_layout_with_geometry(
    config: &HybridConfig,
    geometry: &[HybridStateGeometry],
) -> Result<StateLayout, HybridConfigError> {
    let target_layers = config.layer_schedule.len();
    let mtp_layers = usize::try_from(config.mtp_num_hidden_layers)
        .map_err(|_| invalid("mtp_num_hidden_layers must be non-negative"))?;
    if geometry.len() != target_layers && geometry.len() != target_layers + mtp_layers {
        return Err(invalid(format!(
            "hybrid state geometry has {} layers, expected {} target or {} total units",
            geometry.len(),
            target_layers,
            target_layers + mtp_layers,
        )));
    }
    let history = config
        .linear_conv_kernel_dim
        .checked_sub(1)
        .ok_or_else(|| invalid("linear convolution history underflowed"))?;
    let fixed =
        |value| StateTensorDimension::fixed(value).map_err(|error| invalid(error.to_string()));
    let mut policies = config
        .layer_schedule
        .iter()
        .copied()
        .zip(geometry.iter().copied())
        .enumerate()
        .map(|(layer, (policy, geometry))| match (policy, geometry) {
            (
                HybridLayerPolicy::SelfAttention(attention),
                HybridStateGeometry::FullAttention { key_value_heads },
            ) => LayerCachePolicy::key_value(attention, key_value_heads, config.head_dim)
                .map_err(|error| invalid(error.to_string())),
            (
                HybridLayerPolicy::LinearAttention,
                HybridStateGeometry::LinearAttention {
                    key_heads,
                    value_heads,
                },
            ) => {
                let key_width = key_heads
                    .checked_mul(config.linear_key_head_dim)
                    .ok_or_else(|| invalid("rank-local linear key width overflowed"))?;
                let value_width = value_heads
                    .checked_mul(config.linear_value_head_dim)
                    .ok_or_else(|| invalid("rank-local linear value width overflowed"))?;
                let convolution_width = key_width
                    .checked_mul(2)
                    .and_then(|width| width.checked_add(value_width))
                    .ok_or_else(|| invalid("rank-local convolution width overflowed"))?;
                LayerCachePolicy::fixed_only(vec![
                    StateTensorPolicy::new(
                        StateTensorRole::Convolution { slot: 0 },
                        vec![
                            StateTensorDimension::Batch,
                            fixed(history)?,
                            fixed(convolution_width)?,
                        ],
                        StateTensorDtype::Floating,
                        MutableStateResidency::AlwaysDeviceMutable,
                    )
                    .map_err(|error| invalid(error.to_string()))?,
                    StateTensorPolicy::new(
                        StateTensorRole::Recurrent,
                        vec![
                            StateTensorDimension::Batch,
                            fixed(value_heads)?,
                            fixed(config.linear_key_head_dim)?,
                            fixed(config.linear_value_head_dim)?,
                        ],
                        StateTensorDtype::Float32,
                        MutableStateResidency::LayerScopedOffloadable,
                    )
                    .map_err(|error| invalid(error.to_string()))?,
                ])
                .map_err(|error| invalid(error.to_string()))
            }
            (policy, geometry) => Err(invalid(format!(
                "hybrid state geometry {geometry:?} does not match layer {layer} policy {policy:?}"
            ))),
        })
        .collect::<Result<Vec<_>, _>>()?;
    for depth in 0..mtp_layers {
        let key_value_heads = match geometry.get(target_layers + depth).copied() {
            Some(HybridStateGeometry::FullAttention { key_value_heads }) => key_value_heads,
            Some(other) => {
                return Err(invalid(format!(
                    "hybrid MTP geometry {other:?} is not full attention"
                )))
            }
            None => config.num_key_value_heads,
        };
        policies.push(
            LayerCachePolicy::key_value(AttentionPolicy::Full, key_value_heads, config.head_dim)
                .map_err(|error| invalid(error.to_string()))?,
        );
    }
    let schedule = LayerSchedule::new(config.layer_schedule.len() + mtp_layers, policies)
        .map_err(|error| invalid(error.to_string()))?;
    let mut segments = vec![StateSegmentSpec::new(
        TARGET_STATE_SEGMENT,
        0..target_layers,
        StateSegmentLifetime::Persistent,
        0,
    )
    .map_err(|error| invalid(error.to_string()))?];
    if mtp_layers > 0 {
        segments.push(
            StateSegmentSpec::new(
                PREDICTION_STATE_SEGMENT,
                target_layers..target_layers + mtp_layers,
                StateSegmentLifetime::Persistent,
                -1,
            )
            .map_err(|error| invalid(error.to_string()))?,
        );
    }
    StateLayout::segmented(schedule, segments).map_err(|error| invalid(error.to_string()))
}

fn validate_rope_policy(config: Option<&HashMap<String, Value>>) -> Result<(), HybridConfigError> {
    let Some(config) = config else {
        return Ok(());
    };
    let Some(value) = config.get("type").or_else(|| config.get("rope_type")) else {
        return Ok(());
    };
    let Value::String(kind) = value else {
        return Err(invalid("RoPE type must be a string"));
    };
    if !matches!(
        kind.as_str(),
        "default" | "linear" | "proportional" | "yarn" | "llama3"
    ) {
        return Err(invalid(format!("unsupported RoPE type {kind:?}")));
    }
    Ok(())
}

fn deserialize_optional_fp8<'de, D>(
    deserializer: D,
) -> Result<Option<QwenFp8QuantizationConfig>, D::Error>
where
    D: Deserializer<'de>,
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

fn float_config_value(config: &Option<HashMap<String, Value>>, key: &str) -> Option<f32> {
    config.as_ref()?.get(key).and_then(|value| match value {
        Value::Number(value) => value.as_f64().map(|value| value as f32),
        Value::String(value) => value.parse().ok(),
        _ => None,
    })
}

fn string_config_value<'a>(
    config: &'a Option<HashMap<String, Value>>,
    key: &str,
) -> Option<&'a str> {
    config.as_ref()?.get(key).and_then(Value::as_str)
}

fn rope_config_value(config: Option<HashMap<String, Value>>) -> Option<HashMap<String, RopeValue>> {
    config.map(|config| {
        config
            .into_iter()
            .filter_map(|(key, value)| {
                let value = match value {
                    Value::Number(value) => {
                        value.as_f64().map(|value| RopeValue::Float(value as f32))
                    }
                    Value::String(value) => Some(RopeValue::String(value)),
                    Value::Bool(value) => Some(RopeValue::Bool(value)),
                    _ => None,
                }?;
                Some((key, value))
            })
            .collect()
    })
}

fn canonical_config_map(config: &Option<HashMap<String, Value>>) -> String {
    config.as_ref().map_or_else(String::new, |config| {
        config
            .iter()
            .map(|(key, value)| (key.clone(), value.to_string()))
            .collect::<BTreeMap<_, _>>()
            .into_iter()
            .map(|(key, value)| format!("{key}={value}"))
            .collect::<Vec<_>>()
            .join(";")
    })
}

const fn default_true() -> bool {
    true
}
fn default_text_model_type() -> String {
    "qwen3_5_moe_text".into()
}
fn default_hidden_act() -> String {
    "silu".into()
}
const fn default_head_dim() -> i32 {
    256
}
const fn default_rms_norm_eps() -> f32 {
    1e-6
}
const fn default_linear_conv_kernel_dim() -> i32 {
    4
}
const fn default_linear_key_head_dim() -> i32 {
    128
}
const fn default_linear_value_head_dim() -> i32 {
    128
}
const fn default_linear_num_key_heads() -> i32 {
    16
}
const fn default_linear_num_value_heads() -> i32 {
    32
}
const fn default_moe_intermediate_size() -> i32 {
    512
}
const fn default_shared_expert_intermediate_size() -> i32 {
    512
}
const fn default_num_experts_per_tok() -> i32 {
    8
}
const fn default_num_experts() -> i32 {
    256
}

fn invalid(message: impl Into<String>) -> HybridConfigError {
    HybridConfigError::Invalid(message.into())
}

/// Strict hybrid configuration error.
#[derive(Debug, Clone, thiserror::Error, Eq, PartialEq)]
pub enum HybridConfigError {
    /// The top-level model type is outside the admitted cohort.
    #[error("unsupported hybrid model type {0:?}")]
    UnsupportedModelType(String),
    /// A known model type violates its contract.
    #[error("{0}")]
    Invalid(String),
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use eredu_gguf::MetadataValue;
    use serde_json::json;

    use super::*;

    fn text_config(model_type: &str) -> Value {
        json!({
            "model_type": model_type,
            "vocab_size": 64,
            "hidden_size": 32,
            "num_hidden_layers": 4,
            "mtp_num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 128,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "intermediate_size": 48,
            "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 24,
            "num_experts_per_tok": 2,
            "num_experts": 8,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
        })
    }

    #[test]
    fn normalizes_next_dense_moe_and_conditional_forms_to_one_text_policy() {
        let next = model_args_from_config_value(&text_config("qwen3_next")).unwrap();
        assert_eq!(next.text.variant, HybridVariant::Qwen3Next);
        assert_eq!(next.text.mtp_num_hidden_layers, 1);
        assert_eq!(
            fused_projection_widths(&next.text).unwrap(),
            ([8, 8, 16, 16], 2)
        );

        let dense = model_args_from_config_value(&text_config("qwen3_5_text")).unwrap();
        assert_eq!(dense.text.variant, HybridVariant::Qwen35Dense);
        assert!(!dense.text.is_moe());

        let mut conditional = json!({
            "model_type": "qwen3_5_moe",
            "text_config": text_config("qwen3_5_moe_text"),
            "image_token_id": 60,
            "video_token_id": 61,
            "vision_config": {
                "depth": 2, "hidden_size": 16, "intermediate_size": 24,
                "num_heads": 4, "num_position_embeddings": 16,
                "in_channels": 3, "patch_size": 2, "spatial_merge_size": 2,
                "temporal_patch_size": 2, "out_hidden_size": 32
            }
        });
        conditional["text_config"]["intermediate_size"] = json!(0);
        let conditional = model_args_from_config_value(&conditional).unwrap();
        assert_eq!(conditional.text.variant, HybridVariant::Qwen35Moe);
        assert!(conditional.vision.is_some());
    }

    #[test]
    fn prompt_cache_fingerprint_includes_load_time_quantization() {
        let dense = model_args_from_config_value(&text_config("qwen3_next"))
            .unwrap()
            .text;
        let quantized = crate::qwen::hybrid::load_time_quantization(
            &dense,
            eredu_checkpoint::AffineQuantization::new(16, 4)
                .unwrap()
                .into(),
        )
        .unwrap();

        assert_ne!(
            prompt_cache_architecture_fingerprint(&dense),
            prompt_cache_architecture_fingerprint(&quantized)
        );
    }

    #[test]
    fn generic_qwen35_wrapper_uses_nested_moe_architecture() {
        let mut nested = text_config("qwen3_5_moe");
        nested["intermediate_size"] = json!(0);
        let parsed = model_args_from_config_value(&json!({
            "model_type": "qwen3_5",
            "text_config": nested
        }))
        .unwrap();

        assert_eq!(parsed.text.variant, HybridVariant::Qwen35Moe);
        assert!(parsed.text.is_moe());
    }

    #[test]
    fn rejects_head_width_without_a_rotary_pair() {
        let mut value = text_config("qwen3_next");
        value["head_dim"] = json!(1);

        let error = model_args_from_config_value(&value).unwrap_err();

        assert_eq!(
            error.to_string(),
            "hybrid head_dim must be at least 2, got 1"
        );
    }

    #[test]
    fn parses_qwen38_fixture_without_backend_types() {
        let value: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/configs/qwen3.8-27b-1d4bf0f2.json"
        ))
        .unwrap();
        let parsed = model_args_from_config_value(&value).unwrap();
        assert_eq!(parsed.text.model_type, "qwen3_5_text");
        assert_eq!(parsed.text.num_hidden_layers, 64);
        assert_eq!(parsed.text.mtp_num_hidden_layers, 1);
        assert_eq!(parsed.text.rope_dimensions(), 64);
        assert_eq!(parsed.vision.unwrap().layer_count(), 27);
    }

    #[test]
    fn rejects_schedule_media_and_fp8_semantic_drift() {
        let mut invalid = text_config("qwen3_next");
        invalid["linear_num_value_heads"] = json!(3);
        assert!(model_args_from_config_value(&invalid).is_err());

        let mut invalid = text_config("qwen3_5_text");
        invalid["image_token_id"] = json!(60);
        assert!(model_args_from_config_value(&invalid).is_err());

        let mut invalid = text_config("qwen3_5_moe_text");
        invalid["quantization_config"] = json!({
            "quant_method": "fp8", "fmt": "e4m3",
            "activation_scheme": "static", "weight_block_size": [128, 128]
        });
        assert!(model_args_from_config_value(&invalid).is_err());
    }

    #[test]
    fn state_layout_preserves_mixed_recurrent_and_kv_geometry() {
        let parsed = model_args_from_config_value(&text_config("qwen3_next")).unwrap();
        let layout = state_layout(&parsed.text).unwrap();
        assert_eq!(layout.len(), 5);
        assert_eq!(layout.layer(0).unwrap().fixed_state().len(), 2);
        assert!(layout.layer(0).unwrap().attention().is_none());
        assert!(layout.layer(3).unwrap().attention().is_some());
        assert!(layout.layer(4).unwrap().attention().is_some());
        assert_eq!(layout.segments().len(), 2);
        assert_eq!(layout.segments()[0].id().as_str(), TARGET_STATE_SEGMENT);
        assert_eq!(layout.segments()[0].layers(), 0..4);
        assert_eq!(layout.segments()[1].id().as_str(), PREDICTION_STATE_SEGMENT);
        assert_eq!(layout.segments()[1].layers(), 4..5);
    }

    struct Catalog(HashSet<String>);

    impl GgufTensorCatalog for Catalog {
        fn contains(&self, name: &str) -> bool {
            self.0.contains(name)
        }

        fn any(&self, predicate: impl FnMut(&str) -> bool) -> bool {
            self.0.iter().map(String::as_str).any(predicate)
        }
    }

    #[test]
    fn gguf_parser_separates_target_and_mtp_blocks() {
        let architecture = "qwen3next";
        let mut metadata = HashMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String(architecture.into()),
            ),
            (
                format!("{architecture}.block_count"),
                MetadataValue::Uint32(5),
            ),
            (
                format!("{architecture}.nextn_predict_layers"),
                MetadataValue::Uint32(1),
            ),
            (
                format!("{architecture}.embedding_length"),
                MetadataValue::Uint32(32),
            ),
            (
                format!("{architecture}.attention.head_count"),
                MetadataValue::Uint32(4),
            ),
            (
                format!("{architecture}.attention.head_count_kv"),
                MetadataValue::Uint32(2),
            ),
            (
                format!("{architecture}.attention.key_length"),
                MetadataValue::Uint32(8),
            ),
            (
                format!("{architecture}.rope.dimension_count"),
                MetadataValue::Uint32(2),
            ),
            (
                format!("{architecture}.context_length"),
                MetadataValue::Uint32(128),
            ),
            (
                format!("{architecture}.attention.layer_norm_rms_epsilon"),
                MetadataValue::Float32(1e-6),
            ),
            (
                format!("{architecture}.ssm.conv_kernel"),
                MetadataValue::Uint32(4),
            ),
            (
                format!("{architecture}.ssm.state_size"),
                MetadataValue::Uint32(8),
            ),
            (
                format!("{architecture}.ssm.group_count"),
                MetadataValue::Uint32(2),
            ),
            (
                format!("{architecture}.ssm.time_step_rank"),
                MetadataValue::Uint32(4),
            ),
            (
                format!("{architecture}.expert_feed_forward_length"),
                MetadataValue::Uint32(16),
            ),
            (
                format!("{architecture}.expert_shared_feed_forward_length"),
                MetadataValue::Uint32(24),
            ),
            (
                format!("{architecture}.expert_used_count"),
                MetadataValue::Uint32(2),
            ),
            (
                format!("{architecture}.expert_count"),
                MetadataValue::Uint32(8),
            ),
            (
                format!("{architecture}.vocab_size"),
                MetadataValue::Uint32(64),
            ),
        ]);
        metadata.insert(
            format!("{architecture}.full_attention_interval"),
            MetadataValue::Uint32(4),
        );
        let catalog = Catalog(HashSet::from([
            "output.weight".into(),
            "blk.3.attn_q.bias".into(),
        ]));
        let parsed = model_args_from_gguf_catalog(&catalog, &metadata).unwrap();
        assert_eq!(parsed.text.num_hidden_layers, 4);
        assert_eq!(parsed.text.mtp_num_hidden_layers, 0);
        assert_eq!(parsed.text.layer_schedule.len(), 4);
        assert!(parsed.text.attention_bias);
        assert!(!parsed.text.tie_word_embeddings);
    }
}
