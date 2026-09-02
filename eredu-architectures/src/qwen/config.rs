use std::collections::{HashMap, HashSet};
use std::io::Read;

use eredu_checkpoint::WeightQuantization;
use eredu_core::cache::derive_prompt_cache_architecture_fingerprint;
use eredu_core::{AttentionPolicy, LayerSchedule};
use eredu_gguf::{MetadataArray, MetadataValue};
use eredu_nn::RotarySpec;
use serde::Deserialize;
use serde_json::Value;

use crate::{
    decoder::{AttentionProjection, Config},
    rotary::RopeValue,
    GgufArchitecture, GgufTensorCatalog,
};

/// Invalid or unsupported Qwen text configuration.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// Configuration JSON could not be decoded.
    #[error("invalid Qwen configuration: {0}")]
    Json(#[from] serde_json::Error),
    /// Configuration changes unsupported execution semantics or has invalid geometry.
    #[error("{0}")]
    Invalid(String),
}

/// Exact supported Qwen text architecture generation.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum QwenVariant {
    /// Qwen2 and Qwen2.5 with biased Q/K/V projections.
    Qwen2,
    /// Dense Qwen3 with per-head Q/K RMS normalization.
    Qwen3,
    /// Qwen3 with top-k routed gated-product experts.
    Qwen3Moe,
}

impl QwenVariant {
    /// Canonical architecture family published by the model registry.
    pub const fn model_kind(self) -> crate::ModelKind {
        match self {
            Self::Qwen2 => crate::ModelKind::Qwen2,
            Self::Qwen3 | Self::Qwen3Moe => crate::ModelKind::Qwen3,
        }
    }
}

/// Typed context for parsing a standalone or multimodal-embedded text config.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TextConfigContext {
    /// The document itself is a standalone text model configuration.
    Standalone,
    /// A Qwen3-VL artifact embeds a dense Qwen3 text configuration.
    Qwen3Vl,
    /// A Qwen3-VL-MoE artifact embeds a Qwen3-MoE text configuration.
    Qwen3VlMoe,
}

impl TextConfigContext {
    /// Selects the embedded Qwen text policy for a registry-resolved Qwen3-VL GGUF.
    pub fn from_qwen3_vl_gguf_architecture(
        architecture: GgufArchitecture,
    ) -> Result<Self, ConfigError> {
        match architecture {
            GgufArchitecture::Qwen3Vl => Ok(Self::Qwen3Vl),
            GgufArchitecture::Qwen3VlMoe => Ok(Self::Qwen3VlMoe),
            other => Err(invalid(format!(
                "unsupported Qwen3-VL GGUF architecture {other:?}"
            ))),
        }
    }
}

/// Normalized Qwen arguments shared by inspection, checkpoint planning, and execution.
#[derive(Debug, Clone)]
pub struct ModelArgs {
    /// Exact supported architecture variant.
    pub variant: QwenVariant,
    /// Canonical standalone text model type.
    pub model_type: String,
    /// Canonical parameter namespace of the text decoder in its enclosing artifact.
    pub parameter_root: String,
    /// Transformer hidden size.
    pub hidden_size: i32,
    /// Number of decoder layers.
    pub num_hidden_layers: i32,
    /// Dense SwiGLU intermediate width; zero for pure routed-MoE blocks.
    pub intermediate_size: i32,
    /// Number of query heads.
    pub num_attention_heads: i32,
    /// Number of key/value heads.
    pub num_key_value_heads: i32,
    /// Per-head query/key/value width.
    pub head_dim: i32,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// Token vocabulary size.
    pub vocab_size: i32,
    /// Maximum configured sequence length.
    pub max_position_embeddings: i32,
    /// RoPE base frequency.
    pub rope_theta: f32,
    /// Whether the output projection shares the token embedding table.
    pub tie_word_embeddings: bool,
    /// External RoPE scaling metadata retained for identity and plan emission.
    pub rope_scaling: Option<HashMap<String, RopeValue>>,
    /// Exact ordered attention policy for every layer.
    pub attention_schedule: LayerSchedule<AttentionPolicy>,
    /// Routed-expert intermediate width.
    pub moe_intermediate_size: i32,
    /// Total routed expert count.
    pub num_experts: i32,
    /// Selected experts per token.
    pub num_experts_per_tok: i32,
    /// Whether selected expert scores are renormalized.
    pub norm_topk_prob: bool,
    /// Canonical model-wide checkpoint encoding.
    pub quantization: Option<WeightQuantization>,
    /// Exact canonical names using model-wide quantization.
    pub quantized_weights: Option<HashSet<String>>,
    /// Per-parameter encodings for mixed checkpoints.
    pub quantized_weight_configs: Option<HashMap<String, WeightQuantization>>,
}

#[derive(Debug, Deserialize)]
struct ModelArgsSource {
    #[serde(default)]
    model_type: Option<String>,
    hidden_size: i32,
    num_hidden_layers: i32,
    intermediate_size: i32,
    num_attention_heads: i32,
    rms_norm_eps: f32,
    vocab_size: i32,
    num_key_value_heads: i32,
    max_position_embeddings: i32,
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    #[serde(default)]
    head_dim: i32,
    tie_word_embeddings: bool,
    rope_scaling: Option<HashMap<String, RopeValue>>,
    #[serde(default = "default_hidden_act")]
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
}

impl ModelArgs {
    /// Validates all derived geometry and supported execution policy.
    pub fn validate(&self) -> Result<(), ConfigError> {
        validate_model_args(self)
    }

    /// Returns whether decoder blocks use routed experts.
    pub const fn is_moe(&self) -> bool {
        matches!(self.variant, QwenVariant::Qwen3Moe)
    }

    /// Returns the canonical routed observation point for one decoder layer.
    pub fn routed_observation_point(
        &self,
        unit_path: &str,
        _layer: usize,
    ) -> Option<eredu_runtime::RoutedObservationPoint> {
        self.is_moe().then(|| {
            eredu_runtime::RoutedObservationPoint::new(format!("{unit_path}.mlp"), self.num_experts)
        })
    }

    /// Returns the model-wide physical encoding, if any.
    pub fn weight_quantization(&self) -> Option<WeightQuantization> {
        self.quantization
    }

    /// Returns the physical encoding for one canonical parameter identity.
    pub fn weight_quantization_for(&self, name: &str) -> Option<WeightQuantization> {
        if let Some(encoding) = self
            .quantized_weight_configs
            .as_ref()
            .and_then(|encodings| encodings.get(name))
        {
            return Some(*encoding);
        }
        let encoding = self.weight_quantization()?;
        match &self.quantized_weights {
            Some(names) if !names.contains(name) => None,
            _ => Some(encoding),
        }
    }
}

impl Config for ModelArgs {
    fn model_family(&self) -> &'static str {
        self.variant.model_kind().canonical_name()
    }
    fn model_identity(&self) -> &str {
        &self.model_type
    }
    fn architecture_fingerprint(&self) -> String {
        prompt_cache_architecture_fingerprint(self)
    }
    fn parameter_root(&self) -> &str {
        &self.parameter_root
    }
    fn routed_observation_point(
        &self,
        unit_path: &str,
        layer: usize,
    ) -> Option<eredu_runtime::RoutedObservationPoint> {
        ModelArgs::routed_observation_point(self, unit_path, layer)
    }
    fn validate_config(&self) -> Result<(), eredu_nn::Error> {
        self.validate().map_err(eredu_nn::Error::backend)
    }
    fn hidden_size(&self) -> i32 {
        self.hidden_size
    }
    fn num_hidden_layers(&self) -> i32 {
        self.num_hidden_layers
    }
    fn intermediate_size(&self) -> i32 {
        if self.is_moe() {
            self.moe_intermediate_size
        } else {
            self.intermediate_size
        }
    }
    fn num_attention_heads(&self) -> i32 {
        self.num_attention_heads
    }
    fn num_key_value_heads(&self) -> i32 {
        self.num_key_value_heads
    }
    fn head_dim(&self) -> i32 {
        self.head_dim
    }
    fn rms_norm_epsilon(&self) -> f32 {
        self.rms_norm_eps
    }
    fn vocabulary_size(&self) -> i32 {
        self.vocab_size
    }
    fn attention_bias(&self, projection: AttentionProjection) -> bool {
        self.variant == QwenVariant::Qwen2 && !matches!(projection, AttentionProjection::Output)
    }
    fn query_key_norm_epsilon(&self) -> Option<f32> {
        (self.variant != QwenVariant::Qwen2).then_some(self.rms_norm_eps)
    }
    fn mlp_bias(&self) -> bool {
        false
    }
    fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings
    }
    fn attention_schedule(&self) -> &LayerSchedule<AttentionPolicy> {
        &self.attention_schedule
    }
    fn weight_quantization(&self, name: &str) -> Option<WeightQuantization> {
        self.weight_quantization_for(name)
    }
    fn rotary_spec(&self, dimensions: i32) -> RotarySpec {
        RotarySpec {
            dimensions,
            base: self.rope_theta,
            traditional: false,
            algorithm: crate::rotary::normalize_algorithm(self.rope_scaling.as_ref())
                .expect("validated Qwen RoPE algorithm"),
        }
    }
}

/// Reads and validates a standalone Hugging Face Qwen configuration.
pub fn model_args_from_config_reader(reader: impl Read) -> Result<ModelArgs, ConfigError> {
    let value = serde_json::from_reader(reader)?;
    model_args_from_config_value(&value)
}

/// Parses a standalone Hugging Face Qwen configuration.
pub fn model_args_from_config_value(value: &Value) -> Result<ModelArgs, ConfigError> {
    model_args_from_text_config_value(value, TextConfigContext::Standalone)
}

/// Parses Qwen text policy in an explicit standalone or multimodal embedding context.
pub fn model_args_from_text_config_value(
    value: &Value,
    context: TextConfigContext,
) -> Result<ModelArgs, ConfigError> {
    validate_execution_fields(value)?;
    let mut source: ModelArgsSource = serde_json::from_value(value.clone())?;
    let variant = match context {
        TextConfigContext::Standalone => match source.model_type.as_deref() {
            Some("qwen2") => QwenVariant::Qwen2,
            Some("qwen3") => QwenVariant::Qwen3,
            Some("qwen3_moe") => QwenVariant::Qwen3Moe,
            Some(other) => return Err(invalid(format!("unsupported Qwen model type {other:?}"))),
            None => return Err(invalid("standalone Qwen config is missing model_type")),
        },
        TextConfigContext::Qwen3Vl => QwenVariant::Qwen3,
        TextConfigContext::Qwen3VlMoe => QwenVariant::Qwen3Moe,
    };
    let canonical_model_type = match variant {
        QwenVariant::Qwen2 => "qwen2",
        QwenVariant::Qwen3 => "qwen3",
        QwenVariant::Qwen3Moe => "qwen3_moe",
    };
    if context == TextConfigContext::Standalone {
        validate_declared_architectures(value, canonical_model_type)?;
    }
    source.model_type = Some(canonical_model_type.into());
    if source.head_dim == 0
        && source.num_attention_heads > 0
        && source.hidden_size % source.num_attention_heads == 0
    {
        source.head_dim = source.hidden_size / source.num_attention_heads;
    }
    let attention_schedule = hf_attention_schedule(
        value,
        variant,
        usize::try_from(source.num_hidden_layers).map_err(|_| {
            invalid(format!(
                "num_hidden_layers must be positive, got {}",
                source.num_hidden_layers
            ))
        })?,
    )?;
    validate_source_policy(variant, &source)?;
    let quantization = match (source.quantization, source.quantization_config) {
        (Some(first), Some(second)) if first != second => {
            return Err(invalid(
                "Qwen quantization and quantization_config disagree",
            ));
        }
        (Some(value), _) | (_, Some(value)) => Some(value),
        (None, None) => None,
    };
    let args = ModelArgs {
        variant,
        model_type: canonical_model_type.into(),
        parameter_root: match context {
            TextConfigContext::Standalone => "model",
            TextConfigContext::Qwen3Vl | TextConfigContext::Qwen3VlMoe => "model.language_model",
        }
        .into(),
        hidden_size: source.hidden_size,
        num_hidden_layers: source.num_hidden_layers,
        intermediate_size: source.intermediate_size,
        num_attention_heads: source.num_attention_heads,
        num_key_value_heads: source.num_key_value_heads,
        head_dim: source.head_dim,
        rms_norm_eps: source.rms_norm_eps,
        vocab_size: source.vocab_size,
        max_position_embeddings: source.max_position_embeddings,
        rope_theta: source.rope_theta,
        tie_word_embeddings: source.tie_word_embeddings,
        rope_scaling: source.rope_scaling,
        attention_schedule,
        moe_intermediate_size: source.moe_intermediate_size,
        num_experts: source.num_experts,
        num_experts_per_tok: source.num_experts_per_tok,
        norm_topk_prob: source.norm_topk_prob,
        quantization,
        quantized_weights: None,
        quantized_weight_configs: None,
    };
    args.validate()?;
    Ok(args)
}

fn hf_attention_schedule(
    value: &Value,
    variant: QwenVariant,
    layers: usize,
) -> Result<LayerSchedule<AttentionPolicy>, ConfigError> {
    if layers == 0 {
        return Err(invalid("num_hidden_layers must be positive, got 0"));
    }
    let enabled = match value.get("use_sliding_window") {
        None => false,
        Some(value) => value
            .as_bool()
            .ok_or_else(|| invalid("use_sliding_window must be boolean"))?,
    };
    if variant != QwenVariant::Qwen2 {
        if enabled {
            return Err(invalid(
                "Qwen3 does not use Qwen2 sliding-window configuration",
            ));
        }
        return LayerSchedule::all_full(layers).map_err(|error| invalid(error.to_string()));
    }
    if !enabled {
        return LayerSchedule::all_full(layers).map_err(|error| invalid(error.to_string()));
    }
    let window = required_positive_u32(value, "sliding_window")?;
    if window > i32::MAX as u32 {
        return Err(invalid(format!("sliding_window exceeds i32: {window}")));
    }
    let first = required_nonnegative_usize(value, "max_window_layers")?;
    if first >= layers {
        return Err(invalid(format!(
            "max_window_layers must leave at least one sliding layer, got {first} for {layers} layers"
        )));
    }
    let sliding = AttentionPolicy::sliding(window).map_err(|error| invalid(error.to_string()))?;
    LayerSchedule::new(
        layers,
        (0..layers)
            .map(|layer| {
                if layer < first {
                    AttentionPolicy::Full
                } else {
                    sliding
                }
            })
            .collect(),
    )
    .map_err(|error| invalid(error.to_string()))
}

/// Parses normalized Qwen arguments from pure GGUF catalog metadata.
pub fn model_args_from_gguf_catalog(
    arrays: &impl GgufTensorCatalog,
    metadata: &HashMap<String, MetadataValue>,
) -> Result<ModelArgs, ConfigError> {
    model_args_from_gguf_catalog_with_context(arrays, metadata, TextConfigContext::Standalone)
}

/// Parses normalized Qwen arguments from GGUF in an explicit embedding context.
pub fn model_args_from_gguf_catalog_with_context(
    arrays: &impl GgufTensorCatalog,
    metadata: &HashMap<String, MetadataValue>,
    context: TextConfigContext,
) -> Result<ModelArgs, ConfigError> {
    let declared_architecture = gguf_string(metadata, "general.architecture")?;
    let architecture = GgufArchitecture::resolve(&declared_architecture)
        .map_err(|error| invalid(error.to_string()))?;
    let variant = match (context, architecture) {
        (TextConfigContext::Standalone, GgufArchitecture::Qwen2) => QwenVariant::Qwen2,
        (TextConfigContext::Standalone, GgufArchitecture::Qwen3) => QwenVariant::Qwen3,
        (TextConfigContext::Standalone, GgufArchitecture::Qwen3Moe) => QwenVariant::Qwen3Moe,
        (TextConfigContext::Qwen3Vl, GgufArchitecture::Qwen3Vl) => QwenVariant::Qwen3,
        (TextConfigContext::Qwen3VlMoe, GgufArchitecture::Qwen3VlMoe) => QwenVariant::Qwen3Moe,
        (_, other) => {
            return Err(invalid(format!(
                "unsupported Qwen GGUF architecture {other:?} for {context:?}"
            )))
        }
    };
    let architecture = architecture.metadata_name();
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let hidden_size = gguf_i32(metadata, &key("embedding_length"))?;
    let num_hidden_layers = gguf_i32(metadata, &key("block_count"))?;
    let num_attention_heads = gguf_i32(metadata, &key("attention.head_count"))?;
    if num_attention_heads <= 0 {
        return Err(invalid(format!(
            "GGUF attention head count must be positive, got {num_attention_heads}"
        )));
    }
    let num_key_value_heads = gguf_optional_i64(metadata, &key("attention.head_count_kv"))?
        .map(i32::try_from)
        .transpose()
        .map_err(|_| invalid("GGUF KV-head count exceeds i32"))?
        .unwrap_or(num_attention_heads);
    let head_dim = gguf_optional_i64(metadata, &key("attention.key_length"))?
        .map(i32::try_from)
        .transpose()
        .map_err(|_| invalid("GGUF attention key length exceeds i32"))?
        .unwrap_or(hidden_size / num_attention_heads);
    if let Some(value) = gguf_optional_i64(metadata, &key("attention.value_length"))? {
        if value != i64::from(head_dim) {
            return Err(invalid(format!(
                "GGUF value length {value} does not match head dimension {head_dim}"
            )));
        }
    }
    if let Some(value) = gguf_optional_i64(metadata, &key("rope.dimension_count"))? {
        if value != i64::from(head_dim) {
            return Err(invalid(format!(
                "GGUF rotary dimension {value} does not match head dimension {head_dim}"
            )));
        }
    }
    let layers = usize::try_from(num_hidden_layers)
        .map_err(|_| invalid(format!("invalid GGUF block count {num_hidden_layers}")))?;
    let attention_schedule = if variant == QwenVariant::Qwen2 {
        gguf_qwen2_schedule(metadata, architecture, layers)?
    } else {
        LayerSchedule::all_full(layers).map_err(|error| invalid(error.to_string()))?
    };
    let vocab_size = gguf_i32(metadata, &key("vocab_size"))?;
    let is_moe = variant == QwenVariant::Qwen3Moe;
    let args = ModelArgs {
        variant,
        model_type: match variant {
            QwenVariant::Qwen2 => "qwen2",
            QwenVariant::Qwen3 => "qwen3",
            QwenVariant::Qwen3Moe => "qwen3_moe",
        }
        .into(),
        parameter_root: match context {
            TextConfigContext::Standalone => "model",
            TextConfigContext::Qwen3Vl | TextConfigContext::Qwen3VlMoe => "model.language_model",
        }
        .into(),
        hidden_size,
        num_hidden_layers,
        intermediate_size: if is_moe {
            gguf_optional_i64(metadata, &key("feed_forward_length"))?
                .map(i32::try_from)
                .transpose()
                .map_err(|_| invalid("GGUF feed-forward length exceeds i32"))?
                .unwrap_or(0)
        } else {
            gguf_i32(metadata, &key("feed_forward_length"))?
        },
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        rms_norm_eps: gguf_f32(metadata, &key("attention.layer_norm_rms_epsilon"))?,
        vocab_size,
        max_position_embeddings: gguf_i32(metadata, &key("context_length"))?,
        rope_theta: gguf_optional_f32(metadata, &key("rope.freq_base"))?.unwrap_or(1_000_000.0),
        tie_word_embeddings: !arrays.contains("output.weight"),
        rope_scaling: gguf_rope_scaling(metadata, architecture)?,
        attention_schedule,
        moe_intermediate_size: if is_moe {
            gguf_i32(metadata, &key("expert_feed_forward_length"))?
        } else {
            0
        },
        num_experts: if is_moe {
            gguf_i32(metadata, &key("expert_count"))?
        } else {
            0
        },
        num_experts_per_tok: if is_moe {
            gguf_i32(metadata, &key("expert_used_count"))?
        } else {
            0
        },
        norm_topk_prob: is_moe,
        quantization: None,
        quantized_weights: None,
        quantized_weight_configs: None,
    };
    args.validate()?;
    Ok(args)
}

/// Returns the stable cache-compatibility fingerprint for normalized Qwen policy.
pub fn prompt_cache_architecture_fingerprint(args: &ModelArgs) -> String {
    let rope_scaling = args.rope_scaling.as_ref().map_or_else(
        || "none".to_string(),
        |config| {
            let mut entries = config.iter().collect::<Vec<_>>();
            entries.sort_unstable_by_key(|(key, _)| key.as_str());
            entries
                .into_iter()
                .map(|(key, value)| format!("{key}={value:?}"))
                .collect::<Vec<_>>()
                .join(";")
        },
    );
    let mut quantized_weights = args
        .quantized_weights
        .as_ref()
        .map(|weights| weights.iter().cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    quantized_weights.sort_unstable();
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
        args.variant.model_kind().canonical_name(),
        [
            ("variant", format!("{:?}", args.variant)),
            ("model_type", args.model_type.clone()),
            ("parameter_root", args.parameter_root.clone()),
            ("hidden_size", args.hidden_size.to_string()),
            ("num_hidden_layers", args.num_hidden_layers.to_string()),
            ("intermediate_size", args.intermediate_size.to_string()),
            ("num_attention_heads", args.num_attention_heads.to_string()),
            ("num_key_value_heads", args.num_key_value_heads.to_string()),
            ("head_dim", args.head_dim.to_string()),
            (
                "rms_norm_eps",
                format!("{:08x}", args.rms_norm_eps.to_bits()),
            ),
            ("vocab_size", args.vocab_size.to_string()),
            (
                "max_position_embeddings",
                args.max_position_embeddings.to_string(),
            ),
            ("rope_theta", format!("{:08x}", args.rope_theta.to_bits())),
            ("rope_scaling", rope_scaling),
            (
                "attention_schedule",
                args.attention_schedule.fingerprint_component(),
            ),
            (
                "moe_intermediate_size",
                args.moe_intermediate_size.to_string(),
            ),
            ("num_experts", args.num_experts.to_string()),
            ("num_experts_per_tok", args.num_experts_per_tok.to_string()),
            ("norm_topk_prob", args.norm_topk_prob.to_string()),
            ("tie_word_embeddings", args.tie_word_embeddings.to_string()),
            ("quantization", format!("{:?}", args.weight_quantization())),
            ("quantized_weights", quantized_weights.join(";")),
            (
                "quantized_weight_configs",
                quantized_weight_configs.join(";"),
            ),
        ],
    )
}

fn validate_source_policy(
    variant: QwenVariant,
    source: &ModelArgsSource,
) -> Result<(), ConfigError> {
    if source.hidden_act != "silu" {
        return Err(invalid(format!(
            "Qwen requires hidden_act=\"silu\", got {:?}",
            source.hidden_act
        )));
    }
    if source.attention_dropout != 0.0 {
        return Err(invalid(format!(
            "Qwen inference requires attention_dropout=0, got {}",
            source.attention_dropout
        )));
    }
    match (variant, source.attention_bias) {
        (QwenVariant::Qwen2, Some(false)) => {
            return Err(invalid("Qwen2 requires learned Q/K/V projection biases"));
        }
        (QwenVariant::Qwen3 | QwenVariant::Qwen3Moe, Some(true)) => {
            return Err(invalid("Qwen3 does not support biased Q/K/V projections"));
        }
        _ => {}
    }
    if source.mlp_bias == Some(true) {
        return Err(invalid("Qwen does not support biased SwiGLU projections"));
    }
    Ok(())
}

fn validate_model_args(args: &ModelArgs) -> Result<(), ConfigError> {
    if args.parameter_root.is_empty()
        || args.parameter_root.starts_with('.')
        || args.parameter_root.ends_with('.')
        || args.parameter_root.split('.').any(str::is_empty)
    {
        return Err(invalid(format!(
            "invalid Qwen parameter root {:?}",
            args.parameter_root
        )));
    }
    for (name, value) in [
        ("hidden_size", args.hidden_size),
        ("num_hidden_layers", args.num_hidden_layers),
        ("num_attention_heads", args.num_attention_heads),
        ("num_key_value_heads", args.num_key_value_heads),
        ("head_dim", args.head_dim),
        ("vocab_size", args.vocab_size),
        ("max_position_embeddings", args.max_position_embeddings),
    ] {
        if value <= 0 {
            return Err(invalid(format!("{name} must be positive, got {value}")));
        }
    }
    if args.num_attention_heads % args.num_key_value_heads != 0 {
        return Err(invalid(format!(
            "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
            args.num_attention_heads, args.num_key_value_heads
        )));
    }
    let query_width = args
        .num_attention_heads
        .checked_mul(args.head_dim)
        .ok_or_else(|| invalid("query projection width overflows i32"))?;
    args.num_key_value_heads
        .checked_mul(args.head_dim)
        .ok_or_else(|| invalid("key/value projection width overflows i32"))?;
    if args.variant == QwenVariant::Qwen2 && query_width != args.hidden_size {
        return Err(invalid(format!(
            "Qwen2 hidden_size {} does not equal query width {query_width}",
            args.hidden_size
        )));
    }
    if !args.rms_norm_eps.is_finite() || args.rms_norm_eps <= 0.0 {
        return Err(invalid(format!(
            "rms_norm_eps must be finite and positive, got {}",
            args.rms_norm_eps
        )));
    }
    if !args.rope_theta.is_finite() || args.rope_theta <= 0.0 {
        return Err(invalid(format!(
            "rope_theta must be finite and positive, got {}",
            args.rope_theta
        )));
    }
    if args.attention_schedule.len() != args.num_hidden_layers as usize {
        return Err(invalid(
            "attention schedule length does not match decoder layers",
        ));
    }
    if args.variant != QwenVariant::Qwen2
        && args
            .attention_schedule
            .iter()
            .any(|policy| *policy != AttentionPolicy::Full)
    {
        return Err(invalid("Qwen3 requires full attention in every layer"));
    }
    validate_rope_scaling(args.rope_scaling.as_ref())?;
    if args.is_moe() {
        for (name, value) in [
            ("moe_intermediate_size", args.moe_intermediate_size),
            ("num_experts", args.num_experts),
            ("num_experts_per_tok", args.num_experts_per_tok),
        ] {
            if value <= 0 {
                return Err(invalid(format!("{name} must be positive, got {value}")));
            }
        }
        if args.num_experts_per_tok > args.num_experts {
            return Err(invalid("num_experts_per_tok exceeds num_experts"));
        }
    } else if args.intermediate_size <= 0 {
        return Err(invalid(format!(
            "intermediate_size must be positive, got {}",
            args.intermediate_size
        )));
    }
    Ok(())
}

fn validate_execution_fields(value: &Value) -> Result<(), ConfigError> {
    for field in [
        "layer_types",
        "sliding_window_pattern",
        "attention_type",
        "use_qk_norm",
        "qk_norm",
        "attention_chunk_size",
        "value_head_dim",
        "attention_output_bias",
    ] {
        if value.get(field).is_some() {
            return Err(invalid(format!(
                "Qwen config field {field:?} changes decoder execution and is unsupported"
            )));
        }
    }
    if let Some(factor) = value.get("partial_rotary_factor") {
        if factor.as_f64() != Some(1.0) {
            return Err(invalid("Qwen requires partial_rotary_factor=1"));
        }
    }
    if let Some(interleaved) = value.get("rope_interleaved") {
        if interleaved.as_bool() != Some(false) {
            return Err(invalid("Qwen requires rope_interleaved=false"));
        }
    }
    Ok(())
}

fn validate_declared_architectures(value: &Value, model_type: &str) -> Result<(), ConfigError> {
    let Some(declared) = value.get("architectures") else {
        return Ok(());
    };
    let declared = declared
        .as_array()
        .ok_or_else(|| invalid("config architectures must be an array of strings"))?;
    let expected = match model_type {
        "qwen2" => "Qwen2ForCausalLM",
        "qwen3" => "Qwen3ForCausalLM",
        "qwen3_moe" => "Qwen3MoeForCausalLM",
        _ => unreachable!(),
    };
    if declared.is_empty() || declared.iter().any(|item| item.as_str() != Some(expected)) {
        return Err(invalid(format!(
            "model_type {model_type:?} requires architectures [{expected:?}]"
        )));
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
        .ok_or_else(|| invalid("rope_scaling requires string type or rope_type"))?;
    if !matches!(kind, "default" | "linear" | "yarn") {
        return Err(invalid(format!("unsupported Qwen RoPE scaling {kind:?}")));
    }
    if matches!(kind, "linear" | "yarn") {
        let factor = rope_number(scaling, "factor")
            .ok_or_else(|| invalid("scaled Qwen RoPE requires a numeric factor"))?;
        if factor <= 0.0 {
            return Err(invalid("scaled Qwen RoPE factor must be positive"));
        }
    }
    if kind == "yarn"
        && rope_number(scaling, "original_max_position_embeddings").is_none_or(|value| value <= 0.0)
    {
        return Err(invalid(
            "YaRN requires positive original_max_position_embeddings",
        ));
    }
    crate::rotary::normalize_algorithm(Some(scaling))
        .map(|_| ())
        .map_err(invalid)
}

fn rope_number(values: &HashMap<String, RopeValue>, key: &str) -> Option<f32> {
    match values.get(key)? {
        RopeValue::Float(value) if value.is_finite() => Some(*value),
        RopeValue::String(value) => value.parse().ok().filter(|value: &f32| value.is_finite()),
        _ => None,
    }
}

fn gguf_qwen2_schedule(
    metadata: &HashMap<String, MetadataValue>,
    architecture: &str,
    layers: usize,
) -> Result<LayerSchedule<AttentionPolicy>, ConfigError> {
    let window_key = format!("{architecture}.attention.sliding_window");
    let pattern_key = format!("{architecture}.attention.sliding_window_pattern");
    let window = gguf_optional_i64(metadata, &window_key)?;
    let pattern = match metadata.get(&pattern_key) {
        None => None,
        Some(MetadataValue::Array(MetadataArray::Bool(values))) => Some(values.as_slice()),
        Some(_) => return Err(invalid(format!("{pattern_key} must be a boolean array"))),
    };
    if let Some(pattern) = pattern {
        if pattern.len() != layers {
            return Err(invalid(format!(
                "Qwen2 sliding pattern has {} entries for {layers} layers",
                pattern.len()
            )));
        }
    }
    let Some(window) = window else {
        if pattern.is_some_and(|values| values.iter().any(|value| *value)) {
            return Err(invalid("sliding pattern requires a sliding-window size"));
        }
        return LayerSchedule::all_full(layers).map_err(|error| invalid(error.to_string()));
    };
    let window = u32::try_from(window)
        .ok()
        .filter(|window| *window > 0 && *window <= i32::MAX as u32)
        .ok_or_else(|| invalid(format!("invalid Qwen2 sliding window {window}")))?;
    let pattern = pattern.map_or_else(|| vec![true; layers], <[bool]>::to_vec);
    LayerSchedule::from_sliding_pattern(layers, &pattern, Some(window))
        .map_err(|error| invalid(error.to_string()))
}

fn gguf_rope_scaling(
    metadata: &HashMap<String, MetadataValue>,
    architecture: &str,
) -> Result<Option<HashMap<String, RopeValue>>, ConfigError> {
    let Some(kind) = gguf_optional_string(metadata, &format!("{architecture}.rope.scaling.type"))?
    else {
        return Ok(None);
    };
    match kind.as_str() {
        "none" | "default" => Ok(None),
        "linear" => Ok(Some(HashMap::from([
            ("rope_type".into(), RopeValue::String("linear".into())),
            (
                "factor".into(),
                RopeValue::Float(gguf_f32(
                    metadata,
                    &format!("{architecture}.rope.scaling.factor"),
                )?),
            ),
        ]))),
        "yarn" => Ok(Some(HashMap::from([
            ("rope_type".into(), RopeValue::String("yarn".into())),
            (
                "factor".into(),
                RopeValue::Float(gguf_f32(
                    metadata,
                    &format!("{architecture}.rope.scaling.factor"),
                )?),
            ),
            (
                "original_max_position_embeddings".into(),
                RopeValue::Float(gguf_i32(
                    metadata,
                    &format!("{architecture}.rope.scaling.original_context_length"),
                )? as f32),
            ),
            ("truncate".into(), RopeValue::Bool(false)),
        ]))),
        other => Err(invalid(format!(
            "unsupported Qwen GGUF RoPE scaling {other:?}"
        ))),
    }
}

fn required_positive_u32(value: &Value, field: &str) -> Result<u32, ConfigError> {
    let value = value
        .get(field)
        .and_then(Value::as_i64)
        .ok_or_else(|| invalid(format!("{field} must be a positive integer")))?;
    u32::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| invalid(format!("{field} must be positive, got {value}")))
}

fn required_nonnegative_usize(value: &Value, field: &str) -> Result<usize, ConfigError> {
    let value = value
        .get(field)
        .and_then(Value::as_i64)
        .ok_or_else(|| invalid(format!("{field} must be a non-negative integer")))?;
    usize::try_from(value).map_err(|_| invalid(format!("{field} must be non-negative")))
}

fn gguf_string(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<String, ConfigError> {
    gguf_optional_string(metadata, key)?
        .ok_or_else(|| invalid(format!("GGUF metadata is missing {key:?}")))
}

fn gguf_optional_string(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<String>, ConfigError> {
    match metadata.get(key) {
        Some(MetadataValue::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(invalid(format!("GGUF metadata {key:?} has wrong type"))),
        None => Ok(None),
    }
}

fn gguf_i32(metadata: &HashMap<String, MetadataValue>, key: &str) -> Result<i32, ConfigError> {
    i32::try_from(
        gguf_optional_i64(metadata, key)?
            .ok_or_else(|| invalid(format!("GGUF metadata is missing {key:?}")))?,
    )
    .map_err(|_| invalid(format!("GGUF metadata {key:?} exceeds i32")))
}

fn gguf_optional_i64(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<i64>, ConfigError> {
    match metadata.get(key) {
        Some(value) => value
            .as_i64()
            .map(Some)
            .ok_or_else(|| invalid(format!("GGUF metadata {key:?} has wrong type"))),
        None => Ok(None),
    }
}

fn gguf_f32(metadata: &HashMap<String, MetadataValue>, key: &str) -> Result<f32, ConfigError> {
    gguf_optional_f32(metadata, key)?
        .ok_or_else(|| invalid(format!("GGUF metadata is missing {key:?}")))
}

fn gguf_optional_f32(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<f32>, ConfigError> {
    match metadata.get(key) {
        Some(value) => value
            .as_f32()
            .map(Some)
            .ok_or_else(|| invalid(format!("GGUF metadata {key:?} has wrong type"))),
        None => Ok(None),
    }
}

fn invalid(message: impl Into<String>) -> ConfigError {
    ConfigError::Invalid(message.into())
}

fn default_hidden_act() -> String {
    "silu".into()
}

const fn default_rope_theta() -> f32 {
    1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Catalog;

    impl GgufTensorCatalog for Catalog {
        fn contains(&self, _name: &str) -> bool {
            false
        }

        fn any(&self, _predicate: impl FnMut(&str) -> bool) -> bool {
            false
        }
    }

    fn base(model_type: &str) -> Value {
        serde_json::json!({
            "model_type": model_type,
            "hidden_size": 16,
            "num_hidden_layers": 3,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "rms_norm_eps": 0.000001,
            "vocab_size": 64,
            "max_position_embeddings": 128,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": true
        })
    }

    #[test]
    fn distinguishes_qwen2_and_qwen3_attention_policy() {
        let qwen2 = model_args_from_config_value(&base("qwen2")).unwrap();
        let qwen3 = model_args_from_config_value(&base("qwen3")).unwrap();
        assert!(qwen2.attention_bias(AttentionProjection::Query));
        assert!(!qwen2.attention_bias(AttentionProjection::Output));
        assert_eq!(qwen2.query_key_norm_epsilon(), None);
        assert_eq!(qwen3.query_key_norm_epsilon(), Some(0.000001));
    }

    #[test]
    fn prompt_cache_identity_uses_the_registry_family() {
        for (model_type, family) in [("qwen2", "qwen2"), ("qwen3", "qwen3")] {
            let args = model_args_from_config_value(&base(model_type)).unwrap();
            let layout = crate::qwen::state_layout(&args).unwrap();
            let identity = crate::qwen::state_identity(
                &args,
                &layout,
                0,
                eredu_core::cache::PromptCacheTopology::default(),
            )
            .unwrap()
            .prompt_cache_identity(&layout)
            .unwrap();

            assert_eq!(identity.model_family(), family);
        }
    }

    #[test]
    fn gguf_structural_vocabulary_ignores_malformed_tokenizer_metadata() {
        let metadata = HashMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String("qwen3".into()),
            ),
            ("qwen3.embedding_length".into(), MetadataValue::Uint32(16)),
            ("qwen3.block_count".into(), MetadataValue::Uint32(3)),
            (
                "qwen3.attention.head_count".into(),
                MetadataValue::Uint32(4),
            ),
            (
                "qwen3.attention.head_count_kv".into(),
                MetadataValue::Uint32(2),
            ),
            (
                "qwen3.feed_forward_length".into(),
                MetadataValue::Uint32(32),
            ),
            (
                "qwen3.attention.layer_norm_rms_epsilon".into(),
                MetadataValue::Float32(1e-6),
            ),
            ("qwen3.vocab_size".into(), MetadataValue::Uint32(64)),
            ("qwen3.context_length".into(), MetadataValue::Uint32(128)),
            (
                "tokenizer.ggml.tokens".into(),
                MetadataValue::String("malformed tokenizer payload".into()),
            ),
        ]);

        let args = model_args_from_gguf_catalog(&Catalog, &metadata).unwrap();
        assert_eq!(args.vocab_size, 64);
    }

    #[test]
    fn canonicalizes_quantization_alias_and_rejects_conflicts() {
        let mut value = base("qwen3");
        value["quantization_config"] = serde_json::json!({"group_size": 32, "bits": 4});
        let args = model_args_from_config_value(&value).unwrap();
        assert_eq!(
            args.quantization,
            Some(WeightQuantization::Affine(
                eredu_checkpoint::AffineQuantization::new(32, 4).unwrap()
            ))
        );

        value["quantization"] = serde_json::json!({"group_size": 64, "bits": 4});
        let error = model_args_from_config_value(&value).unwrap_err();
        assert_eq!(
            error.to_string(),
            "Qwen quantization and quantization_config disagree"
        );
    }

    #[test]
    fn parses_mixed_full_and_sliding_qwen2_schedule() {
        let mut value = base("qwen2");
        value["use_sliding_window"] = Value::Bool(true);
        value["sliding_window"] = Value::from(32);
        value["max_window_layers"] = Value::from(1);
        let args = model_args_from_config_value(&value).unwrap();
        assert_eq!(args.attention_schedule.get(0), Some(&AttentionPolicy::Full));
        assert_eq!(
            args.attention_schedule
                .get(1)
                .unwrap()
                .window()
                .unwrap()
                .get(),
            32
        );
    }

    #[test]
    fn standalone_parser_rejects_multimodal_aliases() {
        assert!(model_args_from_config_value(&base("qwen3_vl_text")).is_err());
        let args = model_args_from_text_config_value(&base("ignored"), TextConfigContext::Qwen3Vl)
            .unwrap();
        assert_eq!(args.variant, QwenVariant::Qwen3);
        assert_eq!(args.model_type, "qwen3");
    }

    #[test]
    fn qwen3_vl_gguf_context_uses_registry_identity() {
        assert_eq!(
            TextConfigContext::from_qwen3_vl_gguf_architecture(GgufArchitecture::Qwen3Vl).unwrap(),
            TextConfigContext::Qwen3Vl
        );
        assert_eq!(
            TextConfigContext::from_qwen3_vl_gguf_architecture(GgufArchitecture::Qwen3VlMoe)
                .unwrap(),
            TextConfigContext::Qwen3VlMoe
        );
        assert!(
            TextConfigContext::from_qwen3_vl_gguf_architecture(GgufArchitecture::Qwen3).is_err()
        );
    }

    #[test]
    fn validates_qwen3_moe_routing_geometry() {
        let mut value = base("qwen3_moe");
        value["intermediate_size"] = Value::from(0);
        value["moe_intermediate_size"] = Value::from(8);
        value["num_experts"] = Value::from(4);
        value["num_experts_per_tok"] = Value::from(2);
        value["norm_topk_prob"] = Value::Bool(true);
        let args = model_args_from_config_value(&value).unwrap();
        assert!(args.is_moe());
        let point = args.routed_observation_point("model.layers.2", 2).unwrap();
        assert_eq!(point.path(), "model.layers.2.mlp");
        assert_eq!(point.expert_count(), 4);
        assert!(model_args_from_config_value(&base("qwen3"))
            .unwrap()
            .routed_observation_point("model.layers.2", 2)
            .is_none());
    }
}
