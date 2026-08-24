//! Strict portable normalization for released Moshi-family configurations.

use crate::decoder::{
    AttentionProjection, AttentionProjectionLayout, BlockParameterFields, Config as DecoderConfig,
    GatedProjectionLayout,
};
use eredu_checkpoint::WeightQuantization;
use eredu_core::{
    cache::derive_prompt_cache_architecture_fingerprint, AttentionPolicy, LayerSchedule,
    RealtimeConfigError, RealtimeFrameConvention, RealtimeSpeechConfig,
};
use eredu_nn::RotarySpec;
use serde::Deserialize;
use serde_json::Value;
use std::{collections::BTreeMap, io::Read};

/// Stable architecture-family identity shared by native Moshi and PersonaPlex.
pub const MOSHI_FAMILY: &str = "moshi";
/// Only PersonaPlex release admitted by the normalized architecture.
pub const PERSONAPLEX_VERSION: &str = "7b-v1";

const NATIVE_VERSION: &str = "0.1";
const RMS_NORM_EPSILON: f64 = 1e-8;
const MAX_PORTABLE_LAYERS: i64 = 65_536;
const ROOT_PARAMETER_NAMESPACE: &str = "";
const TEMPORAL_PARAMETER_ROOT: &str = "transformer";
const DEPTH_SLICE_ZERO_PARAMETER_ROOT: &str = "depformer.slices.0.transformer";

/// Effective released model selected before backend or checkpoint planning.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum EffectiveModelType {
    /// Native Moshi.
    Moshi,
    /// PersonaPlex using Moshi-family equations.
    PersonaPlex,
}

impl EffectiveModelType {
    /// Stable metadata spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Moshi => "moshi",
            Self::PersonaPlex => "personaplex",
        }
    }
}

/// Normalized artifact profile, independent of model geometry.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum ArtifactProfile {
    /// Built-in defaults for original Moshi v0.1 artifacts with no config.
    NativeV0_1,
    /// Native Moshi values read from an explicit config.
    NativeConfig,
    /// Published PersonaPlex 7B v1 artifact.
    PersonaPlex7bV1,
}

impl ArtifactProfile {
    const fn as_str(self) -> &'static str {
        match self {
            Self::NativeV0_1 => "native_v0_1",
            Self::NativeConfig => "native_config",
            Self::PersonaPlex7bV1 => "personaplex_7b_v1",
        }
    }
}

/// Physical SafeTensors namespace selected by artifact metadata.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum CheckpointLayout {
    /// Original MLX-style Moshi SafeTensors names.
    NativeMlx,
    /// Published PyTorch-style PersonaPlex SafeTensors names.
    PersonaPlexPytorch,
}

impl CheckpointLayout {
    const fn as_str(self) -> &'static str {
        match self {
            Self::NativeMlx => "native_mlx",
            Self::PersonaPlexPytorch => "personaplex_pytorch",
        }
    }
}

/// Positional equation applied by one transformer stack.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum PositionalEncoding {
    /// Traditional rotary position encoding.
    Rope,
    /// No position encoding.
    None,
}

impl PositionalEncoding {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Rope => "rope",
            Self::None => "none",
        }
    }
}

/// Cross-slice ownership of normalized depth parameters.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum ParameterSharing {
    /// Native depth slices own independent parameters.
    IndependentDepthSlices,
    /// PersonaPlex depth normalization parameters are true shared aliases.
    SharedDepthNorms,
}

impl ParameterSharing {
    const fn as_str(self) -> &'static str {
        match self {
            Self::IndependentDepthSlices => "independent_depth_slices",
            Self::SharedDepthNorms => "shared_depth_norms",
        }
    }
}

/// Stable normalized identity used by capability and realtime admission.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MoshiIdentity {
    family: String,
    effective_model_type: EffectiveModelType,
    artifact_profile: ArtifactProfile,
    version: Option<String>,
    architecture_fingerprint: String,
}

impl MoshiIdentity {
    /// Architecture family; always `moshi` for every admitted profile.
    pub fn family(&self) -> &str {
        &self.family
    }

    /// Effective model type retained separately from the common family.
    pub const fn effective_model_type(&self) -> EffectiveModelType {
        self.effective_model_type
    }

    /// Explicit normalized artifact profile.
    pub const fn artifact_profile(&self) -> ArtifactProfile {
        self.artifact_profile
    }

    /// Released profile version, when one is defined by artifact metadata.
    pub fn version(&self) -> Option<&str> {
        self.version.as_deref()
    }

    /// Stable SHA-256 fingerprint of normalized architecture semantics.
    pub fn architecture_fingerprint(&self) -> &str {
        &self.architecture_fingerprint
    }
}

/// Validated transformer geometry usable directly by the shared decoder.
#[derive(Debug, Clone, PartialEq)]
pub struct MoshiTransformerConfig {
    model_identity: String,
    parameter_root: String,
    hidden_size: i32,
    num_hidden_layers: i32,
    num_attention_heads: i32,
    head_dim: i32,
    feed_forward_size: i32,
    gated_hidden_size: i32,
    context: i32,
    attention_window: i32,
    rope_base: f32,
    rms_norm_epsilon: f32,
    positional_encoding: PositionalEncoding,
    vocabulary_size: i32,
    attention_schedule: LayerSchedule<AttentionPolicy>,
    native_quantization: Option<WeightQuantization>,
    parallel_local: bool,
}

impl MoshiTransformerConfig {
    /// Stable identity of this concrete temporal or depth-slice decoder.
    pub fn model_identity(&self) -> &str {
        &self.model_identity
    }

    /// Canonical parameter root used by the shared decoder.
    pub fn parameter_root(&self) -> &str {
        &self.parameter_root
    }

    /// Stable identity of this decoder's complete normalized policy.
    pub fn architecture_fingerprint(&self) -> String {
        derive_prompt_cache_architecture_fingerprint(
            "moshi_shared_decoder",
            [
                ("model_identity", self.model_identity.clone()),
                ("parameter_root", self.parameter_root.clone()),
                ("block_fields", "self_attn:q_proj:k_proj:v_proj:out_proj:sinks:q_norm:k_norm:gating:gate:up:linear_out:norm1:norm2".into()),
                ("hidden_size", self.hidden_size.to_string()),
                ("layers", self.num_hidden_layers.to_string()),
                ("intermediate_size", self.gated_hidden_size.to_string()),
                ("feed_forward_size", self.feed_forward_size.to_string()),
                ("query_heads", self.num_attention_heads.to_string()),
                ("key_value_heads", self.num_attention_heads.to_string()),
                ("head_dim", self.head_dim.to_string()),
                ("context", self.context.to_string()),
                ("attention_window", self.attention_window.to_string()),
                ("rms_norm_epsilon", f32_fingerprint(self.rms_norm_epsilon)),
                ("vocabulary_size", self.vocabulary_size.to_string()),
                ("attention_biases", "q=false,k=false,v=false,o=false".into()),
                ("attention_projection", "component_major_fused:self_attn.in_proj".into()),
                ("learned_attention_sinks", "false".into()),
                ("query_key_norm", "none".into()),
                ("mlp_bias", "false".into()),
                ("gated_projection", "fused:gating.linear_in".into()),
                ("gated_product_policy", "ordinary_silu".into()),
                ("tied_output", "false".into()),
                (
                    "attention_schedule",
                    self.attention_schedule.fingerprint_component(),
                ),
                (
                    "weight_quantization",
                    quantization_fingerprint(self.native_quantization),
                ),
                ("rotary_base", f32_fingerprint(self.rope_base)),
                ("rotary_traditional", "true".into()),
                ("rotary_max_positions", self.context.to_string()),
                ("rotary_scaling", "none".into()),
                (
                    "rotary_enabled",
                    (self.positional_encoding == PositionalEncoding::Rope).to_string(),
                ),
                ("parallel_local", self.parallel_local.to_string()),
            ],
        )
    }

    /// Transformer residual width.
    pub const fn hidden_size(&self) -> i32 {
        self.hidden_size
    }

    /// Transformer block count.
    pub const fn num_hidden_layers(&self) -> i32 {
        self.num_hidden_layers
    }

    /// Self-attention head count.
    pub const fn num_attention_heads(&self) -> i32 {
        self.num_attention_heads
    }

    /// Width of one self-attention head.
    pub const fn head_dim(&self) -> i32 {
        self.head_dim
    }

    /// Raw feed-forward width from model metadata after defaulting.
    pub const fn feed_forward_size(&self) -> i32 {
        self.feed_forward_size
    }

    /// Exact width of each component in the fused SwiGLU projection.
    pub const fn gated_hidden_size(&self) -> i32 {
        self.gated_hidden_size
    }

    /// Configured transformer context before published window adjustment.
    pub const fn context(&self) -> i32 {
        self.context
    }

    /// Exact positive visible-key window used by every layer.
    pub const fn attention_window(&self) -> i32 {
        self.attention_window
    }

    /// Positive rotary base retained even when a stack has no RoPE equation.
    pub const fn rope_base(&self) -> f32 {
        self.rope_base
    }

    /// Positive RMS normalization epsilon.
    pub const fn rms_norm_epsilon(&self) -> f32 {
        self.rms_norm_epsilon
    }

    /// Normalized positional equation.
    pub const fn positional_encoding(&self) -> PositionalEncoding {
        self.positional_encoding
    }

    /// Output vocabulary size, excluding the input-only padding row.
    pub const fn vocabulary_size(&self) -> i32 {
        self.vocabulary_size
    }

    /// Exact all-layer attention schedule.
    pub fn attention_schedule(&self) -> &LayerSchedule<AttentionPolicy> {
        &self.attention_schedule
    }

    pub(crate) fn with_parallel_geometry(
        &self,
        attention_heads: i32,
        gated_hidden_size: i32,
    ) -> Result<Self, MoshiConfigError> {
        if attention_heads <= 0
            || gated_hidden_size <= 0
            || attention_heads > self.num_attention_heads
        {
            return Err(invalid(format!(
                "invalid rank-local Moshi geometry heads={attention_heads}, gated={gated_hidden_size}"
            )));
        }
        let mut local = self.clone();
        local.num_attention_heads = attention_heads;
        local.gated_hidden_size = gated_hidden_size;
        local.parallel_local = true;
        local.validate()?;
        Ok(local)
    }

    fn with_depth_slice_root(&self, slice: usize) -> Result<Self, MoshiConfigError> {
        let mut value = self.clone();
        value.parameter_root = format!("depformer.slices.{slice}.transformer");
        value.model_identity = format!("moshi.depth.{slice}");
        Ok(value)
    }

    fn validate(&self) -> Result<(), MoshiConfigError> {
        if self.parameter_root.is_empty()
            || self.parameter_root.starts_with('.')
            || self.parameter_root.ends_with('.')
            || self.parameter_root.split('.').any(str::is_empty)
        {
            return Err(invalid(format!(
                "invalid Moshi transformer parameter root {:?}",
                self.parameter_root
            )));
        }
        if self.hidden_size <= 0
            || self.num_hidden_layers <= 0
            || self.num_attention_heads <= 0
            || self.head_dim <= 0
            || self.feed_forward_size <= 0
            || self.gated_hidden_size <= 0
            || self.context <= 0
            || self.attention_window <= 0
            || self.vocabulary_size <= 0
        {
            return Err(invalid("Moshi transformer geometry must be positive"));
        }
        if (!self.parallel_local
            && (self.hidden_size % self.num_attention_heads != 0
                || self.hidden_size / self.num_attention_heads != self.head_dim))
            || (self.parallel_local
                && self
                    .num_attention_heads
                    .checked_mul(self.head_dim)
                    .is_none_or(|width| width > self.hidden_size))
        {
            return Err(invalid(
                "Moshi hidden width must divide exactly across attention heads",
            ));
        }
        if !self.rope_base.is_finite() || self.rope_base <= 0.0 {
            return Err(invalid("Moshi RoPE base must be finite and positive"));
        }
        if !self.rms_norm_epsilon.is_finite() || self.rms_norm_epsilon <= 0.0 {
            return Err(invalid(
                "Moshi RMS normalization epsilon must be finite and positive",
            ));
        }
        if self.attention_schedule.len() != self.num_hidden_layers as usize
            || self.attention_schedule.iter().any(|policy| {
                policy.sliding_window_i32().ok().flatten() != Some(self.attention_window)
            })
        {
            return Err(invalid(
                "Moshi attention schedule does not match normalized layers/window",
            ));
        }
        Ok(())
    }
}

impl DecoderConfig for MoshiTransformerConfig {
    fn model_identity(&self) -> &str {
        &self.model_identity
    }

    fn architecture_fingerprint(&self) -> String {
        MoshiTransformerConfig::architecture_fingerprint(self)
    }

    fn parameter_root(&self) -> &str {
        &self.parameter_root
    }

    fn block_parameter_fields(&self) -> BlockParameterFields<'_> {
        BlockParameterFields {
            attention: "self_attn",
            attention_query: "q_proj",
            attention_key: "k_proj",
            attention_value: "v_proj",
            attention_output: "out_proj",
            attention_sinks: "sinks",
            attention_query_norm: "q_norm",
            attention_key_norm: "k_norm",
            feed_forward: "gating",
            feed_forward_gate: "gate",
            feed_forward_up: "up",
            feed_forward_output: "linear_out",
            input_norm: "norm1",
            post_attention_norm: "norm2",
        }
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
        self.gated_hidden_size
    }

    fn num_attention_heads(&self) -> i32 {
        self.num_attention_heads
    }

    fn num_key_value_heads(&self) -> i32 {
        self.num_attention_heads
    }

    fn head_dim(&self) -> i32 {
        self.head_dim
    }

    fn rms_norm_epsilon(&self) -> f32 {
        self.rms_norm_epsilon
    }

    fn vocabulary_size(&self) -> i32 {
        self.vocabulary_size
    }

    fn attention_bias(&self, _projection: AttentionProjection) -> bool {
        false
    }

    fn attention_projection_layout(&self) -> AttentionProjectionLayout<'_> {
        AttentionProjectionLayout::Fused { field: "in_proj" }
    }

    fn mlp_bias(&self) -> bool {
        false
    }

    fn gated_projection_layout(&self) -> GatedProjectionLayout<'_> {
        GatedProjectionLayout::Fused { field: "linear_in" }
    }

    fn tie_word_embeddings(&self) -> bool {
        false
    }

    fn attention_schedule(&self) -> &LayerSchedule<AttentionPolicy> {
        &self.attention_schedule
    }

    fn weight_quantization(&self, _name: &str) -> Option<WeightQuantization> {
        self.native_quantization
    }

    fn rotary_spec(&self, dimensions: i32) -> RotarySpec {
        RotarySpec {
            dimensions,
            base: self.rope_base,
            traditional: true,
            algorithm: eredu_nn::RotaryAlgorithm::Default,
        }
    }

    fn rotary_enabled(&self) -> bool {
        self.positional_encoding == PositionalEncoding::Rope
    }
}

/// One immutable normalized Moshi-family configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct MoshiConfig {
    identity: MoshiIdentity,
    checkpoint_layout: CheckpointLayout,
    frame_schedule: RealtimeSpeechConfig,
    temporal: MoshiTransformerConfig,
    depth_slice_zero: MoshiTransformerConfig,
    text_vocabulary_size: i32,
    audio_vocabulary_size: i32,
    parameter_sharing: ParameterSharing,
    native_quantization: Option<WeightQuantization>,
    parameter_root: String,
}

impl MoshiConfig {
    /// Normalizes absent configuration as original Moshi v0.1 defaults.
    pub fn native_v0_1() -> Result<Self, MoshiConfigError> {
        normalize_native(native_v0_1_source(), ArtifactProfile::NativeV0_1)
    }

    /// Parses an optional JSON string, treating `None` as native v0.1.
    pub fn from_optional_json(config: Option<&str>) -> Result<Self, MoshiConfigError> {
        match config {
            Some(config) => Self::from_json(config),
            None => Self::native_v0_1(),
        }
    }

    /// Parses and strictly normalizes an explicit JSON configuration.
    pub fn from_json(config: &str) -> Result<Self, MoshiConfigError> {
        let value = serde_json::from_str(config)?;
        Self::from_config_value(Some(&value))
    }

    /// Reads and strictly normalizes an explicit JSON configuration.
    pub fn from_reader(reader: impl Read) -> Result<Self, MoshiConfigError> {
        let value = serde_json::from_reader(reader)?;
        Self::from_config_value(Some(&value))
    }

    /// Normalizes a parsed value, treating `None` as native v0.1 defaults.
    pub fn from_config_value(config: Option<&Value>) -> Result<Self, MoshiConfigError> {
        let Some(config) = config else {
            return Self::native_v0_1();
        };
        let object = config
            .as_object()
            .ok_or_else(|| invalid("Moshi config must be a JSON object"))?;
        match object.get("model_type") {
            None | Some(Value::Null) => {
                let source: NativeSource = serde_json::from_value(config.clone())?;
                normalize_native(source, ArtifactProfile::NativeConfig)
            }
            Some(Value::String(value)) if value == "moshi" => {
                let source: NativeSource = serde_json::from_value(config.clone())?;
                normalize_native(source, ArtifactProfile::NativeConfig)
            }
            Some(Value::String(value)) if value == "personaplex" => {
                let source: PersonaPlexSource = serde_json::from_value(config.clone())?;
                normalize_personaplex(source)
            }
            Some(Value::String(value)) => {
                Err(MoshiConfigError::UnsupportedModelType(value.clone()))
            }
            Some(value) => Err(invalid(format!(
                "Moshi model_type must be a string, got {value}"
            ))),
        }
    }

    /// Stable normalized model identity.
    pub fn identity(&self) -> &MoshiIdentity {
        &self.identity
    }

    /// Architecture family; always `moshi`.
    pub fn family(&self) -> &str {
        self.identity.family()
    }

    /// Effective native Moshi or PersonaPlex model type.
    pub const fn effective_model_type(&self) -> EffectiveModelType {
        self.identity.effective_model_type
    }

    /// Explicit normalized artifact profile.
    pub const fn artifact_profile(&self) -> ArtifactProfile {
        self.identity.artifact_profile
    }

    /// Explicit physical checkpoint layout.
    pub const fn checkpoint_layout(&self) -> CheckpointLayout {
        self.checkpoint_layout
    }

    /// Complete validated realtime schedule and frame convention.
    pub fn frame_schedule(&self) -> &RealtimeSpeechConfig {
        &self.frame_schedule
    }

    /// Validated temporal decoder configuration rooted at `transformer`.
    pub fn temporal(&self) -> &MoshiTransformerConfig {
        &self.temporal
    }

    /// Validated depth decoder template rooted at slice zero.
    pub fn depth_template(&self) -> &MoshiTransformerConfig {
        &self.depth_slice_zero
    }

    /// Returns an owned decoder configuration with a canonical per-slice root.
    pub fn depth_transformer(
        &self,
        codebook: usize,
    ) -> Result<MoshiTransformerConfig, MoshiConfigError> {
        if codebook >= self.frame_schedule.depth_audio_codebooks() {
            return Err(invalid(format!(
                "Moshi depth slice {codebook} is outside 0..{}",
                self.frame_schedule.depth_audio_codebooks()
            )));
        }
        self.depth_slice_zero.with_depth_slice_root(codebook)
    }

    /// Text vocabulary excluding the input-only padding row.
    pub const fn text_vocabulary_size(&self) -> i32 {
        self.text_vocabulary_size
    }

    /// Audio vocabulary excluding the input-only padding row.
    pub const fn audio_vocabulary_size(&self) -> i32 {
        self.audio_vocabulary_size
    }

    /// Explicit depth parameter-sharing policy.
    pub const fn parameter_sharing(&self) -> ParameterSharing {
        self.parameter_sharing
    }

    /// Native packed weight encoding declared by artifact metadata.
    pub const fn native_quantization(&self) -> Option<WeightQuantization> {
        self.native_quantization
    }

    /// Returns an execution-target variant with the requested native matrix encoding.
    ///
    /// This does not mutate the source artifact configuration. Callers performing
    /// load-time quantization must continue to use the original configuration for
    /// physical checkpoint planning and use this returned value only to construct
    /// and identify the materialized target model.
    pub fn with_native_quantization(
        &self,
        quantization: Option<WeightQuantization>,
    ) -> Result<Self, MoshiConfigError> {
        if let Some(quantization) = quantization {
            quantization
                .validate()
                .map_err(|error| invalid(format!("invalid native quantization: {error}")))?;
            if quantization.gguf_iquant().is_some() {
                return Err(invalid(
                    "GGUF block quantization is not supported by Moshi SafeTensors",
                ));
            }
        }
        validate_quantization_geometry(quantization, &self.temporal, &self.depth_slice_zero)?;

        let mut target = self.clone();
        target.native_quantization = quantization;
        target.temporal.native_quantization = quantization;
        target.depth_slice_zero.native_quantization = quantization;
        target.identity.architecture_fingerprint = architecture_fingerprint(
            target.identity.effective_model_type,
            target.identity.artifact_profile,
            target.checkpoint_layout,
            target.identity.version.as_deref(),
            &target.frame_schedule,
            &target.temporal,
            &target.depth_slice_zero,
            target.text_vocabulary_size,
            target.audio_vocabulary_size,
            target.parameter_sharing,
            target.native_quantization,
            &target.parameter_root,
        );
        Ok(target)
    }

    /// Canonical family parameter root; empty means the checkpoint root itself.
    pub fn parameter_root(&self) -> &str {
        &self.parameter_root
    }

    /// Stable SHA-256 fingerprint of normalized architecture semantics.
    pub fn architecture_fingerprint(&self) -> &str {
        self.identity.architecture_fingerprint()
    }
}

/// Strict portable Moshi configuration error.
#[derive(Debug, thiserror::Error)]
pub enum MoshiConfigError {
    /// Invalid JSON or unknown/missing strict metadata fields.
    #[error("invalid Moshi configuration JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// Model type does not belong to the admitted Moshi family profiles.
    #[error("unsupported Moshi-family model_type {0:?}")]
    UnsupportedModelType(String),
    /// PersonaPlex metadata does not name the sole admitted release.
    #[error("unsupported PersonaPlex version {0:?}; expected 7b-v1")]
    UnsupportedPersonaPlexVersion(Option<String>),
    /// Invalid normalized geometry or unsupported architecture feature.
    #[error("invalid Moshi architecture: {0}")]
    Invalid(String),
    /// Invalid portable realtime schedule geometry.
    #[error(transparent)]
    Realtime(#[from] RealtimeConfigError),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeSource {
    #[serde(default)]
    model_type: Option<String>,
    dim: i64,
    text_card: i64,
    #[serde(default)]
    existing_text_padding_id: Option<i64>,
    n_q: i64,
    dep_q: i64,
    #[serde(default, alias = "generated_q", alias = "audio_output_codebooks")]
    generated_audio_codebooks: Option<i64>,
    card: i64,
    num_heads: i64,
    num_layers: i64,
    #[serde(default)]
    dim_feedforward: Option<i64>,
    #[serde(default = "default_true")]
    causal: bool,
    #[serde(default = "default_context")]
    context: i64,
    #[serde(default = "default_rope_base")]
    max_period: f64,
    #[serde(default = "default_rms_epsilon", alias = "rms_norm_eps")]
    rms_norm_epsilon: f64,
    positional_embedding: String,
    depformer_dim: i64,
    #[serde(default)]
    depformer_dim_feedforward: Option<i64>,
    depformer_num_heads: i64,
    depformer_num_layers: i64,
    #[serde(default)]
    depformer_context: Option<i64>,
    #[serde(default)]
    depformer_max_period: Option<f64>,
    #[serde(default = "default_rms_epsilon", alias = "depformer_rms_norm_eps")]
    depformer_rms_norm_epsilon: f64,
    depformer_pos_emb: String,
    delays: Vec<i64>,
    #[serde(default)]
    moshi_name: Option<String>,
    #[serde(default)]
    conditioners: BTreeMap<String, Value>,
    #[serde(default)]
    cross_attention: bool,
    #[serde(default)]
    demux_second_stream: bool,
    #[serde(default)]
    depformer_low_rank_embeddings: Option<i64>,
    #[serde(default)]
    extra_heads_num_heads: i64,
    #[serde(default)]
    quantization: Option<WeightQuantization>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct PersonaPlexSource {
    model_type: String,
    version: String,
    #[serde(default)]
    quantization: Option<WeightQuantization>,
}

fn default_true() -> bool {
    true
}

fn default_context() -> i64 {
    3_000
}

fn default_rope_base() -> f64 {
    10_000.0
}

fn default_rms_epsilon() -> f64 {
    RMS_NORM_EPSILON
}

fn native_v0_1_source() -> NativeSource {
    NativeSource {
        model_type: Some("moshi".into()),
        dim: 4_096,
        text_card: 32_000,
        existing_text_padding_id: None,
        n_q: 16,
        dep_q: 8,
        generated_audio_codebooks: None,
        card: 2_048,
        num_heads: 32,
        num_layers: 32,
        dim_feedforward: None,
        causal: true,
        context: 3_000,
        max_period: 10_000.0,
        rms_norm_epsilon: RMS_NORM_EPSILON,
        positional_embedding: "rope".into(),
        depformer_dim: 1_024,
        depformer_dim_feedforward: Some(4_096),
        depformer_num_heads: 16,
        depformer_num_layers: 6,
        depformer_context: Some(8),
        depformer_max_period: Some(10_000.0),
        depformer_rms_norm_epsilon: RMS_NORM_EPSILON,
        depformer_pos_emb: "none".into(),
        delays: released_delays(),
        moshi_name: Some("model.safetensors".into()),
        conditioners: BTreeMap::new(),
        cross_attention: false,
        demux_second_stream: false,
        depformer_low_rank_embeddings: None,
        extra_heads_num_heads: 0,
        quantization: None,
    }
}

fn released_delays() -> Vec<i64> {
    vec![0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
}

fn normalize_personaplex(source: PersonaPlexSource) -> Result<MoshiConfig, MoshiConfigError> {
    if source.model_type != "personaplex" {
        return Err(MoshiConfigError::UnsupportedModelType(source.model_type));
    }
    if source.version != PERSONAPLEX_VERSION {
        return Err(MoshiConfigError::UnsupportedPersonaPlexVersion(Some(
            source.version,
        )));
    }
    let mut values = native_v0_1_source();
    values.model_type = Some("personaplex".into());
    values.existing_text_padding_id = Some(3);
    values.dep_q = 16;
    values.generated_audio_codebooks = Some(8);
    values.dim_feedforward = Some(16_896);
    values.depformer_dim_feedforward = Some(4_224);
    values.quantization = source.quantization;
    normalize(
        values,
        EffectiveModelType::PersonaPlex,
        ArtifactProfile::PersonaPlex7bV1,
        CheckpointLayout::PersonaPlexPytorch,
        RealtimeFrameConvention::AbsoluteDelayedSlots,
        ParameterSharing::SharedDepthNorms,
        Some(PERSONAPLEX_VERSION.into()),
    )
}

fn normalize_native(
    source: NativeSource,
    profile: ArtifactProfile,
) -> Result<MoshiConfig, MoshiConfigError> {
    if let Some(model_type) = source.model_type.as_deref() {
        if model_type != "moshi" {
            return Err(MoshiConfigError::UnsupportedModelType(model_type.into()));
        }
    }
    let version = (profile == ArtifactProfile::NativeV0_1).then(|| NATIVE_VERSION.into());
    normalize(
        source,
        EffectiveModelType::Moshi,
        profile,
        CheckpointLayout::NativeMlx,
        RealtimeFrameConvention::FeedbackAlignedHistory,
        ParameterSharing::IndependentDepthSlices,
        version,
    )
}

#[allow(clippy::too_many_arguments)]
fn normalize(
    source: NativeSource,
    effective_model_type: EffectiveModelType,
    artifact_profile: ArtifactProfile,
    checkpoint_layout: CheckpointLayout,
    frame_convention: RealtimeFrameConvention,
    parameter_sharing: ParameterSharing,
    version: Option<String>,
) -> Result<MoshiConfig, MoshiConfigError> {
    validate_supported_features(&source)?;

    let text_vocabulary_size = positive_i32("text_card", source.text_card)?;
    let audio_vocabulary_size = positive_i32("card", source.card)?;
    text_vocabulary_size
        .checked_add(1)
        .ok_or_else(|| invalid("text embedding vocabulary plus padding row overflowed"))?;
    audio_vocabulary_size
        .checked_add(1)
        .ok_or_else(|| invalid("audio embedding vocabulary plus padding row overflowed"))?;

    let total_codebooks = usize::try_from(positive_i32("n_q", source.n_q)?)
        .map_err(|_| invalid("n_q exceeds usize"))?;
    let depth_codebooks = usize::try_from(positive_i32("dep_q", source.dep_q)?)
        .map_err(|_| invalid("dep_q exceeds usize"))?;
    let generated_codebooks = usize::try_from(positive_i32(
        "generated_audio_codebooks",
        source.generated_audio_codebooks.unwrap_or(source.dep_q),
    )?)
    .map_err(|_| invalid("generated_audio_codebooks exceeds usize"))?;
    if generated_codebooks > depth_codebooks || depth_codebooks > total_codebooks {
        return Err(invalid(format!(
            "codebooks must satisfy 0 < generated <= dep_q <= n_q, got generated={generated_codebooks} dep_q={depth_codebooks} n_q={total_codebooks}"
        )));
    }
    let input_codebooks = total_codebooks
        .checked_sub(generated_codebooks)
        .ok_or_else(|| invalid("input/generated audio partition underflowed"))?;

    let text_padding_token = match source.existing_text_padding_id {
        Some(token) if token >= 0 && token <= i64::from(text_vocabulary_size) => token as i32,
        Some(token) => {
            return Err(invalid(format!(
                "existing text padding id must be in 0..={text_vocabulary_size}, got {token}"
            )))
        }
        None => text_vocabulary_size,
    };
    let delays = source
        .delays
        .iter()
        .enumerate()
        .map(|(slot, delay)| {
            usize::try_from(*delay)
                .map_err(|_| invalid(format!("delay at slot {slot} must be non-negative")))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let native_quantization = source.quantization;
    if let Some(quantization) = native_quantization {
        quantization
            .validate()
            .map_err(|error| invalid(format!("invalid native quantization: {error}")))?;
        if quantization.gguf_iquant().is_some() {
            return Err(invalid(
                "GGUF block quantization is not supported by Moshi SafeTensors",
            ));
        }
    }

    let temporal = normalize_transformer(TransformerSource {
        label: "temporal",
        model_identity: "moshi.temporal".into(),
        parameter_root: TEMPORAL_PARAMETER_ROOT.into(),
        hidden_size: source.dim,
        num_layers: source.num_layers,
        num_heads: source.num_heads,
        feed_forward_size: source.dim_feedforward,
        context: source.context,
        attention_window_delta: 1,
        rope_base: source.max_period,
        rms_norm_epsilon: source.rms_norm_epsilon,
        positional_encoding: PositionalEncoding::Rope,
        vocabulary_size: text_vocabulary_size,
        native_quantization,
    })?;
    let depth = normalize_transformer(TransformerSource {
        label: "depth",
        model_identity: "moshi.depth.0".into(),
        parameter_root: DEPTH_SLICE_ZERO_PARAMETER_ROOT.into(),
        hidden_size: source.depformer_dim,
        num_layers: source.depformer_num_layers,
        num_heads: source.depformer_num_heads,
        feed_forward_size: source.depformer_dim_feedforward,
        context: source.depformer_context.unwrap_or(8),
        attention_window_delta: 0,
        rope_base: source.depformer_max_period.unwrap_or(8.0),
        rms_norm_epsilon: source.depformer_rms_norm_epsilon,
        positional_encoding: PositionalEncoding::None,
        vocabulary_size: audio_vocabulary_size,
        native_quantization,
    })?;
    validate_quantization_geometry(native_quantization, &temporal, &depth)?;

    let frame_schedule = RealtimeSpeechConfig::new(
        total_codebooks,
        input_codebooks,
        generated_codebooks,
        depth_codebooks,
        text_padding_token,
        audio_vocabulary_size,
        frame_convention,
        delays,
    )?;
    let parameter_root = ROOT_PARAMETER_NAMESPACE.to_owned();
    let architecture_fingerprint = architecture_fingerprint(
        effective_model_type,
        artifact_profile,
        checkpoint_layout,
        version.as_deref(),
        &frame_schedule,
        &temporal,
        &depth,
        text_vocabulary_size,
        audio_vocabulary_size,
        parameter_sharing,
        native_quantization,
        &parameter_root,
    );
    let identity = MoshiIdentity {
        family: MOSHI_FAMILY.into(),
        effective_model_type,
        artifact_profile,
        version,
        architecture_fingerprint,
    };
    Ok(MoshiConfig {
        identity,
        checkpoint_layout,
        frame_schedule,
        temporal,
        depth_slice_zero: depth,
        text_vocabulary_size,
        audio_vocabulary_size,
        parameter_sharing,
        native_quantization,
        parameter_root,
    })
}

fn validate_supported_features(source: &NativeSource) -> Result<(), MoshiConfigError> {
    if !source.causal {
        return Err(invalid("non-causal temporal attention is unsupported"));
    }
    if source.positional_embedding != "rope" {
        return Err(invalid(format!(
            "temporal positional embedding must be rope, got {:?}",
            source.positional_embedding
        )));
    }
    if source.depformer_pos_emb != "none" {
        return Err(invalid(format!(
            "depth positional embedding must be none, got {:?}",
            source.depformer_pos_emb
        )));
    }
    if source.cross_attention {
        return Err(invalid("cross attention is unsupported"));
    }
    if !source.conditioners.is_empty() {
        return Err(invalid("conditioners are unsupported"));
    }
    if source.demux_second_stream {
        return Err(invalid("a demultiplexed second text stream is unsupported"));
    }
    if source.depformer_low_rank_embeddings.is_some() {
        return Err(invalid("low-rank depth embeddings are unsupported"));
    }
    if source.extra_heads_num_heads != 0 {
        return Err(invalid("extra output heads are unsupported"));
    }
    let _artifact_filename_is_resolution_only = &source.moshi_name;
    Ok(())
}

struct TransformerSource {
    label: &'static str,
    model_identity: String,
    parameter_root: String,
    hidden_size: i64,
    num_layers: i64,
    num_heads: i64,
    feed_forward_size: Option<i64>,
    context: i64,
    attention_window_delta: i64,
    rope_base: f64,
    rms_norm_epsilon: f64,
    positional_encoding: PositionalEncoding,
    vocabulary_size: i32,
    native_quantization: Option<WeightQuantization>,
}

fn normalize_transformer(
    source: TransformerSource,
) -> Result<MoshiTransformerConfig, MoshiConfigError> {
    let hidden_size = positive_i32(&format!("{} hidden size", source.label), source.hidden_size)?;
    if source.num_layers > MAX_PORTABLE_LAYERS {
        return Err(invalid(format!(
            "{} layer count {} exceeds portable maximum {MAX_PORTABLE_LAYERS}",
            source.label, source.num_layers
        )));
    }
    let num_hidden_layers =
        positive_i32(&format!("{} layer count", source.label), source.num_layers)?;
    let num_attention_heads =
        positive_i32(&format!("{} head count", source.label), source.num_heads)?;
    if hidden_size % num_attention_heads != 0 {
        return Err(invalid(format!(
            "{} hidden size {hidden_size} is not divisible by {num_attention_heads} heads",
            source.label
        )));
    }
    let head_dim = hidden_size / num_attention_heads;
    let default_feed_forward = hidden_size.checked_mul(4).ok_or_else(|| {
        invalid(format!(
            "{} default feed-forward width overflowed",
            source.label
        ))
    })?;
    let feed_forward_size = match source.feed_forward_size {
        Some(value) => positive_i32(&format!("{} feed-forward width", source.label), value)?,
        None => default_feed_forward,
    };
    let gated_hidden_size = if feed_forward_size == default_feed_forward {
        exact_product_division(source.label, hidden_size, 11, 4, "11 * dim / 4")?
    } else {
        exact_product_division(
            source.label,
            feed_forward_size,
            2,
            3,
            "2 * feed_forward / 3",
        )?
    };
    let context = positive_i32(&format!("{} context", source.label), source.context)?;
    let attention_window = i64::from(context)
        .checked_add(source.attention_window_delta)
        .ok_or_else(|| invalid(format!("{} attention window overflowed", source.label)))?;
    let attention_window = positive_i32(
        &format!("{} attention window", source.label),
        attention_window,
    )?;
    let rope_base = finite_positive_f32(&format!("{} RoPE base", source.label), source.rope_base)?;
    let rms_norm_epsilon = finite_positive_f32(
        &format!("{} RMS normalization epsilon", source.label),
        source.rms_norm_epsilon,
    )?;
    let layers = usize::try_from(num_hidden_layers)
        .map_err(|_| invalid(format!("{} layer count exceeds usize", source.label)))?;
    let window = u32::try_from(attention_window)
        .map_err(|_| invalid(format!("{} attention window exceeds u32", source.label)))?;
    let attention_schedule = LayerSchedule::all_sliding(layers, window).map_err(|error| {
        invalid(format!(
            "invalid {} attention schedule: {error}",
            source.label
        ))
    })?;
    let value = MoshiTransformerConfig {
        model_identity: source.model_identity,
        parameter_root: source.parameter_root,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        head_dim,
        feed_forward_size,
        gated_hidden_size,
        context,
        attention_window,
        rope_base,
        rms_norm_epsilon,
        positional_encoding: source.positional_encoding,
        vocabulary_size: source.vocabulary_size,
        attention_schedule,
        native_quantization: source.native_quantization,
        parallel_local: false,
    };
    value.validate()?;
    Ok(value)
}

fn exact_product_division(
    label: &str,
    value: i32,
    multiplier: i32,
    divisor: i32,
    equation: &str,
) -> Result<i32, MoshiConfigError> {
    let numerator = value
        .checked_mul(multiplier)
        .ok_or_else(|| invalid(format!("{label} gated width {equation} overflowed")))?;
    if numerator % divisor != 0 {
        return Err(invalid(format!(
            "{label} gated width requires exact {equation}, got numerator {numerator}"
        )));
    }
    let result = numerator / divisor;
    if result <= 0 {
        return Err(invalid(format!("{label} gated width must be positive")));
    }
    Ok(result)
}

fn validate_quantization_geometry(
    quantization: Option<WeightQuantization>,
    temporal: &MoshiTransformerConfig,
    depth: &MoshiTransformerConfig,
) -> Result<(), MoshiConfigError> {
    let Some(quantization) = quantization else {
        return Ok(());
    };
    let group = quantization.group_size();
    let bits = quantization.bits();
    if group <= 0 || bits <= 0 {
        return Err(invalid(
            "native quantization group and bits must be positive",
        ));
    }
    for (matrix_family, input_width) in [
        ("temporal hidden-input matrices", temporal.hidden_size),
        ("temporal gated-output matrices", temporal.gated_hidden_size),
        ("depth hidden-input matrices", depth.hidden_size),
        ("depth gated-output matrices", depth.gated_hidden_size),
    ] {
        if input_width % group != 0 {
            return Err(invalid(format!(
                "{matrix_family} width {input_width} is not divisible by quantization group {group}"
            )));
        }
        if input_width % 32 != 0 {
            return Err(invalid(format!(
                "{matrix_family} width {input_width} is not divisible by packed word width 32"
            )));
        }
        let packed_bits = input_width.checked_mul(bits).ok_or_else(|| {
            invalid(format!(
                "{matrix_family} packed quantization width overflowed"
            ))
        })?;
        if packed_bits % 32 != 0 {
            return Err(invalid(format!(
                "{matrix_family} width {input_width} at {bits} bits does not form whole packed words"
            )));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn architecture_fingerprint(
    effective_model_type: EffectiveModelType,
    artifact_profile: ArtifactProfile,
    checkpoint_layout: CheckpointLayout,
    version: Option<&str>,
    frame: &RealtimeSpeechConfig,
    temporal: &MoshiTransformerConfig,
    depth: &MoshiTransformerConfig,
    text_vocabulary_size: i32,
    audio_vocabulary_size: i32,
    parameter_sharing: ParameterSharing,
    native_quantization: Option<WeightQuantization>,
    parameter_root: &str,
) -> String {
    let convention = match frame.frame_convention() {
        RealtimeFrameConvention::FeedbackAlignedHistory => "feedback_aligned_history",
        RealtimeFrameConvention::AbsoluteDelayedSlots => "absolute_delayed_slots",
    };
    let delays = frame
        .delays()
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    derive_prompt_cache_architecture_fingerprint(
        MOSHI_FAMILY,
        [
            ("family", MOSHI_FAMILY.into()),
            ("effective_model_type", effective_model_type.as_str().into()),
            ("artifact_profile", artifact_profile.as_str().into()),
            ("profile_version", version.unwrap_or("none").into()),
            ("checkpoint_layout", checkpoint_layout.as_str().into()),
            ("parameter_root", parameter_root.into()),
            ("temporal.parameter_root", temporal.parameter_root.clone()),
            ("temporal.hidden", temporal.hidden_size.to_string()),
            ("temporal.layers", temporal.num_hidden_layers.to_string()),
            ("temporal.heads", temporal.num_attention_heads.to_string()),
            ("temporal.head_dim", temporal.head_dim.to_string()),
            (
                "temporal.feed_forward",
                temporal.feed_forward_size.to_string(),
            ),
            (
                "temporal.gated_hidden",
                temporal.gated_hidden_size.to_string(),
            ),
            ("temporal.context", temporal.context.to_string()),
            ("temporal.window", temporal.attention_window.to_string()),
            ("temporal.rope_base", f32_fingerprint(temporal.rope_base)),
            (
                "temporal.rms_epsilon",
                f32_fingerprint(temporal.rms_norm_epsilon),
            ),
            (
                "temporal.position",
                temporal.positional_encoding.as_str().into(),
            ),
            (
                "temporal.qkv",
                "component_major_fused:self_attn.in_proj".into(),
            ),
            ("temporal.mlp", "silu_gate_times_up:gating.linear_in".into()),
            ("temporal.causal", "true".into()),
            ("temporal.bias", "false".into()),
            ("temporal.text_head", "untied_unbiased".into()),
            (
                "depth.parameter_root",
                "depformer.slices.{codebook}.transformer".into(),
            ),
            ("depth.hidden", depth.hidden_size.to_string()),
            ("depth.layers", depth.num_hidden_layers.to_string()),
            ("depth.heads", depth.num_attention_heads.to_string()),
            ("depth.head_dim", depth.head_dim.to_string()),
            ("depth.feed_forward", depth.feed_forward_size.to_string()),
            ("depth.gated_hidden", depth.gated_hidden_size.to_string()),
            ("depth.context", depth.context.to_string()),
            ("depth.window", depth.attention_window.to_string()),
            ("depth.rope_base", f32_fingerprint(depth.rope_base)),
            ("depth.rms_epsilon", f32_fingerprint(depth.rms_norm_epsilon)),
            ("depth.position", depth.positional_encoding.as_str().into()),
            (
                "depth.qkv",
                "component_major_fused:self_attn.in_proj".into(),
            ),
            ("depth.mlp", "silu_gate_times_up:gating.linear_in".into()),
            ("depth.bias", "false".into()),
            ("text_vocabulary", text_vocabulary_size.to_string()),
            ("audio_vocabulary", audio_vocabulary_size.to_string()),
            ("total_codebooks", frame.total_audio_codebooks().to_string()),
            ("input_codebooks", frame.input_audio_codebooks().to_string()),
            (
                "generated_codebooks",
                frame.generated_audio_codebooks().to_string(),
            ),
            ("depth_codebooks", frame.depth_audio_codebooks().to_string()),
            ("text_padding", frame.text_padding_token().to_string()),
            ("audio_padding", frame.audio_padding_token().to_string()),
            ("frame_convention", convention.into()),
            ("delays", delays),
            ("parameter_sharing", parameter_sharing.as_str().into()),
            (
                "native_quantization",
                quantization_fingerprint(native_quantization),
            ),
        ],
    )
}

fn quantization_fingerprint(value: Option<WeightQuantization>) -> String {
    match value {
        None => "dense".into(),
        Some(WeightQuantization::Affine(value)) => format!(
            "affine:group={}:bits={}:mode={:?}",
            value.group_size, value.bits, value.mode
        ),
        Some(WeightQuantization::MxFp4) => "mxfp4:group=32:bits=4".into(),
        Some(WeightQuantization::GgufIQuant { .. }) => "unsupported_gguf".into(),
    }
}

fn f32_fingerprint(value: f32) -> String {
    format!("{:08x}", value.to_bits())
}

fn positive_i32(name: &str, value: i64) -> Result<i32, MoshiConfigError> {
    if value <= 0 {
        return Err(invalid(format!("{name} must be positive, got {value}")));
    }
    i32::try_from(value).map_err(|_| invalid(format!("{name} {value} exceeds i32")))
}

fn finite_positive_f32(name: &str, value: f64) -> Result<f32, MoshiConfigError> {
    if !value.is_finite() || value <= 0.0 || value > f64::from(f32::MAX) {
        return Err(invalid(format!(
            "{name} must be finite and positive, got {value}"
        )));
    }
    let value = value as f32;
    if !value.is_finite() || value <= 0.0 {
        return Err(invalid(format!(
            "{name} is not representable as positive f32"
        )));
    }
    Ok(value)
}

fn invalid(message: impl Into<String>) -> MoshiConfigError {
    MoshiConfigError::Invalid(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_checkpoint::AffineQuantization;
    use serde_json::{json, Map};

    fn native_json() -> Value {
        json!({
            "model_type": "moshi",
            "dim": 32,
            "text_card": 101,
            "n_q": 4,
            "dep_q": 3,
            "generated_audio_codebooks": 2,
            "card": 64,
            "num_heads": 4,
            "num_layers": 2,
            "dim_feedforward": 48,
            "causal": true,
            "context": 7,
            "max_period": 10000.0,
            "positional_embedding": "rope",
            "depformer_dim": 24,
            "depformer_dim_feedforward": 36,
            "depformer_num_heads": 4,
            "depformer_num_layers": 2,
            "depformer_context": 3,
            "depformer_max_period": 10000.0,
            "depformer_pos_emb": "none",
            "delays": [0, 0, 1, 2, 1]
        })
    }

    fn normalize_value(value: &Value) -> Result<MoshiConfig, MoshiConfigError> {
        MoshiConfig::from_config_value(Some(value))
    }

    fn set(value: &mut Value, key: &str, replacement: Value) {
        value
            .as_object_mut()
            .expect("object")
            .insert(key.into(), replacement);
    }

    #[test]
    fn absent_config_normalizes_native_v0_1_exactly() {
        let config = MoshiConfig::from_optional_json(None).unwrap();
        assert_eq!(config.family(), MOSHI_FAMILY);
        assert_eq!(config.effective_model_type(), EffectiveModelType::Moshi);
        assert_eq!(config.artifact_profile(), ArtifactProfile::NativeV0_1);
        assert_eq!(config.checkpoint_layout(), CheckpointLayout::NativeMlx);
        assert_eq!(config.identity().version(), Some("0.1"));
        assert_eq!(
            config.frame_schedule().frame_convention(),
            RealtimeFrameConvention::FeedbackAlignedHistory
        );
        assert_eq!(config.frame_schedule().total_audio_codebooks(), 16);
        assert_eq!(config.frame_schedule().input_audio_codebooks(), 8);
        assert_eq!(config.frame_schedule().generated_audio_codebooks(), 8);
        assert_eq!(config.frame_schedule().depth_audio_codebooks(), 8);
        assert_eq!(config.frame_schedule().text_padding_token(), 32_000);
        assert_eq!(config.frame_schedule().audio_padding_token(), 2_048);
        assert_eq!(
            config.frame_schedule().delays(),
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
        );
        assert_eq!(config.temporal().gated_hidden_size(), 11_264);
        assert_eq!(config.temporal().attention_window(), 3_001);
        assert_eq!(config.depth_template().gated_hidden_size(), 2_816);
        assert_eq!(config.depth_template().attention_window(), 8);
        assert_eq!(
            config.parameter_sharing(),
            ParameterSharing::IndependentDepthSlices
        );
        assert!(config.architecture_fingerprint().starts_with("sha256:"));
        assert_eq!(config.architecture_fingerprint().len(), 71);
        assert_eq!(
            config.architecture_fingerprint(),
            MoshiConfig::native_v0_1()
                .unwrap()
                .architecture_fingerprint()
        );
    }

    #[test]
    fn explicit_native_accepts_present_or_early_absent_model_type() {
        let explicit = normalize_value(&native_json()).unwrap();
        let mut early = native_json();
        early.as_object_mut().unwrap().remove("model_type");
        let early = normalize_value(&early).unwrap();
        for config in [&explicit, &early] {
            assert_eq!(config.artifact_profile(), ArtifactProfile::NativeConfig);
            assert_eq!(config.identity().version(), None);
            assert_eq!(config.temporal().feed_forward_size(), 48);
            assert_eq!(config.temporal().gated_hidden_size(), 32);
            assert_eq!(config.depth_template().gated_hidden_size(), 24);
            assert_eq!(config.frame_schedule().input_audio_codebooks(), 2);
            assert_eq!(config.frame_schedule().generated_audio_codebooks(), 2);
        }
        assert_eq!(
            explicit.architecture_fingerprint(),
            early.architecture_fingerprint()
        );
    }

    #[test]
    fn personaplex_metadata_is_strict_and_selects_explicit_policy() {
        let config =
            MoshiConfig::from_json(r#"{"model_type":"personaplex","version":"7b-v1"}"#).unwrap();
        assert_eq!(
            config.effective_model_type(),
            EffectiveModelType::PersonaPlex
        );
        assert_eq!(config.artifact_profile(), ArtifactProfile::PersonaPlex7bV1);
        assert_eq!(
            config.checkpoint_layout(),
            CheckpointLayout::PersonaPlexPytorch
        );
        assert_eq!(config.identity().version(), Some("7b-v1"));
        assert_eq!(
            config.frame_schedule().frame_convention(),
            RealtimeFrameConvention::AbsoluteDelayedSlots
        );
        assert_eq!(config.frame_schedule().text_padding_token(), 3);
        assert_eq!(config.frame_schedule().depth_audio_codebooks(), 16);
        assert_eq!(config.frame_schedule().generated_audio_codebooks(), 8);
        assert_eq!(config.temporal().feed_forward_size(), 16_896);
        assert_eq!(config.temporal().gated_hidden_size(), 11_264);
        assert_eq!(config.depth_template().feed_forward_size(), 4_224);
        assert_eq!(config.depth_template().gated_hidden_size(), 2_816);
        assert_eq!(
            config.parameter_sharing(),
            ParameterSharing::SharedDepthNorms
        );
    }

    #[test]
    fn persona_version_and_metadata_surface_are_strict() {
        assert!(matches!(
            MoshiConfig::from_json(r#"{"model_type":"personaplex","version":"8b-v2"}"#),
            Err(MoshiConfigError::UnsupportedPersonaPlexVersion(_))
        ));
        assert!(matches!(
            MoshiConfig::from_json(r#"{"model_type":"personaplex"}"#),
            Err(MoshiConfigError::Json(_))
        ));
        assert!(matches!(
            MoshiConfig::from_json(
                r#"{"model_type":"personaplex","version":"7b-v1","filename":"x"}"#
            ),
            Err(MoshiConfigError::Json(_))
        ));
        assert!(matches!(
            MoshiConfig::from_json(r#"{"model_type":"other"}"#),
            Err(MoshiConfigError::UnsupportedModelType(_))
        ));
    }

    #[test]
    fn transformer_configs_implement_shared_decoder_with_fused_layouts() {
        let config = MoshiConfig::native_v0_1().unwrap();
        let temporal: &dyn DecoderConfig = config.temporal();
        assert_eq!(temporal.parameter_root(), "transformer");
        assert_eq!(
            temporal.block_parameter_fields(),
            BlockParameterFields {
                attention: "self_attn",
                attention_query: "q_proj",
                attention_key: "k_proj",
                attention_value: "v_proj",
                attention_output: "out_proj",
                attention_sinks: "sinks",
                attention_query_norm: "q_norm",
                attention_key_norm: "k_norm",
                feed_forward: "gating",
                feed_forward_gate: "gate",
                feed_forward_up: "up",
                feed_forward_output: "linear_out",
                input_norm: "norm1",
                post_attention_norm: "norm2",
            }
        );
        assert_eq!(
            temporal.attention_projection_layout(),
            AttentionProjectionLayout::Fused { field: "in_proj" }
        );
        assert_eq!(
            temporal.gated_projection_layout(),
            GatedProjectionLayout::Fused { field: "linear_in" }
        );
        assert_eq!(temporal.intermediate_size(), 11_264);
        assert!(temporal.validate_config().is_ok());
        let slice = config.depth_transformer(7).unwrap();
        assert_eq!(slice.parameter_root(), "depformer.slices.7.transformer");
        assert_eq!(slice.model_identity(), "moshi.depth.7");
        assert_eq!(slice.intermediate_size(), 2_816);
        assert!(config.depth_transformer(8).is_err());
    }

    #[test]
    fn shared_decoder_fingerprint_covers_rotary_enablement() {
        let config = MoshiConfig::native_v0_1().unwrap();
        let enabled = DecoderConfig::architecture_fingerprint(config.temporal());
        let mut disabled = config.temporal().clone();
        disabled.positional_encoding = PositionalEncoding::None;

        assert_ne!(enabled, DecoderConfig::architecture_fingerprint(&disabled));
        assert!(enabled.starts_with("sha256:"));
    }

    #[test]
    fn default_and_explicit_feed_forward_rules_are_exact() {
        let mut value = native_json();
        value.as_object_mut().unwrap().remove("dim_feedforward");
        set(&mut value, "dim", json!(32));
        assert_eq!(
            normalize_value(&value)
                .unwrap()
                .temporal()
                .gated_hidden_size(),
            88
        );

        let mut invalid_default = native_json();
        invalid_default
            .as_object_mut()
            .unwrap()
            .remove("dim_feedforward");
        set(&mut invalid_default, "dim", json!(30));
        set(&mut invalid_default, "num_heads", json!(5));
        assert!(normalize_value(&invalid_default).is_err());

        let mut invalid_explicit = native_json();
        set(&mut invalid_explicit, "dim_feedforward", json!(49));
        assert!(normalize_value(&invalid_explicit).is_err());
    }

    #[test]
    fn malformed_positive_divisibility_and_overflow_geometry_is_rejected() {
        for (key, replacement) in [
            ("dim", json!(0)),
            ("text_card", json!(-1)),
            ("n_q", json!(0)),
            ("dep_q", json!(0)),
            ("card", json!(0)),
            ("num_heads", json!(0)),
            ("num_layers", json!(0)),
            ("context", json!(0)),
            ("depformer_dim", json!(0)),
            ("depformer_num_heads", json!(0)),
            ("depformer_num_layers", json!(0)),
            ("depformer_context", json!(0)),
        ] {
            let mut value = native_json();
            set(&mut value, key, replacement);
            assert!(normalize_value(&value).is_err(), "accepted invalid {key}");
        }
        let mut heads = native_json();
        set(&mut heads, "num_heads", json!(3));
        assert!(normalize_value(&heads).is_err());

        let mut overflow = native_json();
        set(&mut overflow, "dim", json!(i32::MAX));
        set(&mut overflow, "num_heads", json!(1));
        overflow.as_object_mut().unwrap().remove("dim_feedforward");
        assert!(normalize_value(&overflow).is_err());

        let mut vocabulary_overflow = native_json();
        set(&mut vocabulary_overflow, "text_card", json!(i32::MAX));
        assert!(normalize_value(&vocabulary_overflow).is_err());

        let mut codebook_overflow = native_json();
        set(
            &mut codebook_overflow,
            "n_q",
            json!(i64::from(i32::MAX) + 1),
        );
        assert!(normalize_value(&codebook_overflow).is_err());

        let mut window_overflow = native_json();
        set(&mut window_overflow, "context", json!(i32::MAX));
        assert!(normalize_value(&window_overflow).is_err());

        let mut layers = native_json();
        set(&mut layers, "num_layers", json!(MAX_PORTABLE_LAYERS + 1));
        assert!(normalize_value(&layers).is_err());
    }

    #[test]
    fn periods_epsilon_and_windows_must_be_finite_positive() {
        for (key, replacement) in [
            ("max_period", json!(0.0)),
            ("max_period", json!(-1.0)),
            ("depformer_max_period", json!(0.0)),
            ("rms_norm_epsilon", json!(0.0)),
            ("depformer_rms_norm_epsilon", json!(-1.0)),
        ] {
            let mut value = native_json();
            set(&mut value, key, replacement);
            assert!(normalize_value(&value).is_err(), "accepted invalid {key}");
        }
        assert!(MoshiConfig::from_json(
            &native_json()
                .to_string()
                .replace("\"max_period\":10000.0", "\"max_period\":1e400")
        )
        .is_err());
    }

    #[test]
    fn codebook_partition_padding_and_delays_are_strict() {
        for (key, replacement) in [
            ("generated_audio_codebooks", json!(0)),
            ("generated_audio_codebooks", json!(4)),
            ("dep_q", json!(5)),
            ("existing_text_padding_id", json!(-1)),
            ("existing_text_padding_id", json!(102)),
            ("delays", json!([0, 1])),
            ("delays", json!([0, 0, -1, 2, 1])),
            ("delays", json!([0, 0, 1, 2, 2147483648_i64])),
        ] {
            let mut value = native_json();
            set(&mut value, key, replacement);
            assert!(normalize_value(&value).is_err(), "accepted invalid {key}");
        }
    }

    #[test]
    fn every_unsupported_token_only_feature_is_rejected() {
        for (key, replacement) in [
            ("causal", json!(false)),
            ("positional_embedding", json!("none")),
            ("depformer_pos_emb", json!("rope")),
            ("cross_attention", json!(true)),
            ("conditioners", json!({"voice": {}})),
            ("demux_second_stream", json!(true)),
            ("depformer_low_rank_embeddings", json!(16)),
            ("extra_heads_num_heads", json!(1)),
        ] {
            let mut value = native_json();
            set(&mut value, key, replacement);
            assert!(
                normalize_value(&value).is_err(),
                "accepted unsupported {key}"
            );
        }
        let mut unknown = native_json();
        set(&mut unknown, "new_feature", json!(true));
        assert!(matches!(
            normalize_value(&unknown),
            Err(MoshiConfigError::Json(_))
        ));
    }

    #[test]
    fn native_quantization_is_validated_against_every_input_width() {
        let config = MoshiConfig::from_json(
            r#"{"model_type":"personaplex","version":"7b-v1","quantization":{"group_size":32,"bits":4,"mode":"affine"}}"#,
        )
        .unwrap();
        assert!(matches!(
            config.native_quantization(),
            Some(WeightQuantization::Affine(_))
        ));

        let mut invalid = native_json();
        set(
            &mut invalid,
            "quantization",
            json!({"group_size": 64, "bits": 4, "mode": "affine"}),
        );
        assert!(normalize_value(&invalid).is_err());

        let mut invalid_mode = native_json();
        set(
            &mut invalid_mode,
            "quantization",
            json!({"group_size": 32, "bits": 4, "mode": "other"}),
        );
        assert!(matches!(
            normalize_value(&invalid_mode),
            Err(MoshiConfigError::Json(_))
        ));
    }

    #[test]
    fn execution_quantization_variant_preserves_source_and_physical_semantics() {
        let mut value = native_json();
        set(&mut value, "depformer_dim", json!(32));
        set(&mut value, "depformer_dim_feedforward", json!(48));
        let source = normalize_value(&value).unwrap();
        let source_fingerprint = source.architecture_fingerprint().to_owned();
        let target = source
            .with_native_quantization(Some(WeightQuantization::Affine(
                AffineQuantization::new(16, 4).unwrap(),
            )))
            .unwrap();

        assert_eq!(source.native_quantization(), None);
        assert_eq!(source.architecture_fingerprint(), source_fingerprint);
        assert_ne!(target.architecture_fingerprint(), source_fingerprint);
        assert_eq!(target.checkpoint_layout(), source.checkpoint_layout());
        assert_eq!(target.artifact_profile(), source.artifact_profile());
        assert_eq!(target.frame_schedule(), source.frame_schedule());
        assert_eq!(target.parameter_sharing(), source.parameter_sharing());
        assert_eq!(
            target.temporal().attention_schedule(),
            source.temporal().attention_schedule()
        );
        assert_eq!(
            target.depth_template().attention_schedule(),
            source.depth_template().attention_schedule()
        );
        assert_eq!(
            target.temporal().native_quantization,
            target.native_quantization()
        );
        assert_eq!(
            target.depth_template().native_quantization,
            target.native_quantization()
        );
    }

    #[test]
    fn execution_quantization_variant_rejects_invalid_packing_and_group_geometry() {
        let source = normalize_value(&native_json()).unwrap();
        let invalid_group = WeightQuantization::Affine(AffineQuantization::new(64, 4).unwrap());
        assert!(source
            .with_native_quantization(Some(invalid_group))
            .is_err());

        let mut narrow = native_json();
        set(&mut narrow, "dim", json!(16));
        set(&mut narrow, "dim_feedforward", json!(24));
        set(&mut narrow, "depformer_dim", json!(16));
        set(&mut narrow, "depformer_dim_feedforward", json!(24));
        let narrow = normalize_value(&narrow).unwrap();
        let invalid_packing = WeightQuantization::Affine(AffineQuantization::new(16, 4).unwrap());
        assert!(narrow
            .with_native_quantization(Some(invalid_packing))
            .is_err());
        assert_eq!(source.native_quantization(), None);
    }

    #[test]
    fn fingerprint_covers_profile_schedule_sharing_and_quantization() {
        let native = MoshiConfig::native_v0_1().unwrap();
        let persona =
            MoshiConfig::from_json(r#"{"model_type":"personaplex","version":"7b-v1"}"#).unwrap();
        let quantized = MoshiConfig::from_json(
            r#"{"model_type":"personaplex","version":"7b-v1","quantization":{"group_size":32,"bits":4,"mode":"mxfp4"}}"#,
        )
        .unwrap();
        assert_ne!(
            native.architecture_fingerprint(),
            persona.architecture_fingerprint()
        );
        assert_ne!(
            persona.architecture_fingerprint(),
            quantized.architecture_fingerprint()
        );

        let mut changed_delay = native_json();
        set(&mut changed_delay, "delays", json!([0, 0, 1, 2, 2]));
        let original = normalize_value(&native_json()).unwrap();
        let changed = normalize_value(&changed_delay).unwrap();
        assert_ne!(
            original.architecture_fingerprint(),
            changed.architecture_fingerprint()
        );
    }

    #[test]
    fn explicit_empty_or_non_object_config_is_not_absent_config() {
        assert!(MoshiConfig::from_json("{}").is_err());
        assert!(MoshiConfig::from_json("[]").is_err());
        assert!(MoshiConfig::from_json(r#"{"model_type":3}"#).is_err());
    }

    #[test]
    fn strict_native_surface_still_accepts_artifact_filename_as_resolution_data() {
        let mut value = native_json();
        let object: &mut Map<String, Value> = value.as_object_mut().unwrap();
        object.insert("moshi_name".into(), json!("alternate.safetensors"));
        let with_name = normalize_value(&value).unwrap();
        let without_name = normalize_value(&native_json()).unwrap();
        assert_eq!(
            with_name.architecture_fingerprint(),
            without_name.architecture_fingerprint()
        );
    }
}
