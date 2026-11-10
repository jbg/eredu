//! Validated Inkling text, media, and prediction configuration.

use std::collections::HashMap;

use eredu_checkpoint::{LinearFormat, WeightQuantization};
use eredu_core::{
    cache::derive_prompt_cache_architecture_fingerprint, AttentionPolicy, InputModalities,
    LayerSchedule,
};
use eredu_gguf::{MetadataArray, MetadataValue};
use serde::Deserialize;

/// Invalid or unsupported Inkling configuration.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// JSON decoding failed.
    #[error("invalid Inkling configuration: {0}")]
    Json(#[from] serde_json::Error),
    /// Normalized geometry is unsupported.
    #[error("{0}")]
    Invalid(String),
}

fn invalid(message: impl Into<String>) -> ConfigError {
    ConfigError::Invalid(message.into())
}

/// Feed-forward implementation selected for one layer.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum FeedForwardPolicy {
    /// Dense SwiGLU block.
    Dense,
    /// Routed and shared sparse experts.
    SparseMoe,
}

/// Exact attention and feed-forward policy for one layer.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub struct LayerPolicy {
    /// Full or sliding attention.
    pub attention: AttentionPolicy,
    /// Dense or sparse feed-forward topology.
    pub feed_forward: FeedForwardPolicy,
}

fn default_true() -> bool {
    true
}
fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_head_dim() -> i32 {
    128
}
fn default_sconv_kernel_size() -> i32 {
    4
}
fn default_rel_extent() -> i32 {
    1024
}
fn default_sliding_window() -> i64 {
    512
}
fn default_route_scale() -> f32 {
    1.0
}
fn default_logit_scale() -> f32 {
    1.0
}
fn default_model_type() -> String {
    "inkling_mm_model".into()
}
fn default_gate_activation() -> String {
    "sigmoid".into()
}
fn default_hidden_activation() -> String {
    "silu".into()
}
fn default_audio_mode() -> String {
    "dmel".into()
}
fn default_vision_encoder_type() -> String {
    "hmlp".into()
}
fn default_image_token_id() -> u32 {
    200_054
}
fn default_audio_token_id() -> u32 {
    200_053
}

#[derive(Debug, Clone, Deserialize)]
struct TextSource {
    #[serde(default)]
    torch_dtype: Option<String>,
    hidden_size: i32,
    num_hidden_layers: i32,
    vocab_size: i32,
    num_attention_heads: i32,
    num_key_value_heads: i32,
    #[serde(default = "default_head_dim")]
    head_dim: i32,
    #[serde(default)]
    swa_num_attention_heads: Option<i32>,
    #[serde(default)]
    swa_num_key_value_heads: Option<i32>,
    #[serde(default)]
    swa_head_dim: Option<i32>,
    #[serde(default = "default_sliding_window", alias = "sliding_window")]
    sliding_window_size: i64,
    #[serde(default)]
    local_layer_ids: Option<Vec<i64>>,
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    #[serde(default)]
    dense_mlp_idx: Option<i64>,
    #[serde(default)]
    mlp_layer_types: Option<Vec<String>>,
    #[serde(default = "default_sconv_kernel_size", alias = "conv_kernel_size")]
    sconv_kernel_size: i32,
    #[serde(default = "default_true")]
    use_sconv: bool,
    #[serde(default = "default_rel_extent")]
    rel_extent: i32,
    d_rel: i32,
    #[serde(default)]
    log_scaling_n_floor: Option<i32>,
    #[serde(default)]
    log_scaling_alpha: f32,
    #[serde(default = "default_rms_norm_eps")]
    rms_norm_eps: f32,
    #[serde(default = "default_true")]
    use_embed_norm: bool,
    #[serde(default)]
    unpadded_vocab_size: Option<i32>,
    #[serde(default = "default_logit_scale")]
    logits_mup_width_multiplier: f32,
    #[serde(default)]
    final_logit_softcapping: Option<f32>,
    #[serde(default)]
    intermediate_size: i32,
    #[serde(default)]
    dense_intermediate_size: Option<i32>,
    #[serde(default)]
    moe_intermediate_size: Option<i32>,
    #[serde(default, alias = "num_experts")]
    n_routed_experts: i32,
    #[serde(default)]
    num_experts_per_tok: i32,
    #[serde(default)]
    n_shared_experts: i32,
    #[serde(default = "default_route_scale")]
    route_scale: f32,
    #[serde(default = "default_true")]
    shared_expert_sink: bool,
    #[serde(default = "default_true")]
    use_gate_bias: bool,
    #[serde(default = "default_true")]
    norm_after_topk: bool,
    #[serde(default = "default_true")]
    use_global_scale: bool,
    #[serde(default = "default_gate_activation")]
    gate_activation: String,
    #[serde(default = "default_hidden_activation")]
    hidden_act: String,
    #[serde(default)]
    attention_dropout: f32,
    #[serde(default)]
    q_bias: bool,
    #[serde(default)]
    o_bias: bool,
    #[serde(default)]
    model_max_length: Option<i32>,
    #[serde(default)]
    weight_quantization: Option<WeightQuantization>,
    #[serde(default)]
    quantized_weight_configs: Option<HashMap<String, WeightQuantization>>,
}

/// Validated Inkling text-decoder geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct TextArgs {
    /// Released dense scalar type metadata.
    pub torch_dtype: Option<String>,
    /// Decoder hidden width.
    pub hidden_size: i32,
    /// Decoder layer count.
    pub num_hidden_layers: i32,
    /// Stored vocabulary size.
    pub vocab_size: i32,
    /// Global query heads.
    pub num_attention_heads: i32,
    /// Global K/V heads.
    pub num_key_value_heads: i32,
    /// Global head width.
    pub head_dim: i32,
    /// Optional sliding query heads.
    pub swa_num_attention_heads: Option<i32>,
    /// Optional sliding K/V heads.
    pub swa_num_key_value_heads: Option<i32>,
    /// Optional sliding head width.
    pub swa_head_dim: Option<i32>,
    /// Authoritative layer schedule.
    pub layer_schedule: LayerSchedule<LayerPolicy>,
    /// Fixed short-convolution kernel.
    pub sconv_kernel_size: i32,
    /// Required short-convolution switch.
    pub use_sconv: bool,
    /// Relative position table extent.
    pub rel_extent: i32,
    /// Relative feature width.
    pub d_rel: i32,
    /// Optional attention log-scaling floor.
    pub log_scaling_n_floor: Option<i32>,
    /// Attention log-scaling exponent.
    pub log_scaling_alpha: f32,
    /// Normalization epsilon.
    pub rms_norm_eps: f32,
    /// Required embedding normalization switch.
    pub use_embed_norm: bool,
    /// Protocol-visible vocabulary truncation.
    pub unpadded_vocab_size: Option<i32>,
    /// MuP logit scaling divisor.
    pub logits_mup_width_multiplier: f32,
    /// Optional final soft cap; released Inkling supports only zero/none.
    pub final_logit_softcapping: Option<f32>,
    /// Default intermediate width.
    pub intermediate_size: i32,
    /// Optional dense width override.
    pub dense_intermediate_size: Option<i32>,
    /// Optional routed/shared expert width override.
    pub moe_intermediate_size: Option<i32>,
    /// Routed expert count.
    pub n_routed_experts: i32,
    /// Selected routed experts.
    pub num_experts_per_tok: i32,
    /// Shared expert count.
    pub n_shared_experts: i32,
    /// Route output scale.
    pub route_scale: f32,
    /// Shared-expert sink policy.
    pub shared_expert_sink: bool,
    /// Learned router correction bias switch.
    pub use_gate_bias: bool,
    /// Post-top-k normalization switch.
    pub norm_after_topk: bool,
    /// Learned global route scale switch.
    pub use_global_scale: bool,
    /// Router nonlinearity.
    pub gate_activation: String,
    /// Expert activation.
    pub hidden_act: String,
    /// Inference attention dropout.
    pub attention_dropout: f32,
    /// Query bias switch.
    pub q_bias: bool,
    /// Output bias switch.
    pub o_bias: bool,
    /// Optional declared context length.
    pub model_max_length: Option<i32>,
    /// Uniform physical encoding.
    pub weight_quantization: Option<WeightQuantization>,
    /// Per-weight mixed encodings.
    pub quantized_weight_configs: Option<HashMap<String, WeightQuantization>>,
}

impl TextArgs {
    /// Dense intermediate width selected by normalized policy.
    pub fn dense_intermediate_size(&self) -> i32 {
        self.dense_intermediate_size
            .unwrap_or(self.intermediate_size)
    }
    /// Expert intermediate width selected by normalized policy.
    pub fn moe_intermediate_size(&self) -> i32 {
        self.moe_intermediate_size.unwrap_or(self.intermediate_size)
    }
    /// Returns one exact layer policy.
    pub fn layer_policy(&self, layer: usize) -> Option<LayerPolicy> {
        self.layer_schedule.get(layer).copied()
    }
    /// Returns whether any normalized decoder layer uses routed experts.
    pub fn has_sparse_moe_layers(&self) -> bool {
        self.layer_schedule
            .iter()
            .any(|policy| policy.feed_forward == FeedForwardPolicy::SparseMoe)
    }
    /// Returns local or global query heads.
    pub fn query_heads(&self, local: bool) -> i32 {
        if local {
            self.swa_num_attention_heads
                .unwrap_or(self.num_attention_heads)
        } else {
            self.num_attention_heads
        }
    }
    /// Returns local or global K/V heads.
    pub fn key_value_heads(&self, local: bool) -> i32 {
        if local {
            self.swa_num_key_value_heads
                .unwrap_or(self.num_key_value_heads)
        } else {
            self.num_key_value_heads
        }
    }
    /// Returns local or global head width.
    pub fn attention_head_dim(&self, local: bool) -> i32 {
        if local {
            self.swa_head_dim.unwrap_or(self.head_dim)
        } else {
            self.head_dim
        }
    }
    /// Returns one weight's normalized physical format.
    pub fn linear_format_for(&self, name: &str) -> LinearFormat {
        // Short-convolution kernels are rank-3 convolution parameters, not
        // matrix projections. The model-wide load-time linear quantization
        // policy must not change their representation.
        if name.ends_with("_sconv.weight")
            || name.ends_with(".moe.router.weight")
            || name.ends_with(".self_attn.rel_proj")
        {
            return LinearFormat::Dense;
        }
        self.quantized_weight_configs
            .as_ref()
            .and_then(|configs| configs.get(name))
            .copied()
            .or(self.weight_quantization)
            .map(Into::into)
            .unwrap_or(LinearFormat::Dense)
    }
}

/// Embedded multi-token prediction geometry and optional overrides.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct MtpConfig {
    /// Prediction depth count.
    #[serde(default)]
    pub num_nextn_predict_layers: i32,
    /// Whether chain hidden values are normalized between depths.
    #[serde(default)]
    pub chain_hidden_post_norm: bool,
    /// Prediction depths using sliding attention.
    #[serde(default)]
    pub local_layer_ids: Vec<usize>,
    /// Global query-head override.
    #[serde(default)]
    pub num_attention_heads: Option<i32>,
    /// Global K/V-head override.
    #[serde(default)]
    pub num_key_value_heads: Option<i32>,
    /// Global head-width override.
    #[serde(default)]
    pub head_dim: Option<i32>,
    /// Local query-head override.
    #[serde(default)]
    pub swa_num_attention_heads: Option<i32>,
    /// Local K/V-head override.
    #[serde(default)]
    pub swa_num_key_value_heads: Option<i32>,
    /// Local head-width override.
    #[serde(default)]
    pub swa_head_dim: Option<i32>,
    /// Dense intermediate override.
    #[serde(default)]
    pub dense_intermediate_size: Option<i32>,
    /// Default intermediate override.
    #[serde(default)]
    pub intermediate_size: Option<i32>,
    /// Short-convolution kernel override.
    #[serde(default)]
    pub sconv_kernel_size: Option<i32>,
    /// Relative table extent override.
    #[serde(default)]
    pub rel_extent: Option<i32>,
    /// Relative feature width override.
    #[serde(default)]
    pub d_rel: Option<i32>,
}

/// Native dMel audio projector configuration.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct AudioPolicy {
    /// Decoder hidden width.
    #[serde(alias = "decoder_dmodel")]
    pub text_hidden_size: i32,
    /// Number of dMel codebooks per frame.
    #[serde(alias = "n_mel_bins")]
    pub num_codebooks: i32,
    /// Codebook vocabulary size.
    #[serde(alias = "mel_vocab_size")]
    pub codebook_size: i32,
    /// Unsupported projection bias switch.
    #[serde(default)]
    pub bias: bool,
    /// Required output normalization switch.
    #[serde(default = "default_true")]
    pub use_audio_norm: bool,
    /// Exact native audio mode.
    #[serde(default = "default_audio_mode")]
    pub audio_mode: String,
    /// Normalization epsilon.
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    /// Per-weight mixed projector encodings.
    #[serde(default)]
    pub quantized_weight_configs: Option<HashMap<String, WeightQuantization>>,
}

/// Validated audio policy with retained typed parameter scopes.
#[derive(Debug, Clone, Deserialize)]
#[serde(try_from = "AudioPolicy")]
pub struct AudioConfig {
    policy: AudioPolicy,
    parameters: std::sync::Arc<super::audio::Parameters>,
}
impl AudioConfig {
    /// Validates normalized source policy and declares parameters once.
    pub fn new(policy: AudioPolicy) -> Result<Self, eredu_nn::Error> {
        let parameters = super::audio::Parameters::new(&policy)?;
        Ok(Self {
            policy,
            parameters: std::sync::Arc::new(parameters),
        })
    }
    /// Normalized source geometry and encoding policy.
    pub fn policy(&self) -> &AudioPolicy {
        &self.policy
    }
    pub(crate) fn parameters(&self) -> &super::audio::Parameters {
        &self.parameters
    }
    /// Projects encoding extensions without changing geometry or execution policy.
    pub fn with_formats(
        &self,
        formats: HashMap<String, WeightQuantization>,
    ) -> Result<Self, eredu_nn::Error> {
        let formats = (!formats.is_empty()).then_some(formats);
        if formats == self.policy.quantized_weight_configs {
            return Ok(self.clone());
        }
        let mut policy = self.policy.clone();
        policy.quantized_weight_configs = formats;
        let mut parameters = self.parameters.as_ref().clone();
        parameters.project_formats(&policy)?;
        Ok(Self {
            policy,
            parameters: std::sync::Arc::new(parameters),
        })
    }
}
impl TryFrom<AudioPolicy> for AudioConfig {
    type Error = eredu_nn::Error;
    fn try_from(policy: AudioPolicy) -> Result<Self, Self::Error> {
        Self::new(policy)
    }
}
impl std::ops::Deref for AudioConfig {
    type Target = AudioPolicy;
    fn deref(&self) -> &Self::Target {
        &self.policy
    }
}
impl eredu_nn::DeclareLogicalParameters for AudioConfig {
    fn logical_parameters(
        &self,
    ) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, eredu_nn::Error> {
        self.parameters.logical_parameters()
    }
}

impl AudioPolicy {
    /// Returns the exact physical format for one dMel weight.
    pub fn linear_format_for(&self, name: &str) -> LinearFormat {
        self.quantized_weight_configs
            .as_ref()
            .and_then(|configs| configs.get(name))
            .copied()
            .map(Into::into)
            .unwrap_or(LinearFormat::Dense)
    }
}

/// Native folded hMLP vision projector configuration.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct VisionPolicy {
    /// Exact native encoder type.
    #[serde(default = "default_vision_encoder_type")]
    pub vision_encoder_type: String,
    /// Decoder hidden width.
    #[serde(alias = "decoder_dmodel")]
    pub text_hidden_size: i32,
    /// Spatial patch edge.
    pub patch_size: i32,
    /// Temporal patch depth.
    pub temporal_patch_size: i32,
    /// Input channel count.
    #[serde(alias = "n_channels")]
    pub num_channels: i32,
    /// hMLP layer count.
    #[serde(alias = "n_layers")]
    pub num_hidden_layers: i32,
    /// Required output normalization switch.
    #[serde(default = "default_true")]
    pub use_vision_norm: bool,
    /// Normalization epsilon.
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    /// Per-weight mixed projector encodings.
    #[serde(default)]
    pub quantized_weight_configs: Option<HashMap<String, WeightQuantization>>,
}

/// Validated vision policy with retained typed parameter scopes.
#[derive(Debug, Clone, Deserialize)]
#[serde(try_from = "VisionPolicy")]
pub struct VisionConfig {
    policy: VisionPolicy,
    parameters: std::sync::Arc<super::vision::Parameters>,
}
impl VisionConfig {
    /// Validates normalized source policy and declares parameters once.
    pub fn new(policy: VisionPolicy) -> Result<Self, eredu_nn::Error> {
        let parameters = super::vision::Parameters::new(&policy)?;
        Ok(Self {
            policy,
            parameters: std::sync::Arc::new(parameters),
        })
    }
    /// Normalized source geometry and encoding policy.
    pub fn policy(&self) -> &VisionPolicy {
        &self.policy
    }
    pub(crate) fn parameters(&self) -> &super::vision::Parameters {
        &self.parameters
    }
    /// Projects encoding extensions without changing geometry or execution policy.
    pub fn with_formats(
        &self,
        formats: HashMap<String, WeightQuantization>,
    ) -> Result<Self, eredu_nn::Error> {
        let formats = (!formats.is_empty()).then_some(formats);
        if formats == self.policy.quantized_weight_configs {
            return Ok(self.clone());
        }
        let mut policy = self.policy.clone();
        policy.quantized_weight_configs = formats;
        let mut parameters = self.parameters.as_ref().clone();
        parameters.project_formats(&policy)?;
        Ok(Self {
            policy,
            parameters: std::sync::Arc::new(parameters),
        })
    }
}
impl TryFrom<VisionPolicy> for VisionConfig {
    type Error = eredu_nn::Error;
    fn try_from(policy: VisionPolicy) -> Result<Self, Self::Error> {
        Self::new(policy)
    }
}
impl std::ops::Deref for VisionConfig {
    type Target = VisionPolicy;
    fn deref(&self) -> &Self::Target {
        &self.policy
    }
}
impl eredu_nn::DeclareLogicalParameters for VisionConfig {
    fn logical_parameters(
        &self,
    ) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, eredu_nn::Error> {
        self.parameters.logical_parameters()
    }
}

impl VisionPolicy {
    /// Returns the exact physical format for one hMLP weight.
    pub fn linear_format_for(&self, name: &str) -> LinearFormat {
        self.quantized_weight_configs
            .as_ref()
            .and_then(|configs| configs.get(name))
            .copied()
            .map(Into::into)
            .unwrap_or(LinearFormat::Dense)
    }
}

#[derive(Debug, Deserialize)]
struct ModelSource {
    #[serde(default = "default_model_type")]
    model_type: String,
    text_config: TextSource,
    #[serde(default)]
    mtp_config: Option<MtpConfig>,
    #[serde(default)]
    audio_config: Option<AudioConfig>,
    #[serde(default)]
    vision_config: Option<VisionConfig>,
    #[serde(default = "default_image_token_id")]
    image_token_id: u32,
    #[serde(default = "default_audio_token_id")]
    audio_token_id: u32,
}

/// Normalized top-level Inkling source and execution policy.
#[derive(Debug, Clone)]
pub struct ModelPolicy {
    /// Exact family identity.
    pub model_type: String,
    /// Text decoder policy.
    pub text_config: TextArgs,
    /// Optional embedded prediction policy.
    pub mtp_config: Option<MtpConfig>,
    /// Optional dMel audio policy.
    pub audio_config: Option<AudioConfig>,
    /// Optional hMLP vision policy.
    pub vision_config: Option<VisionConfig>,
    /// Image placeholder.
    pub image_token_id: u32,
    /// Audio placeholder.
    pub audio_token_id: u32,
}

/// Immutable source with complete retained text, media and prediction parameter declarations.
#[derive(Debug, Clone)]
pub struct ModelArgs {
    policy: ModelPolicy,
    prediction: std::sync::Arc<super::mtp::PredictionParameters>,
    pinned: std::sync::Arc<super::text::PinnedParameters>,
    common: std::sync::Arc<Vec<super::text::CommonParameters>>,
    feed_forward: std::sync::Arc<Vec<super::text::FeedForwardParameters>>,
}
impl ModelArgs {
    /// Validates normalized policy and declares every text and prediction parameter once.
    pub fn new(policy: ModelPolicy) -> Result<Self, ConfigError> {
        policy.validate()?;
        let prediction =
            super::mtp::PredictionParameters::new(&policy).map_err(|e| invalid(e.to_string()))?;
        let pinned = super::text::PinnedParameters::new(&policy.text_config)
            .map_err(|e| invalid(e.to_string()))?;
        let common = policy
            .text_config
            .layer_schedule
            .iter()
            .enumerate()
            .map(|(layer, layer_policy)| {
                super::text::CommonParameters::new_at(
                    &policy.text_config,
                    layer_policy.attention,
                    &format!("model.layers.{layer}"),
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| invalid(e.to_string()))?;
        let feed_forward = (0..policy.text_config.layer_schedule.len())
            .map(|layer| super::text::FeedForwardParameters::new(&policy.text_config, layer))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| invalid(e.to_string()))?;
        let configuration = Self {
            policy,
            prediction: std::sync::Arc::new(prediction),
            pinned: std::sync::Arc::new(pinned),
            common: std::sync::Arc::new(common),
            feed_forward: std::sync::Arc::new(feed_forward),
        };
        configuration.validate_parameters()?;
        Ok(configuration)
    }
    /// Normalized source geometry, media, and format policy.
    pub fn policy(&self) -> &ModelPolicy {
        &self.policy
    }
    pub(crate) fn prediction(&self) -> &super::mtp::PredictionParameters {
        &self.prediction
    }
    pub(crate) fn pinned(&self) -> &super::text::PinnedParameters {
        &self.pinned
    }
    pub(crate) fn common_layers(&self) -> &[super::text::CommonParameters] {
        &self.common
    }
    pub(crate) fn common(
        &self,
        layer: usize,
    ) -> Result<&super::text::CommonParameters, eredu_nn::Error> {
        self.common.get(layer).ok_or_else(|| {
            eredu_nn::Error::backend("Inkling common invocation is outside its declared source")
        })
    }
    pub(crate) fn local_common(
        &self,
        layer: usize,
        local: &TextArgs,
    ) -> Result<super::text::CommonParameters, eredu_nn::Error> {
        let source = self.common(layer)?;
        if local == &self.text_config {
            return Ok(source.clone());
        }
        self.validate_local_text(local)?;
        let windowed = source.attention.policy.window().is_some();
        source.localized(local.query_heads(windowed), local.key_value_heads(windowed))
    }
    fn validate_local_text(&self, local: &TextArgs) -> Result<(), eredu_nn::Error> {
        if local == &self.text_config {
            return Ok(());
        }
        let mut expected = self.text_config.clone();
        expected.num_attention_heads = local.num_attention_heads;
        expected.num_key_value_heads = local.num_key_value_heads;
        expected.swa_num_attention_heads = local.swa_num_attention_heads;
        expected.swa_num_key_value_heads = local.swa_num_key_value_heads;
        expected.dense_intermediate_size = local.dense_intermediate_size;
        expected.moe_intermediate_size = local.moe_intermediate_size;
        if expected != *local
            || local.dense_intermediate_size() <= 0
            || local.dense_intermediate_size() > self.text_config.dense_intermediate_size()
            || local.moe_intermediate_size() <= 0
            || local.moe_intermediate_size() > self.text_config.moe_intermediate_size()
        {
            return Err(eredu_nn::Error::backend("Inkling local common projection changes non-placement policy or exceeds source widths"));
        }
        Ok(())
    }
    pub(crate) fn feed_forward(
        &self,
        layer: usize,
    ) -> Result<&super::text::FeedForwardParameters, eredu_nn::Error> {
        self.feed_forward.get(layer).ok_or_else(|| {
            eredu_nn::Error::backend(
                "Inkling feed-forward invocation is outside its declared source",
            )
        })
    }
    pub(crate) fn dense(
        &self,
        layer: usize,
    ) -> Result<&super::text::DenseParameters, eredu_nn::Error> {
        match self.feed_forward(layer)? {
            super::text::FeedForwardParameters::Dense(p) => Ok(p),
            _ => Err(eredu_nn::Error::backend(
                "Inkling invocation does not own a dense branch",
            )),
        }
    }
    pub(crate) fn sparse(
        &self,
        layer: usize,
    ) -> Result<&super::text::SparseParameters, eredu_nn::Error> {
        match self.feed_forward(layer)? {
            super::text::FeedForwardParameters::Sparse(p) => Ok(p),
            _ => Err(eredu_nn::Error::backend(
                "Inkling invocation does not own a sparse branch",
            )),
        }
    }
    pub(crate) fn sparse_layers(
        &self,
    ) -> impl Iterator<Item = (usize, &super::text::SparseParameters)> {
        self.feed_forward
            .iter()
            .enumerate()
            .filter_map(|(layer, p)| match p {
                super::text::FeedForwardParameters::Sparse(p) => Some((layer, p)),
                _ => None,
            })
    }
    pub(crate) fn local_feed_forward(
        &self,
        layer: usize,
        local: &TextArgs,
    ) -> Result<super::text::FeedForwardParameters, eredu_nn::Error> {
        self.validate_local_text(local)?;
        self.feed_forward(layer)?.localized(local)
    }
    fn text_parameters(
        &self,
    ) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, eredu_nn::Error> {
        use eredu_nn::DeclareLogicalParameters;
        let mut parameters = self.pinned.logical_parameters()?;
        parameters.extend(self.common.logical_parameters()?);
        parameters.extend(self.feed_forward.logical_parameters()?);
        parameters.extend(self.prediction.logical_parameters()?);
        Ok(parameters)
    }
    fn validate_parameters(&self) -> Result<(), ConfigError> {
        use eredu_nn::DeclareLogicalParameters;
        let declarations = self
            .logical_parameters()
            .map_err(|e| invalid(e.to_string()))?;
        eredu_nn::validate_logical_parameter_topology(&declarations)
            .map_err(|e| invalid(e.to_string()))?;
        let text_parameters = self.text_parameters().map_err(|e| invalid(e.to_string()))?;
        crate::format_references::validate_matrices(
            text_parameters,
            self.text_config
                .quantized_weight_configs
                .iter()
                .flat_map(|formats| formats.iter())
                .map(|(name, format)| (name.as_str(), Some((*format).into()))),
        )
        .map_err(|e| invalid(e.to_string()))?;
        Ok(())
    }
    /// Parses external HF policy before declaration construction.
    pub fn from_hf_json(bytes: &[u8]) -> Result<Self, ConfigError> {
        Self::new(ModelPolicy::from_hf_json(bytes)?)
    }
    /// Parses text-only GGUF policy before declaration construction.
    pub fn from_gguf_metadata(
        metadata: &HashMap<String, MetadataValue>,
    ) -> Result<Self, ConfigError> {
        Self::new(ModelPolicy::from_gguf_metadata(metadata)?)
    }
    /// Attaches validated media while retaining the unchanged predictor scopes.
    pub fn with_gguf_projector_metadata(
        &self,
        model_metadata: &HashMap<String, MetadataValue>,
        projector_metadata: &HashMap<String, MetadataValue>,
        formats: HashMap<String, WeightQuantization>,
    ) -> Result<Self, ConfigError> {
        let policy = self.policy.clone().with_gguf_projector_metadata(
            model_metadata,
            projector_metadata,
            formats,
        )?;
        Ok(Self {
            policy,
            prediction: self.prediction.clone(),
            pinned: self.pinned.clone(),
            common: self.common.clone(),
            feed_forward: self.feed_forward.clone(),
        })
    }
    /// Projects selected text encodings while preserving all equation and invocation policy.
    pub(crate) fn with_text_formats(
        &self,
        uniform: Option<WeightQuantization>,
        formats: Option<HashMap<String, WeightQuantization>>,
    ) -> Result<Self, ConfigError> {
        if uniform == self.text_config.weight_quantization
            && formats == self.text_config.quantized_weight_configs
        {
            return Ok(self.clone());
        }
        let mut policy = self.policy.clone();
        policy.text_config.weight_quantization = uniform;
        policy.text_config.quantized_weight_configs = formats;
        policy.validate()?;
        let mut feed_forward = self.feed_forward.as_ref().clone();
        for parameters in &mut feed_forward {
            parameters
                .project_formats(&policy.text_config)
                .map_err(|e| invalid(e.to_string()))?;
        }
        let mut common = self.common.as_ref().clone();
        for parameters in &mut common {
            parameters
                .project_formats(&policy.text_config)
                .map_err(|e| invalid(e.to_string()))?;
        }
        let mut pinned = self.pinned.as_ref().clone();
        pinned
            .project_formats(&policy.text_config)
            .map_err(|e| invalid(e.to_string()))?;
        let mut prediction = self.prediction.as_ref().clone();
        prediction
            .project_formats(&policy.text_config)
            .map_err(|e| invalid(e.to_string()))?;
        let configuration = Self {
            policy,
            prediction: std::sync::Arc::new(prediction),
            pinned: std::sync::Arc::new(pinned),
            common: std::sync::Arc::new(common),
            feed_forward: std::sync::Arc::new(feed_forward),
        };
        configuration.validate_parameters()?;
        Ok(configuration)
    }
    /// Projects selected media encodings without rebuilding text or prediction geometry.
    pub(crate) fn with_media_formats(
        &self,
        audio: HashMap<String, WeightQuantization>,
        vision: HashMap<String, WeightQuantization>,
    ) -> Result<Self, ConfigError> {
        let mut policy = self.policy.clone();
        if let Some(config) = policy.audio_config.as_mut() {
            *config = config
                .with_formats(audio)
                .map_err(|e| invalid(e.to_string()))?;
        } else if !audio.is_empty() {
            return Err(invalid("Inkling audio formats have no declared dMel tower"));
        }
        if let Some(config) = policy.vision_config.as_mut() {
            *config = config
                .with_formats(vision)
                .map_err(|e| invalid(e.to_string()))?;
        } else if !vision.is_empty() {
            return Err(invalid(
                "Inkling vision formats have no declared hMLP tower",
            ));
        }
        policy.validate()?;
        Ok(Self {
            policy,
            prediction: self.prediction.clone(),
            pinned: self.pinned.clone(),
            common: self.common.clone(),
            feed_forward: self.feed_forward.clone(),
        })
    }
    /// Selects the target-only configuration and removes predictor format references.
    pub fn without_prediction(&self) -> Self {
        let mut policy = self.policy.clone();
        policy.mtp_config = None;
        if let Some(formats) = &mut policy.text_config.quantized_weight_configs {
            formats.retain(|name, _| !name.starts_with("model.mtp."));
        }
        Self {
            policy,
            prediction: std::sync::Arc::new(super::mtp::PredictionParameters::empty()),
            pinned: self.pinned.clone(),
            common: self.common.clone(),
            feed_forward: self.feed_forward.clone(),
        }
    }
}
impl eredu_nn::DeclareLogicalParameters for ModelArgs {
    fn logical_parameters(
        &self,
    ) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, eredu_nn::Error> {
        let mut parameters = self.text_parameters()?;
        if let Some(audio) = &self.audio_config {
            parameters.extend(audio.logical_parameters()?);
        }
        if let Some(vision) = &self.vision_config {
            parameters.extend(vision.logical_parameters()?);
        }
        Ok(parameters)
    }
}
impl std::ops::Deref for ModelArgs {
    type Target = ModelPolicy;
    fn deref(&self) -> &Self::Target {
        &self.policy
    }
}

impl ModelPolicy {
    /// Returns the input modalities admitted by this exact family variant.
    pub const fn input_modalities(&self) -> InputModalities {
        let admits_media = self.vision_config.is_some() || self.audio_config.is_some();
        InputModalities {
            text: true,
            image: admits_media,
            audio: admits_media,
            video: false,
        }
    }

    /// Parses and validates one Hugging Face configuration document.
    fn from_hf_json(bytes: &[u8]) -> Result<Self, ConfigError> {
        let source: ModelSource = serde_json::from_slice(bytes)?;
        let schedule = layer_schedule(&source.text_config)?;
        let text_config = source.text_config.into_args(schedule);
        let args = Self {
            model_type: source.model_type,
            text_config,
            mtp_config: source.mtp_config,
            audio_config: source.audio_config,
            vision_config: source.vision_config,
            image_token_id: source.image_token_id,
            audio_token_id: source.audio_token_id,
        };
        Ok(args)
    }

    /// Parses the released Inkling GGUF metadata without a concrete-backend
    /// checkpoint or tensor type. Sibling projector geometry is applied
    /// separately after its own artifact has been admitted.
    fn from_gguf_metadata(metadata: &HashMap<String, MetadataValue>) -> Result<Self, ConfigError> {
        let key = |suffix: &str| format!("inkling.{suffix}");
        let layers = gguf_i32(metadata, &key("block_count"))?;
        if layers <= 0 {
            return Err(invalid(format!(
                "Inkling GGUF block_count must be positive, got {layers}"
            )));
        }
        let pattern =
            gguf_bool_pattern(metadata, &key("attention.sliding_window_pattern"), layers)?;
        let kv_values = metadata
            .get(&key("attention.head_count_kv"))
            .and_then(MetadataValue::to_i64_vec)
            .ok_or_else(|| invalid("Inkling GGUF is missing attention.head_count_kv"))?;
        let kv_values = if kv_values.len() == 1 {
            vec![kv_values[0]; layers as usize]
        } else {
            kv_values
        };
        if kv_values.len() != layers as usize || kv_values.iter().any(|value| *value <= 0) {
            return Err(invalid(
                "Inkling GGUF attention.head_count_kv must contain one positive value per layer",
            ));
        }
        let global_kv = kv_values
            .iter()
            .zip(&pattern)
            .find_map(|(value, local)| (!local).then_some(*value))
            .unwrap_or(kv_values[0]);
        let local_kv = kv_values
            .iter()
            .zip(&pattern)
            .find_map(|(value, local)| local.then_some(*value))
            .unwrap_or(global_kv);
        if kv_values
            .iter()
            .zip(&pattern)
            .any(|(value, local)| *value != if *local { local_kv } else { global_kv })
        {
            return Err(invalid(
                "Inkling GGUF attention.head_count_kv must be uniform within each attention policy",
            ));
        }
        let hidden_size = gguf_i32(metadata, &key("embedding_length"))?;
        let heads = gguf_i32(metadata, &key("attention.head_count"))?;
        if hidden_size <= 0 || heads <= 0 {
            return Err(invalid(
                "Inkling GGUF embedding length and attention head count must be positive",
            ));
        }
        let head_dim = gguf_optional_i32(metadata, &key("attention.key_length"))?
            .unwrap_or(hidden_size / heads);
        let vocabulary = gguf_vocab_size(metadata, &key("vocab_size"))?;
        let sliding_window = gguf_i32(metadata, &key("attention.sliding_window"))?;
        let sliding = u32::try_from(sliding_window)
            .ok()
            .and_then(|window| AttentionPolicy::sliding(window).ok())
            .ok_or_else(|| invalid("Inkling GGUF sliding window must be positive"))?;
        let dense_layers = gguf_i32(metadata, &key("dense_block_count"))?;
        if !(0..=layers).contains(&dense_layers) {
            return Err(invalid(format!(
                "Inkling GGUF dense block count {dense_layers} is outside 0..={layers}"
            )));
        }
        let layer_schedule = LayerSchedule::new(
            layers as usize,
            pattern
                .iter()
                .enumerate()
                .map(|(layer, local)| LayerPolicy {
                    attention: if *local {
                        sliding
                    } else {
                        AttentionPolicy::Full
                    },
                    feed_forward: if layer < dense_layers as usize {
                        FeedForwardPolicy::Dense
                    } else {
                        FeedForwardPolicy::SparseMoe
                    },
                })
                .collect(),
        )
        .map_err(|error| invalid(error.to_string()))?;
        let args = Self {
            model_type: "inkling_mm_model".into(),
            mtp_config: None,
            text_config: TextArgs {
                torch_dtype: None,
                hidden_size,
                num_hidden_layers: layers,
                vocab_size: vocabulary,
                num_attention_heads: heads,
                num_key_value_heads: i32::try_from(global_kv)
                    .map_err(|_| invalid("Inkling global KV heads exceed i32"))?,
                head_dim,
                swa_num_attention_heads: Some(heads),
                swa_num_key_value_heads: Some(
                    i32::try_from(local_kv)
                        .map_err(|_| invalid("Inkling local KV heads exceed i32"))?,
                ),
                swa_head_dim: Some(head_dim),
                layer_schedule,
                sconv_kernel_size: gguf_i32(metadata, &key("shortconv_kernel"))?,
                use_sconv: true,
                rel_extent: gguf_i32(metadata, &key("rel_extent"))?,
                d_rel: gguf_i32(metadata, &key("d_rel"))?,
                log_scaling_n_floor: gguf_optional_i32(metadata, &key("log_scaling_n_floor"))?
                    .filter(|value| *value > 0),
                log_scaling_alpha: gguf_optional_f32(metadata, &key("log_scaling_alpha"))?
                    .unwrap_or(0.0),
                rms_norm_eps: gguf_f32(metadata, &key("attention.layer_norm_rms_epsilon"))?,
                use_embed_norm: true,
                unpadded_vocab_size: gguf_optional_i32(metadata, &key("unpadded_vocab_size"))?,
                logits_mup_width_multiplier: gguf_optional_f32(
                    metadata,
                    &key("logit_scale_denom"),
                )?
                .unwrap_or(1.0),
                final_logit_softcapping: None,
                intermediate_size: gguf_i32(metadata, &key("expert_feed_forward_length"))?,
                dense_intermediate_size: Some(gguf_i32(metadata, &key("feed_forward_length"))?),
                moe_intermediate_size: None,
                n_routed_experts: gguf_i32(metadata, &key("expert_count"))?,
                num_experts_per_tok: gguf_i32(metadata, &key("expert_used_count"))?,
                n_shared_experts: gguf_i32(metadata, &key("expert_shared_count"))?,
                route_scale: gguf_optional_f32(metadata, &key("expert_weights_scale"))?
                    .unwrap_or(1.0),
                shared_expert_sink: true,
                use_gate_bias: true,
                norm_after_topk: true,
                use_global_scale: true,
                gate_activation: "sigmoid".into(),
                hidden_act: "silu".into(),
                attention_dropout: 0.0,
                q_bias: false,
                o_bias: false,
                model_max_length: Some(gguf_i32(metadata, &key("context_length"))?),
                weight_quantization: None,
                quantized_weight_configs: None,
            },
            audio_config: None,
            vision_config: None,
            image_token_id: default_image_token_id(),
            audio_token_id: default_audio_token_id(),
        };
        Ok(args)
    }

    /// Applies an admitted sibling Inkling hMLP/dMel projector catalog to a
    /// text-only GGUF configuration. Physical per-weight formats are supplied
    /// by the binder after header inspection.
    fn with_gguf_projector_metadata(
        mut self,
        model_metadata: &HashMap<String, MetadataValue>,
        projector_metadata: &HashMap<String, MetadataValue>,
        formats: HashMap<String, WeightQuantization>,
    ) -> Result<Self, ConfigError> {
        let (audio_formats, vision_formats) =
            super::checkpoint::partition_mmproj_weight_formats(formats).map_err(invalid)?;
        let architecture = gguf_string(projector_metadata, "general.architecture")?;
        let vision_projector = gguf_string(projector_metadata, "clip.vision.projector_type")?;
        let audio_projector = gguf_string(projector_metadata, "clip.audio.projector_type")?;
        for (key, description) in [
            ("clip.has_vision_encoder", "vision encoder"),
            ("clip.has_audio_encoder", "audio encoder"),
        ] {
            match projector_metadata.get(key) {
                Some(MetadataValue::Bool(true)) => {}
                Some(MetadataValue::Bool(false)) => {
                    return Err(invalid(format!(
                        "Inkling mmproj does not contain its {description}"
                    )))
                }
                Some(_) => return Err(invalid(format!("Inkling mmproj {key:?} must be boolean"))),
                None => return Err(invalid(format!("Inkling mmproj is missing {key:?}"))),
            }
        }
        if architecture != "clip" || vision_projector != "inkling" || audio_projector != "inkling" {
            return Err(invalid(format!(
                "expected an Inkling audio/vision mmproj, got architecture {architecture:?}, vision projector {vision_projector:?}, and audio projector {audio_projector:?}"
            )));
        }
        let vision_hidden = gguf_i32(projector_metadata, "clip.vision.projection_dim")?;
        let audio_hidden = gguf_i32(projector_metadata, "clip.audio.projection_dim")?;
        let audio_embedding = gguf_i32(projector_metadata, "clip.audio.embedding_length")?;
        if [vision_hidden, audio_hidden, audio_embedding]
            .into_iter()
            .any(|width| width != self.text_config.hidden_size)
        {
            return Err(invalid(format!(
                "Inkling mmproj output widths ({vision_hidden}, {audio_hidden}, {audio_embedding}) do not match decoder width {}",
                self.text_config.hidden_size
            )));
        }
        let patch_size = gguf_i32(projector_metadata, "clip.vision.patch_size")?;
        let image_size = gguf_i32(projector_metadata, "clip.vision.image_size")?;
        if image_size != patch_size {
            return Err(invalid(format!(
                "Inkling mmproj image_size {image_size} does not match patch_size {patch_size}"
            )));
        }
        self.vision_config = Some(
            VisionConfig::new(VisionPolicy {
                vision_encoder_type: "hmlp".into(),
                text_hidden_size: vision_hidden,
                patch_size,
                temporal_patch_size: 2,
                num_channels: gguf_i32(projector_metadata, "clip.vision.embedding_length")?,
                num_hidden_layers: gguf_i32(projector_metadata, "clip.vision.block_count")?,
                use_vision_norm: true,
                rms_norm_eps: gguf_optional_f32(
                    projector_metadata,
                    "clip.vision.attention.layer_norm_epsilon",
                )?
                .unwrap_or(default_rms_norm_eps()),
                quantized_weight_configs: (!vision_formats.is_empty()).then_some(vision_formats),
            })
            .map_err(|e| invalid(e.to_string()))?,
        );
        self.audio_config = Some(
            AudioConfig::new(AudioPolicy {
                text_hidden_size: audio_hidden,
                num_codebooks: gguf_i32(projector_metadata, "clip.audio.num_mel_bins")?,
                codebook_size: 16,
                bias: false,
                use_audio_norm: true,
                audio_mode: "dmel".into(),
                rms_norm_eps: gguf_optional_f32(
                    projector_metadata,
                    "clip.audio.attention.layer_norm_epsilon",
                )?
                .unwrap_or(default_rms_norm_eps()),
                quantized_weight_configs: (!audio_formats.is_empty()).then_some(audio_formats),
            })
            .map_err(|e| invalid(e.to_string()))?,
        );
        if let Some(id) = gguf_optional_i32(model_metadata, "inkling.audio_token_id")? {
            self.audio_token_id = u32::try_from(id)
                .map_err(|_| invalid("Inkling audio placeholder id is negative"))?;
        }
        if let Some(id) = gguf_optional_i32(model_metadata, "inkling.image_token_id")? {
            self.image_token_id = u32::try_from(id)
                .map_err(|_| invalid("Inkling image placeholder id is negative"))?;
        }
        self.validate()?;
        Ok(self)
    }

    /// Stable normalized schedule and geometry identity.
    pub fn architecture_fingerprint(&self) -> String {
        derive_prompt_cache_architecture_fingerprint(
            "inkling",
            [
                ("model_type", self.model_type.clone()),
                ("hidden", self.text_config.hidden_size.to_string()),
                ("vocab", self.text_config.vocab_size.to_string()),
                (
                    "schedule",
                    self.text_config
                        .layer_schedule
                        .iter()
                        .map(|policy| format!("{:?}:{:?}", policy.attention, policy.feed_forward))
                        .collect::<Vec<_>>()
                        .join(","),
                ),
                (
                    "state",
                    format!(
                        "{}:{}:{}",
                        self.text_config.sconv_kernel_size,
                        self.text_config.rel_extent,
                        self.text_config.d_rel
                    ),
                ),
                (
                    "text_quantization",
                    format!(
                        "default={:?};overrides={}",
                        self.text_config.weight_quantization,
                        crate::cache_identity::debug_map(
                            self.text_config.quantized_weight_configs.as_ref()
                        )
                    ),
                ),
                (
                    "vision_quantization",
                    self.vision_config.as_ref().map_or_else(
                        || "none".into(),
                        |config| {
                            crate::cache_identity::debug_map(
                                config.quantized_weight_configs.as_ref(),
                            )
                        },
                    ),
                ),
                (
                    "audio_quantization",
                    self.audio_config.as_ref().map_or_else(
                        || "none".into(),
                        |config| {
                            crate::cache_identity::debug_map(
                                config.quantized_weight_configs.as_ref(),
                            )
                        },
                    ),
                ),
            ],
        )
    }

    pub(crate) fn validate(&self) -> Result<(), ConfigError> {
        let text = &self.text_config;
        if self.model_type != "inkling_mm_model" {
            return Err(invalid(format!(
                "expected Inkling model_type inkling_mm_model, got {:?}",
                self.model_type
            )));
        }
        if text
            .torch_dtype
            .as_deref()
            .is_some_and(|dtype| !matches!(dtype, "bfloat16" | "bf16" | "float16" | "float32"))
        {
            return Err(invalid("unsupported Inkling torch_dtype"));
        }
        for (name, value) in [
            ("hidden_size", text.hidden_size),
            ("num_hidden_layers", text.num_hidden_layers),
            ("vocab_size", text.vocab_size),
            ("num_attention_heads", text.num_attention_heads),
            ("num_key_value_heads", text.num_key_value_heads),
            ("head_dim", text.head_dim),
            ("d_rel", text.d_rel),
            ("rel_extent", text.rel_extent),
            ("sconv_kernel_size", text.sconv_kernel_size),
            ("n_routed_experts", text.n_routed_experts),
            ("num_experts_per_tok", text.num_experts_per_tok),
            ("n_shared_experts", text.n_shared_experts),
        ] {
            if value <= 0 {
                return Err(invalid(format!("Inkling {name} must be positive")));
            }
        }
        if usize::try_from(text.num_hidden_layers).ok() != Some(text.layer_schedule.len()) {
            return Err(invalid(
                "Inkling layer schedule differs from declared target depth",
            ));
        }
        if !text.rms_norm_eps.is_finite()
            || text.rms_norm_eps <= 0.0
            || !text.route_scale.is_finite()
            || !text.logits_mup_width_multiplier.is_finite()
            || text.logits_mup_width_multiplier <= 0.0
            || !text.log_scaling_alpha.is_finite()
        {
            return Err(invalid(
                "Inkling normalization and scaling values are invalid",
            ));
        }
        if !text.use_sconv
            || !text.use_embed_norm
            || !text.shared_expert_sink
            || !text.use_gate_bias
            || !text.norm_after_topk
            || !text.use_global_scale
            || text.gate_activation != "sigmoid"
            || text.hidden_act != "silu"
            || text.attention_dropout != 0.0
            || text.q_bias
            || text.o_bias
            || text
                .final_logit_softcapping
                .is_some_and(|value| value != 0.0)
        {
            return Err(invalid(
                "Inkling config uses an unsupported attention, convolution, routing, or logit variant",
            ));
        }
        for local in [false, true] {
            let q = text.query_heads(local);
            let kv = text.key_value_heads(local);
            let head = text.attention_head_dim(local);
            if q <= 0
                || kv <= 0
                || head <= 0
                || q % kv != 0
                || (local && head != text.head_dim)
                || q.checked_mul(head).is_none()
                || kv.checked_mul(head).is_none()
                || q.checked_mul(text.d_rel).is_none()
            {
                return Err(invalid("Inkling attention head geometry is inconsistent"));
            }
        }
        if text.dense_intermediate_size() <= 0
            || text.moe_intermediate_size() <= 0
            || text.num_experts_per_tok > text.n_routed_experts
            || text
                .n_routed_experts
                .checked_add(text.n_shared_experts)
                .is_none()
            || text.moe_intermediate_size().checked_mul(2).is_none()
        {
            return Err(invalid("Inkling dense or expert geometry is inconsistent"));
        }
        if let Some(mtp) = &self.mtp_config {
            let count = usize::try_from(mtp.num_nextn_predict_layers)
                .map_err(|_| invalid("Inkling MTP layer count cannot be negative"))?;
            if mtp.local_layer_ids.iter().any(|layer| *layer >= count) {
                return Err(invalid("Inkling MTP local layer id is out of range"));
            }
            if !mtp.local_layer_ids.is_empty()
                && !text
                    .layer_schedule
                    .iter()
                    .any(|policy| policy.attention.window().is_some())
            {
                return Err(invalid(
                    "Inkling local MTP requires sliding target attention",
                ));
            }
        }
        if self
            .audio_config
            .as_ref()
            .is_some_and(|audio| audio.text_hidden_size != text.hidden_size)
        {
            return Err(invalid("Inkling audio width differs from decoder geometry"));
        }
        if self
            .vision_config
            .as_ref()
            .is_some_and(|vision| vision.text_hidden_size != text.hidden_size)
        {
            return Err(invalid(
                "Inkling vision width differs from decoder geometry",
            ));
        }
        Ok(())
    }
}

impl TextSource {
    fn into_args(self, schedule: LayerSchedule<LayerPolicy>) -> TextArgs {
        TextArgs {
            torch_dtype: self.torch_dtype,
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            vocab_size: self.vocab_size,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            swa_num_attention_heads: self.swa_num_attention_heads,
            swa_num_key_value_heads: self.swa_num_key_value_heads,
            swa_head_dim: self.swa_head_dim,
            layer_schedule: schedule,
            sconv_kernel_size: self.sconv_kernel_size,
            use_sconv: self.use_sconv,
            rel_extent: self.rel_extent,
            d_rel: self.d_rel,
            log_scaling_n_floor: self.log_scaling_n_floor,
            log_scaling_alpha: self.log_scaling_alpha,
            rms_norm_eps: self.rms_norm_eps,
            use_embed_norm: self.use_embed_norm,
            unpadded_vocab_size: self.unpadded_vocab_size,
            logits_mup_width_multiplier: self.logits_mup_width_multiplier,
            final_logit_softcapping: self.final_logit_softcapping,
            intermediate_size: self.intermediate_size,
            dense_intermediate_size: self.dense_intermediate_size,
            moe_intermediate_size: self.moe_intermediate_size,
            n_routed_experts: self.n_routed_experts,
            num_experts_per_tok: self.num_experts_per_tok,
            n_shared_experts: self.n_shared_experts,
            route_scale: self.route_scale,
            shared_expert_sink: self.shared_expert_sink,
            use_gate_bias: self.use_gate_bias,
            norm_after_topk: self.norm_after_topk,
            use_global_scale: self.use_global_scale,
            gate_activation: self.gate_activation,
            hidden_act: self.hidden_act,
            attention_dropout: self.attention_dropout,
            q_bias: self.q_bias,
            o_bias: self.o_bias,
            model_max_length: self.model_max_length,
            weight_quantization: self.weight_quantization,
            quantized_weight_configs: self.quantized_weight_configs,
        }
    }
}

fn gguf_bool_pattern(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
    layers: i32,
) -> Result<Vec<bool>, ConfigError> {
    let values = match metadata.get(key) {
        Some(MetadataValue::Array(MetadataArray::Bool(values))) => values.clone(),
        Some(_) => {
            return Err(invalid(format!(
                "Inkling GGUF {key:?} must be a bool array"
            )))
        }
        None => (0..layers).map(|layer| (layer + 1) % 6 != 0).collect(),
    };
    if values.len() != layers as usize {
        return Err(invalid(format!(
            "Inkling GGUF {key:?} has {} values for {layers} layers",
            values.len()
        )));
    }
    Ok(values)
}

fn gguf_vocab_size(
    metadata: &HashMap<String, MetadataValue>,
    fallback: &str,
) -> Result<i32, ConfigError> {
    gguf_i32(metadata, fallback)
}

fn gguf_i32(metadata: &HashMap<String, MetadataValue>, key: &str) -> Result<i32, ConfigError> {
    gguf_optional_i32(metadata, key)?
        .ok_or_else(|| invalid(format!("Inkling GGUF is missing required key {key:?}")))
}

fn gguf_string(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<String, ConfigError> {
    match metadata.get(key) {
        Some(MetadataValue::String(value)) => Ok(value.clone()),
        Some(_) => Err(invalid(format!("Inkling GGUF {key:?} must be a string"))),
        None => Err(invalid(format!(
            "Inkling GGUF is missing required key {key:?}"
        ))),
    }
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
                .ok_or_else(|| invalid(format!("Inkling GGUF {key:?} must be an i32 scalar")))
        })
        .transpose()
}

fn gguf_f32(metadata: &HashMap<String, MetadataValue>, key: &str) -> Result<f32, ConfigError> {
    gguf_optional_f32(metadata, key)?
        .ok_or_else(|| invalid(format!("Inkling GGUF is missing required key {key:?}")))
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
                .ok_or_else(|| invalid(format!("Inkling GGUF {key:?} must be numeric")))
        })
        .transpose()
}

fn layer_schedule(source: &TextSource) -> Result<LayerSchedule<LayerPolicy>, ConfigError> {
    let layers = usize::try_from(source.num_hidden_layers)
        .ok()
        .filter(|layers| *layers > 0)
        .ok_or_else(|| invalid("Inkling num_hidden_layers must be positive"))?;
    let window = u32::try_from(source.sliding_window_size)
        .ok()
        .filter(|window| *window > 0 && *window <= i32::MAX as u32)
        .ok_or_else(|| invalid("Inkling sliding_window_size is invalid"))?;
    let sliding = AttentionPolicy::sliding(window).map_err(|error| invalid(error.to_string()))?;
    let from_ids = source
        .local_layer_ids
        .as_ref()
        .map(|ids| {
            let mut local = vec![false; layers];
            for id in ids {
                let layer = usize::try_from(*id)
                    .map_err(|_| invalid("Inkling local_layer_ids contains a negative layer"))?;
                if layer >= layers || std::mem::replace(&mut local[layer], true) {
                    return Err(invalid(
                        "Inkling local_layer_ids is out of range or duplicated",
                    ));
                }
            }
            Ok(local)
        })
        .transpose()?;
    let from_types = source
        .layer_types
        .as_ref()
        .map(|types| {
            if types.len() != layers {
                return Err(invalid(
                    "Inkling layer_types length does not match layer count",
                ));
            }
            types
                .iter()
                .map(|kind| match kind.as_str() {
                    "sliding_attention" => Ok(true),
                    "full_attention" => Ok(false),
                    _ => Err(invalid("invalid Inkling layer_types entry")),
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?;
    if matches!((&from_ids, &from_types), (Some(left), Some(right)) if left != right) {
        return Err(invalid(
            "Inkling local_layer_ids conflicts with layer_types",
        ));
    }
    let attention = from_ids
        .or(from_types)
        .unwrap_or_else(|| (0..layers).map(|layer| (layer + 1) % 6 != 0).collect());
    let from_mlp_types = source
        .mlp_layer_types
        .as_ref()
        .map(|types| {
            if types.len() != layers {
                return Err(invalid(
                    "Inkling mlp_layer_types length does not match layer count",
                ));
            }
            types
                .iter()
                .map(|kind| match kind.as_str() {
                    "dense" => Ok(FeedForwardPolicy::Dense),
                    "moe" => Ok(FeedForwardPolicy::SparseMoe),
                    _ => Err(invalid("invalid Inkling mlp_layer_types entry")),
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?;
    let from_threshold = source
        .dense_mlp_idx
        .map(|threshold| {
            let threshold = usize::try_from(threshold)
                .ok()
                .filter(|threshold| *threshold <= layers)
                .ok_or_else(|| invalid("Inkling dense_mlp_idx is invalid"))?;
            Ok::<_, ConfigError>(
                (0..layers)
                    .map(|layer| {
                        if layer < threshold {
                            FeedForwardPolicy::Dense
                        } else {
                            FeedForwardPolicy::SparseMoe
                        }
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .transpose()?;
    if matches!((&from_mlp_types, &from_threshold), (Some(left), Some(right)) if left != right) {
        return Err(invalid(
            "Inkling dense_mlp_idx conflicts with mlp_layer_types",
        ));
    }
    let feed_forward = from_mlp_types
        .or(from_threshold)
        .unwrap_or_else(|| vec![FeedForwardPolicy::SparseMoe; layers]);
    LayerSchedule::new(
        layers,
        attention
            .into_iter()
            .zip(feed_forward)
            .map(|(local, feed_forward)| LayerPolicy {
                attention: if local {
                    sliding
                } else {
                    AttentionPolicy::Full
                },
                feed_forward,
            })
            .collect(),
    )
    .map_err(|error| invalid(error.to_string()))
}
