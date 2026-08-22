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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone, Deserialize)]
pub struct AudioConfig {
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

impl AudioConfig {
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
#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
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

impl VisionConfig {
    /// Returns the fixed released hMLP fold and projection schedule.
    pub fn layer_specs(&self) -> [(i32, i32, i32, i32); 4] {
        let second_stage_width = if self.text_hidden_size == 4096 {
            320
        } else {
            512
        };
        [
            (75, 128, 1, 5),
            (512, second_stage_width, 1, 2),
            (second_stage_width * 16, 4800, 1, 4),
            (9600, self.text_hidden_size, 2, 1),
        ]
    }

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
    #[serde(default)]
    eos_token_id: Option<u32>,
}

/// Released top-level Inkling configuration.
#[derive(Debug, Clone)]
pub struct ModelArgs {
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
    /// Optional end token.
    pub eos_token_id: Option<u32>,
}

impl ModelArgs {
    /// Returns the input modalities admitted by this exact family variant.
    pub const fn input_modalities(&self) -> InputModalities {
        InputModalities {
            text: true,
            image: self.vision_config.is_some(),
            audio: self.audio_config.is_some(),
            video: false,
        }
    }

    /// Parses and validates one Hugging Face configuration document.
    pub fn from_hf_json(bytes: &[u8]) -> Result<Self, ConfigError> {
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
            eos_token_id: source.eos_token_id,
        };
        args.validate()?;
        Ok(args)
    }

    /// Parses the released Inkling GGUF metadata without a backend-owned
    /// checkpoint or array type. Sibling projector geometry is applied
    /// separately after its own artifact has been admitted.
    pub fn from_gguf_metadata(
        metadata: &HashMap<String, MetadataValue>,
    ) -> Result<Self, ConfigError> {
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
            eos_token_id: gguf_eos_token_id(metadata)?,
        };
        args.validate()?;
        Ok(args)
    }

    /// Applies an admitted sibling Inkling hMLP/dMel projector catalog to a
    /// text-only GGUF configuration. Physical per-weight formats are supplied
    /// by the binder after header inspection.
    pub fn with_gguf_projector_metadata(
        mut self,
        model_metadata: &HashMap<String, MetadataValue>,
        projector_metadata: &HashMap<String, MetadataValue>,
        audio_formats: HashMap<String, WeightQuantization>,
        vision_formats: HashMap<String, WeightQuantization>,
    ) -> Result<Self, ConfigError> {
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
        self.vision_config = Some(VisionConfig {
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
        });
        self.audio_config = Some(AudioConfig {
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
        });
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
            ],
        )
    }

    fn validate(&self) -> Result<(), ConfigError> {
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
        if let Some(audio) = &self.audio_config {
            if audio.text_hidden_size != text.hidden_size
                || audio.num_codebooks <= 0
                || audio.codebook_size <= 0
                || audio.bias
                || !audio.use_audio_norm
                || audio.audio_mode != "dmel"
                || audio
                    .num_codebooks
                    .checked_mul(audio.codebook_size)
                    .is_none()
            {
                return Err(invalid("Inkling audio configuration is inconsistent"));
            }
        }
        if let Some(vision) = &self.vision_config {
            if vision.text_hidden_size != text.hidden_size
                || vision.vision_encoder_type != "hmlp"
                || !vision.use_vision_norm
                || (
                    vision.temporal_patch_size,
                    vision.patch_size,
                    vision.num_hidden_layers,
                    vision.num_channels,
                ) != (2, 40, 4, 3)
            {
                return Err(invalid(
                    "Inkling vision configuration is not the released hMLP tower",
                ));
            }
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
    match metadata
        .get("tokenizer.ggml.tokens")
        .and_then(MetadataValue::as_strings)
    {
        Some(tokens) => i32::try_from(tokens.len())
            .map_err(|_| invalid("Inkling GGUF tokenizer vocabulary exceeds i32")),
        None if metadata.contains_key("tokenizer.ggml.tokens") => Err(invalid(
            "Inkling GGUF tokenizer.ggml.tokens metadata has the wrong type",
        )),
        None => gguf_i32(metadata, fallback),
    }
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

fn gguf_eos_token_id(
    metadata: &HashMap<String, MetadataValue>,
) -> Result<Option<u32>, ConfigError> {
    let scalar = metadata
        .get("tokenizer.ggml.eos_token_id")
        .and_then(MetadataValue::as_i64);
    let value = scalar.or_else(|| {
        metadata
            .get("tokenizer.ggml.eos_token_ids")
            .and_then(MetadataValue::to_i64_vec)
            .and_then(|values| values.first().copied())
    });
    value
        .map(|value| {
            u32::try_from(value).map_err(|_| invalid("Inkling GGUF EOS token identity exceeds u32"))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> serde_json::Value {
        serde_json::json!({
          "model_type":"inkling_mm_model","image_token_id":60,"audio_token_id":61,
          "text_config":{
            "hidden_size":16,"num_hidden_layers":3,"vocab_size":64,
            "num_attention_heads":4,"num_key_value_heads":2,"head_dim":4,
            "sliding_window_size":8,"layer_types":["sliding_attention","full_attention","sliding_attention"],
            "mlp_layer_types":["dense","moe","moe"],"sconv_kernel_size":4,
            "d_rel":2,"rel_extent":16,"intermediate_size":32,
            "n_routed_experts":4,"num_experts_per_tok":2,"n_shared_experts":1
          },
          "mtp_config":{"num_nextn_predict_layers":2,"local_layer_ids":[1]},
          "audio_config":{"text_hidden_size":16,"num_codebooks":4,"codebook_size":8},
          "vision_config":{"text_hidden_size":16,"patch_size":40,"temporal_patch_size":2,
            "num_channels":3,"num_hidden_layers":4}
        })
    }

    #[test]
    fn freezes_explicit_attention_and_mlp_schedules() {
        let args = ModelArgs::from_hf_json(&serde_json::to_vec(&config()).unwrap()).unwrap();
        assert_eq!(args.text_config.layer_schedule.len(), 3);
        assert_eq!(
            args.text_config.layer_policy(0).unwrap().feed_forward,
            FeedForwardPolicy::Dense
        );
        assert_eq!(
            args.text_config.layer_policy(1).unwrap().attention,
            AttentionPolicy::Full
        );
        assert_eq!(
            args.input_modalities(),
            InputModalities {
                text: true,
                image: true,
                audio: true,
                video: false,
            }
        );
        assert!(!args.architecture_fingerprint().is_empty());
    }

    #[test]
    fn text_only_variant_does_not_advertise_media_modalities() {
        let mut value = config();
        let object = value.as_object_mut().unwrap();
        object.remove("vision_config");
        object.remove("audio_config");
        let args = ModelArgs::from_hf_json(&serde_json::to_vec(&value).unwrap()).unwrap();
        assert_eq!(args.input_modalities(), InputModalities::TEXT);
    }

    #[test]
    fn rejects_conflicting_schedule_sources() {
        let mut value = config();
        value["text_config"]["local_layer_ids"] = serde_json::json!([0]);
        assert!(ModelArgs::from_hf_json(&serde_json::to_vec(&value).unwrap()).is_err());
        let mut value = config();
        value["text_config"]["dense_mlp_idx"] = 2.into();
        assert!(ModelArgs::from_hf_json(&serde_json::to_vec(&value).unwrap()).is_err());
    }

    #[test]
    fn parses_gguf_metadata_without_backend_types() {
        let mut metadata = HashMap::from([
            ("inkling.block_count".into(), MetadataValue::Uint32(2)),
            ("inkling.embedding_length".into(), MetadataValue::Uint32(16)),
            (
                "inkling.attention.head_count".into(),
                MetadataValue::Uint32(4),
            ),
            (
                "inkling.attention.key_length".into(),
                MetadataValue::Uint32(4),
            ),
            (
                "inkling.attention.sliding_window".into(),
                MetadataValue::Uint32(8),
            ),
            ("inkling.dense_block_count".into(), MetadataValue::Uint32(1)),
            ("inkling.shortconv_kernel".into(), MetadataValue::Uint32(4)),
            ("inkling.rel_extent".into(), MetadataValue::Uint32(16)),
            ("inkling.d_rel".into(), MetadataValue::Uint32(2)),
            (
                "inkling.attention.layer_norm_rms_epsilon".into(),
                MetadataValue::Float32(1e-5),
            ),
            (
                "inkling.expert_feed_forward_length".into(),
                MetadataValue::Uint32(24),
            ),
            (
                "inkling.feed_forward_length".into(),
                MetadataValue::Uint32(32),
            ),
            ("inkling.expert_count".into(), MetadataValue::Uint32(4)),
            ("inkling.expert_used_count".into(), MetadataValue::Uint32(2)),
            (
                "inkling.expert_shared_count".into(),
                MetadataValue::Uint32(1),
            ),
            ("inkling.context_length".into(), MetadataValue::Uint32(128)),
            ("inkling.vocab_size".into(), MetadataValue::Uint32(64)),
        ]);
        metadata.insert(
            "inkling.attention.sliding_window_pattern".into(),
            MetadataValue::Array(MetadataArray::Bool(vec![true, false])),
        );
        metadata.insert(
            "inkling.attention.head_count_kv".into(),
            MetadataValue::Array(MetadataArray::Uint32(vec![1, 2])),
        );
        let args = ModelArgs::from_gguf_metadata(&metadata).unwrap();
        assert_eq!(args.text_config.num_key_value_heads, 2);
        assert_eq!(args.text_config.swa_num_key_value_heads, Some(1));
        assert_eq!(
            args.text_config.layer_policy(0).unwrap().feed_forward,
            FeedForwardPolicy::Dense
        );
        assert_eq!(
            args.text_config.layer_policy(1).unwrap().attention,
            AttentionPolicy::Full
        );
        let projector = HashMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String("clip".into()),
            ),
            (
                "clip.vision.projector_type".into(),
                MetadataValue::String("inkling".into()),
            ),
            (
                "clip.audio.projector_type".into(),
                MetadataValue::String("inkling".into()),
            ),
            ("clip.has_vision_encoder".into(), MetadataValue::Bool(true)),
            ("clip.has_audio_encoder".into(), MetadataValue::Bool(true)),
            (
                "clip.vision.projection_dim".into(),
                MetadataValue::Uint32(16),
            ),
            (
                "clip.audio.projection_dim".into(),
                MetadataValue::Uint32(16),
            ),
            (
                "clip.audio.embedding_length".into(),
                MetadataValue::Uint32(16),
            ),
            ("clip.vision.patch_size".into(), MetadataValue::Uint32(40)),
            ("clip.vision.image_size".into(), MetadataValue::Uint32(40)),
            (
                "clip.vision.embedding_length".into(),
                MetadataValue::Uint32(3),
            ),
            ("clip.vision.block_count".into(), MetadataValue::Uint32(4)),
            ("clip.audio.num_mel_bins".into(), MetadataValue::Uint32(4)),
        ]);
        let args = args
            .with_gguf_projector_metadata(&metadata, &projector, HashMap::new(), HashMap::new())
            .unwrap();
        assert_eq!(args.audio_config.unwrap().num_codebooks, 4);
        assert_eq!(args.vision_config.unwrap().patch_size, 40);
    }
}
