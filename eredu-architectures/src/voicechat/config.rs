//! Declarative admission of the released, nested NeMo VoiceChat configuration.
//!
//! No Python class names in an artifact are imported or executed. Configuration
//! admission establishes the released parameter/equation contract used by generic
//! realtime discovery and construction. Defaults are from Speech revision
//! `097dfe9e2f55baf653b83035868bdc89849f1b47` and Transformers 4.56.0, pinned by
//! that revision's README. The fixture retains inference fields from the official
//! checkpoint config at `443794ea956ef0065f001967ffd00e77f519cb39`.

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};

/// Official VoiceChat artifact revision.
pub const CHECKPOINT_REVISION: &str = "443794ea956ef0065f001967ffd00e77f519cb39";
/// Independent NVIDIA reference implementation revision.
pub const REFERENCE_REVISION: &str = "097dfe9e2f55baf653b83035868bdc89849f1b47";
/// Required base architecture and tokenizer repository.
pub const BASE_MODEL_ID: &str = "nvidia/NVIDIA-Nemotron-Nano-9B-v2";
/// Required immutable base architecture and tokenizer revision.
pub const BASE_MODEL_REVISION: &str = "6533e8de2c68e4536bf7c411d7a3ce5734111476";

const RELEASED: &str = include_str!("fixtures/official-inference-config.json");

/// Configuration admission failure, before any native resource allocation.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// The document is not valid JSON.
    #[error("invalid VoiceChat JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// A required source field is missing or has an invalid type.
    #[error("invalid VoiceChat field {path}: {reason}")]
    InvalidField {
        /// JSON pointer naming the invalid field.
        path: String,
        /// Missing field or type error description.
        reason: String,
    },
    /// This artifact changes the released architecture or equation contract.
    #[error("unsupported VoiceChat policy at {path}: expected {expected}, got {actual}")]
    UnsupportedPolicy {
        /// JSON pointer naming the differing normalized policy.
        path: String,
        /// Required released value.
        expected: Value,
        /// Supplied value.
        actual: Value,
    },
}

/// Sample-exact cadence shared by source audio, perception and generated speech.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct Cadence {
    pub frame_length: f64,
    pub source_sample_rate: usize,
    pub target_sample_rate: usize,
    pub source_samples_per_frame: usize,
    pub target_samples_per_frame: usize,
}

/// Host mel extraction policy, including omitted upstream numerical defaults.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(missing_docs)]
pub struct PreprocessingConfig {
    pub sample_rate: usize,
    pub window_size: f64,
    pub window_stride: f64,
    pub window: String,
    pub normalize: String,
    pub n_fft: usize,
    pub preemph: f64,
    pub features: usize,
    pub lowfreq: f64,
    pub highfreq: f64,
    pub log: bool,
    pub log_zero_guard_type: String,
    pub log_zero_guard_value: f64,
    pub dither: f64,
    pub pad_to: usize,
    pub frame_splicing: usize,
    pub exact_pad: bool,
    pub pad_value: f64,
    pub mag_power: f64,
    pub mel_norm: String,
    pub use_torchaudio: bool,
}

/// Released causal FastConformer equations and cache context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(missing_docs)]
pub struct PerceptionEncoderConfig {
    pub feat_in: usize,
    pub feat_out: i64,
    pub n_layers: usize,
    pub d_model: usize,
    pub causal_downsampling: bool,
    pub subsampling: String,
    pub subsampling_factor: usize,
    pub subsampling_conv_channels: usize,
    pub subsampling_conv_chunking_factor: usize,
    pub reduction: Option<String>,
    pub reduction_position: Option<usize>,
    pub reduction_factor: usize,
    pub ff_expansion_factor: usize,
    pub self_attention_model: String,
    pub n_heads: usize,
    pub att_context_size: [i64; 2],
    pub att_context_style: String,
    pub xscaling: bool,
    pub untie_biases: bool,
    pub pos_emb_max_len: usize,
    pub conv_kernel_size: usize,
    pub conv_norm_type: String,
    pub conv_context_size: String,
    pub use_bias: bool,
    pub global_tokens: usize,
    pub global_tokens_spacing: usize,
    pub global_attn_separate: bool,
}

/// Perception projection connects the unprojected RNNT and Nemotron branches.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[allow(missing_docs)]
pub struct PerceptionConfig {
    pub preprocessor: PreprocessingConfig,
    pub encoder: PerceptionEncoderConfig,
    pub modality_adapter_dim: usize,
    pub output_dim: usize,
}

/// Gemma equations after resolving the exact upstream configuration defaults.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(missing_docs)]
pub struct GemmaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub hidden_activation: String,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rope_local_base_freq: Option<f64>,
    pub attention_bias: bool,
    pub query_pre_attn_scalar: usize,
    pub sliding_window: usize,
    pub layer_types: Vec<String>,
    pub final_logit_softcapping: Option<f64>,
    pub attn_logit_softcapping: Option<f64>,
    pub rope_scaling: Option<Value>,
    pub causal: bool,
}

/// Mixture-of-Gaussians predictor geometry and numerical policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(missing_docs)]
pub struct MogConfig {
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub low_rank: usize,
    pub num_predictions: usize,
    pub min_log_std: f64,
    pub eps: f64,
}

/// EAR-TTS components and iterative decoding policy.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[allow(missing_docs)]
pub struct TtsConfig {
    pub backbone: GemmaConfig,
    pub character_encoder: GemmaConfig,
    pub mog: MogConfig,
    pub latent_size: usize,
    pub codebook_size: usize,
    pub num_quantizers: usize,
    pub context_hidden_size: Option<usize>,
    pub disable_eos_prediction: bool,
    pub num_delay_speech_tokens: usize,
    pub use_audio_prompt_frozen_projection: bool,
    pub use_bos_eos_emb: bool,
    pub use_subword_flag_emb: bool,
    pub use_gated_fusion_for_text_audio: bool,
    pub random_target_masking: bool,
    pub exponent: f64,
    pub subword_mask_exactly_as_eartts: bool,
    pub context_hidden_mask_exactly_as_eartts: bool,
    pub guidance_enabled: bool,
    pub guidance_scale: f64,
    pub noise_scale: f64,
    pub top_p: f64,
    pub force_silence_on_eos: bool,
    pub num_iterations: usize,
    pub eos_threshold: f64,
    pub use_system_prompt: bool,
    pub audio_prompt_duration: f64,
}

/// Released probabilistic RVQ/ConvNeXt/ISTFT codec topology.
///
/// This describes checkpoint geometry; executable codec ownership is in
/// `eredu-codec`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(missing_docs)]
pub struct CodecConfig {
    pub latent_size: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub base_hidden_size: usize,
    pub channel_mult: Vec<usize>,
    pub rates: Vec<usize>,
    pub num_blocks: usize,
    pub kernel_size: usize,
    pub groups: usize,
    pub codebook_size: usize,
    pub num_quantizers: usize,
    pub wav_to_token_ratio: usize,
}

/// RNNT prediction and joint networks over the unprojected encoder output.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[allow(missing_docs)]
pub struct RnntConfig {
    pub vocab_size: usize,
    pub vocabulary: Vec<String>,
    pub pred_hidden: usize,
    pub pred_rnn_layers: usize,
    pub encoder_hidden: usize,
    pub joint_hidden: usize,
    pub activation: String,
    pub blank_as_pad: bool,
    pub blank_id: usize,
    pub normalization_mode: Option<String>,
    pub rnn_hidden_size: i64,
    pub num_extra_outputs: usize,
    pub log_softmax: Option<bool>,
}

/// Fixed text markers resolved against the retained tokenizer during preparation.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[allow(missing_docs)]
pub struct TokenPolicy {
    pub bos: String,
    pub eos: String,
    pub pad: String,
    pub user_bos: String,
    pub user_eos: String,
    pub tool_call_start: String,
    pub tool_call_end: String,
    pub tool_response_end: String,
    pub speech_bos: usize,
    pub speech_eos: usize,
    pub speech_pad: usize,
}

/// Distinct channel fusion, delay and function-serialization policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ChannelPolicy {
    pub delay_source_text_by: usize,
    pub delay_text_channel_by: usize,
    pub duplex_asr_text_weight: f64,
    pub duplex_function_channel_weight: f64,
    pub duplex_text_channel_weight: f64,
    pub duplex_user_channel_weight: f64,
    pub text_embedding_weight: f64,
    pub max_fc_total_tokens: usize,
    pub use_function_head: bool,
    pub use_separate_asr_head: bool,
    pub predict_user_text: bool,
    pub allow_user_text_in_agent_turn: bool,
    pub eval_text_turn_taking: bool,
}

/// Normalized released architecture contract, available without a backend.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[allow(missing_docs)]
pub struct VoiceChatConfig {
    pub cadence: Cadence,
    pub perception: PerceptionConfig,
    pub channels: ChannelPolicy,
    pub tts: TtsConfig,
    pub codec: CodecConfig,
    pub rnnt: RnntConfig,
    pub tokens: TokenPolicy,
    pub speaker_name: String,
}

impl VoiceChatConfig {
    /// Read the official NeMo document; a Hugging Face `model_type` is unnecessary.
    pub fn from_json(source: &str) -> Result<Self, ConfigError> {
        Self::from_value(&serde_json::from_str(source)?)
    }

    /// Normalize declarative source fields and reject changes to the release.
    pub fn from_value(source: &Value) -> Result<Self, ConfigError> {
        let config = normalize(source)?;
        config.validate()?;
        Ok(config)
    }

    /// Validate this contract against the exact released geometry and equations.
    ///
    /// Preparation must call this again if a caller modifies public fields.
    /// Runtime sampling and conversational overrides belong to session policy,
    /// rather than modifying the admitted checkpoint's architecture contract.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let release = normalize(&serde_json::from_str(RELEASED)?)?;
        compare(
            "",
            &serde_json::to_value(release)?,
            &serde_json::to_value(self)?,
        )
    }
}

fn invalid(path: &str, reason: impl Into<String>) -> ConfigError {
    ConfigError::InvalidField {
        path: path.into(),
        reason: reason.into(),
    }
}

fn at<'a>(source: &'a Value, path: &str) -> Result<&'a Value, ConfigError> {
    source
        .pointer(path)
        .ok_or_else(|| invalid(path, "required field is missing"))
}

fn read<T: DeserializeOwned>(source: &Value, path: &str) -> Result<T, ConfigError> {
    serde_json::from_value(at(source, path)?.clone()).map_err(|e| invalid(path, e.to_string()))
}

fn equals(source: &Value, path: &str, expected: Value) -> Result<(), ConfigError> {
    compare(path, &expected, at(source, path)?)
}

fn compare(path: &str, expected: &Value, actual: &Value) -> Result<(), ConfigError> {
    if expected == actual {
        return Ok(());
    }
    if let (Some(a), Some(b)) = (expected.as_array(), actual.as_array()) {
        if a.len() == b.len() {
            for (i, (left, right)) in a.iter().zip(b).enumerate() {
                compare(&format!("{path}/{i}"), left, right)?;
            }
        }
    }
    if let (Some(a), Some(b)) = (expected.as_object(), actual.as_object()) {
        for (k, v) in a {
            compare(&format!("{path}/{k}"), v, b.get(k).unwrap_or(&Value::Null))?;
        }
    }
    Err(ConfigError::UnsupportedPolicy {
        path: path.into(),
        expected: expected.clone(),
        actual: actual.clone(),
    })
}

// Strip only declarative class annotations and known training-only fields.
// Unknown component fields are errors, so an equation override cannot disappear.
fn component<T: DeserializeOwned>(
    source: &Value,
    path: &str,
    defaults: Value,
    ignored: &[&str],
) -> Result<T, ConfigError> {
    let mut value = at(source, path)?.clone();
    let map = value
        .as_object_mut()
        .ok_or_else(|| invalid(path, "expected object"))?;
    map.remove("_target_");
    for key in ignored {
        map.remove(*key);
    }
    if let Some(defaults) = defaults.as_object() {
        for (key, default) in defaults {
            map.entry(key.clone()).or_insert_with(|| default.clone());
        }
    }
    serde_json::from_value(value).map_err(|e| invalid(path, e.to_string()))
}

fn gemma(source: &Value, path: &str, causal: bool) -> Result<GemmaConfig, ConfigError> {
    let layers: usize = read(source, &format!("{path}/num_hidden_layers"))?;
    // Bound before allocating a layer schedule from untrusted configuration.
    if layers == 0 || layers > 256 {
        return Err(invalid(path, "layer count must be in 1..=256"));
    }
    let period = if causal { 6 } else { 2 };
    let schedule: Vec<_> = (0..layers)
        .map(|i| {
            if (i + 1) % period == 0 {
                "full_attention"
            } else {
                "sliding_attention"
            }
        })
        .collect();
    // Transformers 4.56.0 Gemma3TextConfig and T5GemmaModuleConfig.
    component(
        source,
        path,
        json!({
            "hidden_activation":"gelu_pytorch_tanh", "max_position_embeddings":if causal {131072} else {8192},
            "rms_norm_eps":1e-6, "rope_theta":if causal {1000000.0} else {10000.0},
            "rope_local_base_freq":if causal {Some(10000.0)} else {None},
            "attention_bias":false, "query_pre_attn_scalar":256, "sliding_window":4096,
            "layer_types":schedule, "final_logit_softcapping":if causal {None} else {Some(30.0)},
            "attn_logit_softcapping":if causal {None} else {Some(50.0)}, "rope_scaling":null, "causal":causal
        }),
        &["attention_dropout", "use_cache"],
    )
}

fn known_fields(source: &Value, path: &str, names: &[&str]) -> Result<(), ConfigError> {
    let object = at(source, path)?
        .as_object()
        .ok_or_else(|| invalid(path, "expected object"))?;
    for name in object.keys() {
        if name != "_target_" && !names.contains(&name.as_str()) {
            return Err(invalid(
                &format!("{path}/{name}"),
                "unrecognized architecture field",
            ));
        }
    }
    Ok(())
}

fn normalize(source: &Value) -> Result<VoiceChatConfig, ConfigError> {
    const S: &str = "/model/stt/model";
    const G: &str = "/model/speech_generation/model";
    const T: &str = "/model/speech_generation/model/tts_config";
    const R: &str = "/_rnnt_merge_info";
    known_fields(
        source,
        S,
        &[
            "allow_user_text_in_agent_turn",
            "asr_loss_weight",
            "audio_loss_weight",
            "audio_save_path",
            "augment_fc_system_prompt",
            "backchannel_debug",
            "backchannel_debug_max_files",
            "backchannel_debug_path",
            "backchannel_file_path",
            "backchannel_prob",
            "backchannel_prob_scale",
            "backchannel_snr_db",
            "base_model_name",
            "bos_token",
            "cache_key",
            "codec_prob",
            "delay_source_text_by",
            "delay_text_channel_by",
            "duplex_asr_text_weight",
            "duplex_function_channel_weight",
            "duplex_text_channel_weight",
            "duplex_user_channel_weight",
            "embed_tokens_name",
            "eos_token",
            "eval_text_turn_taking",
            "force_align_device",
            "force_align_user_text",
            "force_use_noise_augmentation",
            "freeze_params",
            "function_loss_weight",
            "function_token_loss_weight",
            "incremental_loading",
            "lr_scheduler",
            "mask_sequence_loss",
            "max_fc_total_tokens",
            "micir_aug_path",
            "micir_prob",
            "nemo_text_norm_cache_dir",
            "nemo_text_norm_input_case",
            "nemo_text_norm_lang",
            "nsynth_noise_aug_path",
            "nsynth_noise_prob",
            "old_noise_aug_path",
            "old_noise_max_snr",
            "old_noise_min_snr",
            "old_noise_prob",
            "optimizer",
            "override_tokens",
            "pad_token",
            "perception",
            "predict_user_text",
            "pretrained_asr",
            "pretrained_llm",
            "pretrained_s2s_model",
            "pretrained_weights",
            "prevent_freeze_params",
            "roomir_aug_path",
            "roomir_prob",
            "scoring_asr",
            "text_embedding_weight",
            "text_loss_weight",
            "text_to_text_loss_weight",
            "token_loss_weight",
            "use_cache_for_nemotron",
            "use_codec_aug",
            "use_function_head",
            "use_mic_ir_aug",
            "use_nemo_text_norm",
            "use_nsynth_noise_aug",
            "use_numbers_norm",
            "use_old_noise_aug",
            "use_room_ir_aug",
            "use_separate_asr_head",
        ],
    )?;
    known_fields(
        source,
        G,
        &[
            "audio_save_path",
            "bos_token",
            "codec_config",
            "context_hidden_mask_exactly_as_eartts",
            "eos_token",
            "freeze_params",
            "inference_force_speech_silence_on_eos",
            "inference_guidance_enabled",
            "inference_guidance_scale",
            "inference_noise_scale",
            "inference_top_p_or_k",
            "lr_scheduler",
            "optimizer",
            "pad_token",
            "pretrained_ae_dir",
            "pretrained_lm_name",
            "pretrained_model",
            "pretrained_tts_model",
            "prevent_freeze_params",
            "subword_mask_exactly_as_eartts",
            "tts_config",
            "use_system_prompt",
            "custom_speech_pad_id",
            "custom_speech_eos_id",
            "custom_speech_bos_id",
        ],
    )?;
    known_fields(
        source,
        T,
        &[
            "backbone_config",
            "backbone_config_class",
            "backbone_model_class",
            "backbone_type",
            "cas_config",
            "codebook_size",
            "context_hidden_size",
            "disable_eos_prediction",
            "exponent",
            "label_smoothing",
            "latent_size",
            "max_training_rate",
            "mog_head_config",
            "num_delay_speech_tokens",
            "num_quantizers",
            "p_uncond",
            "quantizer_dropout",
            "random_target_masking",
            "use_audio_prompt_frozen_projection",
            "use_bos_eos_emb",
            "use_gated_fusion_for_text_audio",
            "use_subword_flag_emb",
        ],
    )?;
    known_fields(
        source,
        &format!("{T}/cas_config"),
        &[
            "backbone_config",
            "backbone_config_class",
            "backbone_model_class",
            "backbone_type",
            "pretrained_tokenizer_name",
        ],
    )?;
    known_fields(
        source,
        &format!("{T}/cas_config/backbone_config"),
        &["encoder", "is_encoder_decoder"],
    )?;
    known_fields(
        source,
        &format!("{S}/perception"),
        &[
            "encoder",
            "modality_adapter",
            "output_dim",
            "preprocessor",
            "target",
        ],
    )?;
    known_fields(
        source,
        &format!("{S}/perception/modality_adapter"),
        &["d_model"],
    )?;
    known_fields(
        source,
        &format!("{R}/decoder_config"),
        &[
            "normalization_mode",
            "random_state_sampling",
            "blank_as_pad",
            "prednet",
            "vocab_size",
        ],
    )?;
    known_fields(
        source,
        &format!("{R}/decoder_config/prednet"),
        &[
            "pred_hidden",
            "pred_rnn_layers",
            "t_max",
            "dropout",
            "forget_gate_bias",
            "weights_init_scale",
            "hidden_hidden_bias_scale",
            "rnn_hidden_size",
        ],
    )?;
    known_fields(
        source,
        &format!("{R}/joint_config"),
        &[
            "log_softmax",
            "preserve_memory",
            "fuse_loss_wer",
            "fused_batch_size",
            "jointnet",
            "num_classes",
            "vocabulary",
            "num_extra_outputs",
            "masking_prob",
        ],
    )?;
    known_fields(
        source,
        &format!("{R}/joint_config/jointnet"),
        &[
            "joint_hidden",
            "activation",
            "dropout",
            "encoder_hidden",
            "pred_hidden",
        ],
    )?;
    equals(source, &format!("{S}/pretrained_llm"), json!(BASE_MODEL_ID))?;
    equals(
        source,
        &format!("{G}/pretrained_lm_name"),
        json!(BASE_MODEL_ID),
    )?;
    equals(
        source,
        &format!("{T}/cas_config/pretrained_tokenizer_name"),
        json!(BASE_MODEL_ID),
    )?;
    equals(source, &format!("{T}/backbone_type"), json!("gemma3_text"))?;
    equals(
        source,
        &format!("{T}/cas_config/backbone_type"),
        json!("t5gemma"),
    )?;
    equals(
        source,
        &format!("{T}/cas_config/backbone_config/is_encoder_decoder"),
        json!(false),
    )?;
    for path in [
        format!("{T}/backbone_config_class"),
        format!("{T}/backbone_model_class"),
        format!("{T}/cas_config/backbone_config_class"),
        format!("{T}/cas_config/backbone_model_class"),
    ] {
        equals(source, &path, Value::Null)?;
    }
    for (key, expected) in [
        ("base_model_name", "backbone"),
        ("embed_tokens_name", "embeddings"),
        ("cache_key", "cache_params"),
    ] {
        equals(source, &format!("{S}/{key}"), json!(expected))?;
    }
    let rate: usize = read(source, "/data/source_sample_rate")?;
    let target_rate: usize = read(source, "/data/target_sample_rate")?;
    let frame: f64 = read(source, "/data/frame_length")?;
    for (path, value) in [
        ("/model/stt/data/source_sample_rate", json!(rate)),
        ("/model/stt/data/target_sample_rate", json!(target_rate)),
        (
            "/model/speech_generation/data/source_sample_rate",
            json!(target_rate),
        ),
        (
            "/model/speech_generation/data/target_sample_rate",
            json!(target_rate),
        ),
        ("/model/speech_generation/data/frame_length", json!(frame)),
    ] {
        equals(source, path, value)?;
    }
    if rate != 16000 || target_rate != 22050 || frame != 0.08 {
        return Err(invalid(
            "/data",
            "released cadence is 80 ms at 16000 Hz input and 22050 Hz output",
        ));
    }
    let codec: CodecConfig = component(source, &format!("{G}/codec_config"), json!({}), &[])?;
    let preprocessor = component(
        source,
        &format!("{S}/perception/preprocessor"),
        json!({
            "preemph":0.97, "lowfreq":0.0, "highfreq":8000.0, "log_zero_guard_type":"add",
            "log_zero_guard_value": 0.000000059604644775390625, "exact_pad":false,
            "mag_power":2.0, "mel_norm":"slaney", "use_torchaudio":false
        }),
        &[],
    )?;
    let encoder = component(
        source,
        &format!("{S}/perception/encoder"),
        json!({
            "subsampling_conv_chunking_factor":1, "global_tokens":0, "global_tokens_spacing":1,
            "global_attn_separate":false
        }),
        &[
            "dropout",
            "dropout_att",
            "dropout_emb",
            "dropout_pre_encoder",
            "stochastic_depth_drop_prob",
            "stochastic_depth_mode",
            "stochastic_depth_start_layer",
        ],
    )?;
    let perception = PerceptionConfig {
        preprocessor,
        encoder,
        modality_adapter_dim: read(source, &format!("{S}/perception/modality_adapter/d_model"))?,
        output_dim: read(source, &format!("{S}/perception/output_dim"))?,
    };
    let text_bos: String = read(source, &format!("{S}/bos_token"))?;
    let text_eos: String = read(source, &format!("{S}/eos_token"))?;
    let text_pad: String = read(source, &format!("{S}/pad_token"))?;
    for (name, token) in [
        ("bos_token", &text_bos),
        ("eos_token", &text_eos),
        ("pad_token", &text_pad),
    ] {
        equals(source, &format!("{G}/{name}"), json!(token))?;
        equals(source, &format!("{S}/override_tokens/{name}"), json!(token))?;
    }
    // duplex_ear_tts.py control code properties; duplex_stt_model.py user markers;
    // nemotron_voicechat_inference_wrapper.py function channel markers.
    let speech_pad = codec.codebook_size;
    let speech_eos = speech_pad
        .checked_add(1)
        .ok_or_else(|| invalid("codebook_size", "control code overflow"))?;
    let speech_bos = speech_pad
        .checked_add(2)
        .ok_or_else(|| invalid("codebook_size", "control code overflow"))?;
    for (name, expected) in [
        ("custom_speech_pad_id", speech_pad),
        ("custom_speech_eos_id", speech_eos),
        ("custom_speech_bos_id", speech_bos),
    ] {
        if source.pointer(&format!("{G}/{name}")).is_some() {
            equals(source, &format!("{G}/{name}"), json!(expected))?;
        }
    }
    let tokens = TokenPolicy {
        bos: text_bos,
        eos: text_eos,
        pad: text_pad,
        user_bos: "^".into(),
        user_eos: "$".into(),
        tool_call_start: "<SPECIAL_20>".into(),
        tool_call_end: "<SPECIAL_21>".into(),
        tool_response_end: "<SPECIAL_22>".into(),
        speech_bos,
        speech_eos,
        speech_pad,
    };
    let vocab_size = read(source, &format!("{R}/rnnt_vocab_size"))?;
    equals(
        source,
        &format!("{R}/decoder_config/vocab_size"),
        json!(vocab_size),
    )?;
    equals(
        source,
        &format!("{R}/joint_config/num_classes"),
        json!(vocab_size),
    )?;
    let rnnt = RnntConfig {
        vocab_size,
        blank_id: vocab_size,
        log_softmax: read(source, &format!("{R}/joint_config/log_softmax"))?,
        vocabulary: read(source, &format!("{R}/joint_config/vocabulary"))?,
        pred_hidden: read(source, &format!("{R}/decoder_config/prednet/pred_hidden"))?,
        pred_rnn_layers: read(
            source,
            &format!("{R}/decoder_config/prednet/pred_rnn_layers"),
        )?,
        encoder_hidden: read(source, &format!("{R}/joint_config/jointnet/encoder_hidden"))?,
        joint_hidden: read(source, &format!("{R}/joint_config/jointnet/joint_hidden"))?,
        activation: read(source, &format!("{R}/joint_config/jointnet/activation"))?,
        blank_as_pad: read(source, &format!("{R}/decoder_config/blank_as_pad"))?,
        normalization_mode: read(source, &format!("{R}/decoder_config/normalization_mode"))?,
        rnn_hidden_size: source
            .pointer(&format!("{R}/decoder_config/prednet/rnn_hidden_size"))
            .map(|v| serde_json::from_value(v.clone()))
            .transpose()?
            .unwrap_or(-1),
        num_extra_outputs: source
            .pointer(&format!("{R}/joint_config/num_extra_outputs"))
            .map(|v| serde_json::from_value(v.clone()))
            .transpose()?
            .unwrap_or(0),
    };
    equals(
        source,
        &format!("{R}/joint_config/jointnet/pred_hidden"),
        json!(rnnt.pred_hidden),
    )?;
    let tts = TtsConfig {
        backbone: gemma(source, &format!("{T}/backbone_config"), true)?,
        character_encoder: gemma(
            source,
            &format!("{T}/cas_config/backbone_config/encoder"),
            false,
        )?,
        mog: component(source, &format!("{T}/mog_head_config"), json!({}), &[])?,
        latent_size: read(source, &format!("{T}/latent_size"))?,
        codebook_size: read(source, &format!("{T}/codebook_size"))?,
        num_quantizers: read(source, &format!("{T}/num_quantizers"))?,
        context_hidden_size: read(source, &format!("{T}/context_hidden_size"))?,
        disable_eos_prediction: read(source, &format!("{T}/disable_eos_prediction"))?,
        num_delay_speech_tokens: read(source, &format!("{T}/num_delay_speech_tokens"))?,
        use_audio_prompt_frozen_projection: read(
            source,
            &format!("{T}/use_audio_prompt_frozen_projection"),
        )?,
        use_bos_eos_emb: read(source, &format!("{T}/use_bos_eos_emb"))?,
        use_subword_flag_emb: read(source, &format!("{T}/use_subword_flag_emb"))?,
        use_gated_fusion_for_text_audio: read(
            source,
            &format!("{T}/use_gated_fusion_for_text_audio"),
        )?,
        random_target_masking: read(source, &format!("{T}/random_target_masking"))?,
        exponent: read(source, &format!("{T}/exponent"))?,
        subword_mask_exactly_as_eartts: read(
            source,
            &format!("{G}/subword_mask_exactly_as_eartts"),
        )?,
        context_hidden_mask_exactly_as_eartts: read(
            source,
            &format!("{G}/context_hidden_mask_exactly_as_eartts"),
        )?,
        guidance_enabled: read(source, &format!("{G}/inference_guidance_enabled"))?,
        guidance_scale: read(source, &format!("{G}/inference_guidance_scale"))?,
        noise_scale: read(source, &format!("{G}/inference_noise_scale"))?,
        top_p: read(source, &format!("{G}/inference_top_p_or_k"))?,
        force_silence_on_eos: read(
            source,
            &format!("{G}/inference_force_speech_silence_on_eos"),
        )?,
        // duplex_ear_tts.py::_get_generation_config, not a Moshi depth schedule.
        num_iterations: 8,
        eos_threshold: -3.0,
        use_system_prompt: read(source, &format!("{G}/use_system_prompt"))?,
        audio_prompt_duration: read(
            source,
            "/model/speech_generation/data/audio_prompt_duration",
        )?,
    };
    let speaker_reference: Option<String> = read(source, "/model/inference_speaker_reference")?;
    if speaker_reference.as_deref().is_some_and(|s| !s.is_empty()) {
        return Err(invalid(
            "/model/inference_speaker_reference",
            "released admission requires fixed Aria conditioning",
        ));
    }
    Ok(VoiceChatConfig {
        cadence: Cadence {
            frame_length: frame,
            source_sample_rate: rate,
            target_sample_rate: target_rate,
            source_samples_per_frame: 1280,
            target_samples_per_frame: 1764,
        },
        perception,
        channels: read(source, S)?,
        tts,
        codec,
        rnnt,
        tokens,
        speaker_name: read(source, "/model/inference_speaker_name")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_release_resolves_numerically_material_defaults() {
        let config = VoiceChatConfig::from_json(RELEASED).unwrap();
        // Query scaling is an upstream default of 256, not this model's head dim 72.
        assert_eq!(config.tts.backbone.query_pre_attn_scalar, 256);
        assert_eq!(config.tts.backbone.layer_types[5], "full_attention");
        assert_eq!(config.tts.backbone.layer_types[6], "sliding_attention");
        assert_eq!(
            config.tts.character_encoder.attn_logit_softcapping,
            Some(50.0)
        );
        assert!(!config.tts.character_encoder.causal);
        assert_eq!(config.perception.preprocessor.preemph, 0.97);
        assert_eq!(config.rnnt.blank_id, 1024);
        assert_eq!(config.cadence.target_samples_per_frame, 1764);
        assert_eq!(config.tokens.speech_bos, 1026);
        let decoder = super::super::codec::waveform_decoder_config(&config.codec).unwrap();
        assert_eq!(decoder.neural.rates, [9, 7, 7]);
        assert_eq!(decoder.neural.channel_mult, [4, 2, 1]);
        assert_eq!(decoder.samples_per_frame().unwrap(), 1764);
    }

    #[test]
    fn altered_equations_geometry_and_cross_component_tokens_are_rejected() {
        for (path, replacement) in [
            ("/model/stt/model/perception/encoder/d_model", json!(768)),
            (
                "/model/speech_generation/model/tts_config/backbone_config/query_pre_attn_scalar",
                json!(72),
            ),
            ("/model/speech_generation/model/pad_token", json!("<pad>")),
            (
                "/_rnnt_merge_info/joint_config/jointnet/pred_hidden",
                json!(1024),
            ),
            (
                "/model/speech_generation/model/codec_config/rates",
                json!([7, 7, 8]),
            ),
        ] {
            let mut source: Value = serde_json::from_str(RELEASED).unwrap();
            if let Some(value) = source.pointer_mut(path) {
                *value = replacement;
            } else {
                let (parent, key) = path.rsplit_once('/').unwrap();
                source.pointer_mut(parent).unwrap()[key] = replacement;
            }
            assert!(VoiceChatConfig::from_value(&source).is_err(), "{path}");
        }
    }

    #[test]
    fn declarative_targets_are_inert_but_unknown_equation_fields_are_rejected() {
        let mut source: Value = serde_json::from_str(RELEASED).unwrap();
        source["model"]["stt"]["model"]["perception"]["encoder"]["_target_"] = json!("os.system");
        VoiceChatConfig::from_value(&source).unwrap();
        source["model"]["stt"]["model"]["perception"]["encoder"]["unrecognized_attention_policy"] =
            json!(true);
        assert!(matches!(
            VoiceChatConfig::from_value(&source),
            Err(ConfigError::InvalidField { .. })
        ));
    }
}
