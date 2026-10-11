//! Released VoiceChat checkpoint catalogs and canonical source recipes.
//!
//! Geometry is the pinned 11B release contract. GGUF schemas describe storage;
//! acceptance does not assert that every encoding has a native execution kernel.

use std::collections::BTreeMap;

use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeDtype},
    schema::{
        CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint, GgufTypeConstraint,
        SafetensorsCheckpointPlan, SafetensorsTensorConstraint, StoredDtypeConstraint,
    },
    store::TensorSelection,
    StoredDtype,
};
use eredu_gguf::GgmlType;

use super::config::VoiceChatConfig;

/// Independently retained roles in the released speech artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Component {
    /// Feature buffers, causal encoder and projection into the language model.
    Perception,
    /// Nemotron blocks, token embedding and agent-text output head.
    Backbone,
    /// Function-channel output projection.
    FunctionHead,
    /// Character encoder, speech generator, guidance and speaker conditioning.
    Tts,
    /// Waveform decoder parameters.
    CodecDecoder,
    /// Waveform encoder parameters, absent from the split GGUF bundle.
    CodecEncoder,
    /// Probabilistic residual quantizer parameters.
    CodecQuantizer,
    /// Transcription prediction and joint networks.
    Rnnt,
    /// Speech control and codec-silence token buffers.
    Controls,
}

/// Physical catalogs published at the pinned community GGUF revision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufLayout {
    /// Single `nemotron_voicechat` container with the complete NeMo catalog.
    Complete,
    /// `clip` FastConformer projector container.
    Perception,
    /// Extracted `nemotron_h` language model.
    Backbone,
    /// Separate `voicechat_function_head` container.
    FunctionHead,
    /// `voicechat_tts` generator, quantizer and codec decoder container.
    Tts,
}

impl GgufLayout {
    /// Exact architecture metadata identifying this storage layout.
    pub const fn architecture(self) -> &'static str {
        match self {
            Self::Complete => "nemotron_voicechat",
            Self::Perception => "clip",
            Self::Backbone => "nemotron_h",
            Self::FunctionHead => "voicechat_function_head",
            Self::Tts => "voicechat_tts",
        }
    }
}

/// Resolves a canonical released tensor identity to its component role.
/// Unknown namespaces are not admitted by this classifier or by the catalog.
pub fn component_for_key(key: &str) -> Option<Component> {
    if key.starts_with("stt_model.perception.") {
        Some(Component::Perception)
    } else if key.starts_with("stt_model.llm.")
        || key.starts_with("stt_model.embed_tokens.")
        || key.starts_with("stt_model.lm_head.")
    {
        Some(Component::Backbone)
    } else if key.starts_with("stt_model.function_head.") {
        Some(Component::FunctionHead)
    } else if key.starts_with("stt_model.rnnt_decoder.") || key.starts_with("stt_model.rnnt_joint.")
    {
        Some(Component::Rnnt)
    } else if key.starts_with("tts_model.audio_codec.encoder.") {
        Some(Component::CodecEncoder)
    } else if key.starts_with("tts_model.audio_codec.decoder.") {
        Some(Component::CodecDecoder)
    } else if key.starts_with("tts_model.audio_codec.prvq.") {
        Some(Component::CodecQuantizer)
    } else if key.starts_with("tts_model.tts_model.")
        || key.starts_with("tts_model.audio_prompt_latents.")
    {
        Some(Component::Tts)
    } else if matches!(
        key,
        "tts_model._control_codes" | "tts_model.codec_silence_tokens"
    ) {
        Some(Component::Controls)
    } else {
        None
    }
}

/// Strict released SafeTensors catalog, including affine companions and host buffers.
pub fn safetensors_plan(
    config: &VoiceChatConfig,
    formats: &BTreeMap<String, eredu_checkpoint::LinearFormat>,
) -> Result<SafetensorsCheckpointPlan, String> {
    config.validate().map_err(|e| e.to_string())?;
    SafetensorsCheckpointPlan::new(
        "VoiceChat 11B SafeTensors",
        released_catalog()
            .into_iter()
            .map(|t| {
                if t.integer {
                    Ok(vec![SafetensorsTensorConstraint::required(
                        t.key,
                        t.shape,
                        StoredDtypeConstraint::Exact(StoredDtype::I64),
                    )])
                } else {
                    eredu_checkpoint::schema::matrix_for_linear_format(
                        t.key.clone(),
                        std::iter::empty::<String>(),
                        t.shape,
                        formats
                            .get(&t.key)
                            .copied()
                            .unwrap_or(eredu_checkpoint::LinearFormat::Dense),
                        None,
                    )
                    .map_err(|e| e.to_string())
                }
            })
            .collect::<Result<Vec<_>, String>>()?
            .into_iter()
            .flatten()
            .collect(),
        Vec::new(),
        CatalogPolicy::strict(),
    )
    .map_err(|e| e.to_string())
}

pub(crate) fn validate_linear_formats(
    formats: &BTreeMap<String, eredu_checkpoint::LinearFormat>,
) -> Result<(), String> {
    let eligible: std::collections::BTreeSet<_> = released_catalog()
        .into_iter()
        .filter(|t| t.quantized && t.shape.len() == 2 && t.key.ends_with(".weight"))
        .map(|t| t.key)
        .collect();
    for (name, format) in formats {
        if !eligible.contains(name)
            || !matches!(
                format,
                eredu_checkpoint::LinearFormat::Dense | eredu_checkpoint::LinearFormat::Affine(_)
            )
        {
            return Err(format!("unsupported physical projection format for {name}"));
        }
        format.validate().map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Normalizes explicit per-module affine declarations into canonical weight names.
/// Geometry and companions are admitted by the same matrix schema as dense input.
pub fn configured_formats(
    source: &serde_json::Value,
) -> Result<BTreeMap<String, eredu_checkpoint::LinearFormat>, String> {
    let Some(declarations) = source
        .get("quantization")
        .or_else(|| source.pointer("/mlx_conversion/quantization/modules"))
    else {
        return Ok(BTreeMap::new());
    };
    let object = declarations
        .as_object()
        .ok_or("quantization declarations must be an object")?;
    let eligible: std::collections::BTreeSet<_> = released_catalog()
        .into_iter()
        .filter(|t| t.quantized && t.shape.len() == 2 && t.key.ends_with(".weight"))
        .map(|t| t.key)
        .collect();
    object
        .iter()
        .filter(|(key, _)| !matches!(key.as_str(), "bits" | "group_size" | "mode"))
        .map(|(module, declaration)| {
            let name = format!("{module}.weight");
            if !eligible.contains(&name) {
                return Err(format!("unknown affine projection {module}"));
            }
            let format: eredu_checkpoint::AffineQuantization =
                serde_json::from_value(declaration.clone()).map_err(|e| e.to_string())?;
            format.validate().map_err(|e| e.to_string())?;
            Ok((name, eredu_checkpoint::LinearFormat::Affine(format)))
        })
        .collect()
}

/// Retains physical formats for architecture-declared ordinary projection matrices.
/// Missing tensors remain ordinary schema errors; no payload is read.
pub fn safetensors_formats(
    config: &VoiceChatConfig,
    catalog: &dyn eredu_checkpoint::validation::SafetensorsCatalog,
) -> Result<BTreeMap<String, eredu_checkpoint::LinearFormat>, String> {
    config.validate().map_err(|e| e.to_string())?;
    let keys: std::collections::BTreeSet<_> = catalog.keys().into_iter().collect();
    released_catalog()
        .into_iter()
        .filter(|t| {
            !t.integer
                && t.quantized
                && t.shape.len() == 2
                && t.key.ends_with(".weight")
                && keys.contains(&t.key)
        })
        .map(|t| {
            eredu_checkpoint::schema::infer_matrix_linear_format(catalog, &t.key, &t.shape)
                .map(|format| (t.key, format))
                .map_err(|e| e.to_string())
        })
        .collect()
}

/// Strict GGUF physical plan. Quantized convolutions are admitted as storage,
/// with canonical recipes retaining their logical geometry for bounded conversion.
pub fn gguf_plan(
    config: &VoiceChatConfig,
    layout: GgufLayout,
) -> Result<GgufCheckpointPlan, String> {
    config.validate().map_err(|e| e.to_string())?;
    let tensors = if layout == GgufLayout::Complete {
        released_catalog()
            .into_iter()
            .map(|t| {
                let shape = if t.key.ends_with(".pad_tensor") {
                    vec![1]
                } else {
                    t.shape
                };
                let encoding = if t.integer {
                    GgufTypeConstraint::Exact(GgmlType::I64)
                } else if t.quantized {
                    GgufTypeConstraint::OneOf(vec![GgmlType::F16, GgmlType::Q4_0, GgmlType::Q8_0])
                } else {
                    GgufTypeConstraint::Exact(GgmlType::F16)
                };
                GgufTensorConstraint::required(t.key, shape, encoding)
            })
            .collect()
    } else {
        split_catalog(layout)
            .into_iter()
            .map(|t| {
                GgufTensorConstraint::required(
                    t.source,
                    t.shape,
                    GgufTypeConstraint::Exact(t.encoding),
                )
            })
            .collect()
    };
    GgufCheckpointPlan::new(
        format!("VoiceChat 11B {} GGUF", layout.architecture()),
        tensors,
        Vec::new(),
        CatalogPolicy::strict(),
    )
    .map_err(|e| e.to_string())
}

/// Canonical NeMo parameter recipes for one already validated physical source.
///
/// `None` selects official SafeTensors. Split TTS keeps its two derived character
/// tables in the retained physical source but does not treat them as model
/// parameters. Its two missing scalar padding buffers, RNNT and codec encoder
/// require companion sources when requested. No tensor payload is read here.
pub fn canonical_recipes(
    config: &VoiceChatConfig,
    layout: Option<GgufLayout>,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    config.validate().map_err(|e| e.to_string())?;
    let canonical = released_catalog()
        .into_iter()
        .map(|t| (t.key.clone(), t))
        .collect::<BTreeMap<_, _>>();
    match layout {
        None | Some(GgufLayout::Complete) => Ok(canonical
            .into_iter()
            .map(|(name, _)| {
                let mut recipe = DerivedWeightRecipe::source(&name, TensorSelection::Full);
                if layout.is_some() && name.ends_with(".pad_tensor") {
                    recipe = DerivedWeightRecipe::Reshape {
                        input: Box::new(recipe),
                        shape: Vec::new(),
                    };
                }
                (name, recipe)
            })
            .collect()),
        Some(layout) => split_catalog(layout)
            .into_iter()
            .filter(|t| !t.target.is_empty())
            .map(|t| {
                let target = canonical
                    .get(&t.target)
                    .ok_or_else(|| format!("unknown VoiceChat canonical tensor {}", t.target))?;
                let mut recipe = DerivedWeightRecipe::source(&t.source, TensorSelection::Full);
                if t.shape != target.shape {
                    recipe = DerivedWeightRecipe::Reshape {
                        input: Box::new(recipe),
                        shape: target.shape.clone(),
                    };
                }
                recipe = match t.transform {
                    Transform::Identity => recipe,
                    Transform::NegLog => DerivedWeightRecipe::NegLog {
                        input: Box::new(recipe),
                    },
                    Transform::SubtractOne => DerivedWeightRecipe::SubtractOne {
                        input: Box::new(recipe),
                    },
                };
                if target.integer {
                    recipe = DerivedWeightRecipe::Cast {
                        input: Box::new(recipe),
                        dtype: RecipeDtype::I64,
                    };
                }
                Ok((t.target, recipe))
            })
            .collect(),
    }
}

/// Reconstructs the complete container's declarative NeMo configuration.
/// Values are retained JSON data; Python target names are never executed.
pub fn config_from_gguf_metadata(
    metadata: &BTreeMap<String, eredu_gguf::MetadataValue>,
) -> Result<VoiceChatConfig, String> {
    let mut source = serde_json::Map::new();
    for (key, value) in metadata {
        if let Some(field) = key.strip_prefix("voicechat.config.") {
            let json = value
                .as_str()
                .ok_or_else(|| format!("VoiceChat GGUF {key} must be JSON text"))?;
            source.insert(
                field.to_owned(),
                serde_json::from_str(json)
                    .map_err(|error| format!("invalid VoiceChat GGUF {key}: {error}"))?,
            );
        }
    }
    VoiceChatConfig::from_value(&serde_json::Value::Object(source))
        .map_err(|error| error.to_string())
}

/// Validates architecture and equation metadata before source publication.
///
/// Complete containers carry the original nested NeMo configuration. Split
/// containers carry explicit numerical policy; all inference-relevant fields
/// must agree with the normalized released contract. Float metadata is compared
/// at its GGUF F32 precision. Descriptions and names do not select semantics.
pub fn validate_gguf_metadata(
    config: &VoiceChatConfig,
    layout: GgufLayout,
    metadata: &BTreeMap<String, eredu_gguf::MetadataValue>,
) -> Result<(), String> {
    config.validate().map_err(|e| e.to_string())?;
    for (key, expected) in metadata_expectations(layout) {
        let actual = metadata
            .get(&key)
            .ok_or_else(|| format!("missing VoiceChat GGUF metadata {key}"))?;
        let matches = match &expected {
            serde_json::Value::String(v) => actual.as_str() == Some(v.as_str()),
            serde_json::Value::Bool(v) => actual.as_bool() == Some(*v),
            serde_json::Value::Number(v) if v.is_i64() || v.is_u64() => {
                actual.as_i64() == v.as_i64()
            }
            serde_json::Value::Number(v) => actual.as_f32() == v.as_f64().map(|v| v as f32),
            serde_json::Value::Array(v) => {
                actual.as_array().and_then(|a| a.to_i64_vec())
                    == v.iter().map(|v| v.as_i64()).collect::<Option<Vec<_>>>()
            }
            _ => false,
        };
        if !matches {
            return Err(format!(
                "VoiceChat GGUF metadata {key} disagrees with released policy: expected {expected}, got {actual:?}"
            ));
        }
    }
    if layout == GgufLayout::Complete {
        let retained = config_from_gguf_metadata(metadata)?;
        if &retained != config {
            return Err(
                "VoiceChat GGUF embedded configuration disagrees with prepared configuration"
                    .into(),
            );
        }
    }
    if layout == GgufLayout::Backbone {
        let tokens = metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_strings())
            .filter(|v| v.len() == 131072)
            .ok_or_else(|| "VoiceChat GGUF tokenizer requires exactly 131072 strings".to_owned())?;
        for (index, expected) in [
            (1, &config.tokens.bos),
            (2, &config.tokens.eos),
            (12, &config.tokens.pad),
            (20, &config.tokens.tool_call_start),
            (21, &config.tokens.tool_call_end),
            (22, &config.tokens.tool_response_end),
        ] {
            if &tokens[index] != expected {
                return Err(format!(
                    "VoiceChat GGUF tokenizer token {index} disagrees with {expected}"
                ));
            }
        }
        let types = metadata
            .get("tokenizer.ggml.token_type")
            .and_then(|v| v.as_array())
            .and_then(|v| v.to_i64_vec())
            .filter(|v| v.len() == tokens.len())
            .ok_or_else(|| {
                "VoiceChat GGUF token types must cover the complete vocabulary".to_owned()
            })?;
        if [1, 2, 12, 20, 21, 22].iter().any(|i| types[*i] != 3) {
            return Err("VoiceChat GGUF control tokens must be marked special".into());
        }
        metadata
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.as_strings())
            .filter(|v| v.len() == 269443)
            .ok_or_else(|| {
                "VoiceChat GGUF tokenizer requires the retained 269443 merge entries".to_owned()
            })?;
    }
    Ok(())
}

// These values are part of the exact released storage contract. Config
// validation fixes the same architecture dimensions and upstream defaults.
#[rustfmt::skip]
fn metadata_expectations(layout: GgufLayout) -> BTreeMap<String, serde_json::Value> {
    let expected = match layout {
        GgufLayout::Complete => r#"{
            "general.architecture": "nemotron_voicechat",
            "voicechat.gguf_version": "voicechat-custom-f16-v1",
            "voicechat.source_format": "safetensors",
            "voicechat.source_dtype": "mixed",
            "voicechat.converted_dtype": "F16"
}"#,
        GgufLayout::Perception => r#"{
            "general.architecture": "clip",
            "general.type": "mmproj",
            "clip.has_audio_encoder": true,
            "clip.projector_type": "voicechat",
            "clip.audio.embedding_length": 1024,
            "clip.audio.feed_forward_length": 4096,
            "clip.audio.block_count": 24,
            "clip.audio.attention.head_count": 8,
            "clip.audio.projection_dim": 4480,
            "clip.audio.attention.layer_norm_epsilon": 9.999999747378752e-06,
            "clip.audio.num_mel_bins": 128,
            "clip.audio.subsampling_factor": 8,
            "clip.audio.conv_kernel_size": 9,
            "clip.audio.window_size": 70,
            "general.file_type": 2
}"#,
        GgufLayout::FunctionHead => r#"{
            "general.architecture": "voicechat_function_head",
            "general.type": "model",
            "voicechat.fusion.text_weight": 1.0,
            "voicechat.fusion.audio_weight": 1.0,
            "voicechat.fusion.function_weight": 2.0,
            "voicechat.function.sotc_token_id": 20,
            "voicechat.function.eotc_token_id": 21,
            "voicechat.function.eotr_token_id": 22
}"#,
        GgufLayout::Backbone => r#"{
            "general.architecture": "nemotron_h",
            "general.type": "model",
            "nemotron_h.context_length": 131072,
            "nemotron_h.embedding_length": 4480,
            "nemotron_h.block_count": 56,
            "nemotron_h.vocab_size": 131072,
            "nemotron_h.feed_forward_length": [
                        0,
                        15680,
                        0,
                        15680,
                        0,
                        15680,
                        0,
                        0,
                        15680,
                        0,
                        15680,
                        0,
                        15680,
                        0,
                        0,
                        15680,
                        0,
                        15680,
                        0,
                        15680,
                        0,
                        0,
                        15680,
                        0,
                        15680,
                        0,
                        15680,
                        0,
                        15680,
                        0,
                        0,
                        15680,
                        0,
                        15680,
                        0,
                        15680,
                        0,
                        15680,
                        0,
                        0,
                        15680,
                        0,
                        15680,
                        0,
                        0,
                        15680,
                        0,
                        15680,
                        0,
                        15680,
                        0,
                        15680,
                        0,
                        15680,
                        0,
                        15680
            ],
            "nemotron_h.attention.head_count": 40,
            "nemotron_h.attention.head_count_kv": [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        8,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        8,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        8,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        8,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0
            ],
            "nemotron_h.attention.key_length": 128,
            "nemotron_h.attention.value_length": 128,
            "nemotron_h.attention.layer_norm_rms_epsilon": 9.999999747378752e-06,
            "nemotron_h.attention.layer_norm_epsilon": 9.999999747378752e-06,
            "nemotron_h.ssm.conv_kernel": 4,
            "nemotron_h.ssm.inner_size": 10240,
            "nemotron_h.ssm.state_size": 128,
            "nemotron_h.ssm.group_count": 8,
            "nemotron_h.ssm.time_step_rank": 128,
            "general.file_type": 2,
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.pre": "tekken",
            "tokenizer.ggml.bos_token_id": 1,
            "tokenizer.ggml.eos_token_id": 2,
            "tokenizer.ggml.padding_token_id": 12,
            "tokenizer.ggml.add_bos_token": false,
            "tokenizer.ggml.add_eos_token": false
}"#,
        GgufLayout::Tts => r#"{
            "general.architecture": "voicechat_tts",
            "general.type": "model",
            "voicechat_tts.block_count": 28,
            "voicechat_tts.embedding_length": 1152,
            "voicechat_tts.feed_forward_length": 4608,
            "voicechat_tts.attention.head_count": 16,
            "voicechat_tts.attention.head_count_kv": 16,
            "voicechat_tts.attention.key_length": 72,
            "voicechat_tts.attention.value_length": 72,
            "voicechat_tts.attention.sliding_window": 7500,
            "voicechat_tts.attention.layer_norm_rms_epsilon": 9.999999974752427e-07,
            "general.file_type": 2,
            "voicechat.tts.latent_size": 512,
            "voicechat.tts.codebook_size": 1024,
            "voicechat.tts.num_quantizers": 31,
            "voicechat.tts.num_delay_speech_tokens": 2,
            "voicechat.tts.mog.num_layers": 3,
            "voicechat.tts.mog.low_rank": 64,
            "voicechat.tts.mog.num_predictions": 1024,
            "voicechat.codec.base_hidden_size": 384,
            "voicechat.codec.num_blocks": 3,
            "voicechat.codec.kernel_size": 7,
            "voicechat.codec.n_fft": 16,
            "voicechat.codec.hop_length": 4,
            "voicechat.codec.sample_rate": 22050,
            "voicechat.codec.wav_to_token_ratio": 1764,
            "voicechat.codec.rates": [
                        7,
                        7,
                        9
            ],
            "voicechat.codec.channel_mult": [
                        1,
                        2,
                        4
            ],
            "voicechat.tts.mog.min_log_std": -4.0,
            "voicechat.tts.mog.eps": 9.999999974752427e-07,
            "voicechat.tts.guidance_scale": 0.20000000298023224,
            "voicechat.tts.noise_scale": 0.0010000000474974513,
            "voicechat.tts.speaker": "Aria",
            "voicechat.tts.num_iter": 8,
            "voicechat.tts.top_p": 0.949999988079071,
            "voicechat.tts.exponent": 3.0,
            "voicechat.tts.force_silence_on_eos": true,
            "voicechat_tts.rope.freq_base": 1000000.0,
            "voicechat.tts.rope_local_theta": 10000.0,
            "voicechat.tts.swa_pattern": 6,
            "voicechat.tts.query_pre_attn_scalar": 256,
            "voicechat.tts.cas.rope_theta": 10000.0,
            "voicechat.tts.cas.attn_softcap": 50.0,
            "voicechat.tts.cas.query_pre_attn_scalar": 256,
            "voicechat.tts.text_bos_id": 1,
            "voicechat.tts.text_eos_id": 2,
            "voicechat.tts.text_pad_id": 12
}"#,
    };
    serde_json::from_str(expected).expect("static metadata object")
}

struct ReleasedTensor {
    key: String,
    shape: Vec<usize>,
    integer: bool,
    quantized: bool,
}
#[derive(Clone, Copy)]
enum Transform {
    Identity,
    NegLog,
    SubtractOne,
}
struct SplitTensor {
    source: String,
    target: String,
    shape: Vec<usize>,
    encoding: GgmlType,
    transform: Transform,
}

fn add(
    out: &mut Vec<ReleasedTensor>,
    pattern: &str,
    indices: &[usize],
    shape: &[usize],
    integer: bool,
    quantized: bool,
) {
    let keys = if indices.is_empty() {
        vec![pattern.to_owned()]
    } else {
        indices
            .iter()
            .map(|i| pattern.replace("{}", &i.to_string()))
            .collect()
    };
    out.extend(keys.into_iter().map(|key| ReleasedTensor {
        key,
        shape: shape.to_vec(),
        integer,
        quantized,
    }));
}

// Exact tensor geometry from NVIDIA model.safetensors at
// 443794ea956ef0065f001967ffd00e77f519cb39. Repeated indices denote stored
// parameter identities, not replicated logical invocations. `quantized` marks
// tensors stored as Q4_0/Q8_0 in the complete published GGUF variants.
#[rustfmt::skip]
fn released_catalog() -> Vec<ReleasedTensor> {
    let mut out = Vec::new();
    add(&mut out, "stt_model.embed_tokens.weight", &[], &[131072, 4480], false, true);
    add(&mut out, "stt_model.function_head.weight", &[], &[131072, 4480], false, true);
    add(&mut out, "stt_model.llm.layers.{}.mixer.A_log", &[0, 11, 13, 16, 18, 2, 20, 23, 25, 27, 29, 32, 34, 36, 38, 4, 41, 43, 44, 46, 48, 50, 52, 54, 6, 7, 9], &[128], false, false);
    add(&mut out, "stt_model.llm.layers.{}.mixer.D", &[0, 11, 13, 16, 18, 2, 20, 23, 25, 27, 29, 32, 34, 36, 38, 4, 41, 43, 44, 46, 48, 50, 52, 54, 6, 7, 9], &[128], false, false);
    add(&mut out, "stt_model.llm.layers.{}.mixer.conv1d.bias", &[0, 11, 13, 16, 18, 2, 20, 23, 25, 27, 29, 32, 34, 36, 38, 4, 41, 43, 44, 46, 48, 50, 52, 54, 6, 7, 9], &[12288], false, false);
    add(&mut out, "stt_model.llm.layers.{}.mixer.conv1d.weight", &[0, 11, 13, 16, 18, 2, 20, 23, 25, 27, 29, 32, 34, 36, 38, 4, 41, 43, 44, 46, 48, 50, 52, 54, 6, 7, 9], &[12288, 1, 4], false, false);
    add(&mut out, "stt_model.llm.layers.{}.mixer.down_proj.weight", &[1, 10, 12, 15, 17, 19, 22, 24, 26, 28, 3, 31, 33, 35, 37, 40, 42, 45, 47, 49, 5, 51, 53, 55, 8], &[4480, 15680], false, true);
    add(&mut out, "stt_model.llm.layers.{}.mixer.dt_bias", &[0, 11, 13, 16, 18, 2, 20, 23, 25, 27, 29, 32, 34, 36, 38, 4, 41, 43, 44, 46, 48, 50, 52, 54, 6, 7, 9], &[128], false, false);
    add(&mut out, "stt_model.llm.layers.{}.mixer.in_proj.weight", &[0, 11, 13, 16, 18, 2, 20, 23, 25, 27, 29, 32, 34, 36, 38, 4, 41, 43, 44, 46, 48, 50, 52, 54, 6, 7, 9], &[22656, 4480], false, true);
    add(&mut out, "stt_model.llm.layers.{}.mixer.k_proj.weight", &[14, 21, 30, 39], &[1024, 4480], false, true);
    add(&mut out, "stt_model.llm.layers.{}.mixer.norm.weight", &[0, 11, 13, 16, 18, 2, 20, 23, 25, 27, 29, 32, 34, 36, 38, 4, 41, 43, 44, 46, 48, 50, 52, 54, 6, 7, 9], &[10240], false, false);
    add(&mut out, "stt_model.llm.layers.{}.mixer.o_proj.weight", &[14, 21, 30, 39], &[4480, 5120], false, true);
    add(&mut out, "stt_model.llm.layers.{}.mixer.out_proj.weight", &[0, 11, 13, 16, 18, 2, 20, 23, 25, 27, 29, 32, 34, 36, 38, 4, 41, 43, 44, 46, 48, 50, 52, 54, 6, 7, 9], &[4480, 10240], false, true);
    add(&mut out, "stt_model.llm.layers.{}.mixer.q_proj.weight", &[14, 21, 30, 39], &[5120, 4480], false, true);
    add(&mut out, "stt_model.llm.layers.{}.mixer.up_proj.weight", &[1, 10, 12, 15, 17, 19, 22, 24, 26, 28, 3, 31, 33, 35, 37, 40, 42, 45, 47, 49, 5, 51, 53, 55, 8], &[15680, 4480], false, true);
    add(&mut out, "stt_model.llm.layers.{}.mixer.v_proj.weight", &[14, 21, 30, 39], &[1024, 4480], false, true);
    add(&mut out, "stt_model.llm.layers.{}.norm.weight", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 3, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 4, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 5, 50, 51, 52, 53, 54, 55, 6, 7, 8, 9], &[4480], false, false);
    add(&mut out, "stt_model.llm.norm_f.weight", &[], &[4480], false, false);
    add(&mut out, "stt_model.lm_head.weight", &[], &[131072, 4480], false, true);
    add(&mut out, "stt_model.perception.encoder.layers.{}.conv.batch_norm.bias", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 3, 4, 5, 6, 7, 8, 9], &[1024], false, false);
    add(&mut out, "stt_model.perception.encoder.layers.{}.conv.batch_norm.weight", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 3, 4, 5, 6, 7, 8, 9], &[1024], false, false);
    add(&mut out, "stt_model.perception.encoder.layers.{}.conv.depthwise_conv.weight", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 3, 4, 5, 6, 7, 8, 9], &[1024, 1, 9], false, false);
    add(&mut out, "stt_model.perception.encoder.layers.{}.conv.pointwise_conv1.weight", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 3, 4, 5, 6, 7, 8, 9], &[2048, 1024, 1], false, false);
    add(&mut out, "stt_model.perception.encoder.layers.{}.conv.pointwise_conv2.weight", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 3, 4, 5, 6, 7, 8, 9], &[1024, 1024, 1], false, false);
    add(&mut out, "stt_model.perception.encoder.layers.{}.feed_forward1.linear1.weight", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 3, 4, 5, 6, 7, 8, 9], &[4096, 1024], false, true);
    add(&mut out, "stt_model.perception.encoder.layers.{}.feed_forward1.linear2.weight", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 3, 4, 5, 6, 7, 8, 9], &[1024, 4096], false, true);
    add(&mut out, "stt_model.perception.encoder.layers.{}.feed_forward2.linear1.weight", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 3, 4, 5, 6, 7, 8, 9], &[4096, 1024], false, true);
    add(&mut out, "stt_model.perception.encoder.layers.{}.feed_forward2.linear2.weight", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 3, 4, 5, 6, 7, 8, 9], &[1024, 4096], false, true);
    add(&mut out, "stt_model.perception.encoder.layers.{}.norm_conv.bias", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 3, 4, 5, 6, 7, 8, 9], &[1024], false, false);
    add(&mut out, "stt_model.perception.encoder.layers.{}.norm_conv.weight", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 3, 4, 5, 6, 7, 8, 9], &[1024], false, false);
    add(&mut out, "stt_model.perception.encoder.layers.{}.norm_feed_forward1.bias", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 3, 4, 5, 6, 7, 8, 9], &[1024], false, false);
    add(&mut out, "stt_model.perception.encoder.layers.{}.norm_feed_forward1.weight", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 3, 4, 5, 6, 7, 8, 9], &[1024], false, false);
    add(&mut out, "stt_model.perception.encoder.layers.{}.norm_feed_forward2.bias", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 3, 4, 5, 6, 7, 8, 9], &[1024], false, false);
    add(&mut out, "stt_model.perception.encoder.layers.{}.norm_feed_forward2.weight", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 3, 4, 5, 6, 7, 8, 9], &[1024], false, false);
    add(&mut out, "stt_model.perception.encoder.layers.{}.norm_out.bias", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 3, 4, 5, 6, 7, 8, 9], &[1024], false, false);
    add(&mut out, "stt_model.perception.encoder.layers.{}.norm_out.weight", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 3, 4, 5, 6, 7, 8, 9], &[1024], false, false);
    add(&mut out, "stt_model.perception.encoder.layers.{}.norm_self_att.bias", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 3, 4, 5, 6, 7, 8, 9], &[1024], false, false);
    add(&mut out, "stt_model.perception.encoder.layers.{}.norm_self_att.weight", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 3, 4, 5, 6, 7, 8, 9], &[1024], false, false);
    add(&mut out, "stt_model.perception.encoder.layers.{}.self_attn.linear_k.weight", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 3, 4, 5, 6, 7, 8, 9], &[1024, 1024], false, true);
    add(&mut out, "stt_model.perception.encoder.layers.{}.self_attn.linear_out.weight", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 3, 4, 5, 6, 7, 8, 9], &[1024, 1024], false, true);
    add(&mut out, "stt_model.perception.encoder.layers.{}.self_attn.linear_pos.weight", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 3, 4, 5, 6, 7, 8, 9], &[1024, 1024], false, true);
    add(&mut out, "stt_model.perception.encoder.layers.{}.self_attn.linear_q.weight", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 3, 4, 5, 6, 7, 8, 9], &[1024, 1024], false, true);
    add(&mut out, "stt_model.perception.encoder.layers.{}.self_attn.linear_v.weight", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 3, 4, 5, 6, 7, 8, 9], &[1024, 1024], false, true);
    add(&mut out, "stt_model.perception.encoder.layers.{}.self_attn.pos_bias_u", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 3, 4, 5, 6, 7, 8, 9], &[8, 128], false, true);
    add(&mut out, "stt_model.perception.encoder.layers.{}.self_attn.pos_bias_v", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 3, 4, 5, 6, 7, 8, 9], &[8, 128], false, true);
    add(&mut out, "stt_model.perception.encoder.pre_encode.conv.{}.bias", &[0, 2, 3, 5, 6], &[256], false, false);
    add(&mut out, "stt_model.perception.encoder.pre_encode.conv.{}.weight", &[0, 2, 5], &[256, 1, 3, 3], false, false);
    add(&mut out, "stt_model.perception.encoder.pre_encode.conv.{}.weight", &[3, 6], &[256, 256, 1, 1], false, false);
    add(&mut out, "stt_model.perception.encoder.pre_encode.out.bias", &[], &[1024], false, false);
    add(&mut out, "stt_model.perception.encoder.pre_encode.out.weight", &[], &[1024, 4352], false, true);
    add(&mut out, "stt_model.perception.preprocessor.featurizer.fb", &[], &[1, 128, 257], false, false);
    add(&mut out, "stt_model.perception.preprocessor.featurizer.window", &[], &[400], false, false);
    add(&mut out, "stt_model.perception.proj.bias", &[], &[4480], false, false);
    add(&mut out, "stt_model.perception.proj.weight", &[], &[4480, 1024], false, true);
    add(&mut out, "stt_model.rnnt_decoder.prediction.dec_rnn.lstm.bias_hh_l0", &[], &[2560], false, false);
    add(&mut out, "stt_model.rnnt_decoder.prediction.dec_rnn.lstm.bias_hh_l1", &[], &[2560], false, false);
    add(&mut out, "stt_model.rnnt_decoder.prediction.dec_rnn.lstm.bias_ih_l0", &[], &[2560], false, false);
    add(&mut out, "stt_model.rnnt_decoder.prediction.dec_rnn.lstm.bias_ih_l1", &[], &[2560], false, false);
    add(&mut out, "stt_model.rnnt_decoder.prediction.dec_rnn.lstm.weight_hh_l0", &[], &[2560, 640], false, true);
    add(&mut out, "stt_model.rnnt_decoder.prediction.dec_rnn.lstm.weight_hh_l1", &[], &[2560, 640], false, true);
    add(&mut out, "stt_model.rnnt_decoder.prediction.dec_rnn.lstm.weight_ih_l0", &[], &[2560, 640], false, true);
    add(&mut out, "stt_model.rnnt_decoder.prediction.dec_rnn.lstm.weight_ih_l1", &[], &[2560, 640], false, true);
    add(&mut out, "stt_model.rnnt_decoder.prediction.embed.weight", &[], &[1025, 640], false, true);
    add(&mut out, "stt_model.rnnt_joint.enc.bias", &[], &[640], false, false);
    add(&mut out, "stt_model.rnnt_joint.enc.weight", &[], &[640, 1024], false, true);
    add(&mut out, "stt_model.rnnt_joint.joint_net.{}.bias", &[2], &[1025], false, false);
    add(&mut out, "stt_model.rnnt_joint.joint_net.{}.weight", &[2], &[1025, 640], false, true);
    add(&mut out, "stt_model.rnnt_joint.pred.bias", &[], &[640], false, false);
    add(&mut out, "stt_model.rnnt_joint.pred.weight", &[], &[640, 640], false, true);
    add(&mut out, "tts_model._control_codes", &[], &[3], true, false);
    add(&mut out, "tts_model.audio_codec.decoder.layers.{}.dwconv.bias", &[10, 11, 9], &[384], false, false);
    add(&mut out, "tts_model.audio_codec.decoder.layers.{}.dwconv.bias", &[5, 6, 7], &[768], false, false);
    add(&mut out, "tts_model.audio_codec.decoder.layers.{}.dwconv.bias", &[1, 2, 3], &[1536], false, false);
    add(&mut out, "tts_model.audio_codec.decoder.layers.{}.dwconv.weight", &[10, 11, 9], &[384, 1, 7], false, false);
    add(&mut out, "tts_model.audio_codec.decoder.layers.{}.dwconv.weight", &[5, 6, 7], &[768, 1, 7], false, false);
    add(&mut out, "tts_model.audio_codec.decoder.layers.{}.dwconv.weight", &[1, 2, 3], &[1536, 1, 7], false, false);
    add(&mut out, "tts_model.audio_codec.decoder.layers.{}.norm.bias", &[10, 11, 9], &[384], false, false);
    add(&mut out, "tts_model.audio_codec.decoder.layers.{}.norm.bias", &[5, 6, 7], &[768], false, false);
    add(&mut out, "tts_model.audio_codec.decoder.layers.{}.norm.bias", &[1, 2, 3], &[1536], false, false);
    add(&mut out, "tts_model.audio_codec.decoder.layers.{}.norm.weight", &[10, 11, 9], &[384], false, false);
    add(&mut out, "tts_model.audio_codec.decoder.layers.{}.norm.weight", &[5, 6, 7], &[768], false, false);
    add(&mut out, "tts_model.audio_codec.decoder.layers.{}.norm.weight", &[1, 2, 3], &[1536], false, false);
    add(&mut out, "tts_model.audio_codec.decoder.layers.{}.pwconv1.bias", &[10, 11, 9], &[1536], false, false);
    add(&mut out, "tts_model.audio_codec.decoder.layers.{}.pwconv1.bias", &[5, 6, 7], &[3072], false, false);
    add(&mut out, "tts_model.audio_codec.decoder.layers.{}.pwconv1.bias", &[1, 2, 3], &[6144], false, false);
    add(&mut out, "tts_model.audio_codec.decoder.layers.{}.pwconv1.weight", &[10, 11, 9], &[1536, 384, 1], false, false);
    add(&mut out, "tts_model.audio_codec.decoder.layers.{}.pwconv1.weight", &[5, 6, 7], &[3072, 768, 1], false, false);
    add(&mut out, "tts_model.audio_codec.decoder.layers.{}.pwconv1.weight", &[1, 2, 3], &[6144, 1536, 1], false, false);
    add(&mut out, "tts_model.audio_codec.decoder.layers.{}.pwconv2.bias", &[10, 11, 9], &[384], false, false);
    add(&mut out, "tts_model.audio_codec.decoder.layers.{}.pwconv2.bias", &[5, 6, 7], &[768], false, false);
    add(&mut out, "tts_model.audio_codec.decoder.layers.{}.pwconv2.bias", &[1, 2, 3], &[1536], false, false);
    add(&mut out, "tts_model.audio_codec.decoder.layers.{}.pwconv2.weight", &[10, 11, 9], &[384, 1536, 1], false, false);
    add(&mut out, "tts_model.audio_codec.decoder.layers.{}.pwconv2.weight", &[5, 6, 7], &[768, 3072, 1], false, false);
    add(&mut out, "tts_model.audio_codec.decoder.layers.{}.pwconv2.weight", &[1, 2, 3], &[1536, 6144, 1], false, false);
    add(&mut out, "tts_model.audio_codec.decoder.layers.{}.weight", &[12], &[18, 384, 1], false, false);
    add(&mut out, "tts_model.audio_codec.decoder.layers.{}.weight", &[0], &[512, 1536, 9], false, false);
    add(&mut out, "tts_model.audio_codec.decoder.layers.{}.weight", &[8], &[768, 384, 7], false, false);
    add(&mut out, "tts_model.audio_codec.decoder.layers.{}.weight", &[4], &[1536, 768, 7], false, false);
    add(&mut out, "tts_model.audio_codec.encoder.layers.{}.dwconv.bias", &[1, 2, 3], &[384], false, false);
    add(&mut out, "tts_model.audio_codec.encoder.layers.{}.dwconv.bias", &[5, 6, 7], &[768], false, false);
    add(&mut out, "tts_model.audio_codec.encoder.layers.{}.dwconv.bias", &[10, 11, 9], &[1536], false, false);
    add(&mut out, "tts_model.audio_codec.encoder.layers.{}.dwconv.weight", &[1, 2, 3], &[384, 1, 7], false, false);
    add(&mut out, "tts_model.audio_codec.encoder.layers.{}.dwconv.weight", &[5, 6, 7], &[768, 1, 7], false, false);
    add(&mut out, "tts_model.audio_codec.encoder.layers.{}.dwconv.weight", &[10, 11, 9], &[1536, 1, 7], false, false);
    add(&mut out, "tts_model.audio_codec.encoder.layers.{}.norm.bias", &[1, 2, 3], &[384], false, false);
    add(&mut out, "tts_model.audio_codec.encoder.layers.{}.norm.bias", &[5, 6, 7], &[768], false, false);
    add(&mut out, "tts_model.audio_codec.encoder.layers.{}.norm.bias", &[10, 11, 9], &[1536], false, false);
    add(&mut out, "tts_model.audio_codec.encoder.layers.{}.norm.weight", &[1, 2, 3], &[384], false, false);
    add(&mut out, "tts_model.audio_codec.encoder.layers.{}.norm.weight", &[5, 6, 7], &[768], false, false);
    add(&mut out, "tts_model.audio_codec.encoder.layers.{}.norm.weight", &[10, 11, 9], &[1536], false, false);
    add(&mut out, "tts_model.audio_codec.encoder.layers.{}.pwconv1.bias", &[1, 2, 3], &[1536], false, false);
    add(&mut out, "tts_model.audio_codec.encoder.layers.{}.pwconv1.bias", &[5, 6, 7], &[3072], false, false);
    add(&mut out, "tts_model.audio_codec.encoder.layers.{}.pwconv1.bias", &[10, 11, 9], &[6144], false, false);
    add(&mut out, "tts_model.audio_codec.encoder.layers.{}.pwconv1.weight", &[1, 2, 3], &[1536, 384, 1], false, false);
    add(&mut out, "tts_model.audio_codec.encoder.layers.{}.pwconv1.weight", &[5, 6, 7], &[3072, 768, 1], false, false);
    add(&mut out, "tts_model.audio_codec.encoder.layers.{}.pwconv1.weight", &[10, 11, 9], &[6144, 1536, 1], false, false);
    add(&mut out, "tts_model.audio_codec.encoder.layers.{}.pwconv2.bias", &[1, 2, 3], &[384], false, false);
    add(&mut out, "tts_model.audio_codec.encoder.layers.{}.pwconv2.bias", &[5, 6, 7], &[768], false, false);
    add(&mut out, "tts_model.audio_codec.encoder.layers.{}.pwconv2.bias", &[10, 11, 9], &[1536], false, false);
    add(&mut out, "tts_model.audio_codec.encoder.layers.{}.pwconv2.weight", &[1, 2, 3], &[384, 1536, 1], false, false);
    add(&mut out, "tts_model.audio_codec.encoder.layers.{}.pwconv2.weight", &[5, 6, 7], &[768, 3072, 1], false, false);
    add(&mut out, "tts_model.audio_codec.encoder.layers.{}.pwconv2.weight", &[10, 11, 9], &[1536, 6144, 1], false, false);
    add(&mut out, "tts_model.audio_codec.encoder.layers.{}.weight", &[0], &[384, 18, 1], false, false);
    add(&mut out, "tts_model.audio_codec.encoder.layers.{}.weight", &[12], &[512, 1536, 9], false, false);
    add(&mut out, "tts_model.audio_codec.encoder.layers.{}.weight", &[4], &[768, 384, 7], false, false);
    add(&mut out, "tts_model.audio_codec.encoder.layers.{}.weight", &[8], &[1536, 768, 7], false, false);
    add(&mut out, "tts_model.audio_codec.prvq._variance_list.{}.variance", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 3, 30, 4, 5, 6, 7, 8, 9], &[], false, false);
    add(&mut out, "tts_model.audio_codec.prvq.mus_list.{}", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 3, 30, 4, 5, 6, 7, 8, 9], &[1024, 512], false, true);
    add(&mut out, "tts_model.audio_prompt_latents.Aria", &[], &[1, 37, 1152], false, false);
    add(&mut out, "tts_model.codec_silence_tokens", &[], &[31], true, false);
    add(&mut out, "tts_model.tts_model.audio_prompt_projection_W", &[], &[1152, 1152], false, true);
    add(&mut out, "tts_model.tts_model.backbone.layers.{}.input_layernorm.weight", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 3, 4, 5, 6, 7, 8, 9], &[1152], false, false);
    add(&mut out, "tts_model.tts_model.backbone.layers.{}.mlp.down_proj.weight", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 3, 4, 5, 6, 7, 8, 9], &[1152, 4608], false, true);
    add(&mut out, "tts_model.tts_model.backbone.layers.{}.mlp.gate_proj.weight", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 3, 4, 5, 6, 7, 8, 9], &[4608, 1152], false, true);
    add(&mut out, "tts_model.tts_model.backbone.layers.{}.mlp.up_proj.weight", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 3, 4, 5, 6, 7, 8, 9], &[4608, 1152], false, true);
    add(&mut out, "tts_model.tts_model.backbone.layers.{}.post_attention_layernorm.weight", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 3, 4, 5, 6, 7, 8, 9], &[1152], false, false);
    add(&mut out, "tts_model.tts_model.backbone.layers.{}.post_feedforward_layernorm.weight", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 3, 4, 5, 6, 7, 8, 9], &[1152], false, false);
    add(&mut out, "tts_model.tts_model.backbone.layers.{}.pre_feedforward_layernorm.weight", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 3, 4, 5, 6, 7, 8, 9], &[1152], false, false);
    add(&mut out, "tts_model.tts_model.backbone.layers.{}.self_attn.k_norm.weight", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 3, 4, 5, 6, 7, 8, 9], &[72], false, false);
    add(&mut out, "tts_model.tts_model.backbone.layers.{}.self_attn.k_proj.weight", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 3, 4, 5, 6, 7, 8, 9], &[1152, 1152], false, true);
    add(&mut out, "tts_model.tts_model.backbone.layers.{}.self_attn.o_proj.weight", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 3, 4, 5, 6, 7, 8, 9], &[1152, 1152], false, true);
    add(&mut out, "tts_model.tts_model.backbone.layers.{}.self_attn.q_norm.weight", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 3, 4, 5, 6, 7, 8, 9], &[72], false, false);
    add(&mut out, "tts_model.tts_model.backbone.layers.{}.self_attn.q_proj.weight", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 3, 4, 5, 6, 7, 8, 9], &[1152, 1152], false, true);
    add(&mut out, "tts_model.tts_model.backbone.layers.{}.self_attn.v_proj.weight", &[0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 3, 4, 5, 6, 7, 8, 9], &[1152, 1152], false, true);
    add(&mut out, "tts_model.tts_model.backbone.norm.weight", &[], &[1152], false, false);
    add(&mut out, "tts_model.tts_model.bos_emb", &[], &[1152], false, false);
    add(&mut out, "tts_model.tts_model.embed_code.weight", &[], &[1152, 512], false, true);
    add(&mut out, "tts_model.tts_model.embed_subword.backbone.encoder.layers.{}.mlp.down_proj.weight", &[0], &[1152, 4608], false, true);
    add(&mut out, "tts_model.tts_model.embed_subword.backbone.encoder.layers.{}.mlp.gate_proj.weight", &[0], &[4608, 1152], false, true);
    add(&mut out, "tts_model.tts_model.embed_subword.backbone.encoder.layers.{}.mlp.up_proj.weight", &[0], &[4608, 1152], false, true);
    add(&mut out, "tts_model.tts_model.embed_subword.backbone.encoder.layers.{}.post_feedforward_layernorm.weight", &[0], &[1152], false, false);
    add(&mut out, "tts_model.tts_model.embed_subword.backbone.encoder.layers.{}.post_self_attn_layernorm.weight", &[0], &[1152], false, false);
    add(&mut out, "tts_model.tts_model.embed_subword.backbone.encoder.layers.{}.pre_feedforward_layernorm.weight", &[0], &[1152], false, false);
    add(&mut out, "tts_model.tts_model.embed_subword.backbone.encoder.layers.{}.pre_self_attn_layernorm.weight", &[0], &[1152], false, false);
    add(&mut out, "tts_model.tts_model.embed_subword.backbone.encoder.layers.{}.self_attn.k_proj.weight", &[0], &[1152, 1152], false, true);
    add(&mut out, "tts_model.tts_model.embed_subword.backbone.encoder.layers.{}.self_attn.o_proj.weight", &[0], &[1152, 1152], false, true);
    add(&mut out, "tts_model.tts_model.embed_subword.backbone.encoder.layers.{}.self_attn.q_proj.weight", &[0], &[1152, 1152], false, true);
    add(&mut out, "tts_model.tts_model.embed_subword.backbone.encoder.layers.{}.self_attn.v_proj.weight", &[0], &[1152, 1152], false, true);
    add(&mut out, "tts_model.tts_model.embed_subword.backbone.encoder.norm.weight", &[], &[1152], false, false);
    add(&mut out, "tts_model.tts_model.embed_subword.bos_eos_emb.pad_tensor", &[], &[], true, false);
    add(&mut out, "tts_model.tts_model.embed_subword.bos_eos_emb.special_emb.weight", &[], &[3, 1152], false, true);
    add(&mut out, "tts_model.tts_model.embed_subword.bos_eos_emb.special_flags", &[], &[131072], true, false);
    add(&mut out, "tts_model.tts_model.embed_subword.embed_tokens.weight", &[], &[257, 1152], false, true);
    add(&mut out, "tts_model.tts_model.embed_subword.proj_embedding.weight", &[], &[1152, 1152], false, true);
    add(&mut out, "tts_model.tts_model.embed_subword.subword_flag_emb.cont_emb.weight", &[], &[2, 1152], false, true);
    add(&mut out, "tts_model.tts_model.embed_subword.subword_flag_emb.is_continuation", &[], &[131073], true, false);
    add(&mut out, "tts_model.tts_model.embed_subword.subword_flag_emb.pad_tensor", &[], &[], true, false);
    add(&mut out, "tts_model.tts_model.gated_fusion_audio_text.audio_proj.bias", &[], &[1152], false, false);
    add(&mut out, "tts_model.tts_model.gated_fusion_audio_text.audio_proj.weight", &[], &[1152, 1152], false, true);
    add(&mut out, "tts_model.tts_model.gated_fusion_audio_text.final_norm.weight", &[], &[1152], false, false);
    add(&mut out, "tts_model.tts_model.gated_fusion_audio_text.gate", &[], &[1152], false, false);
    add(&mut out, "tts_model.tts_model.gated_fusion_audio_text.residual_scale", &[], &[], false, false);
    add(&mut out, "tts_model.tts_model.gated_fusion_audio_text.text_proj.bias", &[], &[1152], false, false);
    add(&mut out, "tts_model.tts_model.gated_fusion_audio_text.text_proj.weight", &[], &[1152, 1152], false, true);
    add(&mut out, "tts_model.tts_model.mog_head.low_mat", &[], &[1024, 512, 64], false, true);
    add(&mut out, "tts_model.tts_model.mog_head.mlp_stack.{}.mlp.down_proj.weight", &[0, 1, 2], &[1152, 4608], false, true);
    add(&mut out, "tts_model.tts_model.mog_head.mlp_stack.{}.mlp.gate_proj.weight", &[0, 1, 2], &[4608, 1152], false, true);
    add(&mut out, "tts_model.tts_model.mog_head.mlp_stack.{}.mlp.up_proj.weight", &[0, 1, 2], &[4608, 1152], false, true);
    add(&mut out, "tts_model.tts_model.mog_head.mlp_stack.{}.post_norm.weight", &[0, 1, 2], &[1152], false, false);
    add(&mut out, "tts_model.tts_model.mog_head.mlp_stack.{}.pre_norm.weight", &[0, 1, 2], &[1152], false, false);
    add(&mut out, "tts_model.tts_model.mog_head.mlp_stack.{}.weight", &[3], &[1152], false, false);
    add(&mut out, "tts_model.tts_model.mog_head.proj_else.weight", &[], &[512, 1152], false, true);
    add(&mut out, "tts_model.tts_model.mog_head.proj_logits.weight", &[], &[1024, 1152], false, true);
    add(&mut out, "tts_model.tts_model.mog_head.proj_logs.weight", &[], &[1, 1152], false, true);
    add(&mut out, "tts_model.tts_model.mog_head.proj_mus.weight", &[], &[65536, 1152], false, true);
    add(&mut out, "tts_model.tts_model.null_emb", &[], &[1152], false, false);
    add(&mut out, "tts_model.tts_model.rvq_embs", &[], &[31, 1024, 512], false, false);
    out
}

#[allow(clippy::too_many_arguments)]
fn add_split(
    out: &mut Vec<SplitTensor>,
    source: &str,
    target: &str,
    indices: &[(usize, usize)],
    shape: &[usize],
    encoding: GgmlType,
    transform: Transform,
) {
    out.extend(indices.iter().map(|(src, dst)| SplitTensor {
        source: source.replace("{}", &src.to_string()),
        target: target.replace("{}", &dst.to_string()),
        shape: shape.to_vec(),
        encoding,
        transform,
    }));
}

// Physical shapes and encodings come from the four published directories at
// 89883a05a031557729771f94abb9998e4facdd45. Recipes reverse the converter's
// singleton-axis changes, -exp(A_log), +1 RMSNorm storage and I64-to-I32 casts.
// Empty targets are derived tokenizer character tables, retained as source data.
#[rustfmt::skip]
fn split_catalog(layout: GgufLayout) -> Vec<SplitTensor> {
    let mut out = Vec::new();
    match layout {
        GgufLayout::Complete => {}
        GgufLayout::Perception => {
    add_split(&mut out, "a.blk.{}.attn_k.weight", "stt_model.perception.encoder.layers.{}.self_attn.linear_k.weight", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23)], &[1024, 1024], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "a.blk.{}.attn_out.weight", "stt_model.perception.encoder.layers.{}.self_attn.linear_out.weight", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23)], &[1024, 1024], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "a.blk.{}.attn_q.weight", "stt_model.perception.encoder.layers.{}.self_attn.linear_q.weight", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23)], &[1024, 1024], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "a.blk.{}.attn_v.weight", "stt_model.perception.encoder.layers.{}.self_attn.linear_v.weight", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23)], &[1024, 1024], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "a.blk.{}.conv_dw.weight", "stt_model.perception.encoder.layers.{}.conv.depthwise_conv.weight", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23)], &[1024, 9], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "a.blk.{}.conv_norm.bias", "stt_model.perception.encoder.layers.{}.conv.batch_norm.bias", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23)], &[1024], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "a.blk.{}.conv_norm.weight", "stt_model.perception.encoder.layers.{}.conv.batch_norm.weight", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23)], &[1024], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "a.blk.{}.conv_pw1.weight", "stt_model.perception.encoder.layers.{}.conv.pointwise_conv1.weight", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23)], &[2048, 1024], GgmlType::F16, Transform::Identity);
    add_split(&mut out, "a.blk.{}.conv_pw2.weight", "stt_model.perception.encoder.layers.{}.conv.pointwise_conv2.weight", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23)], &[1024, 1024], GgmlType::F16, Transform::Identity);
    add_split(&mut out, "a.blk.{}.ffn_down.weight", "stt_model.perception.encoder.layers.{}.feed_forward1.linear2.weight", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23)], &[1024, 4096], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "a.blk.{}.ffn_down_1.weight", "stt_model.perception.encoder.layers.{}.feed_forward2.linear2.weight", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23)], &[1024, 4096], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "a.blk.{}.ffn_norm.bias", "stt_model.perception.encoder.layers.{}.norm_feed_forward1.bias", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23)], &[1024], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "a.blk.{}.ffn_norm.weight", "stt_model.perception.encoder.layers.{}.norm_feed_forward1.weight", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23)], &[1024], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "a.blk.{}.ffn_norm_1.bias", "stt_model.perception.encoder.layers.{}.norm_feed_forward2.bias", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23)], &[1024], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "a.blk.{}.ffn_norm_1.weight", "stt_model.perception.encoder.layers.{}.norm_feed_forward2.weight", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23)], &[1024], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "a.blk.{}.ffn_up.weight", "stt_model.perception.encoder.layers.{}.feed_forward1.linear1.weight", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23)], &[4096, 1024], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "a.blk.{}.ffn_up_1.weight", "stt_model.perception.encoder.layers.{}.feed_forward2.linear1.weight", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23)], &[4096, 1024], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "a.blk.{}.linear_pos.weight", "stt_model.perception.encoder.layers.{}.self_attn.linear_pos.weight", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23)], &[1024, 1024], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "a.blk.{}.ln1.bias", "stt_model.perception.encoder.layers.{}.norm_self_att.bias", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23)], &[1024], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "a.blk.{}.ln1.weight", "stt_model.perception.encoder.layers.{}.norm_self_att.weight", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23)], &[1024], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "a.blk.{}.ln2.bias", "stt_model.perception.encoder.layers.{}.norm_out.bias", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23)], &[1024], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "a.blk.{}.ln2.weight", "stt_model.perception.encoder.layers.{}.norm_out.weight", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23)], &[1024], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "a.blk.{}.norm_conv.bias", "stt_model.perception.encoder.layers.{}.norm_conv.bias", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23)], &[1024], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "a.blk.{}.norm_conv.weight", "stt_model.perception.encoder.layers.{}.norm_conv.weight", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23)], &[1024], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "a.blk.{}.pos_bias_u", "stt_model.perception.encoder.layers.{}.self_attn.pos_bias_u", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23)], &[8, 128], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "a.blk.{}.pos_bias_v", "stt_model.perception.encoder.layers.{}.self_attn.pos_bias_v", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23)], &[8, 128], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "a.conv1d.{}.bias", "stt_model.perception.encoder.pre_encode.conv.{}.bias", &[(0, 0), (2, 2), (3, 3), (5, 5), (6, 6)], &[256, 1, 1], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "a.conv1d.{}.weight", "stt_model.perception.encoder.pre_encode.conv.{}.weight", &[(0, 0), (2, 2), (5, 5)], &[256, 1, 3, 3], GgmlType::F16, Transform::Identity);
    add_split(&mut out, "a.conv1d.{}.weight", "stt_model.perception.encoder.pre_encode.conv.{}.weight", &[(3, 3), (6, 6)], &[256, 256, 1, 1], GgmlType::F16, Transform::Identity);
    add_split(&mut out, "a.mel_filters", "stt_model.perception.preprocessor.featurizer.fb", &[(0, 0)], &[128, 257], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "a.pre_encode.out.bias", "stt_model.perception.encoder.pre_encode.out.bias", &[(0, 0)], &[1024], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "a.pre_encode.out.weight", "stt_model.perception.encoder.pre_encode.out.weight", &[(0, 0)], &[1024, 4352], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "a.window", "stt_model.perception.preprocessor.featurizer.window", &[(0, 0)], &[400], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "mm.a.proj.bias", "stt_model.perception.proj.bias", &[(0, 0)], &[4480], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "mm.a.proj.weight", "stt_model.perception.proj.weight", &[(0, 0)], &[4480, 1024], GgmlType::Q4_0, Transform::Identity);
        }
        GgufLayout::FunctionHead => {
    add_split(&mut out, "function_head.weight", "stt_model.function_head.weight", &[(0, 0)], &[131072, 4480], GgmlType::Q4_0, Transform::Identity);
        }
        GgufLayout::Backbone => {
    add_split(&mut out, "blk.{}.attn_k.weight", "stt_model.llm.layers.{}.mixer.k_proj.weight", &[(14, 14), (21, 21), (30, 30), (39, 39)], &[1024, 4480], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "blk.{}.attn_norm.weight", "stt_model.llm.layers.{}.norm.weight", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25), (26, 26), (27, 27), (28, 28), (29, 29), (30, 30), (31, 31), (32, 32), (33, 33), (34, 34), (35, 35), (36, 36), (37, 37), (38, 38), (39, 39), (40, 40), (41, 41), (42, 42), (43, 43), (44, 44), (45, 45), (46, 46), (47, 47), (48, 48), (49, 49), (50, 50), (51, 51), (52, 52), (53, 53), (54, 54), (55, 55)], &[4480], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "blk.{}.attn_output.weight", "stt_model.llm.layers.{}.mixer.o_proj.weight", &[(14, 14), (21, 21), (30, 30), (39, 39)], &[4480, 5120], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "blk.{}.attn_q.weight", "stt_model.llm.layers.{}.mixer.q_proj.weight", &[(14, 14), (21, 21), (30, 30), (39, 39)], &[5120, 4480], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "blk.{}.attn_v.weight", "stt_model.llm.layers.{}.mixer.v_proj.weight", &[(14, 14), (21, 21), (30, 30), (39, 39)], &[1024, 4480], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "blk.{}.ffn_down.weight", "stt_model.llm.layers.{}.mixer.down_proj.weight", &[(1, 1), (3, 3), (5, 5), (8, 8), (10, 10), (12, 12), (15, 15), (17, 17), (19, 19), (22, 22), (24, 24), (26, 26), (28, 28), (31, 31), (33, 33), (35, 35), (37, 37), (40, 40), (42, 42), (45, 45), (47, 47), (49, 49), (51, 51), (53, 53), (55, 55)], &[4480, 15680], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "blk.{}.ffn_up.weight", "stt_model.llm.layers.{}.mixer.up_proj.weight", &[(1, 1), (3, 3), (5, 5), (8, 8), (10, 10), (12, 12), (15, 15), (17, 17), (19, 19), (22, 22), (24, 24), (26, 26), (28, 28), (31, 31), (33, 33), (35, 35), (37, 37), (40, 40), (42, 42), (45, 45), (47, 47), (49, 49), (51, 51), (53, 53), (55, 55)], &[15680, 4480], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "blk.{}.ssm_a", "stt_model.llm.layers.{}.mixer.A_log", &[(0, 0), (2, 2), (4, 4), (6, 6), (7, 7), (9, 9), (11, 11), (13, 13), (16, 16), (18, 18), (20, 20), (23, 23), (25, 25), (27, 27), (29, 29), (32, 32), (34, 34), (36, 36), (38, 38), (41, 41), (43, 43), (44, 44), (46, 46), (48, 48), (50, 50), (52, 52), (54, 54)], &[128, 1], GgmlType::F32, Transform::NegLog);
    add_split(&mut out, "blk.{}.ssm_conv1d.bias", "stt_model.llm.layers.{}.mixer.conv1d.bias", &[(0, 0), (2, 2), (4, 4), (6, 6), (7, 7), (9, 9), (11, 11), (13, 13), (16, 16), (18, 18), (20, 20), (23, 23), (25, 25), (27, 27), (29, 29), (32, 32), (34, 34), (36, 36), (38, 38), (41, 41), (43, 43), (44, 44), (46, 46), (48, 48), (50, 50), (52, 52), (54, 54)], &[12288], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "blk.{}.ssm_conv1d.weight", "stt_model.llm.layers.{}.mixer.conv1d.weight", &[(0, 0), (2, 2), (4, 4), (6, 6), (7, 7), (9, 9), (11, 11), (13, 13), (16, 16), (18, 18), (20, 20), (23, 23), (25, 25), (27, 27), (29, 29), (32, 32), (34, 34), (36, 36), (38, 38), (41, 41), (43, 43), (44, 44), (46, 46), (48, 48), (50, 50), (52, 52), (54, 54)], &[12288, 4], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "blk.{}.ssm_d", "stt_model.llm.layers.{}.mixer.D", &[(0, 0), (2, 2), (4, 4), (6, 6), (7, 7), (9, 9), (11, 11), (13, 13), (16, 16), (18, 18), (20, 20), (23, 23), (25, 25), (27, 27), (29, 29), (32, 32), (34, 34), (36, 36), (38, 38), (41, 41), (43, 43), (44, 44), (46, 46), (48, 48), (50, 50), (52, 52), (54, 54)], &[128, 1], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "blk.{}.ssm_dt.bias", "stt_model.llm.layers.{}.mixer.dt_bias", &[(0, 0), (2, 2), (4, 4), (6, 6), (7, 7), (9, 9), (11, 11), (13, 13), (16, 16), (18, 18), (20, 20), (23, 23), (25, 25), (27, 27), (29, 29), (32, 32), (34, 34), (36, 36), (38, 38), (41, 41), (43, 43), (44, 44), (46, 46), (48, 48), (50, 50), (52, 52), (54, 54)], &[128], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "blk.{}.ssm_in.weight", "stt_model.llm.layers.{}.mixer.in_proj.weight", &[(0, 0), (2, 2), (4, 4), (6, 6), (7, 7), (9, 9), (11, 11), (13, 13), (16, 16), (18, 18), (20, 20), (23, 23), (25, 25), (27, 27), (29, 29), (32, 32), (34, 34), (36, 36), (38, 38), (41, 41), (43, 43), (44, 44), (46, 46), (48, 48), (50, 50), (52, 52), (54, 54)], &[22656, 4480], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "blk.{}.ssm_norm.weight", "stt_model.llm.layers.{}.mixer.norm.weight", &[(0, 0), (2, 2), (4, 4), (6, 6), (7, 7), (9, 9), (11, 11), (13, 13), (16, 16), (18, 18), (20, 20), (23, 23), (25, 25), (27, 27), (29, 29), (32, 32), (34, 34), (36, 36), (38, 38), (41, 41), (43, 43), (44, 44), (46, 46), (48, 48), (50, 50), (52, 52), (54, 54)], &[8, 1280], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "blk.{}.ssm_out.weight", "stt_model.llm.layers.{}.mixer.out_proj.weight", &[(0, 0), (2, 2), (4, 4), (6, 6), (7, 7), (9, 9), (11, 11), (13, 13), (16, 16), (18, 18), (20, 20), (23, 23), (25, 25), (27, 27), (29, 29), (32, 32), (34, 34), (36, 36), (38, 38), (41, 41), (43, 43), (44, 44), (46, 46), (48, 48), (50, 50), (52, 52), (54, 54)], &[4480, 10240], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "output.weight", "stt_model.lm_head.weight", &[(0, 0)], &[131072, 4480], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "output_norm.weight", "stt_model.llm.norm_f.weight", &[(0, 0)], &[4480], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "token_embd.weight", "stt_model.embed_tokens.weight", &[(0, 0)], &[131072, 4480], GgmlType::Q4_0, Transform::Identity);
        }
        GgufLayout::Tts => {
    add_split(&mut out, "blk.{}.attn_k.weight", "tts_model.tts_model.backbone.layers.{}.self_attn.k_proj.weight", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25), (26, 26), (27, 27)], &[1152, 1152], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "blk.{}.attn_k_norm.weight", "tts_model.tts_model.backbone.layers.{}.self_attn.k_norm.weight", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25), (26, 26), (27, 27)], &[72], GgmlType::F32, Transform::SubtractOne);
    add_split(&mut out, "blk.{}.attn_norm.weight", "tts_model.tts_model.backbone.layers.{}.input_layernorm.weight", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25), (26, 26), (27, 27)], &[1152], GgmlType::F32, Transform::SubtractOne);
    add_split(&mut out, "blk.{}.attn_output.weight", "tts_model.tts_model.backbone.layers.{}.self_attn.o_proj.weight", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25), (26, 26), (27, 27)], &[1152, 1152], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "blk.{}.attn_q.weight", "tts_model.tts_model.backbone.layers.{}.self_attn.q_proj.weight", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25), (26, 26), (27, 27)], &[1152, 1152], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "blk.{}.attn_q_norm.weight", "tts_model.tts_model.backbone.layers.{}.self_attn.q_norm.weight", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25), (26, 26), (27, 27)], &[72], GgmlType::F32, Transform::SubtractOne);
    add_split(&mut out, "blk.{}.attn_v.weight", "tts_model.tts_model.backbone.layers.{}.self_attn.v_proj.weight", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25), (26, 26), (27, 27)], &[1152, 1152], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "blk.{}.ffn_down.weight", "tts_model.tts_model.backbone.layers.{}.mlp.down_proj.weight", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25), (26, 26), (27, 27)], &[1152, 4608], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "blk.{}.ffn_gate.weight", "tts_model.tts_model.backbone.layers.{}.mlp.gate_proj.weight", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25), (26, 26), (27, 27)], &[4608, 1152], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "blk.{}.ffn_norm.weight", "tts_model.tts_model.backbone.layers.{}.pre_feedforward_layernorm.weight", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25), (26, 26), (27, 27)], &[1152], GgmlType::F32, Transform::SubtractOne);
    add_split(&mut out, "blk.{}.ffn_up.weight", "tts_model.tts_model.backbone.layers.{}.mlp.up_proj.weight", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25), (26, 26), (27, 27)], &[4608, 1152], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "blk.{}.post_attention_norm.weight", "tts_model.tts_model.backbone.layers.{}.post_attention_layernorm.weight", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25), (26, 26), (27, 27)], &[1152], GgmlType::F32, Transform::SubtractOne);
    add_split(&mut out, "blk.{}.post_ffw_norm.weight", "tts_model.tts_model.backbone.layers.{}.post_feedforward_layernorm.weight", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25), (26, 26), (27, 27)], &[1152], GgmlType::F32, Transform::SubtractOne);
    add_split(&mut out, "codec.dec.blk.{}.dwconv.bias", "tts_model.audio_codec.decoder.layers.{}.dwconv.bias", &[(6, 9), (7, 10), (8, 11)], &[384], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "codec.dec.blk.{}.dwconv.bias", "tts_model.audio_codec.decoder.layers.{}.dwconv.bias", &[(3, 5), (4, 6), (5, 7)], &[768], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "codec.dec.blk.{}.dwconv.bias", "tts_model.audio_codec.decoder.layers.{}.dwconv.bias", &[(0, 1), (1, 2), (2, 3)], &[1536], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "codec.dec.blk.{}.dwconv.weight", "tts_model.audio_codec.decoder.layers.{}.dwconv.weight", &[(6, 9), (7, 10), (8, 11)], &[384, 1, 7], GgmlType::F16, Transform::Identity);
    add_split(&mut out, "codec.dec.blk.{}.dwconv.weight", "tts_model.audio_codec.decoder.layers.{}.dwconv.weight", &[(3, 5), (4, 6), (5, 7)], &[768, 1, 7], GgmlType::F16, Transform::Identity);
    add_split(&mut out, "codec.dec.blk.{}.dwconv.weight", "tts_model.audio_codec.decoder.layers.{}.dwconv.weight", &[(0, 1), (1, 2), (2, 3)], &[1536, 1, 7], GgmlType::F16, Transform::Identity);
    add_split(&mut out, "codec.dec.blk.{}.norm.bias", "tts_model.audio_codec.decoder.layers.{}.norm.bias", &[(6, 9), (7, 10), (8, 11)], &[384], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "codec.dec.blk.{}.norm.bias", "tts_model.audio_codec.decoder.layers.{}.norm.bias", &[(3, 5), (4, 6), (5, 7)], &[768], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "codec.dec.blk.{}.norm.bias", "tts_model.audio_codec.decoder.layers.{}.norm.bias", &[(0, 1), (1, 2), (2, 3)], &[1536], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "codec.dec.blk.{}.norm.weight", "tts_model.audio_codec.decoder.layers.{}.norm.weight", &[(6, 9), (7, 10), (8, 11)], &[384], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "codec.dec.blk.{}.norm.weight", "tts_model.audio_codec.decoder.layers.{}.norm.weight", &[(3, 5), (4, 6), (5, 7)], &[768], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "codec.dec.blk.{}.norm.weight", "tts_model.audio_codec.decoder.layers.{}.norm.weight", &[(0, 1), (1, 2), (2, 3)], &[1536], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "codec.dec.blk.{}.pw1.bias", "tts_model.audio_codec.decoder.layers.{}.pwconv1.bias", &[(6, 9), (7, 10), (8, 11)], &[1536], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "codec.dec.blk.{}.pw1.bias", "tts_model.audio_codec.decoder.layers.{}.pwconv1.bias", &[(3, 5), (4, 6), (5, 7)], &[3072], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "codec.dec.blk.{}.pw1.bias", "tts_model.audio_codec.decoder.layers.{}.pwconv1.bias", &[(0, 1), (1, 2), (2, 3)], &[6144], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "codec.dec.blk.{}.pw1.weight", "tts_model.audio_codec.decoder.layers.{}.pwconv1.weight", &[(6, 9), (7, 10), (8, 11)], &[1536, 384], GgmlType::F16, Transform::Identity);
    add_split(&mut out, "codec.dec.blk.{}.pw1.weight", "tts_model.audio_codec.decoder.layers.{}.pwconv1.weight", &[(3, 5), (4, 6), (5, 7)], &[3072, 768], GgmlType::F16, Transform::Identity);
    add_split(&mut out, "codec.dec.blk.{}.pw1.weight", "tts_model.audio_codec.decoder.layers.{}.pwconv1.weight", &[(0, 1), (1, 2), (2, 3)], &[6144, 1536], GgmlType::F16, Transform::Identity);
    add_split(&mut out, "codec.dec.blk.{}.pw2.bias", "tts_model.audio_codec.decoder.layers.{}.pwconv2.bias", &[(6, 9), (7, 10), (8, 11)], &[384], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "codec.dec.blk.{}.pw2.bias", "tts_model.audio_codec.decoder.layers.{}.pwconv2.bias", &[(3, 5), (4, 6), (5, 7)], &[768], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "codec.dec.blk.{}.pw2.bias", "tts_model.audio_codec.decoder.layers.{}.pwconv2.bias", &[(0, 1), (1, 2), (2, 3)], &[1536], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "codec.dec.blk.{}.pw2.weight", "tts_model.audio_codec.decoder.layers.{}.pwconv2.weight", &[(6, 9), (7, 10), (8, 11)], &[384, 1536], GgmlType::F16, Transform::Identity);
    add_split(&mut out, "codec.dec.blk.{}.pw2.weight", "tts_model.audio_codec.decoder.layers.{}.pwconv2.weight", &[(3, 5), (4, 6), (5, 7)], &[768, 3072], GgmlType::F16, Transform::Identity);
    add_split(&mut out, "codec.dec.blk.{}.pw2.weight", "tts_model.audio_codec.decoder.layers.{}.pwconv2.weight", &[(0, 1), (1, 2), (2, 3)], &[1536, 6144], GgmlType::F16, Transform::Identity);
    add_split(&mut out, "codec.dec.head.weight", "tts_model.audio_codec.decoder.layers.{}.weight", &[(0, 12)], &[18, 384, 1], GgmlType::F16, Transform::Identity);
    add_split(&mut out, "codec.dec.up.{}.weight", "tts_model.audio_codec.decoder.layers.{}.weight", &[(0, 0)], &[512, 1536, 9], GgmlType::F16, Transform::Identity);
    add_split(&mut out, "codec.dec.up.{}.weight", "tts_model.audio_codec.decoder.layers.{}.weight", &[(8, 8)], &[768, 384, 7], GgmlType::F16, Transform::Identity);
    add_split(&mut out, "codec.dec.up.{}.weight", "tts_model.audio_codec.decoder.layers.{}.weight", &[(4, 4)], &[1536, 768, 7], GgmlType::F16, Transform::Identity);
    add_split(&mut out, "codec.rvq.{}.mus", "tts_model.audio_codec.prvq.mus_list.{}", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25), (26, 26), (27, 27), (28, 28), (29, 29), (30, 30)], &[1024, 512], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "codec.rvq.{}.variance", "tts_model.audio_codec.prvq._variance_list.{}.variance", &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25), (26, 26), (27, 27), (28, 28), (29, 29), (30, 30)], &[1], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "codec.rvq_embs", "tts_model.tts_model.rvq_embs", &[(0, 0)], &[31, 1024, 512], GgmlType::F16, Transform::Identity);
    add_split(&mut out, "codec.silence_tokens", "tts_model.codec_silence_tokens", &[(0, 0)], &[31], GgmlType::I32, Transform::Identity);
    add_split(&mut out, "output_norm.weight", "tts_model.tts_model.backbone.norm.weight", &[(0, 0)], &[1152], GgmlType::F32, Transform::SubtractOne);
    add_split(&mut out, "tts.audio_prompt_latents", "tts_model.audio_prompt_latents.Aria", &[(0, 0)], &[1, 37, 1152], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "tts.audio_prompt_proj.weight", "tts_model.tts_model.audio_prompt_projection_W", &[(0, 0)], &[1152, 1152], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "tts.bos_emb", "tts_model.tts_model.bos_emb", &[(0, 0)], &[1152], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "tts.control_codes", "tts_model._control_codes", &[(0, 0)], &[3], GgmlType::I32, Transform::Identity);
    add_split(&mut out, "tts.embed_code.weight", "tts_model.tts_model.embed_code.weight", &[(0, 0)], &[1152, 512], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "tts.fusion.audio_proj.bias", "tts_model.tts_model.gated_fusion_audio_text.audio_proj.bias", &[(0, 0)], &[1152], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "tts.fusion.audio_proj.weight", "tts_model.tts_model.gated_fusion_audio_text.audio_proj.weight", &[(0, 0)], &[1152, 1152], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "tts.fusion.gate", "tts_model.tts_model.gated_fusion_audio_text.gate", &[(0, 0)], &[1152], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "tts.fusion.norm.weight", "tts_model.tts_model.gated_fusion_audio_text.final_norm.weight", &[(0, 0)], &[1152], GgmlType::F32, Transform::SubtractOne);
    add_split(&mut out, "tts.fusion.residual_scale", "tts_model.tts_model.gated_fusion_audio_text.residual_scale", &[(0, 0)], &[1], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "tts.fusion.text_proj.bias", "tts_model.tts_model.gated_fusion_audio_text.text_proj.bias", &[(0, 0)], &[1152], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "tts.fusion.text_proj.weight", "tts_model.tts_model.gated_fusion_audio_text.text_proj.weight", &[(0, 0)], &[1152, 1152], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "tts.mog.blk.{}.ffn_down.weight", "tts_model.tts_model.mog_head.mlp_stack.{}.mlp.down_proj.weight", &[(0, 0), (1, 1), (2, 2)], &[1152, 4608], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "tts.mog.blk.{}.ffn_gate.weight", "tts_model.tts_model.mog_head.mlp_stack.{}.mlp.gate_proj.weight", &[(0, 0), (1, 1), (2, 2)], &[4608, 1152], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "tts.mog.blk.{}.ffn_up.weight", "tts_model.tts_model.mog_head.mlp_stack.{}.mlp.up_proj.weight", &[(0, 0), (1, 1), (2, 2)], &[4608, 1152], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "tts.mog.blk.{}.post_norm.weight", "tts_model.tts_model.mog_head.mlp_stack.{}.post_norm.weight", &[(0, 0), (1, 1), (2, 2)], &[1152], GgmlType::F32, Transform::SubtractOne);
    add_split(&mut out, "tts.mog.blk.{}.pre_norm.weight", "tts_model.tts_model.mog_head.mlp_stack.{}.pre_norm.weight", &[(0, 0), (1, 1), (2, 2)], &[1152], GgmlType::F32, Transform::SubtractOne);
    add_split(&mut out, "tts.mog.else.weight", "tts_model.tts_model.mog_head.proj_else.weight", &[(0, 0)], &[512, 1152], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "tts.mog.logits.weight", "tts_model.tts_model.mog_head.proj_logits.weight", &[(0, 0)], &[1024, 1152], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "tts.mog.logs.weight", "tts_model.tts_model.mog_head.proj_logs.weight", &[(0, 0)], &[1, 1152], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "tts.mog.low_mat", "tts_model.tts_model.mog_head.low_mat", &[(0, 0)], &[1024, 512, 64], GgmlType::F16, Transform::Identity);
    add_split(&mut out, "tts.mog.mus.weight", "tts_model.tts_model.mog_head.proj_mus.weight", &[(0, 0)], &[65536, 1152], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "tts.mog.norm.weight", "tts_model.tts_model.mog_head.mlp_stack.{}.weight", &[(0, 3)], &[1152], GgmlType::F32, Transform::SubtractOne);
    add_split(&mut out, "tts.null_emb", "tts_model.tts_model.null_emb", &[(0, 0)], &[1152], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "tts.subword.blk.{}.attn_k.weight", "tts_model.tts_model.embed_subword.backbone.encoder.layers.{}.self_attn.k_proj.weight", &[(0, 0)], &[1152, 1152], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "tts.subword.blk.{}.attn_norm.weight", "tts_model.tts_model.embed_subword.backbone.encoder.layers.{}.pre_self_attn_layernorm.weight", &[(0, 0)], &[1152], GgmlType::F32, Transform::SubtractOne);
    add_split(&mut out, "tts.subword.blk.{}.attn_output.weight", "tts_model.tts_model.embed_subword.backbone.encoder.layers.{}.self_attn.o_proj.weight", &[(0, 0)], &[1152, 1152], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "tts.subword.blk.{}.attn_q.weight", "tts_model.tts_model.embed_subword.backbone.encoder.layers.{}.self_attn.q_proj.weight", &[(0, 0)], &[1152, 1152], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "tts.subword.blk.{}.attn_v.weight", "tts_model.tts_model.embed_subword.backbone.encoder.layers.{}.self_attn.v_proj.weight", &[(0, 0)], &[1152, 1152], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "tts.subword.blk.{}.ffn_down.weight", "tts_model.tts_model.embed_subword.backbone.encoder.layers.{}.mlp.down_proj.weight", &[(0, 0)], &[1152, 4608], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "tts.subword.blk.{}.ffn_gate.weight", "tts_model.tts_model.embed_subword.backbone.encoder.layers.{}.mlp.gate_proj.weight", &[(0, 0)], &[4608, 1152], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "tts.subword.blk.{}.ffn_norm.weight", "tts_model.tts_model.embed_subword.backbone.encoder.layers.{}.pre_feedforward_layernorm.weight", &[(0, 0)], &[1152], GgmlType::F32, Transform::SubtractOne);
    add_split(&mut out, "tts.subword.blk.{}.ffn_up.weight", "tts_model.tts_model.embed_subword.backbone.encoder.layers.{}.mlp.up_proj.weight", &[(0, 0)], &[4608, 1152], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "tts.subword.blk.{}.post_attention_norm.weight", "tts_model.tts_model.embed_subword.backbone.encoder.layers.{}.post_self_attn_layernorm.weight", &[(0, 0)], &[1152], GgmlType::F32, Transform::SubtractOne);
    add_split(&mut out, "tts.subword.blk.{}.post_ffw_norm.weight", "tts_model.tts_model.embed_subword.backbone.encoder.layers.{}.post_feedforward_layernorm.weight", &[(0, 0)], &[1152], GgmlType::F32, Transform::SubtractOne);
    add_split(&mut out, "tts.subword.char_ids", "", &[(0, 0)], &[891139], GgmlType::I32, Transform::Identity);
    add_split(&mut out, "tts.subword.char_offsets", "", &[(0, 0)], &[131073], GgmlType::I32, Transform::Identity);
    add_split(&mut out, "tts.subword.cont_emb.weight", "tts_model.tts_model.embed_subword.subword_flag_emb.cont_emb.weight", &[(0, 0)], &[2, 1152], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "tts.subword.embed_tokens.weight", "tts_model.tts_model.embed_subword.embed_tokens.weight", &[(0, 0)], &[257, 1152], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "tts.subword.is_continuation", "tts_model.tts_model.embed_subword.subword_flag_emb.is_continuation", &[(0, 0)], &[131073], GgmlType::I32, Transform::Identity);
    add_split(&mut out, "tts.subword.output_norm.weight", "tts_model.tts_model.embed_subword.backbone.encoder.norm.weight", &[(0, 0)], &[1152], GgmlType::F32, Transform::SubtractOne);
    add_split(&mut out, "tts.subword.proj.weight", "tts_model.tts_model.embed_subword.proj_embedding.weight", &[(0, 0)], &[1152, 1152], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "tts.subword.special_emb.weight", "tts_model.tts_model.embed_subword.bos_eos_emb.special_emb.weight", &[(0, 0)], &[3, 1152], GgmlType::Q4_0, Transform::Identity);
    add_split(&mut out, "tts.subword.special_flags", "tts_model.tts_model.embed_subword.bos_eos_emb.special_flags", &[(0, 0)], &[131072], GgmlType::I32, Transform::Identity);
        }
    }
    out
}

#[cfg(test)]
pub(in crate::voicechat) mod tests {
    use super::*;
    use eredu_checkpoint::{
        safetensors::{SafetensorsHeader, SafetensorsHeaderCatalog},
        validation::{
            validate_safetensors_plan, CatalogTensorMetadata, CheckpointIssueKind,
            CheckpointValidation, SafetensorsCatalog,
        },
    };

    fn config() -> VoiceChatConfig {
        VoiceChatConfig::from_json(include_str!("fixtures/official-inference-config.json")).unwrap()
    }

    pub(in crate::voicechat) fn official() -> SafetensorsHeaderCatalog {
        let bytes = include_bytes!("../../tests/fixtures/voicechat/official-header.json");
        let header: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        let payload = header
            .as_object()
            .unwrap()
            .values()
            .map(|t| t["data_offsets"][1].as_u64().unwrap())
            .max()
            .unwrap();
        let mut encoded = (bytes.len() as u64).to_le_bytes().to_vec();
        encoded.extend_from_slice(bytes);
        SafetensorsHeaderCatalog::from_headers(
            &[SafetensorsHeader {
                member: "model.safetensors".into(),
                bytes: encoded,
                file_len: payload + bytes.len() as u64 + 8,
            }],
            None,
        )
        .unwrap()
    }

    struct ChangedCatalog {
        tensors: BTreeMap<String, CatalogTensorMetadata>,
    }
    impl SafetensorsCatalog for ChangedCatalog {
        fn keys(&self) -> Vec<String> {
            self.tensors.keys().cloned().collect()
        }
        fn metadata(&self, key: &str) -> Result<CatalogTensorMetadata, String> {
            self.tensors.get(key).cloned().ok_or_else(|| key.to_owned())
        }
    }

    #[test]
    fn pinned_official_header_admits_all_components_and_rejects_corruption() {
        let original = official();
        let plan = safetensors_plan(&config(), &BTreeMap::new()).unwrap();
        assert_eq!(
            validate_safetensors_plan(&original, &plan),
            CheckpointValidation::Exact
        );
        assert_eq!(original.keys().len(), 1632);
        let mut changed = ChangedCatalog {
            tensors: original
                .keys()
                .into_iter()
                .map(|k| {
                    let m = original.metadata(&k).unwrap();
                    (k, m)
                })
                .collect(),
        };
        changed.tensors.remove("stt_model.rnnt_joint.enc.weight");
        changed
            .tensors
            .get_mut("stt_model.perception.preprocessor.featurizer.fb")
            .unwrap()
            .shape = vec![1, 80, 257];
        changed
            .tensors
            .get_mut("tts_model._control_codes")
            .unwrap()
            .stored_dtype = StoredDtype::F32;
        changed.tensors.insert(
            "unknown.weight".into(),
            CatalogTensorMetadata {
                shape: vec![1],
                stored_dtype: StoredDtype::F32,
            },
        );
        let CheckpointValidation::Invalid(issues) = validate_safetensors_plan(&changed, &plan)
        else {
            panic!("corrupt header was admitted")
        };
        for (key, kind) in [
            (
                "stt_model.rnnt_joint.enc.weight",
                CheckpointIssueKind::MissingTensor,
            ),
            (
                "stt_model.perception.preprocessor.featurizer.fb",
                CheckpointIssueKind::ShapeMismatch,
            ),
            (
                "tts_model._control_codes",
                CheckpointIssueKind::UnsupportedEncoding,
            ),
            ("unknown.weight", CheckpointIssueKind::UnexpectedTensor),
        ] {
            assert!(
                issues
                    .iter()
                    .any(|i| i.tensor_name.as_deref() == Some(key) && i.kind == kind),
                "{key}: {issues:?}"
            );
        }
    }

    // Tensor directories are extracted from pinned published files. This tests
    // the physical contract without fabricating multi-gigabyte tensor payloads.
    #[test]
    fn pinned_gguf_directories_match_exact_storage_contracts() {
        for (layout, fixture) in [
            (
                GgufLayout::Complete,
                include_str!("../../tests/fixtures/voicechat/gguf-q4-tensors.json"),
            ),
            (
                GgufLayout::Complete,
                include_str!("../../tests/fixtures/voicechat/gguf-4-tensors.json"),
            ),
            (
                GgufLayout::Complete,
                include_str!("../../tests/fixtures/voicechat/gguf-5-tensors.json"),
            ),
            (
                GgufLayout::Perception,
                include_str!("../../tests/fixtures/voicechat/gguf-0-tensors.json"),
            ),
            (
                GgufLayout::FunctionHead,
                include_str!("../../tests/fixtures/voicechat/gguf-1-tensors.json"),
            ),
            (
                GgufLayout::Backbone,
                include_str!("../../tests/fixtures/voicechat/gguf-2-tensors.json"),
            ),
            (
                GgufLayout::Tts,
                include_str!("../../tests/fixtures/voicechat/gguf-3-tensors.json"),
            ),
        ] {
            let directory: BTreeMap<String, serde_json::Value> =
                serde_json::from_str(fixture).unwrap();
            let plan = gguf_plan(&config(), layout).unwrap();
            assert_eq!(plan.common_tensors.len(), directory.len());
            for t in plan.common_tensors {
                let actual = &directory[&t.key];
                let shape: Vec<usize> = serde_json::from_value(actual["shape"].clone()).unwrap();
                assert_eq!(t.shape, shape, "{} {layout:?}", t.key);
                assert!(
                    t.encoding
                        .accepts(GgmlType::from_code(actual["type"].as_u64().unwrap() as u32)),
                    "{} {layout:?}",
                    t.key
                );
            }
        }
    }

    fn metadata_fixture(source: &str) -> BTreeMap<String, eredu_gguf::MetadataValue> {
        use eredu_gguf::{MetadataArray, MetadataValue};
        let values: BTreeMap<String, serde_json::Value> = serde_json::from_str(source).unwrap();
        values
            .into_iter()
            .map(|(key, v)| {
                let value = match v {
                    serde_json::Value::String(v) => MetadataValue::String(v),
                    serde_json::Value::Bool(v) => MetadataValue::Bool(v),
                    serde_json::Value::Number(v) if v.is_i64() || v.is_u64() => {
                        MetadataValue::Int64(v.as_i64().unwrap())
                    }
                    serde_json::Value::Number(v) => {
                        MetadataValue::Float32(v.as_f64().unwrap() as f32)
                    }
                    serde_json::Value::Array(v) => MetadataValue::Array(MetadataArray::Int64(
                        v.iter().map(|v| v.as_i64().unwrap()).collect(),
                    )),
                    _ => panic!("unsupported metadata fixture"),
                };
                (key, value)
            })
            .collect()
    }

    #[test]
    fn published_metadata_rejects_equation_and_special_token_mismatches() {
        use eredu_gguf::MetadataValue;
        let config = config();
        for (layout, source, key, mutation) in [
            (
                GgufLayout::Perception,
                include_str!("../../tests/fixtures/voicechat/gguf-0-metadata.json"),
                "clip.audio.window_size",
                MetadataValue::Int32(71),
            ),
            (
                GgufLayout::FunctionHead,
                include_str!("../../tests/fixtures/voicechat/gguf-1-metadata.json"),
                "voicechat.function.eotr_token_id",
                MetadataValue::Int32(23),
            ),
            (
                GgufLayout::Tts,
                include_str!("../../tests/fixtures/voicechat/gguf-3-metadata.json"),
                "voicechat.tts.query_pre_attn_scalar",
                MetadataValue::Int32(72),
            ),
        ] {
            let mut metadata = metadata_fixture(source);
            validate_gguf_metadata(&config, layout, &metadata).unwrap();
            metadata.insert(key.to_owned(), mutation);
            assert!(validate_gguf_metadata(&config, layout, &metadata)
                .unwrap_err()
                .contains(key));
        }
        let mut metadata = metadata_fixture(include_str!(
            "../../tests/fixtures/voicechat/gguf-q4-metadata.json"
        ));
        validate_gguf_metadata(&config, GgufLayout::Complete, &metadata).unwrap();
        let MetadataValue::String(encoded) = &metadata["voicechat.config.data"] else {
            unreachable!()
        };
        let mut data: serde_json::Value = serde_json::from_str(encoded).unwrap();
        data["frame_length"] = serde_json::json!(0.1);
        metadata.insert(
            "voicechat.config.data".into(),
            MetadataValue::String(data.to_string()),
        );
        assert!(validate_gguf_metadata(&config, GgufLayout::Complete, &metadata).is_err());
    }

    #[test]
    fn split_bundle_requires_exact_missing_companions() {
        let config = config();
        let mut available = BTreeMap::new();
        for layout in [
            GgufLayout::Perception,
            GgufLayout::Backbone,
            GgufLayout::FunctionHead,
            GgufLayout::Tts,
        ] {
            for (key, recipe) in canonical_recipes(&config, Some(layout)).unwrap() {
                assert!(available.insert(key, recipe).is_none());
            }
        }
        let missing = official()
            .keys()
            .into_iter()
            .filter(|key| !available.contains_key(key))
            .collect::<Vec<_>>();
        assert_eq!(missing.len(), 93);
        assert_eq!(
            missing
                .iter()
                .filter(|k| component_for_key(k) == Some(Component::Rnnt))
                .count(),
            15
        );
        assert_eq!(
            missing
                .iter()
                .filter(|k| component_for_key(k) == Some(Component::CodecEncoder))
                .count(),
            76
        );
        assert_eq!(
            missing
                .iter()
                .filter(|k| component_for_key(k) == Some(Component::Tts))
                .cloned()
                .collect::<Vec<_>>(),
            [
                "tts_model.tts_model.embed_subword.bos_eos_emb.pad_tensor",
                "tts_model.tts_model.embed_subword.subword_flag_emb.pad_tensor",
            ]
        );
        // These transforms are required by the released converter, rather than
        // inferred from matching names or element counts.
        assert!(matches!(
            available["stt_model.llm.layers.0.mixer.A_log"],
            DerivedWeightRecipe::NegLog { .. }
        ));
        assert!(matches!(
            available["tts_model.tts_model.backbone.layers.0.self_attn.q_norm.weight"],
            DerivedWeightRecipe::SubtractOne { .. }
        ));
        assert!(matches!(
            available["tts_model._control_codes"],
            DerivedWeightRecipe::Cast {
                dtype: RecipeDtype::I64,
                ..
            }
        ));
    }
}
