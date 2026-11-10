//! Released VoiceChat checkpoint catalogs and canonical source recipes.
//!
//! Geometry is the pinned 11B release contract. GGUF schemas describe storage;
//! acceptance does not assert that every encoding has a native execution kernel.

mod backbone;
mod codec_codebooks;
mod codec_neural;
mod conditioning;
mod conformer;
mod function_head;
mod mog;
mod perception_pinned;
mod rvq;
mod transformer;

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

/// Strict released SafeTensors catalog, including affine companions and host buffers.
pub fn safetensors_plan(
    config: &VoiceChatConfig,
    formats: &BTreeMap<String, eredu_checkpoint::LinearFormat>,
) -> Result<SafetensorsCheckpointPlan, String> {
    config.validate().map_err(|e| e.to_string())?;
    SafetensorsCheckpointPlan::new(
        "VoiceChat 11B SafeTensors",
        released_catalog()?
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
    let eligible: std::collections::BTreeSet<_> = released_catalog()?
        .into_iter()
        .filter(ReleasedTensor::accepts_affine_format)
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
    let eligible: std::collections::BTreeSet<_> = released_catalog()?
        .into_iter()
        .filter(ReleasedTensor::accepts_affine_format)
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
    released_catalog()?
        .into_iter()
        .filter(|t| t.accepts_affine_format() && keys.contains(&t.key))
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
        released_catalog()?
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
        split_catalog(layout)?
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

/// A canonical source recipe referencing its architecture-declared component.
#[derive(Debug, Clone)]
pub struct ParameterSourceDeclaration {
    /// Component owning the logical output, independently of its physical artifact.
    pub component: Component,
    /// Exact retained physical-to-logical source transformation.
    pub recipe: DerivedWeightRecipe,
}

/// Canonical NeMo source recipes and component owners for a validated physical source.
///
/// `None` selects official SafeTensors. Split TTS keeps its two derived character
/// tables in the retained physical source but does not treat them as model
/// parameters. Its two missing scalar padding buffers, RNNT and codec encoder
/// require companion sources when requested. No tensor payload is read here.
pub fn parameter_sources(
    config: &VoiceChatConfig,
    layout: Option<GgufLayout>,
) -> Result<BTreeMap<String, ParameterSourceDeclaration>, String> {
    config.validate().map_err(|e| e.to_string())?;
    let canonical = released_catalog()?
        .into_iter()
        .map(|t| (t.key.clone(), t))
        .collect::<BTreeMap<_, _>>();
    match layout {
        None | Some(GgufLayout::Complete) => Ok(canonical
            .into_iter()
            .map(|(name, declaration)| {
                let mut recipe = DerivedWeightRecipe::source(&name, TensorSelection::Full);
                if layout.is_some() && name.ends_with(".pad_tensor") {
                    recipe = DerivedWeightRecipe::Reshape {
                        input: Box::new(recipe),
                        shape: Vec::new(),
                    };
                }
                (
                    name,
                    ParameterSourceDeclaration {
                        component: declaration.component,
                        recipe,
                    },
                )
            })
            .collect()),
        Some(layout) => split_catalog(layout)?
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
                Ok((
                    t.target,
                    ParameterSourceDeclaration {
                        component: target.component,
                        recipe,
                    },
                ))
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
    for (key, expected) in metadata_expectations(layout)? {
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
fn metadata_expectations(layout: GgufLayout) -> Result<BTreeMap<String, serde_json::Value>, String> {
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
        GgufLayout::Backbone => {
            return backbone::metadata_expectations(
                super::backbone::released_declaration().map_err(|e| e.to_string())?,
            );
        }
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
    Ok(serde_json::from_str(expected).expect("static metadata object"))
}

struct ReleasedTensor {
    component: Component,
    key: String,
    shape: Vec<usize>,
    integer: bool,
    quantized: bool,
    // Construction encoding retained for ordinary affine source targets only.
    affine_source_format: Option<eredu_nn::LinearFormatSpec>,
}
impl ReleasedTensor {
    fn accepts_affine_format(&self) -> bool {
        self.quantized && self.affine_source_format.is_some()
    }
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
    component: Component,
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
        component,
        key,
        shape: shape.to_vec(),
        integer,
        quantized,
        affine_source_format: None,
    }));
}

// Exact tensor geometry from NVIDIA model.safetensors at
// 443794ea956ef0065f001967ffd00e77f519cb39. Repeated indices denote stored
// parameter identities, not replicated logical invocations. `quantized` marks
// tensors stored as Q4_0/Q8_0 in the complete published GGUF variants.
#[rustfmt::skip]
fn released_catalog() -> Result<Vec<ReleasedTensor>, String> {
    let mut out = Vec::new();
    add(&mut out, Component::Perception, "stt_model.perception.preprocessor.featurizer.fb", &[], &[1, 128, 257], false, false);
    add(&mut out, Component::Perception, "stt_model.perception.preprocessor.featurizer.window", &[], &[400], false, false);
    add(&mut out, Component::Controls, "tts_model._control_codes", &[], &[3], true, false);
    add(&mut out, Component::Controls, "tts_model.codec_silence_tokens", &[], &[31], true, false);
    add(&mut out, Component::Tts, "tts_model.tts_model.embed_subword.bos_eos_emb.pad_tensor", &[], &[], true, false);
    add(&mut out, Component::Tts, "tts_model.tts_model.embed_subword.bos_eos_emb.special_flags", &[], &[131072], true, false);
    add(&mut out, Component::Tts, "tts_model.tts_model.embed_subword.subword_flag_emb.is_continuation", &[], &[131073], true, false);
    add(&mut out, Component::Tts, "tts_model.tts_model.embed_subword.subword_flag_emb.pad_tensor", &[], &[], true, false);
    use eredu_nn::DeclareLogicalParameters;
    let declaration = super::rnnt_specs::released_declaration()
        .map_err(|e| e.to_string())?;
    let declaration = declaration.spec();
    // The complete GGUF variants explicitly encode these RNNT matrices. Other
    // parameter roles do not imply source-format eligibility or native support.
    let encoded_weights = declaration.units().flat_map(|unit| {
        use super::rnnt_specs::UnitSpec;
        match unit {
            UnitSpec::Embedding(spec) => vec![&spec.weight],
            UnitSpec::Predictor(spec) => vec![&spec.input.weight, &spec.recurrent.weight],
            UnitSpec::Projection(spec) => vec![&spec.weight],
        }
    }).map(|parameter| &parameter.id).collect::<std::collections::BTreeSet<_>>();
    // Predictor matrices retain their released LSTM storage convention; only
    // embedding and projection units admit ordinary affine source companions.
    let affine_weights = declaration.units().filter_map(|unit| {
        use super::rnnt_specs::UnitSpec;
        match unit {
            UnitSpec::Embedding(spec) => Some(&spec.weight.id),
            UnitSpec::Projection(spec) => Some(&spec.weight.id),
            UnitSpec::Predictor(_) => None,
        }
    }).collect::<std::collections::BTreeSet<_>>();
    for parameter in declaration.logical_parameters().map_err(|e| e.to_string())? {
        out.push(ReleasedTensor {
            component: Component::Rnnt,
            key: parameter.parameter.id.as_str().to_owned(),
            shape: parameter.shape,
            affine_source_format: parameter.linear_format
                .filter(|_| affine_weights.contains(&parameter.parameter.id)).cloned(),
            integer: false,
            quantized: encoded_weights.contains(&parameter.parameter.id),
        });
    }
    mog::released(
        &mut out,
        super::tts_mog_specs::released_declaration().map_err(|e| e.to_string())?,
    )?;
    function_head::released(
        &mut out,
        super::function_head_layered::released_declaration().map_err(|e| e.to_string())?,
    )?;
    perception_pinned::released(
        &mut out,
        super::perception_specs::released_declaration().map_err(|e|e.to_string())?,
    )?;
    conformer::released(
        &mut out,
        super::perception_specs::released_declaration().map_err(|e|e.to_string())?,
    )?;
    conditioning::released(
        &mut out,
        super::tts_conditioning_specs::released_declaration().map_err(|e| e.to_string())?,
    )?;
    rvq::released(
        &mut out,
        super::tts_rvq_specs::released_declaration().map_err(|e| e.to_string())?,
    )?;
    let codec = super::codec::released_declarations().map_err(|e| e.to_string())?;
    codec_codebooks::released(&mut out, codec)?;
    codec_neural::released(&mut out, codec)?;
    let transformers=super::tts_transformer_specs::released_declarations().map_err(|e|e.to_string())?;
    transformer::released(&mut out,[&transformers.causal,&transformers.character])?;
    backbone::released(
        &mut out,
        super::backbone::released_declaration().map_err(|e| e.to_string())?,
    )?;
    let mut identities = std::collections::BTreeSet::new();
    for parameter in &out {
        if !identities.insert(&parameter.key) {
            return Err(format!("duplicate declared VoiceChat source identity {}", parameter.key));
        }
    }
    Ok(out)
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
fn split_catalog(layout: GgufLayout) -> Result<Vec<SplitTensor>, String> {
    let mut out = Vec::new();
    match layout {
        GgufLayout::Complete => {}
        GgufLayout::Perception => {
    add_split(&mut out, "a.mel_filters", "stt_model.perception.preprocessor.featurizer.fb", &[(0, 0)], &[128, 257], GgmlType::F32, Transform::Identity);
    add_split(&mut out, "a.window", "stt_model.perception.preprocessor.featurizer.window", &[(0, 0)], &[400], GgmlType::F32, Transform::Identity);
        }
        GgufLayout::FunctionHead => {
        }
        GgufLayout::Backbone => {
        }
        GgufLayout::Tts => {
    add_split(&mut out, "codec.silence_tokens", "tts_model.codec_silence_tokens", &[(0, 0)], &[31], GgmlType::I32, Transform::Identity);
    add_split(&mut out, "tts.control_codes", "tts_model._control_codes", &[(0, 0)], &[3], GgmlType::I32, Transform::Identity);
    add_split(&mut out, "tts.subword.char_ids", "", &[(0, 0)], &[891139], GgmlType::I32, Transform::Identity);
    add_split(&mut out, "tts.subword.char_offsets", "", &[(0, 0)], &[131073], GgmlType::I32, Transform::Identity);
    add_split(&mut out, "tts.subword.is_continuation", "tts_model.tts_model.embed_subword.subword_flag_emb.is_continuation", &[(0, 0)], &[131073], GgmlType::I32, Transform::Identity);
    add_split(&mut out, "tts.subword.special_flags", "tts_model.tts_model.embed_subword.bos_eos_emb.special_flags", &[(0, 0)], &[131072], GgmlType::I32, Transform::Identity);
        }
    }
    if matches!(layout, GgufLayout::Tts | GgufLayout::FunctionHead | GgufLayout::Perception | GgufLayout::Backbone) {
        match layout {
            GgufLayout::Backbone => backbone::split(
                &mut out,
                super::backbone::released_declaration().map_err(|e| e.to_string())?,
            )?,
            GgufLayout::Tts => {
                mog::split(
                    &mut out,
                    super::tts_mog_specs::released_declaration().map_err(|e| e.to_string())?,
                )?;
                conditioning::split(
                    &mut out,
                    super::tts_conditioning_specs::released_declaration().map_err(|e| e.to_string())?,
                )?;
                rvq::split(
                    &mut out,
                    super::tts_rvq_specs::released_declaration().map_err(|e| e.to_string())?,
                )?;
                let codec = super::codec::released_declarations().map_err(|e| e.to_string())?;
                codec_codebooks::split(&mut out, codec)?;
                codec_neural::split(&mut out, codec)?;
                let transformers=super::tts_transformer_specs::released_declarations().map_err(|e|e.to_string())?;
                transformer::split(&mut out,[&transformers.causal,&transformers.character])?;
            }
            GgufLayout::FunctionHead => function_head::split(
                &mut out,
                super::function_head_layered::released_declaration().map_err(|e| e.to_string())?,
            )?,
            GgufLayout::Perception => {
                perception_pinned::split(
                    &mut out,
                    super::perception_specs::released_declaration().map_err(|e|e.to_string())?,
                )?;
                conformer::split(
                    &mut out,
                    super::perception_specs::released_declaration().map_err(|e|e.to_string())?,
                )?;
            }
            _ => unreachable!(),
        }
    }
    Ok(out)
}
