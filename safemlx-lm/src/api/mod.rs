//! High-level model loading, architecture dispatch, and generation requests.
//!
//! Use [`crate::api::LoadedModel`] when you want to load a model directory
//! together with its tokenizer and chat template. Use
//! [`crate::api::load_model`] and [`crate::api::load_tokenizer`] when you
//! want to manage those pieces separately.

use std::{collections::HashMap, num::NonZeroUsize, path::Path};

use safemlx::{
    error::Exception,
    ops::indexing::{NewAxis, TryIndexOp},
    ops::{GgufCheckpoint, GgufMetadata, GgufMetadataArray, GgufMetadataValue},
    random::RandomState,
    Array, Stream,
};
use safemlx_lm_utils::tokenizer::{
    chat_template_kwargs as inspect_chat_template_kwargs, load_model_chat_template_from_file,
    ApplyChatTemplateArgs, Chat, ModelChatTemplate, Tokenizer as ChatTokenizer,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokenizers::Tokenizer;

pub(crate) use crate::nn as common;
use crate::nn::generation::CausalLm;
use crate::runtime::chat::constraints::ConstraintCompiler;
use crate::runtime::chat::{
    prepare_format_profile, resolve_structural_tokens, SemanticRuntimePlan,
};
pub use crate::runtime::chat::{
    CapabilitySupport, ChatCapabilities, ChatTemplateIdentity, ChatTemplateRequest,
    NativeToolSupport, ParallelToolCallPolicy, PreparedChat, SemanticSupport, ToolChoice,
};
use crate::runtime::checkpoint::gguf::{self as gguf_tokenizer, GgufTokenizer};
use crate::runtime::checkpoint::quantization::WeightQuantization;
use crate::runtime::distributed::topology::ParallelTopology;
use crate::runtime::execution::inspection::ActivationObserver;
use crate::runtime::generation::sampler::{
    CheckpointConfiguredSampler, ConstrainedSampler, DefaultSampler, Sampler, SpeculativeSampler,
};
use crate::runtime::generation::streaming::{
    drive_committed_generation_cancellable, CommittedTokenPipeline, CommittedTokenSource,
    RawTokenDecoder, TokenDecoderBackend,
};
pub use crate::runtime::generation::streaming::{
    FinishReason, GenerationCancellationToken, SemanticEvent,
};
pub(crate) use crate::runtime::media::input;
use crate::runtime::media::PreparedModelInput;
#[cfg(feature = "media-processing")]
use crate::runtime::media::{load_processor, ChatMediaBinding, ModelProcessor, ProcessorInput};
use crate::{
    error::Error,
    runtime::attention::LayerSchedule,
    runtime::cache::residency::{
        validate_prompt_cache_model_identity, CacheResidencyPolicy, CacheResidencyPool,
        CacheResidencyReport, LayerCachePolicy, PagedCacheOptions, PromptCacheDescriptor,
        PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
    },
    runtime::cache::{ConcatKeyValueCache, PagedKeyValueCache},
    runtime::generation::speculative::{
        DrafterKind, LoadedDrafter, MtpBatchOutput, MtpCache, MtpCapability, MtpCheckpointKind,
        MtpConfig, MtpExecutionStreams, MtpScheduler, MtpSchedulerOptions, MtpSchedulerStats,
        MtpSemanticState, MtpStats,
    },
};

/// DeepSeek-V3 and DeepSeek-R1 decoder support.
pub(crate) use crate::architectures::deepseek_v3::model as deepseek_v3;
/// DeepSeek-V4 compressed sparse-attention decoder support.
pub(crate) use crate::architectures::deepseek_v4::model as deepseek_v4;
pub(crate) use crate::architectures::gemma4::assistant as gemma4_assistant;
pub(crate) use crate::architectures::gemma4::audio as gemma4_audio;
/// Gemma 4 text model support.
pub(crate) use crate::architectures::gemma4::model as gemma4;
pub(crate) use crate::architectures::gemma4::multimodal as gemma4_multimodal;
pub(crate) use crate::architectures::gemma4::vision as gemma4_vision;
/// OpenAI GPT-OSS sparse decoder architecture.
pub(crate) use crate::architectures::gpt_oss::model as gpt_oss;
/// Thinking Machines Lab Inkling multimodal decoder support.
pub(crate) use crate::architectures::inkling::model as inkling;
/// Moonshot Kimi Linear hybrid KDA/MLA sparse decoder support.
pub(crate) use crate::architectures::kimi_linear::model as kimi_linear;
/// Liquid AI LFM2/LFM2.5 dense and MoE text model support.
pub(crate) use crate::architectures::lfm2::model as lfm2;
/// Llama decoder-only model support.
pub(crate) use crate::architectures::llama::model as llama;
/// Moshi token language-model support.
///
/// This module operates on pre-tokenized Mimi streams. It intentionally does
/// not implement audio encoding, decoding, or realtime device I/O.
pub(crate) use crate::architectures::moshi::model as moshi;
/// PersonaPlex realtime speech-to-speech token model support.
///
/// This module operates on pre-tokenized Mimi streams and hybrid prompt tokens.
/// It intentionally does not implement audio encoding, decoding, or realtime
/// device I/O.
pub(crate) use crate::architectures::moshi::personaplex;
/// Meta Muse-Glimmer dense multimodal decoder support.
pub(crate) use crate::architectures::muse_glimmer;
/// Nemotron-H hybrid Mamba2/attention/MoE config support.
pub(crate) use crate::architectures::nemotron_h::model as nemotron_h;
/// Shared Qwen2/Qwen2.5/Qwen3 dense decoder support.
use crate::architectures::qwen::dense as dense_qwen;
/// Qwen3.5 dense and MoE text model support.
pub(crate) use crate::architectures::qwen::hybrid::qwen3_5;
/// Qwen3-Next hybrid attention/MoE text model support.
pub(crate) use crate::architectures::qwen::hybrid::qwen3_next;
/// Qwen3-VL multimodal conditional-generation support.
pub(crate) use crate::architectures::qwen::vl::model as qwen3_vl;
/// Qwen3-VL-MoE multimodal conditional-generation support.
pub(crate) use crate::architectures::qwen::vl::moe as qwen3_vl_moe;
pub(crate) use crate::architectures::qwen::vl::vision as qwen_vl;

#[derive(Debug, Clone, Deserialize)]
struct ModelMetadata {
    model_type: String,
    #[serde(default)]
    text_config: Option<TextModelMetadata>,
}

#[derive(Debug, Clone, Deserialize)]
struct TextModelMetadata {
    #[serde(default)]
    model_type: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct EosTokenMetadata {
    #[serde(default)]
    eos_token_id: Option<TokenIdOrIds>,
    #[serde(default)]
    text_config: Option<TextEosTokenMetadata>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct TextEosTokenMetadata {
    #[serde(default)]
    eos_token_id: Option<TokenIdOrIds>,
}

/// Sampling values declared by a checkpoint's `generation_config.json`.
///
/// Missing fields remain distinguishable from explicit values so applications
/// can layer request overrides without losing provenance.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
pub struct CheckpointGenerationConfig {
    /// Whether the checkpoint recommends stochastic sampling.
    #[serde(default)]
    pub do_sample: Option<bool>,
    /// Recommended sampling temperature.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Recommended top-k cutoff.
    #[serde(default)]
    pub top_k: Option<i32>,
    /// Recommended nucleus-sampling probability.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Recommended minimum-probability filter.
    #[serde(default)]
    pub min_p: Option<f32>,
    /// Recommended maximum number of newly generated tokens.
    #[serde(default)]
    pub max_new_tokens: Option<usize>,
}

/// Per-request overrides applied to checkpoint generation defaults.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GenerationConfigOverrides {
    /// Overrides stochastic-versus-greedy selection.
    pub do_sample: Option<bool>,
    /// Overrides temperature; zero selects greedy decoding.
    pub temperature: Option<f32>,
    /// Overrides top-k filtering; zero disables it.
    pub top_k: Option<i32>,
    /// Overrides top-p filtering; one disables it.
    pub top_p: Option<f32>,
    /// Overrides min-p filtering; zero disables it.
    pub min_p: Option<f32>,
    /// Overrides the maximum number of newly generated tokens.
    pub max_new_tokens: Option<usize>,
}

/// Fully resolved generation settings used by SafeMLX samplers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolvedGenerationConfig {
    /// Whether the effective policy samples stochastically.
    pub do_sample: bool,
    /// Effective sampling temperature.
    pub temperature: f32,
    /// Effective top-k cutoff.
    pub top_k: i32,
    /// Effective top-p probability.
    pub top_p: f32,
    /// Effective min-p probability.
    pub min_p: f32,
    /// Effective checkpoint or request token limit, when declared.
    pub max_new_tokens: Option<usize>,
}

impl ResolvedGenerationConfig {
    /// Builds the corresponding configurable SafeMLX sampler.
    pub fn sampler(self) -> crate::runtime::generation::sampler::GenerationSampler {
        crate::runtime::generation::sampler::GenerationSampler::new()
            .top_k(self.top_k)
            .top_p(self.top_p)
            .min_p(self.min_p)
    }
}

fn read_checkpoint_generation_config(
    sidecar_dir: &Path,
) -> Result<Option<CheckpointGenerationConfig>, Error> {
    let path = sidecar_dir.join("generation_config.json");
    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    Ok(Some(serde_json::from_reader(file)?))
}

fn resolve_generation_config(
    checkpoint: Option<&CheckpointGenerationConfig>,
    overrides: GenerationConfigOverrides,
) -> Result<ResolvedGenerationConfig, Error> {
    let checkpoint_present = checkpoint.is_some();
    let checkpoint = checkpoint.cloned().unwrap_or_default();
    let (do_sample, temperature) = if let Some(do_sample) = overrides.do_sample {
        if do_sample {
            (
                true,
                overrides
                    .temperature
                    .or(checkpoint.temperature)
                    .unwrap_or(1.0),
            )
        } else {
            (false, 0.0)
        }
    } else if let Some(temperature) = overrides.temperature {
        // A request-level temperature is itself an explicit sampling override,
        // including when the checkpoint recommends greedy decoding.
        (temperature > 0.0, temperature)
    } else if checkpoint.do_sample.unwrap_or(false) {
        (true, checkpoint.temperature.unwrap_or(1.0))
    } else {
        (false, 0.0)
    };
    let resolved = ResolvedGenerationConfig {
        do_sample,
        temperature,
        top_k: overrides
            .top_k
            .or(checkpoint.top_k)
            .unwrap_or(if checkpoint_present { 50 } else { 40 }),
        top_p: overrides
            .top_p
            .or(checkpoint.top_p)
            .unwrap_or(if checkpoint_present { 1.0 } else { 0.95 }),
        min_p: overrides
            .min_p
            .or(checkpoint.min_p)
            .unwrap_or(if checkpoint_present { 0.0 } else { 0.05 }),
        max_new_tokens: overrides.max_new_tokens.or(checkpoint.max_new_tokens),
    };
    if !resolved.temperature.is_finite() || resolved.temperature < 0.0 {
        return Err(Error::GenerationConfig(format!(
            "temperature must be finite and non-negative, got {}",
            resolved.temperature
        )));
    }
    if resolved.do_sample && resolved.temperature == 0.0 {
        return Err(Error::GenerationConfig(
            "do_sample=true requires a temperature greater than zero".into(),
        ));
    }
    if resolved.top_k < 0 {
        return Err(Error::GenerationConfig(format!(
            "top_k must be non-negative, got {}",
            resolved.top_k
        )));
    }
    if !resolved.top_p.is_finite() || !(0.0..=1.0).contains(&resolved.top_p) {
        return Err(Error::GenerationConfig(format!(
            "top_p must be between zero and one, got {}",
            resolved.top_p
        )));
    }
    if !resolved.min_p.is_finite() || !(0.0..=1.0).contains(&resolved.min_p) {
        return Err(Error::GenerationConfig(format!(
            "min_p must be between zero and one, got {}",
            resolved.min_p
        )));
    }
    if resolved.max_new_tokens == Some(0) {
        return Err(Error::GenerationConfig(
            "max_new_tokens must be positive when supplied".into(),
        ));
    }
    Ok(resolved)
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum TokenIdOrIds {
    Single(u32),
    Multiple(Vec<u32>),
}

impl TokenIdOrIds {
    fn into_vec(self) -> Vec<u32> {
        match self {
            Self::Single(id) => vec![id],
            Self::Multiple(ids) => ids,
        }
    }
}

fn append_unique_eos_token_ids(output: &mut Vec<u32>, ids: impl IntoIterator<Item = u32>) {
    for id in ids {
        if !output.contains(&id) {
            output.push(id);
        }
    }
}

fn merge_eos_token_id_sources(
    sources: impl IntoIterator<Item = impl IntoIterator<Item = u32>>,
) -> Vec<u32> {
    let mut output = Vec::new();
    for source in sources {
        append_unique_eos_token_ids(&mut output, source);
    }
    output
}

fn read_optional_eos_token_metadata(path: &Path) -> Result<Option<EosTokenMetadata>, Error> {
    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    Ok(Some(serde_json::from_reader(file)?))
}

fn eos_token_ids_from_sidecar_dir(sidecar_dir: &Path) -> Result<Vec<u32>, Error> {
    let mut output = Vec::new();
    for filename in ["config.json", "generation_config.json"] {
        let Some(metadata) = read_optional_eos_token_metadata(&sidecar_dir.join(filename))? else {
            continue;
        };
        if let Some(ids) = metadata.eos_token_id {
            append_unique_eos_token_ids(&mut output, ids.into_vec());
        }
        if let Some(ids) = metadata
            .text_config
            .and_then(|text_config| text_config.eos_token_id)
        {
            append_unique_eos_token_ids(&mut output, ids.into_vec());
        }
    }
    Ok(output)
}

pub(crate) fn gguf_eos_token_ids(
    metadata: &std::collections::HashMap<String, GgufMetadataValue>,
) -> Result<Vec<u32>, Error> {
    const KEY: &str = "tokenizer.ggml.eos_token_id";
    let Some(value) = metadata.get(KEY) else {
        return Ok(Vec::new());
    };

    fn invalid(value: impl std::fmt::Display) -> Error {
        Error::UnsupportedArchitecture(format!(
            "GGUF metadata key \"tokenizer.ggml.eos_token_id\" contains invalid EOS token id {value}; expected an integer from 0 through {}",
            u32::MAX
        ))
    }

    let values = match value {
        GgufMetadataValue::Uint64(value) => {
            return u32::try_from(*value)
                .map(|value| vec![value])
                .map_err(|_| invalid(value));
        }
        GgufMetadataValue::Array(GgufMetadataArray::Uint64(values)) => {
            return values
                .iter()
                .map(|&value| u32::try_from(value).map_err(|_| invalid(value)))
                .collect();
        }
        value => value.to_i64_vec().ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "GGUF metadata key {KEY:?} must be an integer or integer array"
            ))
        })?,
    };
    values
        .into_iter()
        .map(|value| u32::try_from(value).map_err(|_| invalid(value)))
        .collect()
}

mod config;
pub(crate) use config::{
    ensure_executable_load_options, validate_load_policy, ArtifactLoadKind, GgufArchitecture,
};
#[cfg(test)]
pub(crate) use config::{resolve_model_config, ResolvedModelConfig};
pub use config::{ModelKind, ModelLoadOptions};

mod automatic;
pub use automatic::{
    discover_hardware, execution_plan_load_options, plan_automatic_execution, AllocatorTelemetry,
    AutomaticPlanRequest, AutomaticPlanner, AutomaticPlannerPolicy, BackendKind, DevicePlan,
    DraftPlacementPlan, DraftingPlan, DurationSeconds, ExecutionPlan, ExecutionPlanReport,
    ExecutionTelemetry, ExpertCachePlan, ExpertCacheTelemetry, HardwareBackendProfile,
    HardwareDeviceProfile, HardwareMemorySemantics, HardwareProfile, ModelResourceProfile,
    ObservationKind, Observed, ParallelismPlan, PlanExplanation, PlanExplanationEntry,
    PlanExplanationLevel, ResidencyPlan, ResidencyTelemetry, TimingTelemetry, TransferTelemetry,
    WeightTransformationPlan, AUTOMATIC_SCHEMA_VERSION,
};

mod capability;
pub use capability::{
    available_memory, Admission, AdmissionRejection, AdmissionRequest, AdmissionResult,
    AvailableMemory, CacheStateStrategy, CapabilityError, CapabilityValue, EstimationCompleteness,
    InputModalities, InputTokenCount, MeasurementKind, ModelCapabilities, PhysicalMemorySemantics,
    RuntimeStateEstimate, SlidingWindowLayerCount, StateMemoryAssumptions, StaticMemoryReport,
};

mod dispatch;
use dispatch::validate_gemma4_drafter;
pub use dispatch::{Model, ModelCache, ModelGenerate};

mod request;
use request::{
    prepare_chat_from_parts, with_prepared_chat_runtime, ModelGenerateTokenSource,
    PreparedChatSemanticState, PreparedChatTokenDecoder, ResolvedPreparedChatGenerationSettings,
};
pub use request::{
    PreparedChatEmbeddedMtpBatchRequest, PreparedChatEmbeddedMtpGenerationRequest,
    PreparedChatGenerationOutput, PreparedChatGenerationRequest, PreparedChatGenerationSettings,
    PreparedChatInput, PreparedChatMtpBatchLane, PreparedChatMtpBatchOutput,
    PreparedChatMtpBatchRequest, PreparedChatMtpGenerationOptions, PreparedChatMtpGenerationOutput,
    PreparedChatMtpGenerationRequest, TextDecoder,
};
/// Codec-free realtime speech-to-speech token APIs.
pub mod realtime;

mod loaded;
use loaded::load_gguf_model_data;
pub(crate) use loaded::validate_gguf_quantization_source;
pub use loaded::LoadedModel;

mod inspection;
pub(crate) mod structural;
pub use inspection::{
    inspect_model, ArtifactKind, ArtifactModality, ArtifactTensorEncoding, InspectionIssue,
    InspectionIssueCode, InspectionReadiness, InspectionRequirement, InspectionSeverity,
    ModelInspectionOptions, ModelInspectionReport,
};

/// Loads only the model weights and architecture from a model directory.
pub fn load_model(
    model_dir: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    load_model_with_options(
        model_dir,
        ModelLoadOptions::default(),
        stream,
        weights_stream,
    )
}

/// Loads only the model weights and architecture using shared load options.
pub fn load_model_with_options(
    model_dir: impl AsRef<Path>,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    ensure_executable_load_options(options)?;
    if is_gguf_file(model_dir) {
        return Ok(load_gguf_model_data(model_dir, false, options, stream, weights_stream)?.model);
    }
    let metadata = read_model_metadata(model_dir)?;
    let kind = ModelKind::from_model_type(&effective_model_type(&metadata))?;
    match kind {
        ModelKind::PersonaPlex => Err(Error::UnsupportedArchitecture(
            "PersonaPlex is a realtime speech-to-speech token model; use architectures::moshi::personaplex::load_model".into(),
        )),
        _ => load_model_for_kind(kind, model_dir, options, stream, weights_stream),
    }
}

fn load_model_for_kind(
    kind: ModelKind,
    model_dir: &Path,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    validate_load_policy(kind, ArtifactLoadKind::Safetensors, options)?;
    if let (Some(expert_cache), Some(non_expert)) = (
        options.weight_residency.expert_cache(),
        options.weight_residency.non_experts(),
    ) {
        return match kind {
            ModelKind::KimiLinear => Ok(Model::KimiLinear(
                crate::architectures::kimi_linear::layerwise::load_kimi_linear_expert_cache_model(
                    model_dir, non_expert, expert_cache, options.quantization, stream, weights_stream,
                )?,
            )),
            ModelKind::DeepSeekV3 => Ok(Model::DeepSeekV3(
                crate::architectures::deepseek_v3::layerwise::load_deepseek_v3_expert_cache_model(
                    model_dir, non_expert, expert_cache, options.quantization, stream, weights_stream,
                )?,
            )),
            ModelKind::DeepSeekV4 => Ok(Model::DeepSeekV4Layerwise(Box::new(
                crate::architectures::deepseek_v4::layerwise::load_deepseek_v4_expert_cache_model(
                    model_dir,
                    non_expert,
                    expert_cache,
                    options.quantization,
                    stream,
                    weights_stream,
                )?,
            ))),
            ModelKind::GptOss => Ok(Model::GptOss(
                crate::architectures::gpt_oss::layerwise::load_gpt_oss_expert_cache_model(
                    model_dir, non_expert, expert_cache, options.quantization, stream, weights_stream,
                )?,
            )),
            ModelKind::Inkling => Ok(Model::Inkling(
                crate::architectures::inkling::layerwise::load_inkling_expert_cache_model(
                    model_dir, non_expert, expert_cache, options.quantization, stream, weights_stream,
                )?,
            )),
            ModelKind::Lfm2 => Ok(Model::Lfm2(
                crate::architectures::lfm2::layerwise::load_lfm2_expert_cache_model(
                    model_dir, non_expert, expert_cache, options.quantization, stream, weights_stream,
                )?,
            )),
            ModelKind::NemotronH => Ok(Model::NemotronH(
                crate::architectures::nemotron_h::layerwise::load_nemotron_h_expert_cache_model(
                    model_dir, non_expert, expert_cache, options.quantization, stream, weights_stream,
                )?,
            )),
            ModelKind::Qwen2 => Err(Error::UnsupportedArchitecture(
                "Qwen2 is dense and does not support sparse expert-cache residency".into(),
            )),
            ModelKind::Qwen3 => Ok(Model::DenseQwen(
                crate::architectures::qwen::dense::layerwise::load_qwen3_expert_cache_model(
                    model_dir, non_expert, expert_cache, options.quantization, stream, weights_stream,
                )?,
            )),
            ModelKind::Qwen3Next => Ok(Model::Qwen3Next(
                crate::architectures::qwen::hybrid::layerwise::load_qwen3_next_expert_cache_model(
                    model_dir, non_expert, expert_cache, options.quantization, stream, weights_stream,
                )?,
            )),
            ModelKind::Qwen3VlMoe => Ok(Model::Qwen3VlMoe(
                crate::architectures::qwen::vl::layerwise::load_qwen3_vl_expert_cache_model(
                    model_dir, non_expert, expert_cache, options.quantization, stream, weights_stream,
                )?,
            )),
            ModelKind::Qwen35 => Ok(Model::Qwen35(
                crate::architectures::qwen::hybrid::layerwise::load_qwen35_expert_cache_model(
                    model_dir, non_expert, expert_cache, options.quantization, stream, weights_stream,
                )?,
            )),
            _ => Err(Error::UnsupportedArchitecture(format!(
                "independent expert caching requires a supported safetensors MoE architecture, not {}",
                kind.model_type_name()
            ))),
        };
    }
    let execution = options.weight_residency.layers();
    if let Some(quantization) = options.quantization {
        quantization.validate()?;
    }
    match kind {
        ModelKind::DeepSeekV3 => Ok(Model::DeepSeekV3(
            crate::architectures::deepseek_v3::layerwise::load_deepseek_v3_layerwise_model(
                model_dir,
                execution,
                options.quantization,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::DeepSeekV4 => Ok(Model::DeepSeekV4Layerwise(Box::new(
            crate::architectures::deepseek_v4::layerwise::load_deepseek_v4_layerwise_model(
                model_dir,
                execution,
                options.quantization,
                stream,
                weights_stream,
            )?,
        ))),
        ModelKind::Gemma4 => Ok(Model::Gemma4(Box::new(
            crate::architectures::gemma4::layerwise::load_gemma4_layerwise_model(
                model_dir,
                execution,
                options.quantization,
                stream,
                weights_stream,
            )?,
        ))),
        ModelKind::Inkling => Ok(Model::Inkling(
            crate::architectures::inkling::layerwise::load_inkling_layerwise_model(
                model_dir,
                execution,
                options.quantization,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::KimiLinear => Ok(Model::KimiLinear(
            crate::architectures::kimi_linear::layerwise::load_kimi_linear_layerwise_model(
                model_dir,
                execution,
                options.quantization,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Llama => Ok(Model::Llama(
            crate::architectures::llama::layerwise::load_llama_model(
                model_dir,
                crate::architectures::llama::layerwise::LlamaLoadOptions {
                    weight_residency: execution.weight_residency(),
                    quantization: options.quantization,
                },
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::MuseGlimmer => Ok(Model::MuseGlimmer(
            crate::architectures::muse_glimmer::layerwise::load_safetensors(
                model_dir,
                execution,
                options.quantization,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Qwen2 | ModelKind::Qwen3 => Ok(Model::DenseQwen(
            crate::architectures::qwen::dense::layerwise::load_safetensors(
                model_dir,
                execution,
                options.quantization,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::GptOss => Ok(Model::GptOss(
            crate::architectures::gpt_oss::layerwise::load_gpt_oss_layerwise_model(
                model_dir,
                execution,
                options.quantization,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Lfm2 => Ok(Model::Lfm2(
            crate::architectures::lfm2::layerwise::load_lfm2_layerwise_model(
                model_dir,
                execution,
                options.quantization,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::NemotronH => Ok(Model::NemotronH(
            crate::architectures::nemotron_h::layerwise::load_nemotron_h_layerwise_model(
                model_dir,
                execution,
                options.quantization,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Qwen3Next => Ok(Model::Qwen3Next(
            crate::architectures::qwen::hybrid::layerwise::load_qwen3_next_layerwise_model(
                model_dir,
                execution,
                options.quantization,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Qwen3Vl => Ok(Model::Qwen3Vl(
            crate::architectures::qwen::vl::layerwise::load_qwen3_vl_layerwise_model(
                model_dir,
                execution,
                options.quantization,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Qwen3VlMoe => Ok(Model::Qwen3VlMoe(
            crate::architectures::qwen::vl::layerwise::load_qwen3_vl_layerwise_model(
                model_dir,
                execution,
                options.quantization,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Qwen35 => Ok(Model::Qwen35(
            crate::architectures::qwen::hybrid::layerwise::load_qwen35_layerwise_model(
                model_dir,
                execution,
                options.quantization,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::PersonaPlex => Err(Error::UnsupportedArchitecture(
            "PersonaPlex bounded layer residency is selected through the realtime loader".into(),
        )),
    }
}

mod tokenizer;
pub(crate) use tokenizer::tokenizer_vocabulary_fingerprint;
pub use tokenizer::{chat_template_kwargs, load_tokenizer};
use tokenizer::{
    effective_model_type, gguf_sidecar_dir, is_gguf_file, load_chat_template,
    load_gguf_tokenizer_from_metadata, load_tokenizer_template_kwargs, read_model_metadata,
};

#[cfg(test)]
mod tests;
