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
    random::RandomState,
    Array, Stream,
};
use safemlx_gguf::{MetadataArray as GgufMetadataArray, MetadataValue as GgufMetadataValue};
use safemlx_lm_utils::tokenizer::{
    chat_template_kwargs as inspect_chat_template_kwargs, load_model_chat_template_from_file,
    ApplyChatTemplateArgs, Chat, ModelChatTemplate, Tokenizer as ChatTokenizer,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokenizers::Tokenizer;

use crate::core::cache::{
    validate_prompt_cache_model_identity, PromptCacheDescriptor, PromptCacheManifest,
    PromptCacheModelIdentity, PromptCacheOptions,
};
use crate::core::generation::{
    resolve_generation_config, CheckpointGenerationConfig, FinishReason,
    GenerationCancellationToken, GenerationConfigOverrides, MtpConfig, MtpSchedulerOptions,
    ResolvedGenerationConfig, SemanticEvent,
};

pub(crate) use crate::nn as common;
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
pub(crate) use crate::runtime::media::input;
use crate::runtime::media::PreparedModelInput;
#[cfg(feature = "media-processing")]
use crate::runtime::media::{load_processor, ChatMediaBinding, ModelProcessor, ProcessorInput};
use crate::{
    core::cache::{CacheResidencyPool, LayerCachePolicy},
    error::Error,
    runtime::attention::LayerSchedule,
    runtime::cache::residency::{CacheResidencyPolicy, CacheResidencyReport, PagedCacheOptions},
    runtime::cache::{ConcatKeyValueCache, PagedKeyValueCache},
    runtime::generation::speculative::{
        DrafterKind, LoadedDrafter, MtpBatchOutput, MtpCache, MtpCapability, MtpCheckpointKind,
        MtpExecutionStreams, MtpScheduler, MtpSchedulerStats, MtpSemanticState, MtpStats,
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
pub(crate) use config::ensure_executable_load_options;
#[cfg(test)]
pub(crate) use config::{resolve_model_config, ResolvedModelConfig};
pub use config::{ModelKind, ModelLoadOptions};
pub use safemlx_lm_core::GgufArchitecture;

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
pub use crate::backend::mlx::MlxGeneration;
use dispatch::validate_gemma4_drafter;
pub use dispatch::{Model, ModelCache};

mod request;
use request::{
    prepare_chat_from_parts, with_prepared_chat_runtime, GenerationTokenSource,
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
pub(crate) use crate::backend::mlx::validate_gguf_quantization_source;
pub use loaded::LoadedModel;

mod inspection;
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
    use safemlx_lm_core::Backend;
    let inspection = safemlx_lm_core::inspect_artifact(model_dir.as_ref())?;
    let plan = safemlx_lm_core::plan_model_preparation(inspection, options.preparation_policy()?)?;
    crate::backend::mlx::MlxBackend::new(stream)
        .prepare_model(crate::backend::mlx::MlxModelConfig {
            plan,
            options,
            weights_stream,
        })
        .map(safemlx_lm_core::PreparedModel::into_inner)
}

mod tokenizer;
pub use tokenizer::{chat_template_kwargs, load_tokenizer};
use tokenizer::{
    effective_model_type, is_gguf_file, load_chat_template, load_gguf_tokenizer_from_metadata,
    load_tokenizer_template_kwargs, read_model_metadata,
};
pub(crate) use tokenizer::{gguf_sidecar_dir, tokenizer_vocabulary_fingerprint};

#[cfg(test)]
mod tests;
