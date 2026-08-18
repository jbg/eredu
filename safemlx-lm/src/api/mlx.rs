//! MLX model loading, architecture dispatch, and generation extensions.
//!
//! Use [`crate::api::LoadedModel`] when you want to load a model directory
//! together with its tokenizer and chat template. Use
//! [`crate::load_model`] and [`crate::api::load_tokenizer`] when you
//! want to manage those pieces separately.
//! Ordinary generation is available for every `TextGenerationBackend`;
//! prepared-chat speculative generation is available through the
//! [`PreparedChatSpeculativeBackend`] capability on the same `LoadedModel<B>`.

use super::portable::{
    LoadedModel, LoadedTextModelConfig, TextDecoder, TextDecoderError, TextModelError,
};

use std::{collections::HashMap, num::NonZeroUsize, path::Path};

use safemlx::{
    error::Exception,
    ops::indexing::{NewAxis, TryIndexOp},
    random::RandomState,
    Array, Stream,
};
use safemlx_gguf::MetadataValue as GgufMetadataValue;
use safemlx_lm_utils::tokenizer::{
    chat_template_kwargs as inspect_chat_template_kwargs, load_model_chat_template_from_file,
    ApplyChatTemplateArgs, Chat,
};
pub use safemlx_lm_utils::tokenizer::{ModelChatTemplate, Tokenizer as ChatTokenizer};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokenizers::Tokenizer;

use crate::core::cache::{PromptCacheDescriptor, PromptCacheManifest, PromptCacheOptions};
use crate::core::generation::{
    resolve_generation_config, CheckpointGenerationConfig, FinishReason,
    GenerationCancellationToken, GenerationConfigOverrides, MtpConfig, MtpSchedulerOptions,
    SemanticEvent,
};
use crate::core::{
    MtpBatchOutput, MtpCapability, MtpSchedulerStats, MtpStats, SpeculativeOutputError,
    SpeculativeSemanticState,
};

use crate::backend::mlx::MlxGeneration;
use crate::runtime::chat::constraints::ConstraintCompiler;
use crate::runtime::chat::{
    prepare_format_profile, resolve_structural_tokens, SemanticRuntimePlan,
};
pub use crate::runtime::chat::{
    CapabilitySupport, ChatCapabilities, ChatTemplateIdentity, ChatTemplateRequest,
    NativeToolSupport, ParallelToolCallPolicy, PreparedChat, SemanticSupport, ToolChoice,
};
use crate::runtime::checkpoint::gguf::{self as gguf_tokenizer, GgufTokenizer};
use crate::runtime::execution::inspection::ActivationObserver;
pub use crate::runtime::generation::sampler::ConstraintError;
use crate::runtime::generation::sampler::{
    ConstrainedSampler, DefaultSampler, Sampler, SpeculativeSampler,
};
use crate::runtime::generation::streaming::{
    drive_committed_generation_cancellable, CommittedGenerationError, CommittedTokenPipeline,
    CommittedTokenPipelineError, CommittedTokenSource, RawTokenDecoder, TokenDecoderBackend,
};
pub(crate) use crate::runtime::media::input;
use crate::runtime::media::PreparedModelInput;
#[cfg(feature = "media-processing")]
use crate::runtime::media::{ChatMediaBinding, ModelProcessor, ProcessorInput};
use crate::{
    backend::mlx::speculative::{
        scheduler::MlxMtpScheduler, MlxDrafter, MlxDrafterKind, MlxMtpCache, MtpExecutionStreams,
    },
    core::cache::{CacheResidencyPool, LayerCachePolicy},
    error::Error,
    runtime::attention::LayerSchedule,
    runtime::cache::residency::{CacheResidencyPolicy, CacheResidencyReport, PagedCacheOptions},
};

/// DeepSeek-V3 and DeepSeek-R1 decoder support.
pub(crate) use crate::architectures::deepseek_v3::model as deepseek_v3;
/// Gemma 4 text model support.
pub(crate) use crate::architectures::gemma4::model as gemma4;
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
/// Meta Muse-Glimmer dense multimodal decoder support.
pub(crate) use crate::architectures::muse_glimmer;
/// Nemotron-H hybrid Mamba2/attention/MoE config support.
pub(crate) use crate::architectures::nemotron_h::model as nemotron_h;
/// Shared Qwen2/Qwen2.5/Qwen3 dense decoder support.
use crate::architectures::qwen::dense as dense_qwen;
/// Qwen3.5 dense and MoE text model support.
pub(crate) use crate::architectures::qwen::hybrid::qwen3_5;
/// Qwen3-VL multimodal conditional-generation support.
pub(crate) use crate::architectures::qwen::vl::model as qwen3_vl;

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
) -> Result<Option<CheckpointGenerationConfig>, tokenizer::TextMetadataError> {
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

fn read_optional_eos_token_metadata(
    path: &Path,
) -> Result<Option<EosTokenMetadata>, tokenizer::TextMetadataError> {
    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    Ok(Some(serde_json::from_reader(file)?))
}

fn eos_token_ids_from_sidecar_dir(
    sidecar_dir: &Path,
) -> Result<Vec<u32>, tokenizer::TextMetadataError> {
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
) -> Result<Vec<u32>, tokenizer::TextMetadataError> {
    const KEY: &str = "tokenizer.ggml.eos_token_id";
    safemlx_lm_core::gguf_u32_metadata_values(KEY, metadata.get(KEY))
        .map_err(|error| tokenizer::TextMetadataError::UnsupportedArchitecture(error.to_string()))
}

use crate::backend::mlx::ModelLoadOptions;
pub use safemlx_lm_core::GgufArchitecture;
pub use safemlx_lm_core::ModelKind;

#[path = "automatic.rs"]
mod automatic;
pub use automatic::{
    discover_hardware, execution_plan_load_options, plan_automatic_execution, AllocatorTelemetry,
    AutomaticPlanRequest, AutomaticPlanner, AutomaticPlannerPolicy, DurationSeconds,
    ExecutionPlanReport, ExecutionTelemetry, ExpertCacheTelemetry, HardwareBackendProfile,
    HardwareDeviceProfile, HardwareMemorySemantics, HardwareProfile, ModelResourceProfile,
    ObservationKind, Observed, PlanExplanation, PlanExplanationEntry, PlanExplanationLevel,
    ResidencyTelemetry, TimingTelemetry, TransferTelemetry, AUTOMATIC_SCHEMA_VERSION,
};
pub use safemlx_lm_core::{
    BackendId, DevicePlan, DraftPlacementPlan, DraftingPlan, ExecutionPlan, ExpertCachePlan,
    ResidencyPlan, WeightTransformationPlan,
};

#[path = "capability.rs"]
mod capability;
pub use capability::{
    available_memory, Admission, AdmissionRejection, AdmissionRequest, AdmissionResult,
    AvailableMemory, CacheStateStrategy, CapabilityError, CapabilityValue, EstimationCompleteness,
    InputModalities, InputTokenCount, MeasurementKind, ModelCapabilities, PhysicalMemorySemantics,
    RuntimeStateEstimate, SlidingWindowLayerCount, StateMemoryAssumptions, StaticMemoryReport,
};

use crate::backend::mlx::{validate_gemma4_drafter, Model, ModelCache};

#[path = "request.rs"]
mod request;
use request::{
    prepare_chat_from_parts, prepared_chat_control_runtime, with_prepared_chat_runtime,
    BackendGenerationTokenSource, PreparedChatSemanticState, PreparedChatSetupError,
    PreparedChatTokenDecoder, ResolvedPreparedChatGenerationSettings,
};
pub use request::{
    PreparedChatDraft, PreparedChatError, PreparedChatGenerationOutput,
    PreparedChatGenerationRequest, PreparedChatGenerationSettings, PreparedChatInput,
    PreparedChatMtpBatchLane, PreparedChatMtpBatchOutput, PreparedChatMtpBatchRequest,
    PreparedChatMtpGenerationOptions, PreparedChatMtpGenerationOutput,
    PreparedChatMtpGenerationRequest, PreparedChatSpeculativeBackend,
};

impl PreparedChatSpeculativeBackend for crate::backend::mlx::MlxBackend<'static> {
    type Drafter = MlxDrafter;
    type SpeculativeError = Error;

    fn mtp_capability(model: &LoadedModel<Self>) -> MtpCapability {
        model.mlx_mtp_capability()
    }

    fn execute_prepared_chat_mtp<'a, F>(
        model: &mut LoadedModel<Self>,
        request: PreparedChatMtpGenerationRequest<'a, Self, Self::Drafter, F>,
    ) -> Result<PreparedChatMtpGenerationOutput, Error>
    where
        F: FnMut(SemanticEvent),
    {
        model.execute_prepared_chat_mtp_mlx(request)
    }

    fn execute_prepared_chat_mtp_batch<'a>(
        model: &mut LoadedModel<Self>,
        request: PreparedChatMtpBatchRequest<'a, Self, Self::Drafter>,
    ) -> Result<PreparedChatMtpBatchOutput, Error> {
        model.execute_prepared_chat_mtp_batch_mlx(request)
    }
}

#[path = "loaded.rs"]
mod loaded;
pub(crate) use crate::backend::mlx::validate_gguf_quantization_source;
pub use loaded::LoadedModelLoadError;

#[path = "inspection.rs"]
mod inspection;
pub use inspection::{
    inspect_model, ArtifactKind, ArtifactModality, ArtifactTensorEncoding, InspectionIssue,
    InspectionIssueCode, InspectionReadiness, InspectionRequirement, InspectionSeverity,
    ModelInspectionOptions, ModelInspectionReport,
};

#[path = "tokenizer.rs"]
mod tokenizer;
pub(crate) use tokenizer::gguf_sidecar_dir;
pub use tokenizer::{chat_template_kwargs, load_tokenizer, TextMetadataError};
use tokenizer::{
    is_gguf_file, load_chat_template, load_gguf_tokenizer_from_metadata, load_tokenizer_for_kind,
    load_tokenizer_template_kwargs,
};

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
