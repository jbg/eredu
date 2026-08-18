//! MLX model loading, architecture dispatch, and generation extensions.
//!
//! Use [`crate::api::LoadedModel`] when you want to load a model directory
//! together with its tokenizer and chat template. Use
//! [`crate::load_model`] and [`crate::api::load_tokenizer`] when you
//! want to manage those pieces separately.
//! Ordinary generation is available for every `TextGenerationBackend`;
//! prepared-chat speculative generation is available through the
//! [`PreparedChatSpeculativeBackend`] capability on the same `LoadedModel<B>`.

use super::metadata::{
    eos_token_ids_from_sidecar_dir, gguf_eos_token_ids, merge_eos_token_id_sources,
    read_checkpoint_generation_config,
};
use super::portable::{LoadedModel, LoadedTextModelConfig, TextDecoderError, TextModelError};
use super::request::{
    prepare_chat_from_parts, prepared_chat_control_runtime, with_prepared_chat_runtime,
    BackendGenerationTokenSource, PreparedChatDraft, PreparedChatError,
    PreparedChatGenerationOutput, PreparedChatGenerationRequest, PreparedChatGenerationSettings,
    PreparedChatInput, PreparedChatMtpBatchLane, PreparedChatMtpBatchOutput,
    PreparedChatMtpBatchRequest, PreparedChatMtpGenerationOptions, PreparedChatMtpGenerationOutput,
    PreparedChatMtpGenerationRequest, PreparedChatSemanticState, PreparedChatSetupError,
    PreparedChatSpeculativeBackend, PreparedChatTokenDecoder,
    ResolvedPreparedChatGenerationSettings,
};
use super::tokenizer::{
    gguf_sidecar_dir, load_chat_template, load_gguf_tokenizer_from_metadata,
    load_tokenizer_for_kind, load_tokenizer_template_kwargs, TextMetadataError,
};

use std::{num::NonZeroUsize, path::Path};

use safemlx::{
    error::Exception,
    ops::indexing::{NewAxis, TryIndexOp},
    random::RandomState,
    Array, Stream,
};
use safemlx_gguf::MetadataValue as GgufMetadataValue;
use safemlx_lm_utils::gguf::GgufTokenizer;
use safemlx_lm_utils::tokenizer::{
    chat_template_kwargs as inspect_chat_template_kwargs, ApplyChatTemplateArgs, Chat,
};
pub use safemlx_lm_utils::tokenizer::{ModelChatTemplate, Tokenizer as ChatTokenizer};
use serde::Serialize;

use crate::core::cache::{PromptCacheDescriptor, PromptCacheManifest, PromptCacheOptions};
use crate::core::generation::{
    resolve_generation_config, FinishReason, GenerationCancellationToken,
    GenerationConfigOverrides, MtpConfig, MtpSchedulerOptions, SemanticEvent,
};
use crate::core::{MtpBatchOutput, MtpCapability, MtpStats, SpeculativeSemanticState};

use crate::backend::mlx::MlxGeneration;
use crate::runtime::chat::constraints::ConstraintCompiler;
use crate::runtime::chat::{
    ChatTemplateIdentity, ChatTemplateRequest, PreparedChat, SemanticSupport,
};
use crate::runtime::execution::inspection::ActivationObserver;
use crate::runtime::generation::sampler::{
    ConstrainedSampler, DefaultSampler, Sampler, SpeculativeSampler,
};
use crate::runtime::generation::streaming::{
    drive_committed_generation_cancellable, CommittedGenerationError, CommittedTokenPipeline,
    CommittedTokenPipelineError, RawTokenDecoder,
};
pub(crate) use crate::runtime::media::input;
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
/// Thinking Machines Lab Inkling multimodal decoder support.
pub(crate) use crate::architectures::inkling::model as inkling;
/// Nemotron-H hybrid Mamba2/attention/MoE config support.
pub(crate) use crate::architectures::nemotron_h::model as nemotron_h;
/// Qwen3.5 dense and MoE text model support.
pub(crate) use crate::architectures::qwen::hybrid::qwen3_5;
pub use safemlx_lm_core::GgufArchitecture;
pub use safemlx_lm_core::ModelKind;

pub use safemlx_lm_core::ArtifactFormat;

use crate::backend::mlx::{validate_gemma4_drafter, Model, ModelCache};

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
pub use loaded::LoadedModelLoadError;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
