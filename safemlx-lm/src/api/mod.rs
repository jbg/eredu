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

use crate::chat::{prepare_format_profile, resolve_structural_tokens, ToolRuntimePlan};
pub use crate::chat::{
    ChatTemplateIdentity, ChatTemplateRequest, NativeToolSupport, ParallelToolCallPolicy,
    PreparedChat, ToolChoice,
};
use crate::nn::generation::CausalLm;
#[cfg(feature = "media-processing")]
use crate::processor::{load_processor, ModelProcessor, PreparedModelInput, ProcessorInput};
use crate::runtime::checkpoint::gguf::{self as gguf_tokenizer, GgufTokenizer};
use crate::runtime::checkpoint::quantization::WeightQuantization;
use crate::runtime::distributed::topology::ParallelTopology;
use crate::runtime::execution::inspection::ActivationObserver;
use crate::sampler::{ConstrainedSampler, DefaultSampler, Sampler, SpeculativeSampler};
use crate::streaming::{
    drive_committed_generation, CommittedTokenPipeline, CommittedTokenSource, FinishReason,
    RawTokenDecoder, SemanticEvent, TokenDecoderBackend,
};
use crate::tool_constraints::ConstraintCompiler;
use crate::{
    error::Error,
    mtp::{
        LoadedDrafter, MtpBatchOutput, MtpCache, MtpCapability, MtpCheckpointKind, MtpConfig,
        MtpExecutionStreams, MtpScheduler, MtpSchedulerOptions, MtpSchedulerStats,
        MtpSemanticState, MtpStats,
    },
    runtime::cache::residency::{
        open_prompt_cache, validate_prompt_cache_model_identity, CacheResidencyManager,
        CacheResidencyPolicy, CacheResidencyReport, PagedCacheOptions, PromptCacheDescriptor,
        PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
    },
    runtime::cache::{ConcatKeyValueCache, PagedKeyValueCache, SlidingKeyValueCache},
    runtime::execution::layerwise::{LayerExecutionLoadOptions, WeightResidency},
};

/// Shared building blocks used by multiple decoder-only model families.
pub mod common;
/// DeepSeek-V3 and DeepSeek-R1 decoder support.
pub use crate::architectures::deepseek_v3::model as deepseek_v3;
pub(crate) use crate::architectures::gemma4::assistant as gemma4_assistant;
pub(crate) use crate::architectures::gemma4::audio as gemma4_audio;
/// Gemma 4 text model support.
pub use crate::architectures::gemma4::model as gemma4;
pub(crate) use crate::architectures::gemma4::multimodal as gemma4_multimodal;
pub(crate) use crate::architectures::gemma4::vision as gemma4_vision;
/// OpenAI GPT-OSS sparse decoder architecture.
pub use crate::architectures::gpt_oss::model as gpt_oss;
/// Thinking Machines Lab Inkling multimodal decoder support.
pub use crate::architectures::inkling::model as inkling;
/// Typed runtime input support.
pub mod input;
/// Liquid AI LFM2/LFM2.5 dense and MoE text model support.
pub use crate::architectures::lfm2::model as lfm2;
/// Llama decoder-only model support.
pub use crate::architectures::llama::model as llama;
/// Moshi token language-model support.
///
/// This module operates on pre-tokenized Mimi streams. It intentionally does
/// not implement audio encoding, decoding, or realtime device I/O.
pub use crate::architectures::moshi::model as moshi;
/// PersonaPlex realtime speech-to-speech token model support.
///
/// This module operates on pre-tokenized Mimi streams and hybrid prompt tokens.
/// It intentionally does not implement audio encoding, decoding, or realtime
/// device I/O.
pub use crate::architectures::moshi::personaplex;
/// Nemotron-H hybrid Mamba2/attention/MoE config support.
pub use crate::architectures::nemotron_h::model as nemotron_h;
/// Qwen3.5 MoE text model support.
pub use crate::architectures::qwen::hybrid::qwen3_5 as qwen3_5_moe;
/// Qwen3-Next hybrid attention/MoE text model support.
pub use crate::architectures::qwen::hybrid::qwen3_next;
/// Qwen3 decoder-only model support.
pub use crate::architectures::qwen::qwen3::model as qwen3;
/// Qwen3-VL multimodal conditional-generation support.
pub use crate::architectures::qwen::vl::model as qwen3_vl;
/// Qwen3-VL-MoE multimodal conditional-generation support.
pub use crate::architectures::qwen::vl::moe as qwen3_vl_moe;
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
pub use config::{
    check_model_config, check_model_config_json, check_model_dir, ModelConfigSupport, ModelKind,
    ModelLoadOptions, SupportedModelConfig,
};

mod dispatch;
use dispatch::validate_gemma4_drafter;
pub use dispatch::{Model, ModelCache, ModelGenerate};

mod request;
use request::{
    prepare_chat_from_parts, with_prepared_chat_runtime, ModelGenerateTokenSource,
    PreparedChatSemanticState, PreparedChatTokenDecoder,
};
pub use request::{
    PreparedChatEmbeddedMtpBatchRequest, PreparedChatEmbeddedMtpGenerationRequest,
    PreparedChatGenerationOutput, PreparedChatGenerationRequest, PreparedChatGenerationSettings,
    PreparedChatMtpBatchLane, PreparedChatMtpBatchOutput, PreparedChatMtpBatchRequest,
    PreparedChatMtpGenerationOptions, PreparedChatMtpGenerationOutput,
    PreparedChatMtpGenerationRequest, TextDecoder,
};

mod loaded;
pub(crate) use loaded::validate_gguf_quantization_source;
pub use loaded::LoadedModel;
use loaded::{final_token_logits, load_gguf_model_data};

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
            "PersonaPlex is a realtime speech-to-speech token model; use models::personaplex::load_model".into(),
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
    ensure_executable_load_options(options)?;
    if let WeightResidency::SparseExpertCacheWithDenseLayers(combined) = options.weight_residency {
        if options.quantization.is_some() {
            return Err(Error::Quantization(format!(
                "load-time quantization is unsupported for {} sparse expert caching with dense disk streaming; use a matching checkpoint-native packed format",
                kind.model_type_name()
            )));
        }
        let expert_cache = combined.expert_cache;
        let non_expert = combined.non_expert;
        return match kind {
            ModelKind::DeepSeekV3 => Ok(Model::DeepSeekV3Layerwise(
                crate::deepseek_v3::load_deepseek_v3_sparse_expert_cache_model_with_dense_layers(
                    model_dir, expert_cache, non_expert, stream, weights_stream,
                )?,
            )),
            ModelKind::GptOss => Ok(Model::GptOssLayerwise(
                crate::gpt_oss::load_gpt_oss_sparse_expert_cache_model_with_dense_layers(
                    model_dir, expert_cache, non_expert, stream, weights_stream,
                )?,
            )),
            ModelKind::Inkling => Ok(Model::InklingLayerwise(
                crate::inkling::load_inkling_sparse_expert_cache_model_with_dense_layers(
                    model_dir, expert_cache, non_expert, stream, weights_stream,
                )?,
            )),
            ModelKind::Lfm2 => Ok(Model::Lfm2Layerwise(
                crate::lfm2::load_lfm2_sparse_expert_cache_model_with_dense_layers(
                    model_dir, expert_cache, non_expert, stream, weights_stream,
                )?,
            )),
            ModelKind::NemotronH => Ok(Model::NemotronHLayerwise(
                crate::nemotron_h::load_nemotron_h_sparse_expert_cache_model_with_dense_layers(
                    model_dir, expert_cache, non_expert, stream, weights_stream,
                )?,
            )),
            ModelKind::Qwen3 => Ok(Model::Qwen3Layerwise(
                crate::qwen3::load_qwen3_sparse_expert_cache_model_with_dense_layers(
                    model_dir, expert_cache, non_expert, stream, weights_stream,
                )?,
            )),
            ModelKind::Qwen3Next => Ok(Model::Qwen3NextLayerwise(
                crate::qwen_hybrid::load_qwen3_next_sparse_expert_cache_model_with_dense_layers(
                    model_dir, expert_cache, non_expert, stream, weights_stream,
                )?,
            )),
            ModelKind::Qwen3VlMoe => Ok(Model::Qwen3VlMoeLayerwise(
                crate::qwen3_vl::load_qwen3_vl_sparse_expert_cache_model_with_dense_layers(
                    model_dir, expert_cache, non_expert, stream, weights_stream,
                )?,
            )),
            ModelKind::Qwen35Moe => Ok(Model::Qwen35MoeLayerwise(
                crate::qwen_hybrid::load_qwen35_sparse_expert_cache_model_with_dense_layers(
                    model_dir, expert_cache, non_expert, stream, weights_stream,
                )?,
            )),
            _ => Err(Error::UnsupportedArchitecture(format!(
                "sparse expert caching with dense disk streaming requires a supported safetensors MoE architecture, not {}",
                kind.model_type_name()
            ))),
        };
    }
    if let WeightResidency::SparseExpertCache(expert_cache) = options.weight_residency {
        if options.quantization.is_some() {
            return Err(Error::Quantization(format!(
                "load-time quantization is unsupported for {} sparse expert caching; use a matching checkpoint-native packed format",
                kind.model_type_name()
            )));
        }
        return match kind {
            ModelKind::DeepSeekV3 => Ok(Model::DeepSeekV3Layerwise(
                crate::deepseek_v3::load_deepseek_v3_sparse_expert_cache_model(
                    model_dir,
                    expert_cache,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::GptOss => Ok(Model::GptOssLayerwise(
                crate::gpt_oss::load_gpt_oss_sparse_expert_cache_model(
                    model_dir,
                    expert_cache,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Inkling => Ok(Model::InklingLayerwise(
                crate::inkling::load_inkling_sparse_expert_cache_model(
                    model_dir,
                    expert_cache,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Lfm2 => Ok(Model::Lfm2Layerwise(
                crate::lfm2::load_lfm2_sparse_expert_cache_model(
                    model_dir,
                    expert_cache,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::NemotronH => Ok(Model::NemotronHLayerwise(
                crate::nemotron_h::load_nemotron_h_sparse_expert_cache_model(
                    model_dir,
                    expert_cache,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Qwen3 => Ok(Model::Qwen3Layerwise(
                crate::qwen3::load_qwen3_sparse_expert_cache_model(
                    model_dir,
                    expert_cache,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Qwen3Next => Ok(Model::Qwen3NextLayerwise(
                crate::qwen_hybrid::load_qwen3_next_sparse_expert_cache_model(
                    model_dir,
                    expert_cache,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Qwen3VlMoe => Ok(Model::Qwen3VlMoeLayerwise(
                crate::qwen3_vl::load_qwen3_vl_sparse_expert_cache_model(
                    model_dir,
                    expert_cache,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Qwen35Moe => Ok(Model::Qwen35MoeLayerwise(
                crate::qwen_hybrid::load_qwen35_sparse_expert_cache_model(
                    model_dir,
                    expert_cache,
                    stream,
                    weights_stream,
                )?,
            )),
            _ => Err(Error::UnsupportedArchitecture(format!(
                "sparse expert caching requires a supported safetensors MoE architecture, not {}",
                kind.model_type_name()
            ))),
        };
    }
    let layerwise: Option<LayerExecutionLoadOptions> = match options.weight_residency {
        WeightResidency::LayerwiseHost(options) => Some(options.into()),
        WeightResidency::DenseDiskStream(options) => Some(options.into()),
        _ => None,
    };
    if let Some(layerwise) = layerwise {
        if options.quantization.is_some() {
            return Err(Error::Quantization(format!(
                "load-time quantization is unsupported for {} layer streaming; use a matching checkpoint-native packed format",
                kind.model_type_name()
            )));
        }
        return match kind {
            ModelKind::DeepSeekV3 => Ok(Model::DeepSeekV3Layerwise(
                crate::deepseek_v3::load_deepseek_v3_layerwise_model(
                    model_dir,
                    layerwise,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Gemma4 => Ok(Model::Gemma4Layerwise(
                crate::gemma4::load_gemma4_layerwise_model(
                    model_dir,
                    layerwise,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Inkling => Ok(Model::InklingLayerwise(
                crate::inkling::load_inkling_layerwise_model(
                    model_dir,
                    layerwise,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Llama => Ok(Model::LlamaLayerwise(crate::llama::load_llama_model(
                model_dir,
                crate::llama::LlamaLoadOptions {
                    weight_residency: match layerwise {
                        LayerExecutionLoadOptions::LayerwiseHost(options) => {
                            WeightResidency::LayerwiseHost(options)
                        }
                        LayerExecutionLoadOptions::DenseDiskStream(options) => {
                            WeightResidency::DenseDiskStream(options)
                        }
                    },
                },
                stream,
                weights_stream,
            )?)),
            ModelKind::Qwen3 => Ok(Model::Qwen3Layerwise(
                crate::qwen3::load_qwen3_layerwise_model(
                    model_dir,
                    layerwise,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::GptOss => Ok(Model::GptOssLayerwise(
                crate::gpt_oss::load_gpt_oss_layerwise_model(
                    model_dir,
                    layerwise,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Lfm2 => Ok(Model::Lfm2Layerwise(
                crate::lfm2::load_lfm2_layerwise_model(
                    model_dir,
                    layerwise,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::NemotronH => Ok(Model::NemotronHLayerwise(
                crate::nemotron_h::load_nemotron_h_layerwise_model(
                    model_dir,
                    layerwise,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Qwen3Next => Ok(Model::Qwen3NextLayerwise(
                crate::qwen_hybrid::load_qwen3_next_layerwise_model(
                    model_dir,
                    layerwise,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Qwen3Vl => Ok(Model::Qwen3VlLayerwise(
                crate::qwen3_vl::load_qwen3_vl_layerwise_model(
                    model_dir,
                    layerwise,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Qwen3VlMoe => Ok(Model::Qwen3VlMoeLayerwise(
                crate::qwen3_vl::load_qwen3_vl_layerwise_model(
                    model_dir,
                    layerwise,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Qwen35Moe => Ok(Model::Qwen35MoeLayerwise(
                crate::qwen_hybrid::load_qwen35_layerwise_model(
                    model_dir,
                    layerwise,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::PersonaPlex => Err(Error::UnsupportedArchitecture(
                "PersonaPlex bounded layer residency is selected through the realtime loader"
                    .into(),
            )),
        };
    }
    if let Some(quantization) = options.quantization {
        quantization.validate()?;
        return match kind {
            ModelKind::DeepSeekV3 => Ok(Model::DeepSeekV3(
                deepseek_v3::load_model_quantized(
                    model_dir,
                    quantization,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Gemma4 => Ok(Model::Gemma4(gemma4::load_gemma4_model_quantized(
                model_dir,
                quantization,
                stream,
                weights_stream,
            )?)),
            ModelKind::GptOss => Ok(Model::GptOss(gpt_oss::load_model_quantized(
                model_dir,
                quantization,
                stream,
                weights_stream,
            )?)),
            ModelKind::Inkling => Err(Error::Quantization(
                "Inkling affine/MXFP4 on-load quantization is unavailable because its routed experts use packed rank-3 grouped-matmul weights without a matching quantized grouped-matmul implementation".into(),
            )),
            ModelKind::Llama => Ok(Model::Llama(llama::load_resident_llama_model_quantized(
                model_dir,
                quantization,
                stream,
                weights_stream,
            )?)),
            ModelKind::Lfm2 => Ok(Model::Lfm2(lfm2::load_model_quantized(
                model_dir,
                quantization,
                stream,
                weights_stream,
            )?)),
            ModelKind::Qwen3 => Ok(Model::Qwen3(qwen3::load_qwen3_model_quantized(
                model_dir,
                quantization,
                stream,
                weights_stream,
            )?)),
            ModelKind::Qwen3Next => Ok(Model::Qwen3Next(
                qwen3_next::load_qwen3_next_model_quantized(
                    model_dir,
                    quantization,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Qwen3Vl => Ok(Model::Qwen3Vl(
                qwen3_vl::load_qwen3_vl_model_quantized(
                    model_dir,
                    quantization,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Qwen3VlMoe => Ok(Model::Qwen3VlMoe(
                qwen3_vl_moe::load_qwen3_vl_moe_model_quantized(
                    model_dir,
                    quantization,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::NemotronH => Err(Error::Quantization(
                "Nemotron-H affine on-load quantization is unavailable because its routed experts use packed rank-3 grouped-matmul weights without an affine grouped-matmul implementation".into(),
            )),
            ModelKind::Qwen35Moe => Ok(Model::Qwen35Moe(
                qwen3_5_moe::load_qwen3_5_moe_model_quantized(
                    model_dir,
                    quantization,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::PersonaPlex => Err(Error::UnsupportedArchitecture(
                "PersonaPlex must be loaded through the realtime API".into(),
            )),
        };
    }

    match kind {
        ModelKind::DeepSeekV3 => Ok(Model::DeepSeekV3(deepseek_v3::load_model(
            model_dir,
            stream,
            weights_stream,
        )?)),
        ModelKind::Gemma4 => Ok(Model::Gemma4(gemma4::load_gemma4_model(
            model_dir,
            stream,
            weights_stream,
        )?)),
        ModelKind::GptOss => Ok(Model::GptOss(gpt_oss::load_model(
            model_dir,
            stream,
            weights_stream,
        )?)),
        ModelKind::Inkling => Ok(Model::Inkling(inkling::load_model(
            model_dir,
            stream,
            weights_stream,
        )?)),
        ModelKind::Llama => Ok(Model::Llama(llama::load_resident_llama_model(
            model_dir,
            stream,
            weights_stream,
        )?)),
        ModelKind::Lfm2 => Ok(Model::Lfm2(lfm2::load_model(
            model_dir,
            stream,
            weights_stream,
        )?)),
        ModelKind::NemotronH => Ok(Model::NemotronH(nemotron_h::load_nemotron_h_model(
            model_dir,
            stream,
            weights_stream,
        )?)),
        ModelKind::Qwen3 => Ok(Model::Qwen3(qwen3::load_qwen3_model(
            model_dir,
            stream,
            weights_stream,
        )?)),
        ModelKind::Qwen3Next => Ok(Model::Qwen3Next(qwen3_next::load_qwen3_next_model(
            model_dir,
            stream,
            weights_stream,
        )?)),
        ModelKind::Qwen3Vl => Ok(Model::Qwen3Vl(qwen3_vl::load_qwen3_vl_model(
            model_dir,
            stream,
            weights_stream,
        )?)),
        ModelKind::Qwen3VlMoe => Ok(Model::Qwen3VlMoe(qwen3_vl_moe::load_qwen3_vl_moe_model(
            model_dir,
            stream,
            weights_stream,
        )?)),
        ModelKind::Qwen35Moe => Ok(Model::Qwen35Moe(qwen3_5_moe::load_qwen3_5_moe_model(
            model_dir,
            stream,
            weights_stream,
        )?)),
        ModelKind::PersonaPlex => Err(Error::UnsupportedArchitecture(
            "PersonaPlex must be loaded through the realtime API".into(),
        )),
    }
}

mod tokenizer;
pub use tokenizer::{chat_template_kwargs, load_tokenizer};
use tokenizer::{
    effective_model_type, gguf_sidecar_dir, is_gguf_file, load_chat_template,
    load_gguf_tokenizer_from_metadata, load_tokenizer_template_kwargs, read_model_metadata,
};

#[cfg(test)]
mod tests;
