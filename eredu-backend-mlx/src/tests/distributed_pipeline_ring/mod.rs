#![cfg(unix)]

use std::{
    collections::BTreeMap,
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Child, Command, Output, Stdio},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

use crate::native::{ExecutionContext, MlxModelInput, MlxModelSession};
use crate::{
    backend::runtime::{
        execution::layerwise::open_safetensors_weight_store,
        media::{input::InputPayload, PreparedModelInput},
    },
    backend::{
        nn::shared::{
            neutral_parameter_refs, neutral_parameter_refs_mut, MlxModule, MlxNeuralBackend,
        },
        DeviceAssignment, MlxBackend,
    },
    tests::support::checkpoint_fixtures,
};
use crate::{MlxLoadRequest, MlxTensor};
use eredu_architectures::gpt_oss;
use eredu_architectures::qwen::hybrid as qwen_hybrid;
use eredu_core::cache::{PromptCacheDescriptor, PromptCacheOptions};
use eredu_core::{
    load_model, residency::OffloadConfig, BackendSession as _, DevicePlan, DraftPlacementPlan,
    DraftingPlan, ExecutionPlan, ExternalDraftArtifact, FinishReason, GenerationCancellationToken,
    InputExtent, InputMetadataKey, InputModality, ModelRuntime, ObservationRequest, SemanticEvent,
    SpeculativeCapability, SpeculativeConfig, SpeculativeDraft, SpeculativeExecutionTopology,
    SpeculativeGenerationBackend, SpeculativeGenerationBatchRequest, SpeculativeGenerationLane,
    SpeculativeOutputError, SpeculativeSemanticState, SpeculativeTokenFilterController,
    TextGenerationConfig, TokenFilter, TokenFilterController, TokenOutput as _,
    TokenizerCompatibilityProof,
};
use eredu_gguf::{
    GgmlType, MetadataArray, MetadataValue as GgufMetadataValue, TensorInput, Writer,
};
use eredu_nn::{ParameterMetadata, ParameterVisitor, ParameterVisitorMut, Parameterized};
use eredu_runtime::{
    CacheResidencyPolicy, DefaultSampler, DenseDiskStreamLoadOptions, LayerwiseLoadOptions,
    OrdinaryWeightResidency, PagedCacheOptions, ParameterBankLoadOptions, WeightResidency,
};
use safemlx::{
    distributed::{self, Backend},
    ops::{indexing::TryIndexOp, stack_axis},
    Array, Device, DeviceType, Dtype as MlxDtype, Stream,
};

fn ring_completion_policy() -> eredu_runtime::CommunicationCompletionPolicy {
    eredu_runtime::CommunicationCompletionPolicy::new(
        Duration::from_secs(30),
        eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
    )
    .unwrap()
}
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

const WORKER_RANK: &str = "EREDU_PIPELINE_RING_WORKER";
const CHECKPOINT_DIR: &str = "EREDU_PIPELINE_CHECKPOINT";
const FIXTURE_FAMILY: &str = "EREDU_PIPELINE_FIXTURE_FAMILY";
const DENSE_STREAM: &str = "EREDU_PIPELINE_DENSE_STREAM";
const LAYERWISE_HOST: &str = "EREDU_PIPELINE_LAYERWISE_HOST";
const PROMPT_CACHE_ROOT: &str = "EREDU_PIPELINE_PROMPT_CACHE";
const CARTESIAN_AXES: &str = "EREDU_PIPELINE_CARTESIAN_AXES";
const EXPERT_CACHE: &str = "EREDU_PIPELINE_EXPERT_CACHE";
const EXPERT_CACHE_EVICTION: &str = "EREDU_PIPELINE_EXPERT_CACHE_EVICTION";
const REQUANTIZE: &str = "EREDU_PIPELINE_REQUANTIZE";
const FINAL_OUTPUT_INTERVENTION: &str = "EREDU_PIPELINE_FINAL_OUTPUT_INTERVENTION";
const OPAQUE_SESSION: &str = "EREDU_PIPELINE_OPAQUE_SESSION";
const PREDICTION_FREE_TARGET: &str = "EREDU_PIPELINE_PREDICTION_FREE_TARGET";
const PREPARED_SPECULATIVE_CAPABILITY: &str = "EREDU_PIPELINE_PREPARED_SPECULATIVE_CAPABILITY";
const EXPECTED_UNSUPPORTED_DIRECT_PARTITION: &str =
    "EREDU_PIPELINE_EXPECTED_UNSUPPORTED_DIRECT_PARTITION";
const OPAQUE_INSPECTION: &str = "EREDU_PIPELINE_OPAQUE_INSPECTION";
const OPAQUE_TEXT_GENERATION: &str = "EREDU_PIPELINE_OPAQUE_TEXT_GENERATION";
const OPAQUE_MUSE_IMAGE: &str = "EREDU_PIPELINE_OPAQUE_MUSE_IMAGE";
const OPAQUE_INKLING_MEDIA: &str = "EREDU_PIPELINE_OPAQUE_INKLING_MEDIA";
const OPAQUE_QWEN_CONDITIONAL_MEDIA: &str = "EREDU_PIPELINE_OPAQUE_QWEN_CONDITIONAL_MEDIA";
const OPAQUE_INKLING_MTP: &str = "EREDU_PIPELINE_OPAQUE_INKLING_MTP";
const OPAQUE_QWEN_HYBRID_MTP: &str = "EREDU_PIPELINE_OPAQUE_QWEN_HYBRID_MTP";
const OPAQUE_NEMOTRON_H_MTP: &str = "EREDU_PIPELINE_OPAQUE_NEMOTRON_H_MTP";
const OPAQUE_DEEPSEEK_MTP_TARGET: &str = "EREDU_PIPELINE_OPAQUE_DEEPSEEK_MTP_TARGET";
const OPAQUE_DEEPSEEK_DSPARK_TARGET: &str = "EREDU_PIPELINE_OPAQUE_DEEPSEEK_DSPARK_TARGET";
const OPAQUE_GEMMA4_MEDIA: &str = "EREDU_PIPELINE_OPAQUE_GEMMA4_MEDIA";
const OPAQUE_QWEN3_VL_MEDIA: &str = "EREDU_PIPELINE_OPAQUE_QWEN3_VL_MEDIA";
const QWEN_HYBRID_PROMPT_CACHE: &str = "EREDU_PIPELINE_QWEN_HYBRID_PROMPT_CACHE";
const PROMPT_CACHE_PREPARE_FAILURE: &str = "EREDU_PIPELINE_PROMPT_CACHE_PREPARE_FAILURE";

fn input_part(
    modality: InputModality,
    payload: InputPayload,
    metadata: impl IntoIterator<Item = (InputMetadataKey, Array)>,
    extents: impl IntoIterator<Item = InputExtent>,
) -> crate::backend::runtime::media::input::InputPart {
    crate::backend::runtime::media::input::input_part(modality, payload, metadata, extents).unwrap()
}

fn text_input_part(tokens: &Array) -> crate::backend::runtime::media::input::InputPart {
    input_part(
        InputModality::Text,
        InputPayload::TokenIds(tokens.clone()),
        [],
        [],
    )
}

#[derive(Debug, Clone, Copy, Default)]
struct AllowAllTokens;

impl TokenFilterController for AllowAllTokens {
    type Error = std::convert::Infallible;

    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        Ok(TokenFilter::All)
    }

    fn commit_token(&mut self, _token_id: u32) -> Result<(), Self::Error> {
        Ok(())
    }

    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        Ok(false)
    }
}

impl SpeculativeTokenFilterController for AllowAllTokens {
    fn filter_at(&self, _history: &[u32]) -> Result<TokenFilter, Self::Error> {
        Ok(TokenFilter::All)
    }

    fn prefix_is_complete(&self, _history: &[u32]) -> Result<bool, Self::Error> {
        Ok(false)
    }
}

#[derive(Clone, Default)]
struct TokenOnlySemanticState {
    events: Vec<SemanticEvent>,
}

impl SpeculativeSemanticState for TokenOnlySemanticState {
    fn fork_box(&self) -> Result<Box<dyn SpeculativeSemanticState>, SpeculativeOutputError> {
        let mut fork = self.clone();
        fork.events.clear();
        Ok(Box::new(fork))
    }

    fn push_token(&mut self, token: u32) -> Result<bool, SpeculativeOutputError> {
        self.events
            .push(SemanticEvent::TextDelta(token.to_string()));
        Ok(false)
    }

    fn finish(&mut self, reason: FinishReason) -> Result<(), SpeculativeOutputError> {
        self.events.push(SemanticEvent::Finished { reason });
        Ok(())
    }

    fn cancel(&mut self) -> Result<(), SpeculativeOutputError> {
        self.events.push(SemanticEvent::Finished {
            reason: FinishReason::Cancelled,
        });
        Ok(())
    }

    fn take_events(&mut self) -> Vec<SemanticEvent> {
        std::mem::take(&mut self.events)
    }
}

fn run_neutral_embedded_mtp<'world>(
    runtime: &mut ModelRuntime<MlxBackend<'world>>,
    prompt: crate::composition::mlx::MlxModelInput,
    config: SpeculativeConfig,
) -> Result<eredu_core::SpeculativeGenerationOutput, crate::backend::error::Error> {
    execute_neutral_embedded_mtp(runtime, prompt, config).0
}

fn execute_neutral_embedded_mtp<'world>(
    runtime: &mut ModelRuntime<MlxBackend<'world>>,
    prompt: crate::composition::mlx::MlxModelInput,
    config: SpeculativeConfig,
) -> (
    Result<eredu_core::SpeculativeGenerationOutput, crate::backend::error::Error>,
    usize,
) {
    let sampling = eredu_core::resolve_generation_config(
        None,
        eredu_core::GenerationConfigOverrides {
            max_new_tokens: Some(config.max_tokens),
            temperature: Some(config.temperature),
            ..Default::default()
        },
    )
    .unwrap();
    let publications = Arc::new(AtomicUsize::new(0));
    let published = Arc::clone(&publications);
    let output = <MlxBackend<'world> as SpeculativeGenerationBackend>::with_speculative_execution(
        runtime,
        SpeculativeGenerationBatchRequest::new(
            SpeculativeDraft::Embedded,
            vec![SpeculativeGenerationLane::new(
                prompt,
                TextGenerationConfig::new(sampling),
                config,
                AllowAllTokens,
                Box::<TokenOnlySemanticState>::default(),
                GenerationCancellationToken::new(),
                Box::new(move |_| {
                    published.fetch_add(1, Ordering::Relaxed);
                }),
            )],
            [0; 32],
        ),
        eredu_runtime::RunSpeculativeGeneration::default(),
    )
    .map(|output| output.into_requests().into_iter().next().unwrap());
    (output, publications.load(Ordering::Relaxed))
}

fn synthetic_prediction_input(
    parts: &[crate::backend::runtime::media::input::InputPart],
    token_ids: &[u32],
) -> MlxModelInput {
    MlxModelInput::from(crate::backend::runtime::media::input::ModelInput::new(
        parts,
    ))
    .with_semantic_content_fingerprint(eredu_core::cache::prompt_cache_token_fingerprint(token_ids))
    .expect("synthetic prediction input must have an exact cache identity")
}

fn has_selected_embedded_prediction(session: &mut MlxModelSession) -> bool {
    session
        .neutral_prediction_target_mut()
        .expect("prediction fixture must use a neutral target")
        .has_embedded_prediction()
}

include!("unit_and_worker.rs");
include!("fixtures.rs");
include!("suites.rs");
include!("process.rs");
