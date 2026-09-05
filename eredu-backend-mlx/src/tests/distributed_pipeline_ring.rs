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
    composition::checkpoint_fixtures,
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

#[test]
fn pipeline_activation_dtype_comes_from_wire_contract_not_weights() {
    let checkpoint = tempfile::tempdir().unwrap();
    write_fixture(checkpoint.path());
    let topology = crate::test_parallel_rank(0, 1, 2, 1);
    let wire_contract =
        eredu_runtime::PipelineWireContract::new(eredu_runtime::PipelineActivationDtype::Bfloat16);
    let request = MlxLoadRequest::with_parallel(
        topology,
        DeviceAssignment::new(DeviceType::Cpu, 0),
        wire_contract,
        4,
        4096,
        ring_completion_policy(),
    );
    let inspection = eredu_architectures::configuration::inspect_artifact(checkpoint.path())
        .expect("fixture inspection");
    let policy = request.preparation_policy().unwrap();
    let selected =
        crate::composition::mlx::loading::select_preparation(&inspection, request, policy)
            .expect("public neutral selection");

    assert_eq!(
        selected.partitioned_activation_dtype(),
        Some(wire_contract.activation_dtype())
    );
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum FixtureFamily {
    Llama,
    Mistral,
    DeepSeek,
    DeepSeekV4,
    DeepSeekGguf,
    Gemma,
    MuseGlimmer,
    Qwen2,
    Qwen2Gguf,
    Qwen3,
    Qwen3Gguf,
    Qwen3Moe,
    Qwen3MoeTied,
    Qwen3MoeGguf,
    GptOss,
    GptOssGguf,
    Lfm2,
    Lfm2Moe,
    Lfm2MoeGguf,
    KimiLinear,
    KimiLinearGguf,
    NemotronH,
    NemotronHGguf,
    Qwen3Next,
    Qwen3NextMoe,
    Qwen35,
    Qwen35Moe,
    Qwen35Multimodal,
    Qwen35ZeroPrediction,
    Qwen35MoeMultimodal,
    Qwen3Vl,
    Qwen3VlMoe,
    Inkling,
    InklingDense,
    InklingDenseMultimodal,
    InklingMultimodal,
    InklingGguf,
}

impl FixtureFamily {
    const fn name(self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Mistral => "mistral",
            Self::DeepSeek => "deepseek",
            Self::DeepSeekV4 => "deepseek-v4",
            Self::DeepSeekGguf => "deepseek-gguf",
            Self::Gemma => "gemma",
            Self::MuseGlimmer => "muse-glimmer",
            Self::Qwen2 => "qwen2",
            Self::Qwen2Gguf => "qwen2-gguf",
            Self::Qwen3 => "qwen3",
            Self::Qwen3Gguf => "qwen3-gguf",
            Self::Qwen3Moe => "qwen3-moe",
            Self::Qwen3MoeTied => "qwen3-moe-tied",
            Self::Qwen3MoeGguf => "qwen3-moe-gguf",
            Self::GptOss => "gpt-oss",
            Self::GptOssGguf => "gpt-oss-gguf",
            Self::Lfm2 => "lfm2",
            Self::Lfm2Moe => "lfm2-moe",
            Self::Lfm2MoeGguf => "lfm2-moe-gguf",
            Self::KimiLinear => "kimi-linear",
            Self::KimiLinearGguf => "kimi-linear-gguf",
            Self::NemotronH => "nemotron-h",
            Self::NemotronHGguf => "nemotron-h-gguf",
            Self::Qwen3Next => "qwen3-next",
            Self::Qwen3NextMoe => "qwen3-next-moe",
            Self::Qwen35 => "qwen3.5",
            Self::Qwen35Moe => "qwen3.5-moe",
            Self::Qwen35Multimodal => "qwen3.5-multimodal",
            Self::Qwen35ZeroPrediction => "qwen3.5-zero-prediction",
            Self::Qwen35MoeMultimodal => "qwen3.5-moe-multimodal",
            Self::Qwen3Vl => "qwen3-vl",
            Self::Qwen3VlMoe => "qwen3-vl-moe",
            Self::Inkling => "inkling",
            Self::InklingDense => "inkling-dense",
            Self::InklingDenseMultimodal => "inkling-dense-multimodal",
            Self::InklingMultimodal => "inkling-multimodal",
            Self::InklingGguf => "inkling-gguf",
        }
    }

    fn parse(value: &str) -> Self {
        for family in [
            Self::Llama,
            Self::Mistral,
            Self::DeepSeek,
            Self::DeepSeekV4,
            Self::DeepSeekGguf,
            Self::Gemma,
            Self::MuseGlimmer,
            Self::Qwen2,
            Self::Qwen2Gguf,
            Self::Qwen3,
            Self::Qwen3Gguf,
            Self::Qwen3Moe,
            Self::Qwen3MoeTied,
            Self::Qwen3MoeGguf,
            Self::GptOss,
            Self::GptOssGguf,
            Self::Lfm2,
            Self::Lfm2Moe,
            Self::Lfm2MoeGguf,
            Self::KimiLinear,
            Self::KimiLinearGguf,
            Self::NemotronH,
            Self::NemotronHGguf,
            Self::Qwen3Next,
            Self::Qwen3NextMoe,
            Self::Qwen35,
            Self::Qwen35Moe,
            Self::Qwen35Multimodal,
            Self::Qwen35ZeroPrediction,
            Self::Qwen35MoeMultimodal,
            Self::Qwen3Vl,
            Self::Qwen3VlMoe,
            Self::Inkling,
            Self::InklingDense,
            Self::InklingDenseMultimodal,
            Self::InklingMultimodal,
            Self::InklingGguf,
        ] {
            if family.name() == value {
                return family;
            }
        }
        panic!("unexpected pipeline fixture family {value:?}")
    }

    fn layer_count(self) -> usize {
        match self {
            Self::Llama
            | Self::Mistral
            | Self::DeepSeek
            | Self::DeepSeekV4
            | Self::DeepSeekGguf
            | Self::Qwen2
            | Self::Qwen2Gguf
            | Self::Qwen3
            | Self::Qwen3Gguf
            | Self::Qwen3Moe
            | Self::Qwen3MoeTied
            | Self::Qwen3MoeGguf
            | Self::GptOss
            | Self::GptOssGguf
            | Self::Lfm2
            | Self::Lfm2Moe
            | Self::Lfm2MoeGguf
            | Self::KimiLinear
            | Self::KimiLinearGguf
            | Self::Qwen3Next
            | Self::Qwen3NextMoe
            | Self::Qwen35
            | Self::Qwen35Moe
            | Self::Qwen35Multimodal => 2,
            Self::Qwen35ZeroPrediction
            | Self::Qwen35MoeMultimodal
            | Self::Qwen3Vl
            | Self::Qwen3VlMoe
            | Self::MuseGlimmer => 2,
            Self::Gemma | Self::NemotronH | Self::NemotronHGguf => 4,
            Self::Inkling
            | Self::InklingDense
            | Self::InklingDenseMultimodal
            | Self::InklingMultimodal
            | Self::InklingGguf => 3,
        }
    }

    fn stage_range(self, rank: usize) -> std::ops::Range<usize> {
        match (self, rank) {
            (Self::Gemma, 0) => 0..1,
            (Self::Gemma, 1) => 1..4,
            (Self::NemotronH | Self::NemotronHGguf, 0) => 0..2,
            (Self::NemotronH | Self::NemotronHGguf, 1) => 2..4,
            (
                Self::Inkling
                | Self::InklingDense
                | Self::InklingDenseMultimodal
                | Self::InklingMultimodal
                | Self::InklingGguf,
                0,
            ) => 0..2,
            (
                Self::Inkling
                | Self::InklingDense
                | Self::InklingDenseMultimodal
                | Self::InklingMultimodal
                | Self::InklingGguf,
                1,
            ) => 2..3,
            (_, rank) => rank..rank + 1,
        }
    }

    fn expert_layer_count(self, range: std::ops::Range<usize>) -> usize {
        match self {
            Self::DeepSeek
            | Self::DeepSeekGguf
            | Self::Lfm2Moe
            | Self::Lfm2MoeGguf
            | Self::KimiLinear
            | Self::KimiLinearGguf => range.filter(|index| *index == 1).count(),
            Self::DeepSeekV4 => range.len(),
            Self::Inkling | Self::InklingMultimodal | Self::InklingGguf => {
                range.filter(|index| matches!(*index, 1 | 2)).count()
            }
            Self::InklingDense | Self::InklingDenseMultimodal => 0,
            Self::NemotronH => range.filter(|index| *index == 2).count(),
            Self::NemotronHGguf => range.filter(|index| matches!(*index, 1 | 2)).count(),
            _ => range.len(),
        }
    }

    fn effective_model_type(self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Mistral => "mistral",
            Self::DeepSeek | Self::DeepSeekGguf => "deepseek_v3",
            Self::DeepSeekV4 => "deepseek_v4",
            Self::Gemma => "gemma4_text",
            Self::MuseGlimmer => "muse_glimmer_text",
            Self::Qwen2 | Self::Qwen2Gguf => "qwen2",
            Self::Qwen3 | Self::Qwen3Gguf => "qwen3",
            Self::Qwen3Moe | Self::Qwen3MoeTied | Self::Qwen3MoeGguf => "qwen3_moe",
            Self::GptOss | Self::GptOssGguf => "gpt_oss",
            Self::Lfm2 => "lfm2",
            Self::Lfm2Moe | Self::Lfm2MoeGguf => "lfm2_moe",
            Self::KimiLinear | Self::KimiLinearGguf => "kimi_linear",
            Self::NemotronH | Self::NemotronHGguf => "nemotron_h",
            Self::Qwen3Next | Self::Qwen3NextMoe => "qwen3_next",
            Self::Qwen35 | Self::Qwen35Multimodal | Self::Qwen35ZeroPrediction => "qwen3_5_text",
            Self::Qwen35Moe | Self::Qwen35MoeMultimodal => "qwen3_5_moe_text",
            Self::Qwen3Vl => "qwen3_vl_text",
            Self::Qwen3VlMoe => "qwen3_vl_moe_text",
            Self::Inkling
            | Self::InklingDense
            | Self::InklingDenseMultimodal
            | Self::InklingMultimodal
            | Self::InklingGguf => "inkling_mm_model",
        }
    }

    const fn needs_opaque_reference(self) -> bool {
        matches!(
            self,
            Self::Llama
                | Self::Mistral
                | Self::Gemma
                | Self::MuseGlimmer
                | Self::Qwen2
                | Self::Qwen2Gguf
                | Self::Qwen3
                | Self::Qwen3Gguf
                | Self::Qwen3Moe
                | Self::GptOss
                | Self::KimiLinear
                | Self::NemotronH
                | Self::Qwen3Next
                | Self::Qwen35
                | Self::Qwen35Multimodal
                | Self::Qwen35ZeroPrediction
                | Self::Qwen3Vl
                | Self::DeepSeek
                | Self::DeepSeekV4
                | Self::InklingDense
                | Self::InklingDenseMultimodal
        )
    }

    const fn is_multimodal(self) -> bool {
        matches!(
            self,
            Self::InklingMultimodal
                | Self::InklingDenseMultimodal
                | Self::Qwen35Multimodal
                | Self::Qwen35ZeroPrediction
                | Self::Qwen35MoeMultimodal
                | Self::Qwen3Vl
                | Self::Qwen3VlMoe
        )
    }

    const fn comparison_tolerance(self) -> f32 {
        match self {
            Self::DeepSeekGguf => 3e-3,
            Self::DeepSeekV4 | Self::KimiLinear | Self::KimiLinearGguf => 1e-3,
            Self::Qwen3Next
            | Self::Qwen3NextMoe
            | Self::Qwen35
            | Self::Qwen35Moe
            | Self::Qwen35Multimodal
            | Self::Qwen35ZeroPrediction
            | Self::Qwen35MoeMultimodal => 2e-3,
            Self::NemotronH | Self::NemotronHGguf => 1e-3,
            _ if self.is_multimodal() => 5e-4,
            _ => 1e-4,
        }
    }
}

#[test]
fn pipeline_ring_worker() {
    let Some(rank) = std::env::var_os(WORKER_RANK) else {
        return;
    };
    let expected_rank: usize = rank.to_string_lossy().parse().unwrap();
    let checkpoint = PathBuf::from(std::env::var_os(CHECKPOINT_DIR).unwrap());
    let family = FixtureFamily::parse(&std::env::var(FIXTURE_FAMILY).unwrap());
    let prompt_cache_root = PathBuf::from(std::env::var_os(PROMPT_CACHE_ROOT).unwrap());
    let native_group = distributed::init(true, Backend::Ring).unwrap();
    let cartesian_axes = std::env::var(CARTESIAN_AXES).ok();
    let (tensor_parallel_size, pipeline_parallel_size, expert_parallel_size) =
        match cartesian_axes.as_deref() {
            None => (1, 2, 1),
            Some("tp") => (2, 1, 1),
            Some("ep") => (1, 1, 2),
            Some("tp-pp") => (2, 2, 1),
            Some("tp-ep") => (2, 1, 2),
            Some("pp-ep") => (1, 2, 2),
            Some("tp-pp-ep") => (2, 2, 2),
            Some(other) => panic!("unexpected Cartesian pipeline axes {other:?}"),
        };
    let topology = eredu_core::ParallelRankTopology::new(
        eredu_core::ParallelTopology::new(
            tensor_parallel_size,
            pipeline_parallel_size,
            expert_parallel_size,
            1,
        )
        .unwrap(),
        expected_rank,
    )
    .unwrap();
    assert_eq!(topology.global_rank(), expected_rank);
    let pipeline_rank = topology.pipeline_parallel_rank();
    let neutral_gemma_config =
        (family == FixtureFamily::Gemma && std::env::var_os(OPAQUE_SESSION).is_some()).then(|| {
            let config: serde_json::Value = serde_json::from_slice(
                &std::fs::read(checkpoint.join("config.json")).expect("Gemma config"),
            )
            .expect("Gemma JSON config");
            config
        });
    let neutral_gemma_layers = neutral_gemma_config.as_ref().map(|config| {
        config["text_config"]["num_hidden_layers"]
            .as_u64()
            .expect("Gemma text layer count") as usize
    });
    let neutral_prediction_target_layers = ((family == FixtureFamily::Inkling
        && std::env::var_os(OPAQUE_INKLING_MTP).is_some())
        || (family == FixtureFamily::Qwen35Multimodal
            && std::env::var_os(OPAQUE_QWEN_HYBRID_MTP).is_some())
        || (family == FixtureFamily::NemotronH
            && std::env::var_os(OPAQUE_NEMOTRON_H_MTP).is_some()))
    .then(|| {
        let config: serde_json::Value = serde_json::from_slice(
            &std::fs::read(checkpoint.join("config.json")).expect("prediction target config"),
        )
        .expect("prediction target JSON config");
        if family == FixtureFamily::NemotronH {
            config["num_hidden_layers"]
                .as_u64()
                .expect("prediction target layer count") as usize
        } else {
            config["text_config"]["num_hidden_layers"]
                .as_u64()
                .expect("prediction target layer count") as usize
        }
    });
    let neutral_qwen_vl_config =
        (matches!(family, FixtureFamily::Qwen3Vl | FixtureFamily::Qwen3VlMoe)
            && std::env::var_os(OPAQUE_QWEN3_VL_MEDIA).is_some())
        .then(|| {
            serde_json::from_slice::<serde_json::Value>(
                &std::fs::read(checkpoint.join("config.json")).expect("Qwen3-VL config"),
            )
            .expect("Qwen3-VL JSON config")
        });
    let local_layer_range =
        if let Some(layers) = neutral_gemma_layers.or(neutral_prediction_target_layers) {
            layers * pipeline_rank / pipeline_parallel_size
                ..layers * (pipeline_rank + 1) / pipeline_parallel_size
        } else if pipeline_parallel_size == 1 {
            0..family.layer_count()
        } else {
            family.stage_range(pipeline_rank)
        };
    let public_output_owner = topology
        .topology()
        .rank_for(eredu_core::ParallelCoordinates::new(
            0,
            pipeline_parallel_size - 1,
            0,
            topology.data_parallel_rank(),
        ))
        .unwrap();
    let owns_public_output = expected_rank == public_output_owner;
    let device = DeviceAssignment::new(DeviceType::Cpu, 0);
    let stream = Stream::new_with_device(&device.device().unwrap());
    if std::env::var_os(OPAQUE_SESSION).is_some() {
        let dense_composite_neutral = matches!(
            family,
            FixtureFamily::MuseGlimmer
                | FixtureFamily::InklingDense
                | FixtureFamily::InklingDenseMultimodal
                | FixtureFamily::Qwen35ZeroPrediction
        ) && matches!(
            cartesian_axes.as_deref(),
            None | Some("tp") | Some("pp") | Some("tp-pp")
        );
        let dense_composite_neutral = dense_composite_neutral
            || (family == FixtureFamily::Qwen35Multimodal
                && std::env::var_os(OPAQUE_QWEN_HYBRID_MTP).is_some()
                && matches!(
                    cartesian_axes.as_deref(),
                    None | Some("tp") | Some("pp") | Some("tp-pp")
                ));
        let dense_composite_auxiliary_units = dense_composite_neutral
            .then(|| {
                serde_json::from_slice::<serde_json::Value>(
                    &std::fs::read(checkpoint.join("config.json")).expect("dense composite config"),
                )
                .expect("dense composite JSON config")
            })
            .map_or(0, |config| match family {
                FixtureFamily::MuseGlimmer => {
                    let depth = config["vision_config"]["num_hidden_layers"]
                        .as_u64()
                        .unwrap_or(0) as usize;
                    if pipeline_rank == 0 {
                        depth
                    } else {
                        0
                    }
                }
                FixtureFamily::Qwen35ZeroPrediction | FixtureFamily::Qwen35Multimodal => {
                    let depth = config["vision_config"]["depth"].as_u64().unwrap_or(0) as usize;
                    depth * (pipeline_rank + 1) / pipeline_parallel_size
                        - depth * pipeline_rank / pipeline_parallel_size
                }
                _ => 0,
            });
        let routed_neutral = (matches!(
            family,
            FixtureFamily::Qwen3Moe
                | FixtureFamily::Qwen3MoeGguf
                | FixtureFamily::GptOss
                | FixtureFamily::GptOssGguf
                | FixtureFamily::DeepSeek
                | FixtureFamily::DeepSeekGguf
                | FixtureFamily::NemotronH
                | FixtureFamily::NemotronHGguf
                | FixtureFamily::Lfm2MoeGguf
                | FixtureFamily::KimiLinearGguf
                | FixtureFamily::Qwen3VlMoe
                | FixtureFamily::Inkling
                | FixtureFamily::InklingMultimodal
        ) || (family == FixtureFamily::DeepSeekV4
            && (std::env::var_os(PREDICTION_FREE_TARGET).is_some()
                || std::env::var_os(PREPARED_SPECULATIVE_CAPABILITY).is_some())))
            && matches!(
                cartesian_axes.as_deref(),
                None | Some("tp")
                    | Some("ep")
                    | Some("tp-pp")
                    | Some("tp-ep")
                    | Some("pp-ep")
                    | Some("tp-pp-ep")
            );
        let prove_prepared_communication_lifecycle = dense_composite_neutral
            || routed_neutral
            || (family == FixtureFamily::Qwen3Vl
                && matches!(cartesian_axes.as_deref(), None | Some("tp") | Some("tp-pp")))
            || (matches!(
                family,
                FixtureFamily::Llama
                    | FixtureFamily::Mistral
                    | FixtureFamily::Qwen2
                    | FixtureFamily::Qwen2Gguf
                    | FixtureFamily::Qwen3
                    | FixtureFamily::Qwen3Gguf
                    | FixtureFamily::KimiLinear
                    | FixtureFamily::NemotronH
                    | FixtureFamily::Lfm2
                    | FixtureFamily::Gemma
            ) && matches!(
                cartesian_axes.as_deref(),
                None | Some("tp") | Some("pp") | Some("tp-pp")
            ));
        let prove_direct_expert_communication =
            cartesian_axes.as_deref() == Some("ep") && !routed_neutral;
        if prove_prepared_communication_lifecycle || prove_direct_expert_communication {
            crate::composition::mlx::path_instrumentation::reset();
        }
        let backend = crate::native::distributed_backend(&stream, &stream, &native_group);
        let selected_paged = PagedCacheOptions::new(1, 32768, 32768, 1)
            .unwrap()
            .with_full_attention(true);
        let dense_stream = std::env::var_os(DENSE_STREAM).is_some();
        let layerwise_host = std::env::var_os(LAYERWISE_HOST).is_some();
        assert!(!(dense_stream && layerwise_host));
        let load_options = if std::env::var_os(REQUANTIZE).is_some() {
            let request = if family == FixtureFamily::NemotronH {
                eredu_core::QuantizationRequest::MxFp4
            } else {
                eredu_core::QuantizationRequest::Affine {
                    group_size: 32,
                    bits: 4,
                }
            };
            MlxLoadRequest::with_quantization(request).with_parallel_topology(
                topology,
                device,
                eredu_runtime::PipelineWireContract::new(
                    eredu_runtime::PipelineActivationDtype::Float32,
                ),
                4,
                4096,
                ring_completion_policy(),
            )
        } else {
            MlxLoadRequest::with_parallel(
                topology,
                device,
                eredu_runtime::PipelineWireContract::new(
                    eredu_runtime::PipelineActivationDtype::Float32,
                ),
                4,
                4096,
                ring_completion_policy(),
            )
        };
        let load_options = if std::env::var_os(EXPERT_CACHE).is_some() {
            let ordinary = if dense_stream {
                OrdinaryWeightResidency::DenseDiskStream(
                    DenseDiskStreamLoadOptions::new(u64::MAX, u64::MAX, 1, 1).unwrap(),
                )
            } else if layerwise_host {
                OrdinaryWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
                    OffloadConfig::new(None, None, 1).unwrap(),
                ))
            } else {
                OrdinaryWeightResidency::FullyResident
            };
            let bank = if std::env::var_os(EXPERT_CACHE_EVICTION).is_some() {
                ParameterBankLoadOptions::new(
                    OffloadConfig::new(Some(12_288), Some(0), 1).unwrap(),
                    u64::MAX,
                    1 << 30,
                )
                .unwrap()
            } else {
                ParameterBankLoadOptions::default()
            };
            load_options.with_weight_residency(WeightResidency::with_independent_parameter_banks(
                ordinary, bank,
            ))
        } else if dense_stream {
            load_options.with_weight_residency(WeightResidency::dense_disk_stream(
                DenseDiskStreamLoadOptions::new(u64::MAX, u64::MAX, 1, 1).unwrap(),
            ))
        } else if layerwise_host {
            load_options.with_weight_residency(WeightResidency::layerwise_host(
                LayerwiseLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap()),
            ))
        } else {
            load_options
        }
        .with_state_residency(CacheResidencyPolicy::Paged(selected_paged));
        if std::env::var_os(OPAQUE_DEEPSEEK_MTP_TARGET).is_some()
            && family == FixtureFamily::DeepSeekV4
        {
            let inspection = eredu_architectures::configuration::inspect_artifact(&checkpoint)
                .expect("V4 prediction artifact inspection");
            let policy = load_options.preparation_policy().unwrap();
            let selected = crate::composition::mlx::loading::select_preparation(
                &inspection,
                load_options.clone(),
                policy,
            )
            .expect("V4 prediction target selection");
            assert_eq!(
                selected.prediction_extension_kind(),
                Some(
                    eredu_architectures::configuration::PredictionExtensionKind::DeepSeekV4Embedded
                )
            );
            assert!(selected.realized_communication_manifest().is_some());
            assert!(selected.rank_context().is_some());
        }
        let model = match load_model(&backend, &checkpoint, load_options) {
            Ok(_) if std::env::var_os(EXPECTED_UNSUPPORTED_DIRECT_PARTITION).is_some() => {
                panic!("unsupported direct partition route unexpectedly loaded")
            }
            Ok(model) => model,
            Err(error) if std::env::var_os(EXPECTED_UNSUPPORTED_DIRECT_PARTITION).is_some() => {
                assert!(
                    error
                        .to_string()
                        .contains("has no neutral production implementation"),
                    "unsupported direct partition failed for an unexpected reason: {error}"
                );
                return;
            }
            Err(error) => panic!("failed to load Ring fixture: {error}"),
        };
        if prove_prepared_communication_lifecycle {
            assert_eq!(
                crate::composition::mlx::path_instrumentation::snapshot().payload_opens,
                1,
                "included dense decoder must open its admitted payload store exactly once"
            );
            assert_eq!(
                crate::composition::mlx::path_instrumentation::communication_realization_attempts(),
                1,
                "included dense decoder must realize its prepared communication exactly once before payload construction"
            );
            assert_eq!(
                crate::composition::mlx::path_instrumentation::manifest_communication_realization_attempts(),
                1,
                "eligible dense-decoder TP/PP must realize its neutral manifest exactly once"
            );
            if matches!(
                cartesian_axes.as_deref(),
                None | Some("tp")
                    | Some("pp")
                    | Some("ep")
                    | Some("tp-pp")
                    | Some("tp-ep")
                    | Some("pp-ep")
                    | Some("tp-pp-ep")
            ) {
                assert_eq!(
                    crate::composition::mlx::path_instrumentation::neutral_partitioned_constructions(),
                    1,
                    "eligible dense-decoder TP/PP must construct the neutral partitioned session"
                );
                assert_eq!(
                    crate::composition::mlx::path_instrumentation::snapshot().unit_constructions,
                    local_layer_range.len()
                        + neutral_gemma_config.as_ref().map_or(0, |config| {
                            if pipeline_rank == 0 {
                                ["vision_config", "audio_config"]
                                    .iter()
                                    .map(|root| {
                                        config[*root]["num_hidden_layers"].as_u64().unwrap_or(0)
                                            as usize
                                    })
                                    .sum::<usize>()
                            } else {
                                0
                            }
                        })
                        + neutral_qwen_vl_config.as_ref().map_or(0, |config| {
                            let depth =
                                config["vision_config"]["depth"].as_u64().unwrap_or(0) as usize;
                            depth * (pipeline_rank + 1) / pipeline_parallel_size
                                - depth * pipeline_rank / pipeline_parallel_size
                        })
                        + dense_composite_auxiliary_units,
                    "neutral partition construction must bind every local unit exactly once"
                );
                assert_eq!(
                    crate::composition::mlx::path_instrumentation::snapshot().materializations,
                    usize::from(std::env::var_os(REQUANTIZE).is_some()),
                    "neutral construction must execute exactly the selected transform groups"
                );
            }
            if neutral_gemma_layers.is_some() && pipeline_parallel_size > 1 {
                let counts = crate::composition::mlx::path_instrumentation::snapshot();
                let has_media = neutral_gemma_config.as_ref().is_some_and(|config| {
                    config.get("vision_config").is_some() || config.get("audio_config").is_some()
                });
                assert_eq!(
                    counts.local_static_bindings,
                    if has_media && pipeline_rank == 0 {
                        12
                    } else if pipeline_rank == 0 {
                        1
                    } else {
                        2
                    },
                    "only exact first-owner ingress or last-owner output statics may be bound"
                );
                assert_eq!(
                    counts.excluded_local_static_parameters,
                    if has_media && pipeline_rank != 0 {
                        12
                    } else if pipeline_rank == 0 {
                        2
                    } else {
                        1
                    },
                    "every unowned static definition must remain unbound and unread"
                );
            }
            if neutral_qwen_vl_config.is_some() {
                let counts = crate::composition::mlx::path_instrumentation::snapshot();
                assert!(
                    counts.local_static_bindings > 0,
                    "Qwen3-VL must bind its selected stage-local static tasks"
                );
                if pipeline_parallel_size == 1 {
                    assert_eq!(
                        counts.excluded_local_static_parameters, 0,
                        "Qwen3-VL pure TP owns every selected static task"
                    );
                } else {
                    assert!(
                        counts.excluded_local_static_parameters > 0,
                        "Qwen3-VL PP must leave non-owned static parameters unbound"
                    );
                }
            }
        }
        if prove_direct_expert_communication {
            assert_eq!(
                crate::composition::mlx::path_instrumentation::communication_realization_attempts(),
                1
            );
            assert_eq!(
                crate::composition::mlx::path_instrumentation::manifest_communication_realization_attempts(),
                0,
                "direct expert communication must not realize a partition manifest"
            );
        }
        let expected_effective_model_type = family.effective_model_type();
        assert_eq!(model.effective_model_type(), expected_effective_model_type);
        if dense_stream {
            let report = model.dense_stream_report().unwrap().unwrap();
            assert_eq!(report.planned_layer_count(), local_layer_range.len());
            let streamed = report
                .residency()
                .units()
                .iter()
                .filter(|unit| unit.planned_tier() == eredu_core::residency::MemoryTier::Disk)
                .collect::<Vec<_>>();
            assert_eq!(streamed.len(), local_layer_range.len());
            assert!(streamed
                .iter()
                .all(|unit| !unit.host_resident() && !unit.device_resident()));
        }
        if layerwise_host {
            assert!(model.dense_stream_report().unwrap().is_none());
            let report = model.residency_report().unwrap().unwrap();
            let layerwise = report
                .units()
                .iter()
                .filter(|unit| unit.planned_tier() == eredu_core::residency::MemoryTier::Host)
                .collect::<Vec<_>>();
            assert_eq!(layerwise.len(), local_layer_range.len());
            assert!(layerwise
                .iter()
                .all(|unit| unit.host_resident() && !unit.device_resident()));
        }
        let expected_speculative_capability = model.speculative_capability_for_test();
        if std::env::var_os(OPAQUE_DEEPSEEK_MTP_TARGET).is_some() {
            assert_eq!(
                expected_speculative_capability,
                SpeculativeCapability::Ready {
                    draft_source: eredu_core::SpeculativeDraftSource::Embedded,
                },
                "the complete prediction artifact must retain its embedded-draft capability on the neutral target"
            );
        }
        let mut runtime = eredu_core::ModelRuntime::from_prepared(backend, model).unwrap();
        let distributed_view =
            <MlxBackend<'_> as eredu_core::DistributedBackend>::distributed_session(
                runtime.session(),
            )
            .expect("partitioned MLX session must retain its selected communication view");
        let distributed_descriptor = eredu_core::DistributedSession::descriptor(distributed_view);
        assert_eq!(
            distributed_descriptor.world_size(),
            topology.topology().world_size()
        );
        assert_eq!(distributed_descriptor.rank(), topology.global_rank());
        if prove_prepared_communication_lifecycle {
            assert_eq!(
                crate::composition::mlx::path_instrumentation::communication_realization_attempts(),
                1,
                "session creation must consume the prepared communicator without recreating it"
            );
        }
        if std::env::var_os(OPAQUE_TEXT_GENERATION).is_some() {
            let sampling = eredu_core::resolve_generation_config(
                None,
                eredu_core::GenerationConfigOverrides {
                    max_new_tokens: Some(3),
                    ..Default::default()
                },
            )
            .unwrap();
            let generated = eredu_core::TextGeneration::new(
                &mut runtime,
                vec![1, 2],
                eredu_core::TextGenerationConfig::new(sampling),
            )
            .unwrap()
            .map(|token| token.unwrap().token_id().unwrap())
            .collect::<Vec<_>>();
            assert_eq!(generated.len(), 3);
            return;
        }
        assert_eq!(
            runtime.session().effective_model_type(),
            expected_effective_model_type
        );
        assert_eq!(
            <MlxBackend<'_> as eredu_core::SpeculativeGenerationBackend>::speculative_capability(
                &runtime
            ),
            expected_speculative_capability
        );
        use crate::backend::runtime::media::input::{InputPayload, ModelInput};
        let capability_tokens = Array::from_slice(&[1u32, 2], &[1, 2]);
        let capability_parts = [text_input_part(&capability_tokens)];
        let capability_input = ModelInput::new(&capability_parts).into();
        let capabilities =
            <MlxBackend<'_> as eredu_core::ModelCapabilityBackend>::model_capabilities(&runtime)
                .unwrap();
        assert_eq!(
            capabilities.effective_model_type,
            expected_effective_model_type
        );
        let counted = <MlxBackend<'_> as eredu_core::ModelCapabilityBackend>::count_prepared_input(
            &runtime,
            &capability_input,
        )
        .unwrap();
        assert_eq!(counted.text_tokens, 2);
        assert_eq!(counted.model_positions, 2);
        let state = <MlxBackend<'_> as eredu_core::ModelCapabilityBackend>::estimate_runtime_state(
            &runtime, counted, 2, 1,
        )
        .unwrap();
        assert_eq!(state.assumptions.requested_positions, 4);
        assert!(matches!(
            eredu_core::apply_admission_policy(
                &capabilities,
                eredu_core::AdmissionRequest {
                    input: counted,
                    max_output_tokens: 2,
                    batch_size: 1,
                    safety_reserve_bytes: 0,
                    application_memory_budget_bytes: None,
                    require_complete_estimate: false,
                },
                state,
                None,
            )
            .unwrap(),
            eredu_core::AdmissionResult::Admitted(_)
        ));
        <MlxBackend<'_> as eredu_core::ModelCapabilityBackend>::static_memory(&runtime).unwrap();
        if std::env::var_os(PREPARED_SPECULATIVE_CAPABILITY).is_some() {
            return;
        }
        let image_mode = std::env::var_os(OPAQUE_MUSE_IMAGE).is_some();
        let inkling_media_mode = std::env::var_os(OPAQUE_INKLING_MEDIA).is_some();
        let inkling_mtp_mode = std::env::var_os(OPAQUE_INKLING_MTP).is_some();
        let qwen_hybrid_mtp_mode = std::env::var_os(OPAQUE_QWEN_HYBRID_MTP).is_some();
        let qwen_hybrid_prompt_cache_mode = std::env::var_os(QWEN_HYBRID_PROMPT_CACHE).is_some();
        let nemotron_h_mtp_mode = std::env::var_os(OPAQUE_NEMOTRON_H_MTP).is_some();
        let deepseek_mtp_target_mode = std::env::var_os(OPAQUE_DEEPSEEK_MTP_TARGET).is_some();
        let deepseek_dspark_target_mode = std::env::var_os(OPAQUE_DEEPSEEK_DSPARK_TARGET).is_some();
        let gemma4_media_mode = std::env::var_os(OPAQUE_GEMMA4_MEDIA).is_some();
        let qwen3_vl_media_mode = std::env::var_os(OPAQUE_QWEN3_VL_MEDIA).is_some();
        let qwen_conditional_media_mode = std::env::var_os(OPAQUE_QWEN_CONDITIONAL_MEDIA).is_some();
        let prompt = Array::from_slice(&[1u32, 2], &[1, 2]);
        let text_before = Array::from_slice(&[1u32], &[1, 1]);
        let text_after = Array::from_slice(&[2u32], &[1, 1]);
        let image_grid = Array::from_slice(&[1i32, 2, 2], &[1, 3]);
        let image_pixels = Array::from_slice(&[0.01f32; 48], &[4, 12]);
        let inkling_image = Array::from_slice(&[0.01f32; 16], &[1, 1, 16]);
        let inkling_audio = Array::from_slice(&[0u32, 1, 2, 3, 4, 5], &[1, 3, 2]);
        let gemma4_patches = Array::from_slice(&[0.01f32; 192], &[1, 4, 48]);
        let gemma4_positions = Array::from_slice(&[0i32, 0, 0, 1, 1, 0, 1, 1], &[1, 4, 2]);
        let gemma4_grid = Array::from_slice(&[1i32, 2, 2], &[1, 3]);
        let gemma4_audio = Array::from_slice(&[0.01f32; 512], &[1, 4, 128]);
        let gemma4_audio_mask = Array::from_slice(&[true, true, true, true], &[1, 4]);
        let qwen3_vl_grid = Array::from_slice(&[1i32, 2, 4], &[1, 3]);
        let qwen3_vl_pixels = Array::from_slice(&[0.01f32; 96], &[8, 12]);
        let parts = if image_mode {
            vec![
                text_input_part(&text_before),
                input_part(
                    InputModality::Image,
                    InputPayload::Tensor(image_pixels.clone()),
                    [(InputMetadataKey::PatchGrid, image_grid.clone())],
                    [],
                ),
                text_input_part(&text_after),
            ]
        } else if inkling_media_mode {
            vec![
                text_input_part(&prompt),
                input_part(
                    InputModality::Image,
                    InputPayload::Embeddings(inkling_image.clone()),
                    [],
                    [],
                ),
                input_part(
                    InputModality::Audio,
                    InputPayload::Tensor(inkling_audio.clone()),
                    [],
                    [],
                ),
            ]
        } else if gemma4_media_mode {
            vec![
                text_input_part(&prompt),
                input_part(
                    InputModality::Image,
                    InputPayload::Tensor(gemma4_patches.clone()),
                    [
                        (InputMetadataKey::PatchGrid, gemma4_grid.clone()),
                        (InputMetadataKey::PatchPositions, gemma4_positions.clone()),
                    ],
                    [InputExtent::PatchGrid {
                        time: 1,
                        height: 2,
                        width: 2,
                    }],
                ),
                input_part(
                    InputModality::Audio,
                    InputPayload::Tensor(gemma4_audio.clone()),
                    [(InputMetadataKey::AudioMask, gemma4_audio_mask.clone())],
                    [InputExtent::AudioValidFrames(4)],
                ),
            ]
        } else if qwen3_vl_media_mode || qwen_conditional_media_mode {
            vec![
                text_input_part(&prompt),
                input_part(
                    InputModality::Image,
                    InputPayload::Tensor(qwen3_vl_pixels.clone()),
                    [(InputMetadataKey::PatchGrid, qwen3_vl_grid.clone())],
                    [],
                ),
            ]
        } else {
            vec![text_input_part(&prompt)]
        };
        let prefix_tokens = if image_mode {
            vec![1, 22, 2]
        } else if inkling_media_mode {
            vec![1, 2, 21, 20, 20, 20]
        } else if gemma4_media_mode {
            vec![1, 2, 30, 31]
        } else if qwen3_vl_media_mode || qwen_conditional_media_mode {
            vec![1, 2, 42, 42]
        } else {
            vec![1, 2]
        };
        let reference_input =
            PreparedModelInput::from_model_input(ModelInput::new(&parts)).unwrap();
        let reference = (((tensor_parallel_size == 2
            && (pipeline_parallel_size == 1
                || (pipeline_parallel_size == 2
                    && matches!(
                        family,
                        FixtureFamily::Llama
                            | FixtureFamily::Mistral
                            | FixtureFamily::Qwen2
                            | FixtureFamily::Qwen2Gguf
                            | FixtureFamily::Qwen3
                            | FixtureFamily::Qwen3Gguf
                            | FixtureFamily::KimiLinear
                            | FixtureFamily::Qwen3Moe
                            | FixtureFamily::GptOss
                            | FixtureFamily::DeepSeek
                    ))))
            || (tensor_parallel_size == 1
                && pipeline_parallel_size == 2
                && matches!(
                    family,
                    FixtureFamily::Llama
                        | FixtureFamily::Mistral
                        | FixtureFamily::Qwen2
                        | FixtureFamily::Qwen2Gguf
                        | FixtureFamily::Qwen3
                        | FixtureFamily::Qwen3Gguf
                        | FixtureFamily::KimiLinear
                        | FixtureFamily::Qwen3Moe
                        | FixtureFamily::GptOss
                        | FixtureFamily::DeepSeek
                )))
            && (expert_parallel_size == 1
                || matches!(
                    family,
                    FixtureFamily::DeepSeek
                        | FixtureFamily::DeepSeekV4
                        | FixtureFamily::Qwen3Moe
                        | FixtureFamily::GptOss
                ))
            && family.needs_opaque_reference()
            && std::env::var_os(OPAQUE_DEEPSEEK_MTP_TARGET).is_none()
            && std::env::var_os(OPAQUE_QWEN_HYBRID_MTP).is_none()
            && std::env::var_os(OPAQUE_NEMOTRON_H_MTP).is_none())
        .then(|| resident_reference_for_prepared(&checkpoint, &reference_input));
        let reference_tolerance = if image_mode || gemma4_media_mode || qwen_conditional_media_mode
        {
            5e-4
        } else {
            family.comparison_tolerance()
        };
        let neutral_forwards_before =
            crate::composition::mlx::path_instrumentation::snapshot().forwards;
        if std::env::var_os(OPAQUE_INSPECTION).is_some() {
            let identity = runtime.session().prompt_cache_model_identity().unwrap();
            let layer_root = if family == FixtureFamily::Gemma {
                "model.language_model.layers"
            } else {
                "model.layers"
            };
            let expected = format!("{layer_root}.{}.output", identity.global_layer_start());
            let inspected = runtime
                .inspect_prefill(ModelInput::new(&parts).into(), &ObservationRequest::all())
                .unwrap();
            assert!(
                inspected.observations.get(&expected).is_some(),
                "rank {expected_rank} missing canonical {expected:?} in {:?}",
                inspected.observations
            );
            assert!(
                inspected
                    .observations
                    .iter()
                    .all(|(path, _)| !path.starts_with("text_decoder.")),
                "rank {expected_rank} returned a synthetic group/index path: {:?}",
                inspected.observations
            );
            assert_eq!(
                inspected
                    .observations
                    .get(eredu_core::MODEL_LOGITS_OBSERVATION_PATH)
                    .is_some(),
                owns_public_output
            );
            return;
        }
        if inkling_mtp_mode
            || qwen_hybrid_mtp_mode
            || nemotron_h_mtp_mode
            || deepseek_mtp_target_mode
        {
            if inkling_mtp_mode || qwen_hybrid_mtp_mode || nemotron_h_mtp_mode {
                let identity = runtime.session().prompt_cache_model_identity().unwrap();
                assert!(
                    !identity.layer_prefix_offsets().contains(&-1),
                    "the neutral target cache must not absorb adapter-owned prediction state"
                );
            }
            let max_tokens = 3;
            let proposal_capacity = if deepseek_dspark_target_mode { 2 } else { 1 };
            if deepseek_mtp_target_mode {
                let vocabulary_size = if family == FixtureFamily::DeepSeekV4 {
                    16
                } else {
                    8
                };
                let invalid_prompt = Array::from_slice(&[vocabulary_size], &[1, 1]);
                let invalid_parts = [text_input_part(&invalid_prompt)];
                let error = match run_neutral_embedded_mtp(
                    &mut runtime,
                    synthetic_prediction_input(
                        &invalid_parts,
                        &[u32::try_from(vocabulary_size).unwrap()],
                    ),
                    SpeculativeConfig {
                        max_tokens,
                        max_draft_tokens: 1,
                        temperature: 0.0,
                        eos_token_ids: Vec::new(),
                    },
                ) {
                    Ok(_) => panic!("out-of-domain target token unexpectedly entered prediction"),
                    Err(error) => error,
                };
                assert!(
                    error
                        .to_string()
                        .contains(&format!("token ID is outside 0..{vocabulary_size}")),
                    "invalid prediction target failed for an unexpected reason: {error}"
                );
            }
            let output = run_neutral_embedded_mtp(
                &mut runtime,
                synthetic_prediction_input(&parts, &prefix_tokens),
                SpeculativeConfig {
                    max_tokens,
                    max_draft_tokens: proposal_capacity,
                    temperature: 0.0,
                    eos_token_ids: Vec::new(),
                },
            )
            .unwrap();
            assert_eq!(output.token_ids().len(), max_tokens);
            assert_eq!(output.stats().emitted_tokens(), max_tokens);
            assert!(output.stats().draft_tokens() > 0);
            if deepseek_dspark_target_mode {
                assert!(
                    output.stats().draft_tokens() >= 2,
                    "the fused DSpark block must propose more than one token"
                );
                assert!(output.stats().rounds() > 0);
                assert_eq!(
                    output.stats().accept_lens().len(),
                    output.stats().rounds(),
                    "every DSpark verification round must reach transactional commit"
                );
                assert!(output.stats().target_tokens() > max_tokens);
                assert!(output.stats().scheduler_turns() >= output.stats().rounds());
            }
            if deepseek_mtp_target_mode || qwen_hybrid_mtp_mode || nemotron_h_mtp_mode {
                let replay = run_neutral_embedded_mtp(
                    &mut runtime,
                    synthetic_prediction_input(&parts, &prefix_tokens),
                    SpeculativeConfig {
                        max_tokens,
                        max_draft_tokens: 1,
                        temperature: 0.0,
                        eos_token_ids: Vec::new(),
                    },
                )
                .unwrap();
                assert_eq!(
                    replay.token_ids(),
                    output.token_ids(),
                    "a fresh prediction lane must not inherit target or extension cache state"
                );
                assert_eq!(replay.stats().emitted_tokens(), max_tokens);
                assert!(replay.stats().draft_tokens() > 0);
            }
            return;
        }
        if qwen_hybrid_prompt_cache_mode {
            let prompt_input = synthetic_prediction_input(&parts, &prefix_tokens);
            let descriptor;
            let rank_prompt_cache = prompt_cache_root.join(format!("rank-{expected_rank}"));
            {
                let (backend, session) = runtime.parts_mut();
                session
                    .prefill(backend, prompt_input.clone())
                    .unwrap()
                    .wait()
                    .unwrap();
                descriptor = PromptCacheDescriptor::from_model_identity(
                    session.prompt_cache_model_identity().unwrap(),
                    "opaque-ring-qwen-hybrid-prediction",
                    prompt_input
                        .cache_identity()
                        .expect("Qwen Hybrid prefix has an exact identity")
                        .prefix_content_fingerprint(),
                    1,
                )
                .unwrap();
                session
                    .save_prompt_cache(
                        backend,
                        &rank_prompt_cache,
                        descriptor.clone(),
                        &prefix_tokens,
                        &PromptCacheOptions::default(),
                    )
                    .unwrap();
                session
                    .decode(backend, Array::from_slice(&[3_u32], &[1, 1]))
                    .unwrap()
                    .wait()
                    .unwrap();
                session
                    .load_prompt_cache_for_input(
                        backend,
                        &rank_prompt_cache,
                        &descriptor,
                        &prefix_tokens,
                        &prompt_input,
                    )
                    .unwrap();
            }
            let suffix = [3_u32];
            let suffix_tensor = Array::from_slice(&suffix, &[1, 1]);
            let suffix_parts = [text_input_part(&suffix_tensor)];
            let output = run_neutral_embedded_mtp(
                &mut runtime,
                synthetic_prediction_input(&suffix_parts, &suffix),
                SpeculativeConfig {
                    max_tokens: 3,
                    max_draft_tokens: 1,
                    temperature: 0.0,
                    eos_token_ids: Vec::new(),
                },
            )
            .unwrap();
            assert_eq!(output.token_ids().len(), 3);
            assert!(output.stats().draft_tokens() > 0);
            assert!(output.stats().rounds() > 0);
            return;
        }
        let (backend, session) = runtime.parts_mut();
        if neutral_gemma_layers.is_some() || neutral_qwen_vl_config.is_some() {
            let unsupported_image = Array::from_slice(&[0.0f32; 4], &[1, 1, 4]);
            let malformed_parts = [input_part(
                InputModality::Image,
                InputPayload::Tensor(unsupported_image),
                [],
                [],
            )];
            let before = crate::composition::mlx::path_instrumentation::snapshot();
            let error = match session.prefill(backend, ModelInput::new(&malformed_parts).into()) {
                Ok(_) => panic!("unselected Gemma image input unexpectedly entered execution"),
                Err(error) => error,
            };
            let expected = if neutral_gemma_layers.is_some() && !gemma4_media_mode {
                "prepared input modality image is outside the selected composite modalities {Text}"
            } else {
                "unsupported prepared input"
            };
            assert!(
                error.to_string().contains(expected),
                "malformed prepared input failed for an unexpected reason: {error}"
            );
            let after = crate::composition::mlx::path_instrumentation::snapshot();
            assert_eq!(after.forwards, before.forwards);
            assert_eq!(after.completions, before.completions);
        }
        let prompt_input = (dense_composite_neutral
            || neutral_gemma_layers.is_some()
            || neutral_qwen_vl_config.is_some()
            || inkling_media_mode
            || matches!(
                family,
                FixtureFamily::Inkling | FixtureFamily::Qwen35Multimodal
            ))
        .then(|| {
            MlxModelInput::from(ModelInput::new(&parts)).with_semantic_content_fingerprint(
                eredu_core::cache::prompt_cache_token_fingerprint(&prefix_tokens),
            )
        })
        .transpose()
        .unwrap();
        let submitted_input = prompt_input
            .clone()
            .unwrap_or_else(|| ModelInput::new(&parts).into());
        let mut output = session
            .prefill(backend, submitted_input)
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(output.logits().is_some(), owns_public_output);
        if let (Some(actual), Some((expected, _))) = (output.logits(), &reference) {
            assert_final_logits_close(actual.as_array(), expected, reference_tolerance);
        }
        let descriptor = PromptCacheDescriptor::from_model_identity(
            session.prompt_cache_model_identity().unwrap(),
            "opaque-ring-fixture",
            prompt_input.as_ref().map_or_else(
                || format!("tokens:{prefix_tokens:?}"),
                |input| {
                    input
                        .cache_identity()
                        .expect("neutral Gemma input carries its exact prepared identity")
                        .prefix_content_fingerprint()
                        .to_owned()
                },
            ),
            1,
        )
        .unwrap();
        let rank_prompt_cache = prompt_cache_root.join(format!("rank-{expected_rank}"));
        if std::env::var_os(PROMPT_CACHE_PREPARE_FAILURE).is_some() {
            if expected_rank == 0 {
                std::fs::write(
                    &rank_prompt_cache,
                    b"block rank-local cache directory creation",
                )
                .unwrap();
            }
            let error = session
                .save_prompt_cache(
                    backend,
                    &rank_prompt_cache,
                    descriptor.clone(),
                    &prefix_tokens,
                    &PromptCacheOptions::default(),
                )
                .unwrap_err();
            if expected_rank == 0 {
                assert!(
                    error
                        .to_string()
                        .contains("create reversible prompt cache parent"),
                    "injected rank-local preparation failed for an unexpected reason: {error}"
                );
            } else {
                assert!(
                    error.to_string().contains("another rank failed")
                        && error.to_string().contains("PromptCacheSavePreparation"),
                    "peer preparation failure was not reported causally: {error}"
                );
                assert!(
                    rank_prompt_cache.is_dir()
                        && std::fs::read_dir(&rank_prompt_cache)
                            .unwrap()
                            .next()
                            .is_none(),
                    "successful peer preparation left a published or staged shard"
                );
            }
            let retry = session
                .save_prompt_cache(
                    backend,
                    &rank_prompt_cache,
                    descriptor,
                    &prefix_tokens,
                    &PromptCacheOptions::default(),
                )
                .unwrap_err();
            assert!(
                retry.to_string().contains("session is fenced"),
                "cache-control retry was not rejected by the causal fence: {retry}"
            );
            return;
        }
        session
            .save_prompt_cache(
                backend,
                &rank_prompt_cache,
                descriptor.clone(),
                &prefix_tokens,
                &PromptCacheOptions::default(),
            )
            .unwrap();
        let continuity_token = if reference.is_some() {
            Array::from_slice(&[0u32], &[1, 1])
        } else {
            session
                .sample_and_synchronize(output.logits(), 1, &mut DefaultSampler, 0.0, None, false)
                .unwrap()
                .token
        };
        let uninterrupted = session
            .decode(backend, continuity_token.clone())
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(uninterrupted.logits().is_some(), owns_public_output);
        if let (Some(actual), Some((_, expected))) = (uninterrupted.logits(), &reference) {
            assert_final_logits_close(actual.as_array(), expected, reference_tolerance);
        }
        let uninterrupted_logits = uninterrupted.logits().map(|logits| {
            logits
                .as_array()
                .evaluated()
                .unwrap()
                .as_slice::<f32>()
                .to_vec()
        });
        match prompt_input.as_ref() {
            Some(input) => session
                .load_prompt_cache_for_input(
                    backend,
                    &rank_prompt_cache,
                    &descriptor,
                    &prefix_tokens,
                    input,
                )
                .unwrap(),
            None => session
                .load_prompt_cache(backend, &rank_prompt_cache, &descriptor, &prefix_tokens)
                .unwrap(),
        };
        output = session
            .decode(backend, continuity_token)
            .unwrap()
            .wait()
            .unwrap();
        let restored_logits = output.logits().map(|logits| {
            logits
                .as_array()
                .evaluated()
                .unwrap()
                .as_slice::<f32>()
                .to_vec()
        });
        assert_eq!(uninterrupted_logits, restored_logits);
        for _ in 0..2 {
            assert_eq!(output.logits().is_some(), owns_public_output);
            let token = session
                .sample_and_synchronize(output.logits(), 1, &mut DefaultSampler, 0.0, None, false)
                .unwrap()
                .token;
            output = session.decode(backend, token).unwrap().wait().unwrap();
        }
        assert_eq!(output.logits().is_some(), owns_public_output);
        if prove_prepared_communication_lifecycle {
            assert_eq!(
                crate::composition::mlx::path_instrumentation::snapshot().forwards
                    - neutral_forwards_before,
                5,
                "prefill, uninterrupted/restored decode, and two continued decodes must each traverse the neutral session once"
            );
            if dense_stream || layerwise_host {
                assert_eq!(
                    crate::composition::mlx::path_instrumentation::bounded_unit_acquisitions(),
                    5 * local_layer_range.len(),
                    "each neutral forward must acquire every selected bounded-residency unit exactly once"
                );
            }
            if routed_neutral && expert_parallel_size > 1 {
                let exchanges_per_layer = match family {
                    FixtureFamily::Qwen3Moe
                    | FixtureFamily::Qwen3MoeGguf
                    | FixtureFamily::DeepSeek
                    | FixtureFamily::DeepSeekGguf
                    | FixtureFamily::DeepSeekV4
                    | FixtureFamily::NemotronH
                    | FixtureFamily::NemotronHGguf
                    | FixtureFamily::Lfm2MoeGguf
                    | FixtureFamily::KimiLinearGguf
                    | FixtureFamily::Qwen3VlMoe => 8,
                    FixtureFamily::Inkling | FixtureFamily::InklingMultimodal => 16,
                    FixtureFamily::GptOss | FixtureFamily::GptOssGguf
                        if tensor_parallel_size > 1 =>
                    {
                        9
                    }
                    FixtureFamily::GptOss | FixtureFamily::GptOssGguf => 8,
                    _ => unreachable!("only routed neutral fixtures select expert exchange"),
                };
                let exchanged_layers = if pipeline_parallel_size > 1 {
                    family.expert_layer_count(0..family.layer_count())
                } else {
                    family.expert_layer_count(local_layer_range.clone())
                };
                assert_eq!(
                    crate::composition::mlx::path_instrumentation::variable_all_to_all_submissions(),
                    5 * exchanged_layers * exchanges_per_layer,
                    "every routed layer forward must exchange global/local expert IDs, route tags, activations, scores, coefficients, and the exact inverse results without fallback"
                );
            }
        }
        if dense_stream {
            let report = session.dense_stream_report().unwrap().unwrap();
            assert_eq!(report.prefill_forwards(), 1);
            assert_eq!(report.decode_forwards(), 4);
            assert_eq!(report.planned_layer_count(), local_layer_range.len());
        }
        if layerwise_host {
            let report = session.residency_report().unwrap().unwrap();
            let expected_units = local_layer_range.len();
            let layerwise = report
                .units()
                .iter()
                .filter(|unit| unit.planned_tier() == eredu_core::residency::MemoryTier::Host)
                .collect::<Vec<_>>();
            assert_eq!(layerwise.len(), expected_units);
            assert!(layerwise.iter().all(|unit| unit.host_resident()));
            assert!(
                layerwise
                    .iter()
                    .filter(|unit| unit.device_resident())
                    .count()
                    <= 1
            );
        }
        if std::env::var_os(EXPERT_CACHE).is_some() {
            let report = session.parameter_bank_report().unwrap();
            let expected_owned = match family {
                FixtureFamily::Qwen3Moe | FixtureFamily::Qwen3MoeGguf => {
                    local_layer_range.len() * 4 / expert_parallel_size
                }
                FixtureFamily::GptOss | FixtureFamily::GptOssGguf => {
                    local_layer_range.len() * 2 / expert_parallel_size
                }
                FixtureFamily::DeepSeek | FixtureFamily::DeepSeekGguf => {
                    family.expert_layer_count(local_layer_range.clone()) * 4 / expert_parallel_size
                }
                FixtureFamily::DeepSeekV4 => {
                    family.expert_layer_count(local_layer_range.clone()) * 4 / expert_parallel_size
                }
                FixtureFamily::KimiLinearGguf => {
                    family.expert_layer_count(local_layer_range.clone()) * 4 / expert_parallel_size
                }
                FixtureFamily::Lfm2MoeGguf
                | FixtureFamily::NemotronH
                | FixtureFamily::NemotronHGguf => {
                    family.expert_layer_count(local_layer_range.clone()) * 2 / expert_parallel_size
                }
                _ => unreachable!("only exact routed addressable fixtures select expert caching"),
            };
            if expected_owned == 0 {
                assert!(
                    report.is_none(),
                    "rank with no routed units must not manufacture bank ownership"
                );
                return;
            }
            let report = report.expect("owned independent expert bank must expose live telemetry");
            assert!(report.owned_entries() > 0);
            assert!(report.owned_bytes() > 0);
            let requests =
                report.bulk().device().requests() + report.incremental().device().requests();
            let misses = report.bulk().device().misses() + report.incremental().device().misses();
            assert_eq!(
                report.owned_entries(),
                expected_owned,
                "the addressable bank must own only this PP×EP rank's exact expert entries"
            );
            assert_eq!(requests == 0, misses == 0);
            if std::env::var_os(EXPERT_CACHE_EVICTION).is_some() {
                let evictions =
                    report.bulk().device().evictions() + report.incremental().device().evictions();
                if requests > 0 {
                    assert!(evictions > 0, "bounded expert bank never evicted an entry");
                    assert!(misses > report.owned_entries() as u64);
                }
            }
        }
        if prompt_input.is_some() {
            let wrong_descriptor = PromptCacheDescriptor::from_model_identity(
                session.prompt_cache_model_identity().unwrap(),
                "opaque-ring-fixture",
                "wrong-prepared-input-identity",
                1,
            )
            .unwrap();
            let error = session
                .save_prompt_cache(
                    backend,
                    &rank_prompt_cache,
                    wrong_descriptor,
                    &prefix_tokens,
                    &PromptCacheOptions::default(),
                )
                .unwrap_err();
            assert!(
                error.to_string().contains("prepared input"),
                "wrong prepared-input descriptor failed for an unexpected reason: {error}"
            );
            let retry = session
                .save_prompt_cache(
                    backend,
                    &rank_prompt_cache,
                    descriptor,
                    &prefix_tokens,
                    &PromptCacheOptions::default(),
                )
                .unwrap_err();
            assert!(
                retry.to_string().contains("session is fenced"),
                "cache-control retry was not rejected by the causal fence: {retry}"
            );
        }
    }
}

#[test]
fn complete_qwen3_vl_variants_accept_paged_cache() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    for (expected_effective_model_type, moe) in
        [("qwen3_vl_text", false), ("qwen3_vl_moe_text", true)]
    {
        let checkpoint = tempfile::tempdir().unwrap();
        write_qwen3_vl_fixture(checkpoint.path(), moe);
        let backend = crate::native::backend(&stream, &stream);
        let paged = PagedCacheOptions::new(1, 32768, 32768, 1)
            .unwrap()
            .with_full_attention(true);
        let request =
            MlxLoadRequest::default().with_state_residency(CacheResidencyPolicy::Paged(paged));
        let model = load_model(&backend, checkpoint.path(), request).unwrap();
        assert_eq!(model.effective_model_type(), expected_effective_model_type);
        let runtime = ModelRuntime::from_prepared(backend, model).unwrap();
        assert_eq!(
            runtime.session().effective_model_type(),
            expected_effective_model_type
        );
        assert_eq!(
            <MlxBackend<'_> as eredu_core::ModelCapabilityBackend>::model_capabilities(&runtime)
                .unwrap()
                .effective_model_type,
            expected_effective_model_type
        );

        assert!(runtime
            .session()
            .cache_residency_report()
            .unwrap()
            .is_some());
    }
}

#[test]
fn complete_gemma4_preserves_nested_effective_model_type() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let checkpoint = tempfile::tempdir().unwrap();
    write_gemma_fixture(checkpoint.path());
    let backend = crate::native::backend(&stream, &stream);
    let model = load_model(&backend, checkpoint.path(), MlxLoadRequest::default()).unwrap();
    assert_eq!(model.effective_model_type(), "gemma4_text");
    let runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    assert_eq!(runtime.session().effective_model_type(), "gemma4_text");
    assert_eq!(
        <MlxBackend<'_> as eredu_core::ModelCapabilityBackend>::model_capabilities(&runtime)
            .unwrap()
            .effective_model_type,
        "gemma4_text"
    );
}

#[test]
fn public_replicated_prediction_variants_install_only_the_neutral_extension() {
    fn assert_extension(checkpoint: &std::path::Path, expected_depth: usize) {
        crate::composition::mlx::path_instrumentation::reset();
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let backend = crate::native::backend(&stream, &stream);
        let model = load_model(&backend, checkpoint, MlxLoadRequest::default())
            .unwrap()
            .into_inner();
        assert_eq!(
            crate::composition::mlx::path_instrumentation::snapshot().constructors,
            1
        );
        let mut session = MlxModelSession::from_model(
            model,
            eredu_core::SessionCapabilities::new(true, true, true),
        )
        .unwrap();
        assert_eq!(
            session.speculative_capability(),
            SpeculativeCapability::Ready {
                draft_source: eredu_core::SpeculativeDraftSource::Embedded,
            }
        );
        assert!(has_selected_embedded_prediction(&mut session));
        let target = session.neutral_prediction_target_mut().unwrap();
        assert!(target.has_embedded_prediction());
        let _ = expected_depth;
    }

    let deepseek = tempfile::tempdir().unwrap();
    write_deepseek_fixture_with_prediction(deepseek.path(), 2, 1);
    assert_extension(deepseek.path(), 1);

    let inkling = tempfile::tempdir().unwrap();
    write_inkling_mtp_fixture(inkling.path());
    assert_extension(inkling.path(), 2);

    let qwen = tempfile::tempdir().unwrap();
    write_qwen35_multimodal_fixture(qwen.path(), false);
    assert_extension(qwen.path(), 1);

    let nemotron = tempfile::tempdir().unwrap();
    write_nemotron_mtp_fixture(nemotron.path());
    assert_extension(nemotron.path(), 1);
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires a local MLX Metal device"]
fn public_deepseek_v3_embedded_scheduler_executes_on_metal() {
    let checkpoint = tempfile::tempdir().unwrap();
    write_deepseek_fixture_with_prediction(checkpoint.path(), 2, 1);
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let weights_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let backend = crate::native::backend(&stream, &weights_stream);
    let model = load_model(&backend, checkpoint.path(), MlxLoadRequest::default()).unwrap();
    let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    assert_eq!(
        runtime.session().speculative_capability(),
        SpeculativeCapability::Ready {
            draft_source: eredu_core::SpeculativeDraftSource::Embedded,
        }
    );

    let prompt = Array::from_slice(&[1u32, 2], &[1, 2]);
    let parts = [text_input_part(&prompt)];
    let output = run_neutral_embedded_mtp(
        &mut runtime,
        synthetic_prediction_input(&parts, &[1, 2]),
        SpeculativeConfig {
            max_tokens: 3,
            max_draft_tokens: 1,
            temperature: 0.0,
            eos_token_ids: Vec::new(),
        },
    )
    .unwrap();
    assert_eq!(output.token_ids().len(), 3);
    assert!(output.stats().draft_tokens() > 0);
}

#[test]
fn synthetic_prediction_input_binds_prepared_shape_and_token_content() {
    let first_tokens = [1_u32, 2];
    let first = Array::from_slice(&first_tokens, &[1, 2]);
    let first_parts = [text_input_part(&first)];
    let first = synthetic_prediction_input(&first_parts, &first_tokens);
    let same = synthetic_prediction_input(&first_parts, &first_tokens);
    let changed = synthetic_prediction_input(&first_parts, &[2, 1]);
    let reshaped_tokens = Array::from_slice(&first_tokens, &[2, 1]);
    let reshaped_parts = [text_input_part(&reshaped_tokens)];
    let reshaped = synthetic_prediction_input(&reshaped_parts, &first_tokens);

    assert_eq!(first.cache_identity(), same.cache_identity());
    assert_ne!(first.cache_identity(), changed.cache_identity());
    assert_ne!(first.cache_identity(), reshaped.cache_identity());
    assert_eq!(
        first
            .cache_identity()
            .expect("synthetic input identity")
            .semantic_content_fingerprint(),
        eredu_core::cache::prompt_cache_token_fingerprint(&first_tokens)
    );
}

#[derive(Default)]
struct ExternalObservationTrace {
    counts: BTreeMap<String, usize>,
    proposal_logits: Vec<Vec<f32>>,
}

impl ExternalObservationTrace {
    fn record(&mut self, path: &str) {
        *self.counts.entry(path.to_owned()).or_default() += 1;
    }

    fn count(&self, path: &str) -> usize {
        self.counts.get(path).copied().unwrap_or_default()
    }
}

#[derive(Clone, Copy)]
enum ExternalTensorIntervention {
    None,
    Zero,
    ForceToken(usize),
    Fail,
}

fn forced_token_logits(value: &Array, token: usize) -> Result<Array, safemlx::error::Exception> {
    let width = value
        .shape()
        .last()
        .copied()
        .and_then(|width| usize::try_from(width).ok())
        .ok_or_else(|| safemlx::error::Exception::custom("external logits have no vocabulary"))?;
    if token >= width {
        return Err(safemlx::error::Exception::custom(
            "forced external token is outside the vocabulary",
        ));
    }
    let mut values = vec![-100.0_f32; value.size()];
    for row in values.chunks_exact_mut(width) {
        row[token] = 100.0;
    }
    Ok(Array::from_slice(&values, value.shape()))
}

struct ExternalTensorObserver {
    trace: Arc<Mutex<ExternalObservationTrace>>,
    path: Option<&'static str>,
    intervention: ExternalTensorIntervention,
    stream: Stream,
}

impl eredu_runtime::ActivationObserver<MlxTensor, safemlx::error::Exception>
    for ExternalTensorObserver
{
    fn observe(&mut self, path: &str, _value: &MlxTensor) -> Result<(), safemlx::error::Exception> {
        self.trace.lock().unwrap().record(path);
        if self.path == Some(path) && matches!(self.intervention, ExternalTensorIntervention::Fail)
        {
            return Err(safemlx::error::Exception::custom(
                "injected external observation failure",
            ));
        }
        Ok(())
    }

    fn intervene(
        &mut self,
        path: &str,
        value: &MlxTensor,
    ) -> Result<Option<MlxTensor>, safemlx::error::Exception> {
        if self.path != Some(path) {
            return Ok(None);
        }
        match self.intervention {
            ExternalTensorIntervention::None | ExternalTensorIntervention::Fail => Ok(None),
            ExternalTensorIntervention::Zero => Ok(Some(MlxTensor::from_array(
                safemlx::ops::zeros_like(value.as_array(), &self.stream)?,
            ))),
            ExternalTensorIntervention::ForceToken(token) => Ok(Some(MlxTensor::from_array(
                forced_token_logits(value.as_array(), token)?,
            ))),
        }
    }
}

struct ExternalLogitsObserver {
    trace: Arc<Mutex<ExternalObservationTrace>>,
    intervention: ExternalTensorIntervention,
}

impl eredu_runtime::ActivationObserver<Array, safemlx::error::Exception>
    for ExternalLogitsObserver
{
    fn observe(&mut self, path: &str, value: &Array) -> Result<(), safemlx::error::Exception> {
        let mut trace = self.trace.lock().unwrap();
        trace.record(path);
        if path
            == eredu_architectures::external_assistant::EXTERNAL_ASSISTANT_PROPOSAL_LOGITS_OBSERVATION_PATH
        {
            let evaluated = value.evaluated()?;
            trace.proposal_logits.push(
                evaluated
                    .try_to_vec::<f32>()
                    .map_err(|error| safemlx::error::Exception::custom(error.to_string()))?,
            );
            if matches!(self.intervention, ExternalTensorIntervention::Fail) {
                return Err(safemlx::error::Exception::custom(
                    "injected external proposal observation failure",
                ));
            }
        }
        Ok(())
    }

    fn intervene(
        &mut self,
        path: &str,
        value: &Array,
    ) -> Result<Option<Array>, safemlx::error::Exception> {
        if path
            != eredu_architectures::external_assistant::EXTERNAL_ASSISTANT_PROPOSAL_LOGITS_OBSERVATION_PATH
        {
            return Ok(None);
        }
        match self.intervention {
            ExternalTensorIntervention::ForceToken(token) => {
                Ok(Some(forced_token_logits(value, token)?))
            }
            ExternalTensorIntervention::Zero => Ok(Some(Array::from_slice(
                &vec![0.0_f32; value.size()],
                value.shape(),
            ))),
            ExternalTensorIntervention::None | ExternalTensorIntervention::Fail => Ok(None),
        }
    }
}

struct EmbeddedLogitsObserver {
    trace: Arc<Mutex<ExternalObservationTrace>>,
    intervention: ExternalTensorIntervention,
}

impl eredu_runtime::ActivationObserver<Array, safemlx::error::Exception>
    for EmbeddedLogitsObserver
{
    fn observe(&mut self, path: &str, value: &Array) -> Result<(), safemlx::error::Exception> {
        let mut trace = self.trace.lock().unwrap();
        trace.record(path);
        if path == eredu_architectures::speculative_execution::EMBEDDED_PROPOSAL_LOGITS_PATH {
            let evaluated = value.evaluated()?;
            trace.proposal_logits.push(
                evaluated
                    .try_to_vec::<f32>()
                    .map_err(|error| safemlx::error::Exception::custom(error.to_string()))?,
            );
            if matches!(self.intervention, ExternalTensorIntervention::Fail) {
                return Err(safemlx::error::Exception::custom(
                    "injected embedded proposal observation failure",
                ));
            }
        }
        Ok(())
    }

    fn intervene(
        &mut self,
        path: &str,
        value: &Array,
    ) -> Result<Option<Array>, safemlx::error::Exception> {
        if path != eredu_architectures::speculative_execution::EMBEDDED_PROPOSAL_LOGITS_PATH {
            return Ok(None);
        }
        match self.intervention {
            ExternalTensorIntervention::ForceToken(token) => {
                Ok(Some(forced_token_logits(value, token)?))
            }
            ExternalTensorIntervention::Zero => Ok(Some(Array::from_slice(
                &vec![0.0_f32; value.size()],
                value.shape(),
            ))),
            ExternalTensorIntervention::None | ExternalTensorIntervention::Fail => Ok(None),
        }
    }
}

#[test]
fn public_embedded_observers_are_installed_causally_and_transactionally() {
    const TARGET_CAPTURE: &str =
        eredu_architectures::speculative_execution::EMBEDDED_TARGET_CAPTURE_PATH;
    const PREDICTION_OUTPUT: &str =
        eredu_architectures::speculative_execution::EMBEDDED_PREDICTION_OUTPUT_PATH;
    const PROPOSAL_LOGITS: &str =
        eredu_architectures::speculative_execution::EMBEDDED_PROPOSAL_LOGITS_PATH;
    const VERIFICATION_LOGITS: &str =
        eredu_architectures::speculative_execution::EMBEDDED_VERIFICATION_LOGITS_PATH;

    fn run(
        checkpoint: &Path,
        tensor_path: Option<&'static str>,
        tensor_intervention: ExternalTensorIntervention,
        logits_intervention: ExternalTensorIntervention,
    ) -> (
        Result<eredu_core::SpeculativeGenerationOutput, String>,
        usize,
        Arc<Mutex<ExternalObservationTrace>>,
    ) {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let backend = crate::native::backend(&stream, &stream);
        let model = load_model(&backend, checkpoint, MlxLoadRequest::default()).unwrap();
        let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
        let trace = Arc::new(Mutex::new(ExternalObservationTrace::default()));
        runtime
            .session_mut()
            .install_embedded_prediction_observers(
                ExternalTensorObserver {
                    trace: Arc::clone(&trace),
                    path: tensor_path,
                    intervention: tensor_intervention,
                    stream: stream.clone(),
                },
                EmbeddedLogitsObserver {
                    trace: Arc::clone(&trace),
                    intervention: logits_intervention,
                },
            )
            .unwrap();
        let tokens = [1_u32, 2];
        let prompt = Array::from_slice(&tokens, &[1, 2]);
        let parts = [text_input_part(&prompt)];
        let (result, publications) = execute_neutral_embedded_mtp(
            &mut runtime,
            synthetic_prediction_input(&parts, &tokens),
            SpeculativeConfig {
                max_tokens: 3,
                max_draft_tokens: 2,
                temperature: 0.0,
                eos_token_ids: Vec::new(),
            },
        );
        (
            result.map_err(|error| error.to_string()),
            publications,
            trace,
        )
    }

    let checkpoint = tempfile::tempdir().unwrap();
    write_inkling_mtp_fixture(checkpoint.path());
    let (baseline, publications, trace) = run(
        checkpoint.path(),
        None,
        ExternalTensorIntervention::None,
        ExternalTensorIntervention::None,
    );
    let baseline = baseline.unwrap();
    assert!(publications > 0);
    let rounds = baseline.stats().rounds();
    let drafts = baseline.stats().draft_tokens();
    let trace = trace.lock().unwrap();
    assert_eq!(trace.count(TARGET_CAPTURE), rounds + 1);
    assert_eq!(trace.count(PREDICTION_OUTPUT), drafts);
    assert_eq!(trace.count(PROPOSAL_LOGITS), drafts);
    assert_eq!(trace.count(VERIFICATION_LOGITS), rounds);
    drop(trace);

    let (proposal, _, _) = run(
        checkpoint.path(),
        None,
        ExternalTensorIntervention::None,
        ExternalTensorIntervention::ForceToken(7),
    );
    let proposal = proposal.unwrap();
    assert!(
        proposal.token_ids() != baseline.token_ids()
            || proposal.stats().accept_lens() != baseline.stats().accept_lens(),
        "proposal intervention did not reach verification and commit",
    );

    let (verification, _, _) = run(
        checkpoint.path(),
        Some(VERIFICATION_LOGITS),
        ExternalTensorIntervention::ForceToken(6),
        ExternalTensorIntervention::None,
    );
    assert_ne!(verification.unwrap().token_ids(), baseline.token_ids());

    let (failed, failed_publications, failed_trace) = run(
        checkpoint.path(),
        Some(TARGET_CAPTURE),
        ExternalTensorIntervention::Fail,
        ExternalTensorIntervention::None,
    );
    let error = match failed {
        Ok(_) => panic!("injected embedded observation failure unexpectedly succeeded"),
        Err(error) => error,
    };
    assert!(error.contains("injected external observation failure"));
    assert_eq!(failed_publications, 0);
    let failed_trace = failed_trace.lock().unwrap();
    assert_eq!(failed_trace.count(TARGET_CAPTURE), 1);
    assert_eq!(failed_trace.count(PREDICTION_OUTPUT), 0);
    assert_eq!(failed_trace.count(PROPOSAL_LOGITS), 0);
    assert_eq!(failed_trace.count(VERIFICATION_LOGITS), 0);
}

fn execute_public_gemma_external_scheduler(
    target: &Path,
    assistant: &Path,
    target_device: DevicePlan,
    placement: DraftPlacementPlan,
    configure: impl FnOnce(&mut crate::composition::mlx::speculative::MlxDrafter),
) -> (
    Result<eredu_core::SpeculativeGenerationBatchOutput, String>,
    usize,
) {
    let plan = ExecutionPlan::fully_resident(target_device).with_drafting(DraftingPlan::External {
        model: assistant.display().to_string(),
        placement,
        max_draft_tokens: 2,
        lookahead: false,
        adaptive_lookahead: false,
    });
    let factory = crate::composition::mlx::automatic::MlxBackendFactory::default();
    let tokenizer_compatibility = TokenizerCompatibilityProof::prove([7; 32], [7; 32]).unwrap();
    let preparation = eredu_architectures::prepare_external_assistant(assistant).unwrap();
    let inspection = eredu_architectures::configuration::inspect_artifact(target).unwrap();
    let selected = eredu_core::select_execution_plan_target(&factory, &plan, inspection).unwrap();
    let external_artifact = eredu_core::select_execution_plan_drafting(
        &factory,
        &plan,
        &selected,
        Some(ExternalDraftArtifact {
            preparation,
            tokenizer_compatibility,
        }),
    )
    .unwrap();
    let realization = eredu_core::realize_execution_plan_target(&factory, &plan, selected).unwrap();
    let mut runtime = realization.into_runtime().unwrap();

    let mut drafting =
        eredu_core::realize_execution_plan_drafting(&factory, &plan, &runtime, external_artifact)
            .unwrap();
    let mut draft = drafting
        .as_speculative_draft()
        .expect("the external plan must realize a reusable assistant");
    match &mut draft {
        SpeculativeDraft::External(drafter) => configure(drafter),
        _ => panic!("the external plan must expose the materialized assistant"),
    }

    let sampling = eredu_core::resolve_generation_config(
        None,
        eredu_core::GenerationConfigOverrides {
            max_new_tokens: Some(3),
            temperature: Some(0.0),
            ..Default::default()
        },
    )
    .unwrap();
    let prompts = [[1u32, 2], [2u32, 1]]
        .into_iter()
        .map(|tokens| {
            let semantic_content_fingerprint =
                eredu_core::cache::prompt_cache_token_fingerprint(&tokens);
            let tokens = Array::from_slice(&tokens, &[1, 2]);
            let parts = [text_input_part(&tokens)];
            MlxModelInput::from(crate::backend::runtime::media::input::ModelInput::new(
                &parts,
            ))
            .with_semantic_content_fingerprint(semantic_content_fingerprint)
            .unwrap()
        })
        .collect::<Vec<_>>();
    let publications = Arc::new(AtomicUsize::new(0));
    let lanes = prompts
        .into_iter()
        .map(|prompt| {
            let publications = publications.clone();
            SpeculativeGenerationLane::new(
                prompt,
                TextGenerationConfig::new(sampling),
                SpeculativeConfig {
                    max_tokens: 3,
                    max_draft_tokens: 2,
                    temperature: 0.0,
                    eos_token_ids: Vec::new(),
                },
                AllowAllTokens,
                Box::<TokenOnlySemanticState>::default(),
                GenerationCancellationToken::new(),
                Box::new(move |_| {
                    publications.fetch_add(1, Ordering::Relaxed);
                }),
            )
        })
        .collect();
    let output = <MlxBackend<'static> as SpeculativeGenerationBackend>::with_speculative_execution(
        &mut runtime,
        SpeculativeGenerationBatchRequest::new(draft, lanes, [0; 32]),
        eredu_runtime::RunSpeculativeGeneration::default(),
    )
    .map_err(|error| error.to_string());
    (output, publications.load(Ordering::Relaxed))
}

fn run_public_gemma_external_scheduler(
    target: &Path,
    assistant: &Path,
    target_device: DevicePlan,
    placement: DraftPlacementPlan,
    expected_topology: SpeculativeExecutionTopology,
) {
    let (output, _) = execute_public_gemma_external_scheduler(
        target,
        assistant,
        target_device,
        placement,
        |_| {},
    );
    let output = output.unwrap();

    assert_eq!(output.scheduler().execution_topology(), expected_topology);
    assert!(output.scheduler().turns() > 0);
    let requests = output.into_requests();
    assert_eq!(requests.len(), 2);
    for request in requests {
        assert_eq!(request.token_ids().len(), 3);
        assert_eq!(request.stats().emitted_tokens(), 3);
        assert_eq!(request.stats().execution_topology(), expected_topology);
        assert!(
            request.stats().draft_tokens() > 0,
            "assistant did no proposal work"
        );
        assert!(
            request.stats().rounds() > 0,
            "target performed no verification"
        );
        assert_eq!(
            request.stats().accept_lens().len(),
            request.stats().rounds(),
            "each verification round must reach its transaction outcome"
        );
        assert!(
            request.stats().target_tokens() > 2,
            "verification must evaluate more than the two-token prefill"
        );
    }
}

#[test]
fn public_gemma_external_factory_scheduler_supports_target_and_cpu_split_placement() {
    let target = tempfile::tempdir().unwrap();
    let assistant = tempfile::tempdir().unwrap();
    write_gemma_fixture(target.path());
    write_gemma_assistant_fixture(assistant.path());

    run_public_gemma_external_scheduler(
        target.path(),
        assistant.path(),
        DevicePlan::new("mlx", "cpu:0").unwrap(),
        DraftPlacementPlan::Target,
        SpeculativeExecutionTopology::Single,
    );
    run_public_gemma_external_scheduler(
        target.path(),
        assistant.path(),
        DevicePlan::new("mlx", "cpu:0").unwrap(),
        DraftPlacementPlan::Device {
            device: DevicePlan::new("mlx", "cpu:0").unwrap(),
        },
        SpeculativeExecutionTopology::SameDeviceSplit,
    );
}

#[test]
fn public_gemma_external_observers_are_causal_exact_and_transactional() {
    const CAPTURE_PATH: &str = "model.language_model.layers.3.output";
    const PROPOSAL_PATH: &str = eredu_architectures::external_assistant::EXTERNAL_ASSISTANT_PROPOSAL_LOGITS_OBSERVATION_PATH;
    const VERIFICATION_PATH: &str = eredu_architectures::external_assistant::EXTERNAL_ASSISTANT_VERIFICATION_LOGITS_OBSERVATION_PATH;

    fn run(
        target: &Path,
        assistant: &Path,
        tensor_path: Option<&'static str>,
        tensor_intervention: ExternalTensorIntervention,
        proposal_intervention: ExternalTensorIntervention,
    ) -> (
        Result<eredu_core::SpeculativeGenerationBatchOutput, String>,
        usize,
        Arc<Mutex<ExternalObservationTrace>>,
    ) {
        let trace = Arc::new(Mutex::new(ExternalObservationTrace::default()));
        let installed = trace.clone();
        let (result, publications) = execute_public_gemma_external_scheduler(
            target,
            assistant,
            DevicePlan::new("mlx", "cpu:0").unwrap(),
            DraftPlacementPlan::Target,
            move |drafter| {
                drafter.install_external_observers(
                    ExternalTensorObserver {
                        trace: installed.clone(),
                        path: tensor_path,
                        intervention: tensor_intervention,
                        stream: Stream::new_with_device(&Device::new(DeviceType::Cpu, 0)),
                    },
                    ExternalLogitsObserver {
                        trace: installed,
                        intervention: proposal_intervention,
                    },
                );
            },
        );
        (result, publications, trace)
    }

    fn tokens(output: &eredu_core::SpeculativeGenerationBatchOutput) -> Vec<Vec<u32>> {
        output
            .requests()
            .iter()
            .map(|request| request.token_ids().to_vec())
            .collect()
    }

    fn accept_lens(output: &eredu_core::SpeculativeGenerationBatchOutput) -> Vec<Vec<usize>> {
        output
            .requests()
            .iter()
            .map(|request| request.stats().accept_lens().to_vec())
            .collect()
    }

    let target = tempfile::tempdir().unwrap();
    let assistant = tempfile::tempdir().unwrap();
    write_gemma_fixture(target.path());
    write_gemma_assistant_fixture(assistant.path());

    let (baseline, baseline_publications, baseline_trace) = run(
        target.path(),
        assistant.path(),
        None,
        ExternalTensorIntervention::None,
        ExternalTensorIntervention::None,
    );
    let baseline = baseline.unwrap();
    assert!(baseline_publications > 0);
    let baseline_tokens = tokens(&baseline);
    let total_drafts = baseline
        .requests()
        .iter()
        .map(|request| request.stats().draft_tokens())
        .sum::<usize>();
    let total_rounds = baseline
        .requests()
        .iter()
        .map(|request| request.stats().rounds())
        .sum::<usize>();
    let baseline_trace = baseline_trace.lock().unwrap();
    assert_eq!(
        baseline_trace.count(eredu_core::MODEL_LOGITS_OBSERVATION_PATH),
        2
    );
    assert_eq!(baseline_trace.count(PROPOSAL_PATH), total_drafts);
    assert!(baseline_trace.count(VERIFICATION_PATH) >= total_rounds);
    assert_eq!(
        baseline_trace.count(CAPTURE_PATH),
        baseline_trace.count(eredu_core::MODEL_LOGITS_OBSERVATION_PATH)
            + baseline_trace.count(VERIFICATION_PATH),
        "each target capture must be observed exactly once for each target forward",
    );
    let baseline_proposal = baseline_trace
        .proposal_logits
        .first()
        .expect("baseline assistant must produce proposal logits")
        .clone();
    drop(baseline_trace);

    let (capture_perturbed, _, capture_trace) = run(
        target.path(),
        assistant.path(),
        Some(CAPTURE_PATH),
        ExternalTensorIntervention::Zero,
        ExternalTensorIntervention::None,
    );
    capture_perturbed.unwrap();
    let capture_trace = capture_trace.lock().unwrap();
    assert_ne!(
        capture_trace
            .proposal_logits
            .first()
            .expect("capture-intervened assistant must produce proposal logits"),
        &baseline_proposal,
        "intervened target capture did not reach assistant proposal computation",
    );
    assert_eq!(
        capture_trace.count(CAPTURE_PATH),
        capture_trace.count(eredu_core::MODEL_LOGITS_OBSERVATION_PATH)
            + capture_trace.count(VERIFICATION_PATH),
    );
    drop(capture_trace);

    let (proposal_perturbed, _, proposal_trace) = run(
        target.path(),
        assistant.path(),
        None,
        ExternalTensorIntervention::None,
        ExternalTensorIntervention::ForceToken(31),
    );
    let proposal_perturbed = proposal_perturbed.unwrap();
    assert!(
        tokens(&proposal_perturbed) != baseline_tokens
            || accept_lens(&proposal_perturbed) != accept_lens(&baseline),
        "forced proposal logits did not affect verification or commit behavior",
    );
    assert_eq!(
        proposal_trace.lock().unwrap().count(PROPOSAL_PATH),
        proposal_perturbed
            .requests()
            .iter()
            .map(|request| request.stats().draft_tokens())
            .sum::<usize>(),
    );

    let (verification_perturbed, _, verification_trace) = run(
        target.path(),
        assistant.path(),
        Some(VERIFICATION_PATH),
        ExternalTensorIntervention::ForceToken(30),
        ExternalTensorIntervention::None,
    );
    let verification_perturbed = verification_perturbed.unwrap();
    assert_ne!(tokens(&verification_perturbed), baseline_tokens);
    assert!(verification_trace.lock().unwrap().count(VERIFICATION_PATH) > 0);

    let (final_perturbed, _, final_trace) = run(
        target.path(),
        assistant.path(),
        Some(eredu_core::MODEL_LOGITS_OBSERVATION_PATH),
        ExternalTensorIntervention::ForceToken(29),
        ExternalTensorIntervention::None,
    );
    let final_perturbed = final_perturbed.unwrap();
    assert!(final_perturbed
        .requests()
        .iter()
        .all(|request| request.token_ids().first() == Some(&29)),);
    assert_eq!(
        final_trace
            .lock()
            .unwrap()
            .count(eredu_core::MODEL_LOGITS_OBSERVATION_PATH),
        2,
    );

    let (failed, failed_publications, failed_trace) = run(
        target.path(),
        assistant.path(),
        Some(CAPTURE_PATH),
        ExternalTensorIntervention::Fail,
        ExternalTensorIntervention::None,
    );
    let error = match failed {
        Ok(_) => panic!("injected external observation failure unexpectedly succeeded"),
        Err(error) => error,
    };
    assert!(error.contains("injected external observation failure"));
    assert_eq!(
        failed_publications, 0,
        "failed observation published output"
    );
    let failed_trace = failed_trace.lock().unwrap();
    assert_eq!(failed_trace.count(CAPTURE_PATH), 1);
    assert_eq!(failed_trace.count(PROPOSAL_PATH), 0);
    assert_eq!(failed_trace.count(VERIFICATION_PATH), 0);
    assert_eq!(
        failed_trace.count(eredu_core::MODEL_LOGITS_OBSERVATION_PATH),
        0
    );

    let (verification_failed, verification_failed_publications, verification_failed_trace) = run(
        target.path(),
        assistant.path(),
        Some(VERIFICATION_PATH),
        ExternalTensorIntervention::Fail,
        ExternalTensorIntervention::None,
    );
    let error = match verification_failed {
        Ok(_) => panic!("injected verification observation failure unexpectedly succeeded"),
        Err(error) => error,
    };
    assert!(error.contains("injected external observation failure"));
    let verification_failed_trace = verification_failed_trace.lock().unwrap();
    assert_eq!(verification_failed_trace.count(VERIFICATION_PATH), 1);
    assert!(verification_failed_trace.count(PROPOSAL_PATH) > 0);
    assert_eq!(
        verification_failed_publications,
        verification_failed_trace.count(eredu_core::MODEL_LOGITS_OBSERVATION_PATH),
        "failed verification published beyond each lane's committed first token",
    );
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires local MLX Metal execution"]
fn public_gemma_external_factory_scheduler_supports_metal_target_cpu_assistant() {
    if !safemlx::metal::is_available().unwrap_or(false) {
        eprintln!("skipping Gemma external cross-device proof: MLX Metal is unavailable");
        return;
    }
    let target = tempfile::tempdir().unwrap();
    let assistant = tempfile::tempdir().unwrap();
    write_gemma_fixture(target.path());
    write_gemma_assistant_fixture(assistant.path());

    run_public_gemma_external_scheduler(
        target.path(),
        assistant.path(),
        DevicePlan::new("mlx", "metal:0").unwrap(),
        DraftPlacementPlan::Device {
            device: DevicePlan::new("mlx", "cpu:0").unwrap(),
        },
        SpeculativeExecutionTopology::CrossDeviceSplit,
    );
}

#[test]
fn gemma_external_assistant_capture_uses_neutral_target_and_rolls_back_failure() {
    use eredu_architectures::composite_execution::{
        ExternalPredictionCaptureRequest, ExternalPredictionTargetCapture,
        ExternalPredictionTargetOperation,
    };

    crate::composition::mlx::path_instrumentation::reset();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let checkpoint = tempfile::tempdir().unwrap();
    write_gemma_fixture(checkpoint.path());
    let backend = crate::native::backend(&stream, &stream);
    let model = load_model(&backend, checkpoint.path(), MlxLoadRequest::default())
        .unwrap()
        .into_inner();
    assert_eq!(
        crate::composition::mlx::path_instrumentation::snapshot().constructors,
        1
    );
    let mut session = MlxModelSession::from_model(
        model,
        eredu_core::SessionCapabilities::new(true, true, true),
    )
    .unwrap();
    let target = session
        .neutral_prediction_target_mut()
        .unwrap()
        .external_prediction_mut()
        .expect("Gemma target must expose the external-assistant capability");
    let mut cache = target.prepare_external_prediction_target_cache().unwrap();
    let initial_offset = cache.generation().unwrap();
    let tokens = Array::from_slice(&[1u32, 2], &[1, 2]);
    let parts = [text_input_part(&tokens)];
    let invalid = ExternalPredictionCaptureRequest::Gemma4SharedAttention {
        final_hidden_path: "model.language_model.layers.99.output".into(),
    };
    let error = target
        .prefill_external_prediction_target(
            crate::backend::runtime::media::input::ModelInput::new(&parts),
            &invalid,
            &mut cache,
        )
        .unwrap_err();
    assert!(error.to_string().contains("did not reach capture path"));
    assert_eq!(cache.generation().unwrap(), initial_offset);

    let request = ExternalPredictionCaptureRequest::Gemma4SharedAttention {
        final_hidden_path: "model.language_model.layers.3.output".into(),
    };
    let (logits, capture) = target
        .prefill_external_prediction_target(
            crate::backend::runtime::media::input::ModelInput::new(&parts),
            &request,
            &mut cache,
        )
        .unwrap();
    assert_eq!(logits.as_array().shape(), [1, 2, 32]);
    assert_eq!(cache.generation().unwrap(), 2);
    let ExternalPredictionTargetCapture::Gemma4 { hidden, shared_kv } = capture else {
        panic!("Gemma target returned the wrong external-assistant capture")
    };
    assert_eq!(hidden.as_array().shape(), [1, 2, 8]);
    assert!(!shared_kv.is_empty());
    assert!(shared_kv.iter().all(|(_, keys, values)| {
        keys.as_array().dim(-2) == 2 && values.as_array().dim(-2) == 2
    }));
    let proposal = MlxTensor::from_array(Array::from_slice(&[3u32], &[1, 1]));
    let embedding = target
        .apply_external_prediction_target_operation(
            ExternalPredictionTargetOperation::TokenEmbeddings(&proposal),
        )
        .unwrap();
    assert_eq!(embedding.as_array().shape(), [1, 1, 8]);
}

#[test]
fn muse_external_assistant_capture_uses_neutral_target_in_exact_layer_order() {
    use eredu_architectures::composite_execution::{
        ExternalPredictionCaptureRequest, ExternalPredictionTargetCapture,
        ExternalPredictionTargetOperation,
    };

    crate::composition::mlx::path_instrumentation::reset();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let checkpoint = tempfile::tempdir().unwrap();
    write_muse_glimmer_tensor_parallel_fixture(checkpoint.path());
    let backend = crate::native::backend(&stream, &stream);
    let model = load_model(&backend, checkpoint.path(), MlxLoadRequest::default())
        .unwrap()
        .into_inner();
    assert_eq!(
        crate::composition::mlx::path_instrumentation::snapshot().constructors,
        1
    );
    let mut session = MlxModelSession::from_model(
        model,
        eredu_core::SessionCapabilities::new(true, true, true),
    )
    .unwrap();
    let target = session
        .neutral_prediction_target_mut()
        .unwrap()
        .external_prediction_mut()
        .expect("Muse-Glimmer target must expose the external-assistant capability");
    let mut cache = target.prepare_external_prediction_target_cache().unwrap();
    let initial_offset = cache.generation().unwrap();
    let tokens = Array::from_slice(&[1u32, 2], &[1, 2]);
    let parts = [text_input_part(&tokens)];
    let invalid = ExternalPredictionCaptureRequest::MuseGlimmerDFlash {
        target_layers: vec![0, 1].into_boxed_slice(),
        target_paths: vec!["missing.0".into(), "missing.1".into()].into_boxed_slice(),
    };
    let error = target
        .prefill_external_prediction_target(
            crate::backend::runtime::media::input::ModelInput::new(&parts),
            &invalid,
            &mut cache,
        )
        .unwrap_err();
    assert!(error.to_string().contains("did not reach capture path"));
    assert_eq!(cache.generation().unwrap(), initial_offset);

    let request = ExternalPredictionCaptureRequest::MuseGlimmerDFlash {
        target_layers: vec![0, 1].into_boxed_slice(),
        target_paths: vec![
            "model.layers.0.output".into(),
            "model.layers.1.output".into(),
        ]
        .into_boxed_slice(),
    };
    let (logits, capture) = target
        .prefill_external_prediction_target(
            crate::backend::runtime::media::input::ModelInput::new(&parts),
            &request,
            &mut cache,
        )
        .unwrap();
    assert_eq!(logits.as_array().shape(), [1, 2, 32]);
    assert_eq!(cache.generation().unwrap(), 2);
    let ExternalPredictionTargetCapture::MuseGlimmerDFlash { target_states } = capture else {
        panic!("Muse-Glimmer target returned the wrong external-assistant capture")
    };
    assert_eq!(target_states.len(), 2);
    assert!(target_states
        .iter()
        .all(|state| state.as_array().shape() == [1, 2, 16]));
    let proposal = MlxTensor::from_array(Array::from_slice(&[3u32], &[1, 1]));
    let embedding = target
        .apply_external_prediction_target_operation(
            ExternalPredictionTargetOperation::TokenEmbeddings(&proposal),
        )
        .unwrap();
    assert_eq!(embedding.as_array().shape(), [1, 1, 16]);
    let projected = target
        .apply_external_prediction_target_operation(
            ExternalPredictionTargetOperation::ProjectLogits(&target_states[1]),
        )
        .unwrap();
    assert_eq!(projected.as_array().shape(), [1, 2, 32]);
}

#[test]
fn complete_family_adapters_return_final_output_interventions() {
    fn write_qwen(directory: &Path) {
        write_qwen_fixture(directory, "qwen3");
    }

    let fixtures: [(&str, fn(&Path)); 7] = [
        ("Llama", write_fixture),
        ("Qwen", write_qwen),
        ("DeepSeek", |directory| write_deepseek_fixture(directory, 2)),
        ("Gemma 4", write_gemma4_tensor_parallel_fixture),
        ("Inkling", write_inkling_fixture),
        ("Muse-Glimmer", write_muse_glimmer_tensor_parallel_fixture),
        ("Kimi Linear", write_kimi_linear_fixture),
    ];
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));

    for (family, write_fixture) in fixtures {
        struct ReplacingLogits {
            observed: bool,
        }

        impl eredu_runtime::ActivationObserver<MlxTensor, safemlx::error::Exception> for ReplacingLogits {
            fn observe(
                &mut self,
                path: &str,
                _value: &MlxTensor,
            ) -> Result<(), safemlx::error::Exception> {
                self.observed |= path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH;
                Ok(())
            }

            fn intervene(
                &mut self,
                path: &str,
                value: &MlxTensor,
            ) -> Result<Option<MlxTensor>, safemlx::error::Exception> {
                Ok(
                    (path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH).then(|| {
                        let value = value.as_array();
                        MlxTensor::from_array(Array::from_iter(
                            std::iter::repeat_n(41.0f32, value.size()),
                            value.shape(),
                        ))
                    }),
                )
            }
        }

        let checkpoint = tempfile::tempdir().unwrap();
        write_fixture(checkpoint.path());
        let backend = crate::native::backend(&stream, &stream);
        let model = load_model(&backend, checkpoint.path(), MlxLoadRequest::default())
            .unwrap_or_else(|error| panic!("{family} load failed: {error}"))
            .into_inner();
        let mut session = MlxModelSession::from_model(
            model,
            eredu_core::SessionCapabilities::new(true, true, true),
        )
        .unwrap();
        let tokens = Array::from_slice(&[1u32, 2], &[1, 2]);
        let parts = [text_input_part(&tokens)];
        let mut observer = ReplacingLogits { observed: false };
        let output = session
            .submit_prefill_with_observer(
                &backend,
                crate::backend::runtime::media::input::ModelInput::new(&parts).into(),
                &mut observer,
            )
            .unwrap_or_else(|error| panic!("{family} observed forward failed: {error}"));
        let output = output.wait().unwrap();
        assert!(observer.observed, "{family} did not report final logits");
        assert!(
            output
                .evaluated()
                .unwrap()
                .as_slice::<f32>()
                .iter()
                .all(|value| *value == 41.0),
            "{family} ignored the intervention"
        );
    }
}

fn resident_reference_for_prepared(
    checkpoint: &Path,
    prepared: &PreparedModelInput,
) -> (Vec<f32>, Vec<f32>) {
    let checkpoint = checkpoint.to_path_buf();
    let prepared = prepared.clone();
    std::thread::Builder::new()
        .name("resident-reference".into())
        .spawn(move || {
            let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
            resident_reference_for_prepared_inner(&checkpoint, &prepared, &stream)
        })
        .expect("resident-reference fixture thread")
        .join()
        .expect("resident-reference fixture thread panicked")
}

fn resident_reference_for_prepared_inner(
    checkpoint: &Path,
    prepared: &PreparedModelInput,
    stream: &Stream,
) -> (Vec<f32>, Vec<f32>) {
    let backend = crate::native::backend(stream, stream);
    let model = eredu_core::load_model(&backend, checkpoint, MlxLoadRequest::default())
        .unwrap()
        .into_inner();
    let mut session = MlxModelSession::from_model(
        model,
        eredu_core::SessionCapabilities::new(true, true, true),
    )
    .unwrap();
    let parts = prepared.input_parts();
    let prefill = session
        .prefill(
            &backend,
            crate::backend::runtime::media::input::ModelInput::new(parts).into(),
        )
        .unwrap()
        .wait()
        .unwrap()
        .into_logits()
        .unwrap()
        .into_array()
        .evaluated()
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    let token = Array::from_slice(&[0u32], &[1, 1]);
    let decode = session
        .decode(&backend, token)
        .unwrap()
        .wait()
        .unwrap()
        .into_logits()
        .unwrap()
        .into_array()
        .evaluated()
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    (prefill, decode)
}

fn assert_final_logits_close(actual: &Array, expected: &[f32], tolerance: f32) {
    let actual = actual.evaluated().unwrap();
    let values = actual.as_slice::<f32>();
    assert!(values.len() >= expected.len());
    let actual = &values[values.len() - expected.len()..];
    assert_eq!(actual.len(), expected.len());
    assert!(actual
        .iter()
        .zip(expected)
        .all(|(actual, expected)| (actual - expected).abs() <= tolerance),
        "pipeline logits diverged from the resident reference: actual={actual:?}, expected={expected:?}"
    );
}

struct ChildGuard {
    children: Vec<Child>,
}

impl ChildGuard {
    fn finish(mut self) -> Vec<Output> {
        self.children
            .drain(..)
            .map(|child| child.wait_with_output().unwrap())
            .collect()
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        for child in &mut self.children {
            let _ = child.kill();
        }
        for child in &mut self.children {
            let _ = child.wait();
        }
    }
}

fn write_f32_shard(path: &Path, tensors: &[(&str, Vec<usize>, f32)]) {
    let buffers = tensors
        .iter()
        .map(|(_, shape, value)| {
            let count = shape.iter().product::<usize>();
            (0..count)
                .flat_map(|_| value.to_le_bytes())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let views = tensors
        .iter()
        .zip(&buffers)
        .map(|((name, shape, _), bytes)| {
            (
                *name,
                TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
            )
        });
    serialize_to_file(views, None, path).unwrap();
}

fn write_fixture(directory: &Path) {
    write_llama_compatible_fixture(directory, "llama");
}

fn write_mistral_fixture(directory: &Path) {
    write_llama_compatible_fixture(directory, "mistral");
}

fn write_unindexed_llama_compatible_fixture(directory: &Path, model_type: &str) {
    write_llama_compatible_fixture(directory, model_type);
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut arrays = BTreeMap::new();
    for shard in [
        "input.safetensors",
        "layer-0.safetensors",
        "layer-1.safetensors",
        "output.safetensors",
    ] {
        arrays.extend(Array::load_safetensors(directory.join(shard), &stream).unwrap());
    }
    Array::save_safetensors(
        arrays.iter().map(|(name, value)| (name.as_str(), value)),
        None,
        directory.join("model.safetensors"),
    )
    .unwrap();
    std::fs::remove_file(directory.join("model.safetensors.index.json")).unwrap();
}

fn write_llama_compatible_gguf(path: &Path, architecture: &str) {
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let metadata = BTreeMap::from([
        (
            "general.architecture".into(),
            GgufMetadataValue::String(architecture.into()),
        ),
        ("general.file_type".into(), GgufMetadataValue::Uint32(0)),
        (key("block_count"), GgufMetadataValue::Uint32(2)),
        (key("embedding_length"), GgufMetadataValue::Uint32(4)),
        (key("attention.head_count"), GgufMetadataValue::Uint32(2)),
        (key("attention.head_count_kv"), GgufMetadataValue::Uint32(2)),
        (key("feed_forward_length"), GgufMetadataValue::Uint32(4)),
        (
            key("attention.layer_norm_rms_epsilon"),
            GgufMetadataValue::Float32(1e-5),
        ),
        (key("vocab_size"), GgufMetadataValue::Uint32(4)),
        (key("context_length"), GgufMetadataValue::Uint32(32)),
        (key("rope.freq_base"), GgufMetadataValue::Float32(10_000.0)),
    ]);
    let vector = vec![0_u8; 4 * std::mem::size_of::<f32>()];
    let matrix = vec![0_u8; 16 * std::mem::size_of::<f32>()];
    let vector_dimensions: &[u64] = &[4];
    let matrix_dimensions: &[u64] = &[4, 4];
    let mut names = vec![
        "token_embd.weight".to_owned(),
        "output_norm.weight".to_owned(),
    ];
    for layer in 0..2 {
        names.extend(
            [
                "attn_norm.weight",
                "ffn_norm.weight",
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            ]
            .map(|suffix| format!("blk.{layer}.{suffix}")),
        );
    }
    let tensors = names
        .iter()
        .map(|name| {
            let vector_tensor = name.ends_with("norm.weight");
            TensorInput {
                name,
                dimensions: if vector_tensor {
                    vector_dimensions
                } else {
                    matrix_dimensions
                },
                ggml_type: GgmlType::F32,
                data: if vector_tensor { &vector } else { &matrix },
            }
        })
        .collect::<Vec<_>>();
    Writer::default()
        .write(std::fs::File::create(path).unwrap(), &metadata, &tensors)
        .unwrap();
}

fn write_llama_compatible_fixture(directory: &Path, model_type: &str) {
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "model_type": model_type,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 64,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "head_dim": 8,
            "rms_norm_eps": 0.00001,
            "vocab_size": 64,
            "max_position_embeddings": 32,
            "tie_word_embeddings": false,
            "attention_bias": false,
            "mlp_bias": false,
            "attention_schedule": ["full", {"sliding": {"window": 2}}]
        }))
        .unwrap(),
    )
    .unwrap();
    write_f32_shard(
        &directory.join("input.safetensors"),
        &[("model.embed_tokens.weight", vec![64, 64], 0.01)],
    );
    for layer in 0..2 {
        let prefix = format!("model.layers.{layer}");
        let names = [
            (
                format!("{prefix}.self_attn.q_proj.weight"),
                vec![64, 64],
                0.01,
            ),
            (
                format!("{prefix}.self_attn.k_proj.weight"),
                vec![64, 64],
                0.01,
            ),
            (
                format!("{prefix}.self_attn.v_proj.weight"),
                vec![64, 64],
                0.01,
            ),
            (
                format!("{prefix}.self_attn.o_proj.weight"),
                vec![64, 64],
                0.01,
            ),
            (format!("{prefix}.mlp.gate_proj.weight"), vec![64, 64], 0.01),
            (format!("{prefix}.mlp.up_proj.weight"), vec![64, 64], 0.01),
            (format!("{prefix}.mlp.down_proj.weight"), vec![64, 64], 0.01),
            (format!("{prefix}.input_layernorm.weight"), vec![64], 1.0),
            (
                format!("{prefix}.post_attention_layernorm.weight"),
                vec![64],
                1.0,
            ),
        ];
        let borrowed = names
            .iter()
            .map(|(name, shape, value)| (name.as_str(), shape.clone(), *value))
            .collect::<Vec<_>>();
        write_f32_shard(
            &directory.join(format!("layer-{layer}.safetensors")),
            &borrowed,
        );
    }
    write_f32_shard(
        &directory.join("output.safetensors"),
        &[
            ("model.norm.weight", vec![64], 1.0),
            ("lm_head.weight", vec![64, 64], 0.01),
        ],
    );
    let mut weight_map = serde_json::Map::new();
    weight_map.insert(
        "model.embed_tokens.weight".into(),
        serde_json::json!("input.safetensors"),
    );
    for layer in 0..2 {
        for suffix in [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        ] {
            weight_map.insert(
                format!("model.layers.{layer}.{suffix}"),
                serde_json::json!(format!("layer-{layer}.safetensors")),
            );
        }
    }
    weight_map.insert(
        "model.norm.weight".into(),
        serde_json::json!("output.safetensors"),
    );
    weight_map.insert(
        "lm_head.weight".into(),
        serde_json::json!("output.safetensors"),
    );
    std::fs::write(
        directory.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({
            "metadata": {},
            "weight_map": weight_map
        }))
        .unwrap(),
    )
    .unwrap();
}

fn write_deepseek_fixture(directory: &Path, layers: i32) {
    write_deepseek_fixture_with_prediction(directory, layers, 0);
}

fn write_deepseek_fixture_with_prediction(directory: &Path, layers: i32, prediction_layers: i32) {
    let config = serde_json::json!({
        "model_type": "deepseek_v3",
        "hidden_size": 8,
        "intermediate_size": 16,
        "moe_intermediate_size": 4,
        "num_hidden_layers": layers,
        "num_attention_heads": 2,
        "vocab_size": 8,
        "rms_norm_eps": 0.000001,
        "max_position_embeddings": 64,
        "rope_theta": 10000.0,
        "q_lora_rank": null,
        "kv_lora_rank": 4,
        "qk_nope_head_dim": 2,
        "qk_rope_head_dim": 2,
        "v_head_dim": 2,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "n_group": 2,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "scoring_func": "sigmoid",
        "norm_topk_prob": true,
        "routed_scaling_factor": 1.0,
        "num_nextn_predict_layers": prediction_layers,
        "split_kv_b": false,
        "tie_word_embeddings": false
    });
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let args = eredu_architectures::deepseek::parse_v3_config(&config).unwrap();
    struct Collector<'a> {
        stream: &'a Stream,
        arrays: Vec<(String, Array)>,
    }
    impl<'tensor> ParameterVisitor<'tensor, MlxTensor> for Collector<'_> {
        fn visit(&mut self, metadata: ParameterMetadata, parameter: &'tensor MlxTensor) {
            let name = metadata.id.to_string();
            let shape = parameter.as_array().shape().to_vec();
            let value = if name.ends_with("norm.weight") {
                Array::ones::<f32>(&shape, self.stream).unwrap()
            } else {
                Array::full::<f32>(&shape, Array::from_f32(0.01), self.stream).unwrap()
            };
            self.arrays.push((name, value));
        }
    }
    type Backend = MlxNeuralBackend;
    let architecture =
        eredu_architectures::deepseek::v3::Model::<Backend>::new(args.clone(), stream).unwrap();
    let mut collector = Collector {
        stream,
        arrays: Vec::new(),
    };
    architecture
        .static_modules()
        .visit_parameters(&mut collector);
    for layer in 0..usize::try_from(layers).unwrap() {
        eredu_architectures::deepseek::block::V3Block::<Backend>::new(&args, layer, stream)
            .unwrap()
            .visit_parameters(&mut collector);
    }
    for depth in 0..usize::try_from(prediction_layers).unwrap() {
        eredu_architectures::deepseek::mtp::V3PredictionLayer::<Backend>::new(&args, depth, stream)
            .unwrap()
            .visit_parameters(&mut collector);
    }
    let mut arrays = Vec::new();
    let width = args.moe_intermediate_size;
    for (name, value) in collector.arrays {
        if let Some(prefix) = name.strip_suffix(".experts.gate_up_proj") {
            for expert in 0..args.n_routed_experts {
                let packed = value.try_index_device(expert, stream).unwrap();
                arrays.push((
                    format!("{prefix}.experts.{expert}.gate_proj.weight"),
                    packed.try_index_device((0..width, ..), stream).unwrap(),
                ));
                arrays.push((
                    format!("{prefix}.experts.{expert}.up_proj.weight"),
                    packed
                        .try_index_device((width..2 * width, ..), stream)
                        .unwrap(),
                ));
            }
        } else if let Some(prefix) = name.strip_suffix(".experts.down_proj") {
            for expert in 0..args.n_routed_experts {
                arrays.push((
                    format!("{prefix}.experts.{expert}.down_proj.weight"),
                    value.try_index_device(expert, stream).unwrap(),
                ));
            }
        } else {
            arrays.push((name.to_string(), value));
        }
    }
    Array::save_safetensors(
        arrays.iter().map(|(name, value)| (name.as_str(), value)),
        None,
        directory.join("model-00001-of-00001.safetensors"),
    )
    .unwrap();
    let weight_map = arrays
        .iter()
        .map(|(name, _)| (name.clone(), "model-00001-of-00001.safetensors".to_owned()))
        .collect::<BTreeMap<_, _>>();
    std::fs::write(
        directory.join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(&serde_json::json!({ "weight_map": weight_map })).unwrap(),
    )
    .unwrap();
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
    assert!(arrays
        .iter()
        .all(|(_, value)| value.dtype() == MlxDtype::Float32));
}

pub(crate) fn write_deepseek_v4_fixture(directory: &Path, prediction_layers: u64) {
    write_deepseek_v4_fixture_kind(directory, prediction_layers, false)
}

pub(crate) fn write_deepseek_v4_dspark_fixture(directory: &Path) {
    write_deepseek_v4_fixture_kind(directory, 1, true)
}

fn write_deepseek_v4_fixture_kind(directory: &Path, prediction_layers: u64, dspark: bool) {
    let compress_ratios = if prediction_layers == 0 {
        vec![0, 4]
    } else {
        vec![0, 4, 0]
    };
    let mut config = serde_json::json!({
        "model_type": "deepseek_v4",
        "hidden_size": 16,
        "moe_intermediate_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "qk_rope_head_dim": 4,
        "q_lora_rank": 8,
        "o_lora_rank": 8,
        "o_groups": 2,
        "vocab_size": 16,
        "rms_norm_eps": 0.000001,
        "max_position_embeddings": 64,
        "sliding_window": 8,
        "compress_ratios": compress_ratios,
        "index_n_heads": 2,
        "index_head_dim": 4,
        "index_topk": 2,
        "hc_mult": 2,
        "hc_sinkhorn_iters": 2,
        "hc_eps": 0.000001,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 1,
        "num_hash_layers": 1,
        "norm_topk_prob": true,
        "routed_scaling_factor": 1.0,
        "num_nextn_predict_layers": prediction_layers
    });
    if dspark {
        config["dspark_block_size"] = 2.into();
        config["dspark_noise_token_id"] = 0.into();
        config["dspark_target_layer_ids"] = serde_json::json!([0, 1]);
        config["dspark_markov_rank"] = 4.into();
    }
    let args = eredu_architectures::deepseek::parse_v4_config(&config).unwrap();
    let plan = eredu_architectures::deepseek::v4_safetensors_plan(&args).unwrap();
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let arrays = plan
        .common_tensors
        .iter()
        .map(|tensor| {
            let shape = tensor
                .shape
                .iter()
                .map(|dimension| i32::try_from(*dimension).unwrap())
                .collect::<Vec<_>>();
            let value = if matches!(
                tensor.dtype,
                eredu_checkpoint::schema::StoredDtypeConstraint::Exact(
                    eredu_checkpoint::StoredDtype::I32
                )
            ) {
                Array::zeros::<i32>(&shape, stream).unwrap()
            } else if tensor.key.ends_with("norm.weight") {
                Array::ones::<f32>(&shape, stream).unwrap()
            } else {
                Array::full::<f32>(&shape, Array::from_f32(0.01), stream).unwrap()
            };
            (tensor.key.clone(), value)
        })
        .collect::<Vec<_>>();
    Array::save_safetensors(
        arrays.iter().map(|(name, value)| (name.as_str(), value)),
        None,
        directory.join("model.safetensors"),
    )
    .unwrap();
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
}

fn gemma_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "gemma4",
        "tie_word_embeddings": true,
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 8,
            "num_hidden_layers": 4,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "rms_norm_eps": 0.000001,
            "vocab_size": 32,
            "pad_token_id": 0,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "rope_theta": 10000.0,
            "head_dim": 4,
            "attention_bias": false,
            "hidden_size_per_layer_input": 4,
            "vocab_size_per_layer_input": 32,
            "num_kv_shared_layers": 1,
            "layer_types": [
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention"
            ],
            "sliding_window": 8,
            "final_logit_softcapping": 4.0
        }
    })
}

fn write_gemma_fixture(directory: &Path) {
    let config = gemma_config();
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let args = eredu_architectures::gemma4::FamilyConfig::from_hf_json(
        &serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    type Architecture =
        eredu_architectures::gemma4::LayeredModel<crate::backend::nn::shared::MlxNeuralBackend>;
    type State = crate::backend::runtime::cache::state::MlxHybridState;
    struct Collector<'a> {
        stream: &'a Stream,
        arrays: Vec<(String, Array)>,
    }
    impl<'tensor> ParameterVisitor<'tensor, MlxTensor> for Collector<'_> {
        fn visit(&mut self, metadata: ParameterMetadata, parameter: &'tensor MlxTensor) {
            let parameter = parameter.as_array();
            let value = if metadata.id.as_str().ends_with("norm.weight") {
                Array::ones::<f32>(parameter.shape(), self.stream).unwrap()
            } else {
                Array::full::<f32>(parameter.shape(), Array::from_f32(0.01), self.stream).unwrap()
            };
            self.arrays.push((metadata.id.to_string(), value));
        }
    }
    let architecture = Architecture::new(args, stream).unwrap();
    let mut collector = Collector {
        stream,
        arrays: Vec::new(),
    };
    <Architecture as eredu_runtime::LayeredArchitecture<
        crate::backend::nn::shared::MlxNeuralBackend,
        State,
    >>::static_modules(&architecture)
    .visit_parameters(&mut collector);
    for group in 0..3 {
        let count = <Architecture as eredu_runtime::LayeredArchitecture<
            crate::backend::nn::shared::MlxNeuralBackend,
            State,
        >>::group_unit_count(&architecture, group)
        .unwrap();
        for index in 0..count {
            <Architecture as eredu_runtime::LayeredArchitecture<
                crate::backend::nn::shared::MlxNeuralBackend,
                State,
            >>::build_unit(&architecture, group, index, stream)
            .unwrap()
            .visit_parameters(&mut collector);
        }
    }
    let arrays = collector.arrays;
    Array::save_safetensors(
        arrays.iter().map(|(name, value)| (name.as_str(), value)),
        None,
        directory.join("model.safetensors"),
    )
    .unwrap();
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
    assert!(arrays
        .iter()
        .all(|(_, value)| value.dtype() == MlxDtype::Float32));
}

fn write_gemma_assistant_fixture(directory: &Path) {
    let config = serde_json::json!({
        "model_type": "gemma4_assistant",
        "backbone_hidden_size": 8,
        "use_ordered_embeddings": false,
        "tie_word_embeddings": true,
        "block_size": 3,
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "rms_norm_eps": 0.000001,
            "vocab_size": 32,
            "max_position_embeddings": 128,
            "rope_theta": 10000.0,
            "tie_word_embeddings": true,
            "attention_k_eq_v": false,
            "layer_types": ["full_attention"]
        }
    });
    let config_bytes = serde_json::to_vec_pretty(&config).unwrap();
    let assistant = eredu_architectures::gemma4::AssistantConfig::from_json(&config_bytes).unwrap();
    let plan = eredu_architectures::gemma4::assistant_safetensors_plan(&assistant).unwrap();
    assert!(plan.layout_groups.is_empty());

    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let arrays = plan
        .common_tensors
        .iter()
        .map(|tensor| {
            let shape = tensor
                .shape
                .iter()
                .copied()
                .map(i32::try_from)
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            let value = if tensor.key.ends_with("norm.weight") {
                Array::ones::<f32>(&shape, stream).unwrap()
            } else {
                Array::full::<f32>(&shape, Array::from_f32(0.01), stream).unwrap()
            };
            (tensor.key.clone(), value)
        })
        .collect::<Vec<_>>();
    Array::save_safetensors(
        arrays.iter().map(|(name, value)| (name.as_str(), value)),
        None,
        directory.join("model.safetensors"),
    )
    .unwrap();
    std::fs::write(directory.join("config.json"), config_bytes).unwrap();
}

fn qwen_config(model_type: &str) -> serde_json::Value {
    let is_moe = model_type == "qwen3_moe";
    let mut config = serde_json::json!({
        "architectures": [match model_type {
            "qwen2" => "Qwen2ForCausalLM",
            "qwen3" => "Qwen3ForCausalLM",
            "qwen3_moe" => "Qwen3MoeForCausalLM",
            _ => panic!("unsupported Qwen pipeline fixture model type {model_type}"),
        }],
        "model_type": model_type,
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "intermediate_size": if is_moe { 0 } else { 64 },
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "rms_norm_eps": 0.000001,
        "vocab_size": 32,
        "max_position_embeddings": 128,
        "rope_theta": 10000.0,
        "tie_word_embeddings": false,
        "attention_bias": model_type == "qwen2",
        "mlp_bias": false,
        "moe_intermediate_size": if is_moe { 32 } else { 0 },
        "num_experts": if is_moe { 4 } else { 0 },
        "num_experts_per_tok": if is_moe { 2 } else { 0 },
        "norm_topk_prob": is_moe
    });
    if model_type == "qwen2" {
        config["use_sliding_window"] = serde_json::json!(true);
        config["sliding_window"] = serde_json::json!(3);
        config["max_window_layers"] = serde_json::json!(1);
    }
    config
}

fn write_qwen_fixture(directory: &Path, model_type: &str) {
    write_qwen_fixture_with_tied_head(directory, model_type, false);
}

fn write_indexed_qwen_fixture(directory: &Path, model_type: &str) {
    write_qwen_fixture(directory, model_type);
    let source = directory.join("model.safetensors");
    let shard = directory.join("model-00001-of-00001.safetensors");
    std::fs::rename(source, &shard).unwrap();
    let bytes = std::fs::read(&shard).unwrap();
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let weight_map = tensors
        .names()
        .into_iter()
        .map(|name| (name.to_owned(), "model-00001-of-00001.safetensors"))
        .collect::<BTreeMap<_, _>>();
    std::fs::write(
        directory.join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(&serde_json::json!({ "weight_map": weight_map })).unwrap(),
    )
    .unwrap();
}

fn write_qwen_fixture_with_tied_head(directory: &Path, model_type: &str, tied: bool) {
    let mut config = qwen_config(model_type);
    config["tie_word_embeddings"] = serde_json::json!(tied);
    write_qwen_config_fixture(directory, config);
}

fn write_qwen_requantized_tp_fixture(directory: &Path) {
    let mut config = qwen_config("qwen3");
    config["hidden_size"] = serde_json::json!(64);
    config["num_attention_heads"] = serde_json::json!(8);
    config["num_key_value_heads"] = serde_json::json!(4);
    config["intermediate_size"] = serde_json::json!(128);
    config["vocab_size"] = serde_json::json!(64);
    write_qwen_config_fixture(directory, config);
}

fn qwen_fixture_arrays(
    args: &eredu_architectures::qwen::ModelArgs,
    stream: &Stream,
) -> Vec<(String, Array)> {
    struct Collector<'a> {
        stream: &'a Stream,
        arrays: Vec<(String, Array)>,
    }
    impl<'tensor> ParameterVisitor<'tensor, MlxTensor> for Collector<'_> {
        fn visit(&mut self, metadata: ParameterMetadata, parameter: &'tensor MlxTensor) {
            let name = metadata.id.to_string();
            let shape = parameter.as_array().shape().to_vec();
            let value = if name.ends_with("norm.weight") {
                Array::ones::<f32>(&shape, self.stream).unwrap()
            } else {
                let ordinal = name.bytes().fold(0u32, |sum, byte| sum + u32::from(byte)) % 17;
                Array::full::<f32>(
                    &shape,
                    Array::from_f32(0.002 + ordinal as f32 * 0.0003),
                    self.stream,
                )
                .unwrap()
            };
            self.arrays.push((name, value));
        }
    }

    let architecture = eredu_architectures::qwen::RoutedLayeredModel::<MlxNeuralBackend>::new(
        args.clone(),
        stream,
    )
    .unwrap();
    let mut collector = Collector {
        stream,
        arrays: Vec::new(),
    };
    architecture
        .static_modules()
        .visit_parameters(&mut collector);
    for layer in 0..args.num_hidden_layers as usize {
        eredu_architectures::qwen::new_routed_block::<MlxNeuralBackend>(args, layer, stream)
            .unwrap()
            .visit_parameters(&mut collector);
    }
    collector.arrays
}

fn write_qwen_config_fixture(directory: &Path, config: serde_json::Value) {
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let args = eredu_architectures::qwen::model_args_from_config_value(&config).unwrap();
    let arrays = qwen_fixture_arrays(&args, stream);
    Array::save_safetensors(
        arrays.iter().map(|(name, value)| (name.as_str(), value)),
        None,
        directory.join("model.safetensors"),
    )
    .unwrap();
}

fn write_qwen3_moe_gguf_fixture(path: &Path) {
    write_qwen_gguf_fixture(path, "qwen3_moe");
}

fn write_qwen_gguf_fixture(path: &Path, model_type: &str) {
    let config = qwen_config(model_type);
    std::fs::write(
        path.parent().unwrap().join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let args = eredu_architectures::qwen::model_args_from_config_value(&config).unwrap();
    let arrays = qwen_fixture_arrays(&args, stream);
    let mut specs = Vec::new();
    for (runtime_name, value) in &arrays {
        let runtime_name = runtime_name.to_string();
        if let Some(prefix) = runtime_name.strip_suffix(".mlp.experts.gate_up_proj") {
            let gate = value
                .try_index_device((.., ..args.moe_intermediate_size, ..), stream)
                .unwrap();
            let up = value
                .try_index_device((.., args.moe_intermediate_size.., ..), stream)
                .unwrap();
            let prefix = prefix.replace("model.layers.", "blk.");
            specs.push(gguf_tensor_from_array(
                format!("{prefix}.ffn_gate_exps.weight"),
                &gate,
            ));
            specs.push(gguf_tensor_from_array(
                format!("{prefix}.ffn_up_exps.weight"),
                &up,
            ));
            continue;
        }
        if let Some(prefix) = runtime_name.strip_suffix(".mlp.experts.down_proj") {
            specs.push(gguf_tensor_from_array(
                format!(
                    "{}.ffn_down_exps.weight",
                    prefix.replace("model.layers.", "blk.")
                ),
                value,
            ));
            continue;
        }
        let name = runtime_name
            .replace("model.layers.", "blk.")
            .replace("self_attn.q_norm", "attn_q_norm")
            .replace("self_attn.k_norm", "attn_k_norm")
            .replace("self_attn.q_proj", "attn_q")
            .replace("self_attn.k_proj", "attn_k")
            .replace("self_attn.v_proj", "attn_v")
            .replace("self_attn.o_proj", "attn_output")
            .replace("input_layernorm", "attn_norm")
            .replace("post_attention_layernorm", "ffn_norm")
            .replace("mlp.gate.weight", "ffn_gate_inp.weight")
            .replace("mlp.gate_proj", "ffn_gate")
            .replace("mlp.up_proj", "ffn_up")
            .replace("mlp.down_proj", "ffn_down")
            .replace("model.embed_tokens", "token_embd")
            .replace("model.norm", "output_norm")
            .replace("lm_head", "output");
        specs.push(gguf_tensor_from_array(name, value));
    }
    let architecture = match model_type {
        "qwen2" => "qwen2",
        "qwen3" => "qwen3",
        "qwen3_moe" => "qwen3moe",
        _ => panic!("unsupported Qwen GGUF fixture model type {model_type}"),
    };
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let mut metadata = BTreeMap::from([
        (
            "general.architecture".into(),
            GgufMetadataValue::String(architecture.into()),
        ),
        ("general.file_type".into(), GgufMetadataValue::Uint32(0)),
        (
            key("embedding_length"),
            GgufMetadataValue::Uint32(args.hidden_size as u32),
        ),
        (
            key("block_count"),
            GgufMetadataValue::Uint32(args.num_hidden_layers as u32),
        ),
        (
            key("attention.head_count"),
            GgufMetadataValue::Uint32(args.num_attention_heads as u32),
        ),
        (
            key("attention.head_count_kv"),
            GgufMetadataValue::Uint32(args.num_key_value_heads as u32),
        ),
        (
            key("attention.key_length"),
            GgufMetadataValue::Uint32(args.head_dim as u32),
        ),
        (
            key("attention.layer_norm_rms_epsilon"),
            GgufMetadataValue::Float32(args.rms_norm_eps),
        ),
        (
            key("context_length"),
            GgufMetadataValue::Uint32(args.max_position_embeddings as u32),
        ),
        (
            key("rope.freq_base"),
            GgufMetadataValue::Float32(args.rope_theta),
        ),
        (
            key("vocab_size"),
            GgufMetadataValue::Uint32(args.vocab_size as u32),
        ),
    ]);
    if args.is_moe() {
        metadata.insert(
            key("expert_feed_forward_length"),
            GgufMetadataValue::Uint32(args.moe_intermediate_size as u32),
        );
        metadata.insert(
            key("expert_count"),
            GgufMetadataValue::Uint32(args.num_experts as u32),
        );
        metadata.insert(
            key("expert_used_count"),
            GgufMetadataValue::Uint32(args.num_experts_per_tok as u32),
        );
    } else {
        metadata.insert(
            key("feed_forward_length"),
            GgufMetadataValue::Uint32(args.intermediate_size as u32),
        );
    }
    let tensors = specs
        .iter()
        .map(|tensor| TensorInput {
            name: &tensor.name,
            dimensions: &tensor.dimensions,
            ggml_type: GgmlType::F32,
            data: &tensor.data,
        })
        .collect::<Vec<_>>();
    Writer::default()
        .write(std::fs::File::create(path).unwrap(), &metadata, &tensors)
        .unwrap();
}

fn write_gpt_oss_fixture(directory: &Path) {
    let config = serde_json::json!({
        "model_type": "gpt_oss",
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "vocab_size": 64,
        "num_local_experts": 2,
        "num_experts_per_tok": 1,
        "rms_norm_eps": 0.00001,
        "sliding_window": 3,
        "max_position_embeddings": 128,
        "rope_theta": 150000.0,
        "layer_types": ["sliding_attention", "full_attention"],
        "quantization_config": {"quant_method": "mxfp4"},
        "swiglu_limit": 7.0
    });
    let args = gpt_oss::model_args_from_config_value(&config).unwrap();
    let plan = gpt_oss::safetensors_plan(&args).unwrap();
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let arrays = plan
        .common_tensors
        .iter()
        .map(|tensor| {
            let shape = tensor
                .shape
                .iter()
                .map(|dimension| i32::try_from(*dimension).unwrap())
                .collect::<Vec<_>>();
            let value = match &tensor.dtype {
                eredu_checkpoint::schema::StoredDtypeConstraint::Exact(
                    eredu_checkpoint::StoredDtype::U8,
                ) if tensor.key.ends_with("_scales") => {
                    Array::full::<u8>(&shape, Array::from_slice(&[127u8], &[]), stream).unwrap()
                }
                eredu_checkpoint::schema::StoredDtypeConstraint::Exact(
                    eredu_checkpoint::StoredDtype::U8,
                ) => Array::full::<u8>(&shape, Array::from_slice(&[0x11u8], &[]), stream).unwrap(),
                _ if tensor.key.ends_with("norm.weight") => {
                    Array::ones::<f32>(&shape, stream).unwrap()
                }
                _ => {
                    let ordinal = tensor
                        .key
                        .bytes()
                        .fold(0u32, |sum, byte| sum + u32::from(byte))
                        % 17;
                    Array::full::<f32>(
                        &shape,
                        Array::from_f32(0.002 + ordinal as f32 * 0.0003),
                        stream,
                    )
                    .unwrap()
                }
            };
            (tensor.key.clone(), value)
        })
        .collect::<Vec<_>>();
    Array::save_safetensors(
        arrays.iter().map(|(name, value)| (name.as_str(), value)),
        None,
        directory.join("model.safetensors"),
    )
    .unwrap();
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "requires local MLX Metal execution"]
fn replicated_inspection_dispatches_gpt_oss_and_nemotron_h_observers() {
    fn inspect(
        write_fixture: impl FnOnce(&Path) -> PathBuf,
        options: MlxLoadRequest,
        expected_observation: &str,
    ) {
        let checkpoint = tempfile::tempdir().unwrap();
        let artifact = write_fixture(checkpoint.path());
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = execution.stream();
        let backend = crate::native::backend(stream, stream);
        let model = load_model(&backend, artifact, options).unwrap();
        let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
        let tokens = Array::from_slice(&[1_u32, 2], &[1, 2]);
        let parts = [text_input_part(&tokens)];
        let input = crate::composition::mlx::MlxModelInput::from(
            crate::backend::runtime::media::input::ModelInput::new(&parts),
        );
        let inspected = runtime
            .inspect_prefill(input, &ObservationRequest::all())
            .unwrap();
        assert!(
            inspected.observations.get(expected_observation).is_some(),
            "missing {expected_observation:?} in {:?}",
            inspected.observations
        );
        let decode = Array::from_slice(&[3_u32], &[1, 1]);
        let inspected = runtime
            .inspect_decode(decode, &ObservationRequest::all())
            .unwrap();
        assert!(
            inspected.observations.get(expected_observation).is_some(),
            "decode missing {expected_observation:?} in {:?}",
            inspected.observations
        );
    }

    inspect(
        |directory| {
            write_gpt_oss_fixture(directory);
            directory.to_path_buf()
        },
        MlxLoadRequest::default().with_weight_residency(
            WeightResidency::with_independent_parameter_banks(
                OrdinaryWeightResidency::FullyResident,
                ParameterBankLoadOptions::default(),
            ),
        ),
        "model.layers.0.output",
    );
    inspect(
        |directory| {
            write_nemotron_fixture(directory);
            directory.to_path_buf()
        },
        MlxLoadRequest::default(),
        "model.layers.0.output",
    );
}

struct QuantizedGgufFixtureTensor {
    name: String,
    dimensions: Vec<u64>,
    ggml_type: GgmlType,
    data: Vec<u8>,
}

fn mxfp4_payload(elements: u64, phase: usize) -> Vec<u8> {
    assert_eq!(elements % 32, 0);
    let mut data = Vec::with_capacity((elements / 32) as usize * 17);
    for block in 0..elements / 32 {
        data.push(127 + ((block as usize + phase) % 3) as u8);
        data.extend((0..16).map(|index| {
            let low = ((index + phase) % 7 + 1) as u8;
            let high = ((index * 3 + phase) % 7 + 1) as u8;
            low | (high << 4)
        }));
    }
    data
}

pub(crate) fn write_gpt_oss_gguf_fixture(path: &Path) {
    let metadata = BTreeMap::from([
        (
            "general.architecture".into(),
            GgufMetadataValue::String("gpt-oss".into()),
        ),
        ("general.file_type".into(), GgufMetadataValue::Uint32(39)),
        (
            "gpt-oss.embedding_length".into(),
            GgufMetadataValue::Uint32(64),
        ),
        ("gpt-oss.block_count".into(), GgufMetadataValue::Uint32(2)),
        (
            "gpt-oss.expert_feed_forward_length".into(),
            GgufMetadataValue::Uint32(96),
        ),
        (
            "gpt-oss.attention.head_count".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "gpt-oss.attention.head_count_kv".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "gpt-oss.attention.key_length".into(),
            GgufMetadataValue::Uint32(32),
        ),
        (
            "gpt-oss.attention.layer_norm_rms_epsilon".into(),
            GgufMetadataValue::Float32(0.00001),
        ),
        (
            "gpt-oss.attention.sliding_window".into(),
            GgufMetadataValue::Uint32(3),
        ),
        (
            "gpt-oss.context_length".into(),
            GgufMetadataValue::Uint32(128),
        ),
        (
            "gpt-oss.rope.freq_base".into(),
            GgufMetadataValue::Float32(150000.0),
        ),
        ("gpt-oss.expert_count".into(), GgufMetadataValue::Uint32(2)),
        (
            "gpt-oss.expert_used_count".into(),
            GgufMetadataValue::Uint32(1),
        ),
        ("gpt-oss.vocab_size".into(), GgufMetadataValue::Uint32(64)),
    ]);
    let f32_tensor = |name: String, dimensions: Vec<u64>, phase: usize| {
        let values = patterned_values(
            usize::try_from(dimensions.iter().product::<u64>()).unwrap(),
            0.003,
            phase,
        );
        QuantizedGgufFixtureTensor {
            name,
            dimensions,
            ggml_type: GgmlType::F32,
            data: values.into_iter().flat_map(f32::to_le_bytes).collect(),
        }
    };
    let mxfp4_tensor = |name: String, dimensions: Vec<u64>, phase: usize| {
        let elements = dimensions.iter().product();
        QuantizedGgufFixtureTensor {
            name,
            dimensions,
            ggml_type: GgmlType::MxFp4,
            data: mxfp4_payload(elements, phase),
        }
    };
    let mut tensors = vec![f32_tensor("token_embd.weight".into(), vec![64, 64], 0)];
    for layer in 0..2 {
        let prefix = format!("blk.{layer}");
        let phase = layer * 20;
        tensors.extend([
            f32_tensor(format!("{prefix}.attn_norm.weight"), vec![64], phase + 1),
            f32_tensor(
                format!("{prefix}.attn_post_norm.weight"),
                vec![64],
                phase + 2,
            ),
            f32_tensor(format!("{prefix}.attn_q.weight"), vec![64, 128], phase + 3),
            f32_tensor(format!("{prefix}.attn_q.bias"), vec![128], phase + 4),
            f32_tensor(format!("{prefix}.attn_k.weight"), vec![64, 64], phase + 5),
            f32_tensor(format!("{prefix}.attn_k.bias"), vec![64], phase + 6),
            f32_tensor(format!("{prefix}.attn_v.weight"), vec![64, 64], phase + 7),
            f32_tensor(format!("{prefix}.attn_v.bias"), vec![64], phase + 8),
            f32_tensor(
                format!("{prefix}.attn_output.weight"),
                vec![128, 64],
                phase + 9,
            ),
            f32_tensor(format!("{prefix}.attn_output.bias"), vec![64], phase + 10),
            f32_tensor(format!("{prefix}.attn_sinks.weight"), vec![4], phase + 11),
            f32_tensor(
                format!("{prefix}.ffn_gate_inp.weight"),
                vec![64, 2],
                phase + 12,
            ),
            f32_tensor(format!("{prefix}.ffn_gate_inp.bias"), vec![2], phase + 13),
            mxfp4_tensor(
                format!("{prefix}.ffn_gate_exps.weight"),
                vec![64, 96, 2],
                phase + 14,
            ),
            f32_tensor(
                format!("{prefix}.ffn_gate_exps.bias"),
                vec![96, 2],
                phase + 15,
            ),
            mxfp4_tensor(
                format!("{prefix}.ffn_up_exps.weight"),
                vec![64, 96, 2],
                phase + 16,
            ),
            f32_tensor(
                format!("{prefix}.ffn_up_exps.bias"),
                vec![96, 2],
                phase + 17,
            ),
            mxfp4_tensor(
                format!("{prefix}.ffn_down_exps.weight"),
                vec![96, 64, 2],
                phase + 18,
            ),
            f32_tensor(
                format!("{prefix}.ffn_down_exps.bias"),
                vec![64, 2],
                phase + 19,
            ),
        ]);
    }
    tensors.extend([
        f32_tensor("output_norm.weight".into(), vec![64], 41),
        f32_tensor("output.weight".into(), vec![64, 64], 42),
    ]);
    let inputs = tensors
        .iter()
        .map(|tensor| TensorInput {
            name: &tensor.name,
            dimensions: &tensor.dimensions,
            ggml_type: tensor.ggml_type,
            data: &tensor.data,
        })
        .collect::<Vec<_>>();
    Writer::default()
        .write(std::fs::File::create(path).unwrap(), &metadata, &inputs)
        .unwrap();
}

fn write_lfm2_pipeline_fixture(directory: &Path, moe: bool) {
    let config = serde_json::json!({
        "model_type": if moe { "lfm2_moe" } else { "lfm2" },
        "architectures": [if moe { "Lfm2MoeForCausalLM" } else { "Lfm2ForCausalLM" }],
        "vocab_size": 16,
        "hidden_size": 12,
        "intermediate_size": 17,
        "num_hidden_layers": 2,
        "num_attention_heads": 6,
        "num_key_value_heads": 3,
        "max_position_embeddings": 64,
        "norm_eps": 0.00001,
        "layer_types": ["conv", "full_attention"],
        "conv_L_cache": 3,
        "conv_bias": true,
        "block_auto_adjust_ff_dim": false,
        "tie_word_embeddings": false,
        "moe_intermediate_size": if moe { 9 } else { 0 },
        "num_dense_layers": if moe { 1 } else { 0 },
        "num_experts": if moe { 2 } else { 0 },
        "num_experts_per_tok": if moe { 1 } else { 0 },
        "norm_topk_prob": moe,
        "use_expert_bias": moe
    });
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let args = eredu_architectures::lfm2::model_args_from_config_value(&config).unwrap();
    let mut model =
        MlxModule::new(checkpoint_fixtures::Lfm2CheckpointTemplate::new(args, stream).unwrap());
    for (name, parameter) in neutral_parameter_refs_mut(&mut model).flatten() {
        let shape = parameter.shape().to_vec();
        *parameter = if name.ends_with("norm.weight") {
            Array::ones::<f32>(&shape, stream).unwrap()
        } else {
            let ordinal = name.bytes().fold(0u32, |sum, byte| sum + u32::from(byte)) % 17;
            Array::full::<f32>(
                &shape,
                Array::from_f32(0.002 + ordinal as f32 * 0.0003),
                stream,
            )
            .unwrap()
        };
    }
    let arrays = neutral_parameter_refs(&model, false)
        .flatten()
        .into_iter()
        .map(|(name, value)| (name.to_string(), value.clone()))
        .collect::<Vec<_>>();
    Array::save_safetensors(
        arrays.iter().map(|(name, value)| (name.as_str(), value)),
        None,
        directory.join("model-00001-of-00001.safetensors"),
    )
    .unwrap();
    let weight_map = arrays
        .iter()
        .map(|(name, _)| (name.clone(), "model-00001-of-00001.safetensors".to_owned()))
        .collect::<BTreeMap<_, _>>();
    std::fs::write(
        directory.join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(&serde_json::json!({ "weight_map": weight_map })).unwrap(),
    )
    .unwrap();
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
}

fn write_lfm2_moe_gguf_fixture(path: &Path) {
    let config = serde_json::json!({
        "model_type": "lfm2_moe",
        "architectures": ["Lfm2MoeForCausalLM"],
        "vocab_size": 16,
        "hidden_size": 12,
        "intermediate_size": 17,
        "num_hidden_layers": 2,
        "num_attention_heads": 6,
        "num_key_value_heads": 3,
        "max_position_embeddings": 64,
        "norm_eps": 0.00001,
        "layer_types": ["conv", "full_attention"],
        "conv_L_cache": 3,
        "conv_bias": true,
        "block_auto_adjust_ff_dim": false,
        "tie_word_embeddings": false,
        "moe_intermediate_size": 9,
        "num_dense_layers": 1,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "norm_topk_prob": true,
        "use_expert_bias": true
    });
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let args = eredu_architectures::lfm2::model_args_from_config_value(&config).unwrap();
    let mut model = MlxModule::new(
        checkpoint_fixtures::Lfm2CheckpointTemplate::new(args.clone(), stream).unwrap(),
    );
    initialize_fixture(&mut model, stream);
    let mut specs = Vec::new();
    for (runtime_name, value) in neutral_parameter_refs(&model, false).flatten() {
        let runtime_name = runtime_name.to_string();
        let layer_name = |name: &str| {
            name.replace("model.layers.", "blk.")
                .replace(".conv.conv.", ".shortconv.conv.")
                .replace(".conv.in_proj.", ".shortconv.in_proj.")
                .replace(".conv.out_proj.", ".shortconv.out_proj.")
                .replace(".self_attn.q_layernorm.", ".attn_q_norm.")
                .replace(".self_attn.k_layernorm.", ".attn_k_norm.")
                .replace(".self_attn.q_proj.", ".attn_q.")
                .replace(".self_attn.k_proj.", ".attn_k.")
                .replace(".self_attn.v_proj.", ".attn_v.")
                .replace(".self_attn.out_proj.", ".attn_output.")
                .replace(".operator_norm.", ".attn_norm.")
                .replace(".feed_forward.gate.", ".ffn_gate_inp.")
                .replace(".feed_forward.experts.down_proj", ".ffn_down_exps.weight")
                .replace(".feed_forward.w1.", ".ffn_gate.")
                .replace(".feed_forward.w2.", ".ffn_down.")
                .replace(".feed_forward.w3.", ".ffn_up.")
        };
        if runtime_name == "model.embed_tokens.weight" {
            specs.push(gguf_tensor_from_array("token_embd.weight", value));
        } else if runtime_name == "model.embedding_norm.weight" {
            specs.push(gguf_tensor_from_array("token_embd_norm.weight", value));
        } else if runtime_name == "lm_head.weight" {
            specs.push(gguf_tensor_from_array("output.weight", value));
        } else if let Some(prefix) = runtime_name.strip_suffix("feed_forward.experts.gate_up_proj")
        {
            let width = value.dim(1) / 2;
            let gate = value.try_index_device((.., ..width, ..), stream).unwrap();
            let up = value.try_index_device((.., width.., ..), stream).unwrap();
            specs.push(gguf_tensor_from_array(
                layer_name(&format!("{prefix}ffn_gate_exps.weight")),
                &gate,
            ));
            specs.push(gguf_tensor_from_array(
                layer_name(&format!("{prefix}ffn_up_exps.weight")),
                &up,
            ));
        } else if let Some(prefix) = runtime_name.strip_suffix("feed_forward.expert_bias") {
            specs.push(gguf_tensor_from_array(
                layer_name(&format!("{prefix}ffn_exp_probs_b.bias")),
                value,
            ));
        } else if runtime_name.ends_with(".conv.conv.weight") {
            let reshaped = value
                .reshape(&[value.shape()[0], value.shape()[2]], stream)
                .unwrap();
            specs.push(gguf_tensor_from_array(layer_name(&runtime_name), &reshaped));
        } else {
            specs.push(gguf_tensor_from_array(layer_name(&runtime_name), value));
        }
    }
    let key = |suffix: &str| format!("lfm2moe.{suffix}");
    let metadata = BTreeMap::from([
        (
            "general.architecture".into(),
            GgufMetadataValue::String("lfm2moe".into()),
        ),
        ("general.file_type".into(), GgufMetadataValue::Uint32(0)),
        (key("block_count"), GgufMetadataValue::Uint32(2)),
        (key("embedding_length"), GgufMetadataValue::Uint32(12)),
        (key("feed_forward_length"), GgufMetadataValue::Uint32(17)),
        (
            key("expert_feed_forward_length"),
            GgufMetadataValue::Uint32(9),
        ),
        (
            key("leading_dense_block_count"),
            GgufMetadataValue::Uint32(1),
        ),
        (key("expert_count"), GgufMetadataValue::Uint32(2)),
        (key("expert_used_count"), GgufMetadataValue::Uint32(1)),
        (key("expert_weights_norm"), GgufMetadataValue::Uint32(1)),
        (key("attention.head_count"), GgufMetadataValue::Uint32(6)),
        (
            key("attention.head_count_kv"),
            GgufMetadataValue::Array(MetadataArray::Uint32(vec![0, 3])),
        ),
        (
            key("attention.layer_norm_rms_epsilon"),
            GgufMetadataValue::Float32(0.00001),
        ),
        (key("context_length"), GgufMetadataValue::Uint32(64)),
        (key("shortconv.l_cache"), GgufMetadataValue::Uint32(3)),
        (key("rope.freq_base"), GgufMetadataValue::Float32(10_000.0)),
        (key("vocab_size"), GgufMetadataValue::Uint32(16)),
    ]);
    let tensors = specs
        .iter()
        .map(|tensor| TensorInput {
            name: &tensor.name,
            dimensions: &tensor.dimensions,
            ggml_type: GgmlType::F32,
            data: &tensor.data,
        })
        .collect::<Vec<_>>();
    Writer::default()
        .write(std::fs::File::create(path).unwrap(), &metadata, &tensors)
        .unwrap();
}

fn initialize_fixture(model: &mut impl Parameterized<MlxTensor>, stream: &Stream) {
    for (name, parameter) in neutral_parameter_refs_mut(model).flatten() {
        let shape = parameter.shape().to_vec();
        let value = if name.ends_with("norm.weight")
            || name.ends_with("layernorm.weight")
            || name.ends_with("o_norm.weight")
            || name.ends_with("global_scale")
            || name.as_ref() == "model.norm_f.weight"
        {
            Array::ones::<f32>(&shape, stream).unwrap()
        } else if name.ends_with("A_log") {
            Array::full::<f32>(&shape, Array::from_f32(-0.2), stream).unwrap()
        } else {
            let ordinal = name.bytes().fold(0u32, |sum, byte| sum + u32::from(byte)) % 29;
            Array::full::<f32>(
                &shape,
                Array::from_f32(0.002 + ordinal as f32 * 0.0002),
                stream,
            )
            .unwrap()
        };
        *parameter = value.as_dtype(parameter.dtype(), stream).unwrap();
    }
}

fn save_parameter_fixture(
    directory: &Path,
    config: &serde_json::Value,
    model: &impl Parameterized<MlxTensor>,
) {
    let arrays = neutral_parameter_refs(model, false)
        .flatten()
        .into_iter()
        .map(|(name, value)| (name.to_string(), value.clone()))
        .collect::<Vec<_>>();
    save_indexed_pipeline_fixture(directory, &arrays, "model.layers.", 2);
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(config).unwrap(),
    )
    .unwrap();
}

fn save_indexed_pipeline_fixture(
    directory: &Path,
    arrays: &[(String, Array)],
    layer_prefix: &str,
    layer_count: usize,
) {
    let mut weight_map = serde_json::Map::new();
    for layer in 0..layer_count {
        let prefix = format!("{layer_prefix}{layer}.");
        let selected = arrays
            .iter()
            .filter(|(name, _)| name.starts_with(&prefix))
            .collect::<Vec<_>>();
        assert!(!selected.is_empty(), "fixture layer {layer} has no tensors");
        let shard = format!("layer-{layer}.safetensors");
        Array::save_safetensors(
            selected.iter().map(|(name, value)| (name.as_str(), value)),
            None,
            directory.join(&shard),
        )
        .unwrap();
        for (name, _) in selected {
            weight_map.insert(name.clone(), serde_json::json!(shard));
        }
    }
    let static_tensors = arrays
        .iter()
        .filter(|(name, _)| {
            !(0..layer_count).any(|layer| name.starts_with(&format!("{layer_prefix}{layer}.")))
        })
        .collect::<Vec<_>>();
    assert!(!static_tensors.is_empty());
    Array::save_safetensors(
        static_tensors
            .iter()
            .map(|(name, value)| (name.as_str(), value)),
        None,
        directory.join("static.safetensors"),
    )
    .unwrap();
    for (name, _) in static_tensors {
        weight_map.insert(name.clone(), serde_json::json!("static.safetensors"));
    }
    std::fs::write(
        directory.join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "metadata": {},
            "weight_map": weight_map
        }))
        .unwrap(),
    )
    .unwrap();
    let store = open_safetensors_weight_store(directory, 1).unwrap();
    for layer in 0..layer_count {
        let prefix = format!("{layer_prefix}{layer}.");
        for (name, _) in arrays.iter().filter(|(name, _)| name.starts_with(&prefix)) {
            let backing = store.source_metadata(name).unwrap().backing_shard.unwrap();
            assert_eq!(
                backing.file_name().unwrap().to_string_lossy(),
                format!("layer-{layer}.safetensors"),
                "fixture tensor {name} was indexed to the wrong shard"
            );
        }
    }
}

fn kimi_linear_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "kimi_linear",
        "vocab_size": 13,
        "hidden_size": 12,
        "num_hidden_layers": 2,
        "num_attention_heads": 3,
        "num_key_value_heads": 1,
        "intermediate_size": 17,
        "head_dim": 4,
        "model_max_length": 64,
        "rms_norm_eps": 0.00001,
        "rope_theta": 10000.0,
        "linear_attn_config": {
            "kda_layers": [1],
            "full_attn_layers": [2],
            "num_heads": 3,
            "head_dim": 4,
            "short_conv_kernel_size": 2
        },
        "num_experts": 4,
        "moe_intermediate_size": 9,
        "kv_lora_rank": 4,
        "q_lora_rank": null,
        "qk_nope_head_dim": 2,
        "qk_rope_head_dim": 2,
        "v_head_dim": 2,
        "mla_use_nope": true,
        "num_experts_per_token": 2,
        "num_shared_experts": 1,
        "moe_router_activation_func": "sigmoid",
        "moe_renormalize": true,
        "routed_scaling_factor": 1.0,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "use_grouped_topk": true,
        "num_expert_group": 1,
        "topk_group": 1,
        "tie_word_embeddings": false,
        "num_nextn_predict_layers": 0
    })
}

fn write_kimi_linear_fixture(directory: &Path) {
    write_kimi_linear_fixture_from_config(directory, kimi_linear_config());
}

fn write_kimi_linear_dense_fixture(directory: &Path) {
    let mut config = kimi_linear_config();
    config["first_k_dense_replace"] = config["num_hidden_layers"].clone();
    write_kimi_linear_fixture_from_config(directory, config);
}

fn write_kimi_linear_fixture_from_config(directory: &Path, config: serde_json::Value) {
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let args = eredu_architectures::kimi_linear::model_args_from_config_value(&config).unwrap();
    let mut model = MlxModule::new(
        checkpoint_fixtures::KimiLinearCheckpointTemplate::new(args.clone(), stream).unwrap(),
    );
    initialize_fixture(&mut model, stream);
    let mut arrays = Vec::<(String, Array)>::new();
    for (name, value) in neutral_parameter_refs(&model, false).flatten() {
        if name.as_ref() == "model.layers.1.mlp.experts.gate_up_proj" {
            for expert in 0..args.num_experts {
                arrays.push((
                    format!("model.layers.1.block_sparse_moe.experts.{expert}.w1.weight"),
                    value
                        .try_index_device((expert, ..args.moe_intermediate_size, ..), stream)
                        .unwrap(),
                ));
                arrays.push((
                    format!("model.layers.1.block_sparse_moe.experts.{expert}.w3.weight"),
                    value
                        .try_index_device((expert, args.moe_intermediate_size.., ..), stream)
                        .unwrap(),
                ));
            }
            continue;
        }
        if name.as_ref() == "model.layers.1.mlp.experts.down_proj" {
            for expert in 0..args.num_experts {
                arrays.push((
                    format!("model.layers.1.block_sparse_moe.experts.{expert}.w2.weight"),
                    value.try_index_device((expert, .., ..), stream).unwrap(),
                ));
            }
            continue;
        }
        let checkpoint_name =
            if args.has_sparse_moe_layers() && name.starts_with("model.layers.1.mlp.") {
                name.replacen("model.layers.1.mlp.", "model.layers.1.block_sparse_moe.", 1)
            } else {
                name.to_string()
            };
        let value = if checkpoint_name.ends_with("_conv1d.weight") {
            value
                .reshape(
                    &[
                        args.kda_config.num_heads * args.kda_config.head_dim,
                        args.kda_config.short_conv_kernel_size,
                    ],
                    stream,
                )
                .unwrap()
        } else {
            value.clone()
        };
        arrays.push((checkpoint_name, value));
    }
    save_indexed_pipeline_fixture(directory, &arrays, "model.layers.", 2);
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
}

fn nemotron_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "nemotron_h",
        "architectures": ["NemotronHForCausalLM"],
        "vocab_size": 13,
        "hidden_size": 12,
        "intermediate_size": 17,
        "num_hidden_layers": 4,
        "hybrid_override_pattern": "M-E*",
        "num_attention_heads": 6,
        "num_key_value_heads": 3,
        "head_dim": 2,
        "max_position_embeddings": 64,
        "sliding_window": 3,
        "layer_norm_epsilon": 0.00001,
        "norm_eps": 0.00001,
        "mamba_num_heads": 6,
        "mamba_head_dim": 2,
        "n_groups": 3,
        "ssm_state_size": 2,
        "conv_kernel": 3,
        "chunk_size": 2,
        "moe_intermediate_size": 5,
        "moe_shared_expert_intermediate_size": 7,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "n_group": 1,
        "topk_group": 1,
        "tie_word_embeddings": false,
        "torch_dtype": "float32"
    })
}

fn nemotron_quantizable_config() -> serde_json::Value {
    let mut value = nemotron_config();
    value["vocab_size"] = 64.into();
    value["hidden_size"] = 64.into();
    value["intermediate_size"] = 64.into();
    value["num_attention_heads"] = 8.into();
    value["num_key_value_heads"] = 4.into();
    value["head_dim"] = 8.into();
    value["mamba_num_heads"] = 8.into();
    value["mamba_head_dim"] = 8.into();
    value["n_groups"] = 2.into();
    value["moe_intermediate_size"] = 64.into();
    value["moe_shared_expert_intermediate_size"] = 64.into();
    value
}

fn nemotron_public_name(
    runtime: &str,
    args: &eredu_architectures::nemotron_h::ModelArgs,
) -> String {
    if let Some(rest) = runtime.strip_prefix("model.embeddings.") {
        return format!("backbone.embeddings.{rest}");
    }
    if let Some(rest) = runtime.strip_prefix("model.norm_f.") {
        return format!("backbone.norm_f.{rest}");
    }
    for index in 0..args.num_hidden_layers as usize {
        let prefix = format!("model.layers.{index}.");
        let Some(rest) = runtime.strip_prefix(&prefix) else {
            continue;
        };
        if rest.starts_with("norm.") {
            return format!("backbone.layers.{index}.{rest}");
        }
        let field = match args.layer_schedule.get(index).unwrap() {
            eredu_architectures::nemotron_h::LayerPolicy::Mamba => "mamba",
            eredu_architectures::nemotron_h::LayerPolicy::SelfAttention(_) => "attention",
            eredu_architectures::nemotron_h::LayerPolicy::DenseMlp => "mlp",
            eredu_architectures::nemotron_h::LayerPolicy::SparseMoe => "moe",
        };
        let rest = rest.strip_prefix(&format!("{field}.")).unwrap_or(rest);
        return format!("backbone.layers.{index}.mixer.{rest}");
    }
    runtime.to_string()
}

fn write_nemotron_fixture(directory: &Path) {
    write_nemotron_fixture_with_config(directory, nemotron_config());
}

fn write_nemotron_dense_fixture(directory: &Path) {
    let mut config = nemotron_config();
    config["hybrid_override_pattern"] = serde_json::json!("M-**");
    config["intermediate_size"] = 18.into();
    config["num_key_value_heads"] = 2.into();
    config["n_groups"] = 2.into();
    write_nemotron_fixture_with_config(directory, config);
}

fn write_nemotron_mtp_fixture(directory: &Path) {
    let mut config = nemotron_config();
    config["hybrid_override_pattern"] = serde_json::json!("M-**");
    config["intermediate_size"] = 18.into();
    config["num_key_value_heads"] = 2.into();
    config["n_groups"] = 2.into();
    config["num_nextn_predict_layers"] = 1.into();
    config["mtp_hybrid_override_pattern"] = serde_json::json!("*");
    write_nemotron_fixture_with_config(directory, config);
}

fn write_nemotron_quantizable_fixture(directory: &Path) {
    write_nemotron_fixture_with_config(directory, nemotron_quantizable_config());
}

fn write_nemotron_dense_quantizable_fixture(directory: &Path) {
    let mut config = nemotron_quantizable_config();
    config["hybrid_override_pattern"] = serde_json::json!("M-**");
    write_nemotron_fixture_with_config(directory, config);
}

fn write_nemotron_fixture_with_config(directory: &Path, config: serde_json::Value) {
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let args = eredu_architectures::nemotron_h::model_args_from_config_value(&config).unwrap();
    let mut model = MlxModule::new(
        checkpoint_fixtures::NemotronHCheckpointTemplate::new(args.clone(), stream).unwrap(),
    );
    initialize_fixture(&mut model, stream);
    let mut arrays = Vec::<(String, Array)>::new();
    for (name, value) in neutral_parameter_refs(&model, false).flatten() {
        let runtime = name.to_string();
        if runtime.ends_with("moe.experts.up_proj") {
            let prefix = nemotron_public_name(runtime.trim_end_matches(".up_proj"), &args);
            for expert in 0..args.n_routed_experts {
                arrays.push((
                    format!("{prefix}.{expert}.up_proj.weight"),
                    value.try_index_device((expert, .., ..), stream).unwrap(),
                ));
            }
        } else if runtime.ends_with("moe.experts.down_proj") {
            let prefix = nemotron_public_name(runtime.trim_end_matches(".down_proj"), &args);
            for expert in 0..args.n_routed_experts {
                arrays.push((
                    format!("{prefix}.{expert}.down_proj.weight"),
                    value.try_index_device((expert, .., ..), stream).unwrap(),
                ));
            }
        } else {
            arrays.push((nemotron_public_name(&runtime, &args), value.clone()));
        }
    }
    save_indexed_pipeline_fixture(directory, &arrays, "backbone.layers.", 4);
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
}

fn write_nemotron_h_moe_gguf_fixture(path: &Path) {
    let mut config = nemotron_config();
    config["hybrid_override_pattern"] = serde_json::json!("MEE*");
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let args = eredu_architectures::nemotron_h::model_args_from_config_value(&config).unwrap();
    let mut model = MlxModule::new(
        checkpoint_fixtures::NemotronHCheckpointTemplate::new(args.clone(), stream).unwrap(),
    );
    initialize_fixture(&mut model, stream);
    let mut specs = Vec::new();
    for (runtime_name, value) in neutral_parameter_refs(&model, false).flatten() {
        let runtime_name = runtime_name.to_string();
        let gguf_name = if runtime_name == "model.embeddings.weight" {
            "token_embd.weight".to_string()
        } else if runtime_name == "model.norm_f.weight" {
            "output_norm.weight".to_string()
        } else if runtime_name == "lm_head.weight" {
            "output.weight".to_string()
        } else if let Some(rest) = runtime_name.strip_prefix("model.layers.") {
            let (layer, parameter) = rest.split_once('.').unwrap();
            let parameter = parameter
                .strip_prefix("norm.")
                .map_or_else(|| parameter.to_string(), |rest| format!("attn_norm.{rest}"));
            let parameter = parameter
                .replace("mamba.norm.", "ssm_norm.")
                .replace("mamba.in_proj.", "ssm_in.")
                .replace("mamba.conv1d.", "ssm_conv1d.")
                .replace("mamba.dt_bias", "ssm_dt.bias")
                .replace("mamba.A_log", "ssm_a")
                .replace("mamba.D", "ssm_d")
                .replace("mamba.out_proj.", "ssm_out.")
                .replace("attention.q_proj.", "attn_q.")
                .replace("attention.k_proj.", "attn_k.")
                .replace("attention.v_proj.", "attn_v.")
                .replace("attention.o_proj.", "attn_output.")
                .replace("moe.gate.e_score_correction_bias", "exp_probs_b.bias")
                .replace("moe.gate.", "ffn_gate_inp.")
                .replace("moe.experts.up_proj", "ffn_up_exps.weight")
                .replace("moe.experts.down_proj", "ffn_down_exps.weight")
                .replace("moe.shared_experts.up_proj.", "ffn_up_shexp.")
                .replace("moe.shared_experts.down_proj.", "ffn_down_shexp.");
            format!("blk.{layer}.{parameter}")
        } else {
            panic!("unmapped Nemotron-H GGUF fixture tensor {runtime_name}")
        };
        if gguf_name.ends_with(".ssm_conv1d.weight") {
            let reshaped = value
                .reshape(&[value.shape()[0], value.shape()[2]], stream)
                .unwrap();
            specs.push(gguf_tensor_from_array(gguf_name, &reshaped));
        } else if gguf_name.ends_with(".ssm_a") {
            let negative =
                Array::full::<f32>(value.shape(), Array::from_f32(-0.8), stream).unwrap();
            specs.push(gguf_tensor_from_array(gguf_name, &negative));
        } else {
            specs.push(gguf_tensor_from_array(gguf_name, value));
        }
    }
    let key = |suffix: &str| format!("nemotron_h_moe.{suffix}");
    let metadata = BTreeMap::from([
        (
            "general.architecture".into(),
            GgufMetadataValue::String("nemotron_h_moe".into()),
        ),
        ("general.file_type".into(), GgufMetadataValue::Uint32(0)),
        (key("block_count"), GgufMetadataValue::Uint32(4)),
        (key("embedding_length"), GgufMetadataValue::Uint32(12)),
        (
            key("feed_forward_length"),
            GgufMetadataValue::Array(MetadataArray::Uint32(vec![0, 17, 17, 0])),
        ),
        (
            key("attention.head_count_kv"),
            GgufMetadataValue::Array(MetadataArray::Uint32(vec![0, 0, 0, 3])),
        ),
        (key("attention.head_count"), GgufMetadataValue::Uint32(6)),
        (key("attention.key_length"), GgufMetadataValue::Uint32(2)),
        (
            key("attention.layer_norm_rms_epsilon"),
            GgufMetadataValue::Float32(0.00001),
        ),
        (
            key("attention.sliding_window"),
            GgufMetadataValue::Uint32(3),
        ),
        (key("context_length"), GgufMetadataValue::Uint32(64)),
        (key("ssm.inner_size"), GgufMetadataValue::Uint32(12)),
        (key("ssm.time_step_rank"), GgufMetadataValue::Uint32(6)),
        (key("ssm.state_size"), GgufMetadataValue::Uint32(2)),
        (key("ssm.group_count"), GgufMetadataValue::Uint32(3)),
        (key("ssm.conv_kernel"), GgufMetadataValue::Uint32(3)),
        (key("expert_count"), GgufMetadataValue::Uint32(2)),
        (key("expert_shared_count"), GgufMetadataValue::Uint32(1)),
        (
            key("expert_feed_forward_length"),
            GgufMetadataValue::Uint32(5),
        ),
        (
            key("expert_shared_feed_forward_length"),
            GgufMetadataValue::Uint32(7),
        ),
        (key("expert_used_count"), GgufMetadataValue::Uint32(2)),
        (key("expert_weights_norm"), GgufMetadataValue::Uint32(1)),
        (key("expert_group_count"), GgufMetadataValue::Uint32(1)),
        (key("expert_group_used_count"), GgufMetadataValue::Uint32(1)),
        (key("vocab_size"), GgufMetadataValue::Uint32(13)),
    ]);
    let tensors = specs
        .iter()
        .map(|tensor| TensorInput {
            name: &tensor.name,
            dimensions: &tensor.dimensions,
            ggml_type: GgmlType::F32,
            data: &tensor.data,
        })
        .collect::<Vec<_>>();
    Writer::default()
        .write(std::fs::File::create(path).unwrap(), &metadata, &tensors)
        .unwrap();
}

fn qwen_hybrid_config(model_type: &str) -> serde_json::Value {
    serde_json::json!({
        "architectures": [if model_type == "qwen3_next" { "Qwen3NextForCausalLM" } else { "Qwen3_5ForCausalLM" }],
        "model_type": model_type,
        "vocab_size": 64,
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "mtp_num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 128,
        "rms_norm_eps": 0.00001,
        "intermediate_size": 32,
        "num_experts": 0,
        "linear_conv_kernel_dim": 3,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "layer_types": ["linear_attention", "full_attention"],
        "tie_word_embeddings": false
    })
}

fn qwen_hybrid_moe_config(model_type: &str) -> serde_json::Value {
    let mut config = qwen_hybrid_config(model_type);
    config["architectures"] = serde_json::json!([if model_type == "qwen3_next" {
        "Qwen3NextForCausalLM"
    } else {
        "Qwen3_5MoeForCausalLM"
    }]);
    config["intermediate_size"] = serde_json::json!(0);
    config["moe_intermediate_size"] = serde_json::json!(8);
    config["shared_expert_intermediate_size"] = serde_json::json!(8);
    config["num_experts"] = serde_json::json!(2);
    config["num_experts_per_tok"] = serde_json::json!(1);
    config["norm_topk_prob"] = serde_json::json!(true);
    config
}

fn write_qwen_hybrid_fixture(directory: &Path, model_type: &str) {
    let config = qwen_hybrid_config(model_type);
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let parsed = qwen_hybrid::model_args_from_config_value(&config).unwrap();
    let mut model = MlxModule::new(
        crate::composition::checkpoint_fixtures::QwenHybridCheckpointTemplate::new(
            parsed.text,
            stream,
        )
        .unwrap(),
    );
    initialize_fixture(&mut model, stream);
    save_parameter_fixture(directory, &config, &model);
}

fn write_qwen_hybrid_moe_fixture(directory: &Path, model_type: &str) {
    let config = qwen_hybrid_moe_config(model_type);
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let parsed = qwen_hybrid::model_args_from_config_value(&config).unwrap();
    let mut model = MlxModule::new(
        crate::composition::checkpoint_fixtures::QwenHybridCheckpointTemplate::new(
            parsed.text,
            stream,
        )
        .unwrap(),
    );
    initialize_fixture(&mut model, stream);
    save_parameter_fixture(directory, &config, &model);
}

fn write_qwen35_multimodal_fixture(directory: &Path, moe: bool) {
    write_qwen35_multimodal_fixture_with_prediction(directory, moe, 1);
}

fn write_qwen35_zero_prediction_fixture(directory: &Path) {
    write_qwen35_multimodal_fixture_with_prediction(directory, false, 0);
}

fn write_qwen35_multimodal_fixture_with_prediction(
    directory: &Path,
    moe: bool,
    prediction_layers: usize,
) {
    let mut text_config = if moe {
        qwen_hybrid_moe_config("qwen3_5_moe_text")
    } else {
        qwen_hybrid_config("qwen3_5_text")
    };
    text_config["mtp_num_hidden_layers"] = prediction_layers.into();
    let config = serde_json::json!({
        "architectures": [if moe { "Qwen3_5MoeForConditionalGeneration" } else { "Qwen3_5ForConditionalGeneration" }],
        "model_type": if moe { "qwen3_5_moe" } else { "qwen3_5" },
        "image_token_id": 42,
        "video_token_id": 43,
        "text_config": text_config,
        "vision_config": {
            "depth": 2,
            "hidden_size": 8,
            "hidden_act": "silu",
            "intermediate_size": 8,
            "num_heads": 2,
            "num_position_embeddings": 16,
            "in_channels": 3,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 1,
            "window_size": 8,
            "out_hidden_size": 16,
            "fullatt_block_indexes": [0, 1],
            "deepstack_visual_indexes": []
        }
    });
    std::fs::create_dir_all(directory).unwrap();
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let parsed = qwen_hybrid::model_args_from_config_value(&config).unwrap();
    let mut model = MlxModule::new(
        crate::composition::checkpoint_fixtures::QwenConditionalCheckpointTemplate::new(
            parsed, stream,
        )
        .unwrap(),
    );
    initialize_fixture(&mut model, stream);
    save_parameter_fixture(directory, &config, &model);
}

fn write_qwen3_vl_fixture(directory: &Path, moe: bool) {
    let config = serde_json::json!({
        "architectures": [if moe { "Qwen3VLMoeForConditionalGeneration" } else { "Qwen3VLForConditionalGeneration" }],
        "model_type": if moe { "qwen3_vl_moe" } else { "qwen3_vl" },
        "image_token_id": 42,
        "video_token_id": 43,
        "tie_word_embeddings": false,
        "text_config": {
            "model_type": if moe { "qwen3_vl_moe_text" } else { "qwen3_vl_text" },
            "vocab_size": 64,
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": if moe { 0 } else { 32 },
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 128,
            "rms_norm_eps": 0.000001,
            "rope_theta": 10000.0,
            "moe_intermediate_size": if moe { 8 } else { 0 },
            "num_experts": if moe { 2 } else { 0 },
            "num_experts_per_tok": if moe { 1 } else { 0 },
            "norm_topk_prob": moe,
            "rope_scaling": { "mrope_section": [2, 1, 1] }
        },
        "vision_config": {
            "depth": 2,
            "hidden_size": 8,
            "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": 16,
            "num_heads": 2,
            "num_position_embeddings": 16,
            "in_channels": 3,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 1,
            "out_hidden_size": 16,
            "deepstack_visual_indexes": [0, 1]
        }
    });
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let args = eredu_architectures::qwen::vl::model_args_from_config_value(&config).unwrap();
    let mut model = MlxModule::new(
        crate::composition::checkpoint_fixtures::QwenVlCheckpointTemplate::new(args, stream)
            .unwrap(),
    );
    initialize_fixture(&mut model, stream);
    let arrays = neutral_parameter_refs(&model, false)
        .flatten()
        .into_iter()
        .map(|(name, value)| {
            let canonical = name.to_string();
            let canonical = canonical
                .strip_prefix("model.language_model.model.language_model.")
                .map_or(canonical.clone(), |suffix| {
                    format!("model.language_model.{suffix}")
                });
            (canonical, value.clone())
        })
        .collect::<Vec<_>>();
    save_indexed_pipeline_fixture(directory, &arrays, "model.language_model.layers.", 2);
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
}

fn inkling_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "inkling_mm_model",
        "eos_token_id": 1,
        "text_config": {
            "torch_dtype": "float32",
            "hidden_size": 16,
            "num_hidden_layers": 3,
            "model_max_length": 32,
            "vocab_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "swa_num_attention_heads": 4,
            "swa_num_key_value_heads": 2,
            "swa_head_dim": 4,
            "sliding_window_size": 4,
            "layer_types": ["full_attention", "sliding_attention", "full_attention"],
            "dense_mlp_idx": 1,
            "sconv_kernel_size": 3,
            "d_rel": 4,
            "rel_extent": 8,
            "intermediate_size": 8,
            "dense_intermediate_size": 16,
            "moe_intermediate_size": 8,
            "n_routed_experts": 2,
            "num_experts_per_tok": 1,
            "n_shared_experts": 1,
            "route_scale": 1.0,
            "use_sconv": true,
            "use_embed_norm": true,
            "shared_expert_sink": true,
            "use_gate_bias": true,
            "norm_after_topk": true,
            "use_global_scale": true,
            "gate_activation": "sigmoid",
            "hidden_act": "silu",
            "attention_dropout": 0.0,
            "q_bias": false,
            "o_bias": false,
            "logits_mup_width_multiplier": 2.0,
            "unpadded_vocab_size": 30
        }
    })
}

fn inkling_quantizable_config() -> serde_json::Value {
    let mut value = inkling_config();
    value["text_config"]["hidden_size"] = 32.into();
    value["text_config"]["vocab_size"] = 64.into();
    value["text_config"]["num_attention_heads"] = 4.into();
    value["text_config"]["num_key_value_heads"] = 2.into();
    value["text_config"]["head_dim"] = 8.into();
    value["text_config"]["swa_num_attention_heads"] = 4.into();
    value["text_config"]["swa_num_key_value_heads"] = 2.into();
    value["text_config"]["swa_head_dim"] = 8.into();
    value["text_config"]["d_rel"] = 32.into();
    value["text_config"]["rel_extent"] = 32.into();
    value["text_config"]["intermediate_size"] = 32.into();
    value["text_config"]["dense_intermediate_size"] = 32.into();
    value["text_config"]["moe_intermediate_size"] = 32.into();
    value
}

fn inkling_multimodal_config() -> serde_json::Value {
    let mut config = inkling_config();
    config["audio_config"] = serde_json::json!({
        "text_hidden_size": 16,
        "num_codebooks": 2,
        "codebook_size": 8,
        "bias": false,
        "use_audio_norm": true,
        "audio_mode": "dmel",
        "rms_norm_eps": 1e-6
    });
    config["image_token_id"] = serde_json::json!(21);
    config["audio_token_id"] = serde_json::json!(20);
    config
}

fn inkling_released_name(runtime: &str) -> String {
    if runtime == "lm_head.weight" {
        return "model.llm.unembed.weight".into();
    }
    if let Some(rest) = runtime.strip_prefix("audio.") {
        return format!("model.audio.{rest}");
    }
    if let Some(rest) = runtime.strip_prefix("visual.") {
        return format!("model.visual.{rest}");
    }
    let rest = runtime.strip_prefix("model.").unwrap();
    let mut raw = format!("model.llm.{rest}");
    raw = raw
        .replace("model.llm.embed_tokens.weight", "model.llm.embed.weight")
        .replace(".input_layernorm.weight", ".attn_norm.weight")
        .replace(".post_attention_layernorm.weight", ".mlp_norm.weight")
        .replace(".self_attn.q_proj.weight", ".attn.wq_du.weight")
        .replace(".self_attn.k_proj.weight", ".attn.wk_dv.weight")
        .replace(".self_attn.v_proj.weight", ".attn.wv_dv.weight")
        .replace(".self_attn.r_proj.weight", ".attn.wr_du.weight")
        .replace(".self_attn.o_proj.weight", ".attn.wo_ud.weight")
        .replace(".self_attn.q_norm.weight", ".attn.q_norm.weight")
        .replace(".self_attn.k_norm.weight", ".attn.k_norm.weight")
        .replace(".self_attn.rel_proj", ".attn.rel_logits_proj.proj")
        .replace(".self_attn.k_sconv.weight", ".attn.k_sconv.weight")
        .replace(".self_attn.v_sconv.weight", ".attn.v_sconv.weight")
        .replace(".dense.down_proj.weight", ".mlp.w2_md.weight")
        .replace(".dense_global_scale", ".mlp.global_scale")
        .replace(".moe.router.weight", ".mlp.gate.weight")
        .replace(".moe.router.bias", ".mlp.gate.bias")
        .replace(".moe.router.global_scale", ".mlp.gate.global_scale")
        .replace(".moe.experts.down_proj", ".mlp.experts.w2_weight")
        .replace(
            ".moe.shared_experts.down_proj",
            ".mlp.shared_experts.shared_w2_weight",
        );
    raw
}

fn interleave(gate: &Array, up: &Array, axis: i32, stream: &Stream) -> Array {
    let stacked = stack_axis(&[gate.clone(), up.clone()], axis, stream).unwrap();
    let mut shape = gate.shape().to_vec();
    let row_axis = shape.len() - 2;
    shape[row_axis] *= 2;
    stacked.reshape(&shape, stream).unwrap()
}

fn write_inkling_fixture(directory: &Path) {
    write_inkling_fixture_with_config(directory, inkling_config());
}

fn write_inkling_dense_fixture(directory: &Path) {
    let mut config = inkling_config();
    config["text_config"]["dense_mlp_idx"] = config["text_config"]["num_hidden_layers"].clone();
    write_inkling_fixture_with_config(directory, config);
}

fn write_inkling_dense_multimodal_fixture(directory: &Path) {
    let mut config = inkling_multimodal_config();
    config["text_config"]["dense_mlp_idx"] = config["text_config"]["num_hidden_layers"].clone();
    write_inkling_fixture_with_config(directory, config);
}

fn write_inkling_mtp_fixture(directory: &Path) {
    write_inkling_mtp_fixture_for_pipeline(directory, false);
}

fn write_inkling_pipeline_mtp_fixture(directory: &Path) {
    write_inkling_mtp_fixture_for_pipeline(directory, true);
}

fn write_inkling_mtp_fixture_for_pipeline(directory: &Path, pipeline: bool) {
    let mut config = inkling_config();
    if pipeline {
        config["text_config"]["num_hidden_layers"] = 2.into();
        config["text_config"]["layer_types"] =
            serde_json::json!(["sliding_attention", "full_attention"]);
        config["text_config"]["dense_mlp_idx"] = 1.into();
        config["text_config"]["model_max_length"] = 32.into();
    } else {
        config["text_config"]["num_hidden_layers"] = 1.into();
        config["text_config"]["layer_types"] = serde_json::json!(["sliding_attention"]);
        config["text_config"]["dense_mlp_idx"] = 0.into();
    }
    config["mtp_config"] = serde_json::json!({
        "num_nextn_predict_layers": 2,
        "local_layer_ids": [1],
        "chain_hidden_post_norm": true,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "swa_num_attention_heads": 4,
        "swa_num_key_value_heads": 2,
        "swa_head_dim": 4,
        "dense_intermediate_size": 16,
        "sconv_kernel_size": 3
    });
    std::fs::create_dir_all(directory).unwrap();
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let args = eredu_architectures::inkling::ModelArgs::from_hf_json(
        &serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    type Architecture =
        eredu_architectures::inkling::LayeredModel<crate::backend::nn::shared::MlxNeuralBackend>;
    type State = crate::backend::runtime::cache::state::MlxHybridState;
    let architecture = Architecture::new(args, stream).unwrap();
    let mut arrays = Vec::<(String, Array)>::new();
    struct Collector<'a> {
        stream: &'a Stream,
        arrays: &'a mut Vec<(String, Array)>,
    }
    impl<'tensor> ParameterVisitor<'tensor, MlxTensor> for Collector<'_> {
        fn visit(&mut self, metadata: ParameterMetadata, parameter: &'tensor MlxTensor) {
            let parameter = parameter.as_array();
            self.arrays.push((
                metadata.id.to_string(),
                safemlx::ops::zeros_dtype(parameter.shape(), parameter.dtype(), self.stream)
                    .unwrap(),
            ));
        }
    }
    let mut collector = Collector {
        stream,
        arrays: &mut arrays,
    };
    <Architecture as eredu_runtime::LayeredArchitecture<
        crate::backend::nn::shared::MlxNeuralBackend,
        State,
    >>::static_modules(&architecture)
    .visit_parameters(&mut collector);
    let graph = <Architecture as eredu_runtime::LayeredArchitecture<
        crate::backend::nn::shared::MlxNeuralBackend,
        State,
    >>::execution_graph(&architecture)
    .unwrap();
    for group in 0..graph.groups().len() {
        let count = <Architecture as eredu_runtime::LayeredArchitecture<
            crate::backend::nn::shared::MlxNeuralBackend,
            State,
        >>::group_unit_count(&architecture, group)
        .unwrap();
        for index in 0..count {
            <Architecture as eredu_runtime::LayeredArchitecture<
                crate::backend::nn::shared::MlxNeuralBackend,
                State,
            >>::build_unit(&architecture, group, index, stream)
            .unwrap()
            .visit_parameters(&mut collector);
        }
    }
    Array::save_safetensors(
        arrays.iter().map(|(name, array)| (name.as_str(), array)),
        None,
        directory.join("model.safetensors"),
    )
    .unwrap();
}

fn write_gemma4_tensor_parallel_fixture(directory: &Path) {
    write_gemma4_tensor_parallel_fixture_with_options(directory, false, false);
}

fn write_gemma4_tensor_parallel_fixture_with_tied_embeddings(directory: &Path, tied: bool) {
    write_gemma4_tensor_parallel_fixture_with_options(directory, tied, false);
}

fn write_gemma4_multimodal_tensor_parallel_fixture(directory: &Path) {
    write_gemma4_tensor_parallel_fixture_with_options(directory, false, true);
}

fn write_gemma4_tied_multimodal_tensor_parallel_fixture(directory: &Path) {
    write_gemma4_tensor_parallel_fixture_with_options(directory, true, true);
}

fn write_gemma4_tensor_parallel_fixture_with_options(
    directory: &Path,
    tied: bool,
    multimodal: bool,
) {
    let mut config = serde_json::json!({
        "model_type": "gemma4",
        "tie_word_embeddings": tied,
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 16,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "rms_norm_eps": 0.00001,
            "vocab_size": 32,
            "max_position_embeddings": 128,
            "tie_word_embeddings": tied,
            "attention_k_eq_v": false,
            "layer_types": ["full_attention", "sliding_attention"],
            "sliding_window": 8
        }
    });
    if multimodal {
        config["model_type"] = "gemma4_unified".into();
        config["image_token_id"] = 30.into();
        config["audio_token_id"] = 31.into();
        config["vision_config"] = serde_json::json!({
            "hidden_size": 16,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "patch_size": 4,
            "pooling_kernel_size": 2,
            "position_embedding_size": 4,
            "rms_norm_eps": 0.00001
        });
        config["audio_config"] = serde_json::json!({
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "output_proj_dims": 8,
            "conv_kernel_size": 3,
            "attention_chunk_size": 4,
            "attention_context_left": 5,
            "attention_context_right": 0,
            "attention_invalid_logits_value": -1000000000.0,
            "attention_logit_cap": 50.0,
            "residual_weight": 0.5,
            "rms_norm_eps": 0.00001,
            "subsampling_conv_channels": [4, 8]
        });
    }
    std::fs::create_dir_all(directory).unwrap();
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let args = eredu_architectures::gemma4::FamilyConfig::from_hf_json(
        &serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    type Architecture =
        eredu_architectures::gemma4::LayeredModel<crate::backend::nn::shared::MlxNeuralBackend>;
    type State = crate::backend::runtime::cache::state::MlxHybridState;
    let architecture = Architecture::new(args, stream).unwrap();
    let mut arrays = Vec::<(String, Array)>::new();
    struct Collector<'a> {
        stream: &'a Stream,
        arrays: &'a mut Vec<(String, Array)>,
    }
    impl<'tensor> ParameterVisitor<'tensor, MlxTensor> for Collector<'_> {
        fn visit(&mut self, metadata: ParameterMetadata, parameter: &'tensor MlxTensor) {
            let parameter = parameter.as_array();
            self.arrays.push((
                metadata.id.to_string(),
                safemlx::ops::zeros_dtype(parameter.shape(), parameter.dtype(), self.stream)
                    .unwrap(),
            ));
        }
    }
    let mut collector = Collector {
        stream,
        arrays: &mut arrays,
    };
    <Architecture as eredu_runtime::LayeredArchitecture<
        crate::backend::nn::shared::MlxNeuralBackend,
        State,
    >>::static_modules(&architecture)
    .visit_parameters(&mut collector);
    for group in 0..3 {
        let count = <Architecture as eredu_runtime::LayeredArchitecture<
            crate::backend::nn::shared::MlxNeuralBackend,
            State,
        >>::group_unit_count(&architecture, group)
        .unwrap();
        for index in 0..count {
            <Architecture as eredu_runtime::LayeredArchitecture<
                crate::backend::nn::shared::MlxNeuralBackend,
                State,
            >>::build_unit(&architecture, group, index, stream)
            .unwrap()
            .visit_parameters(&mut collector);
        }
    }
    // MLX's SafeTensors writer promotes rank-zero arrays to `[1]`, while the
    // released clipped media bounds and their neutral schema are true
    // scalars. Preserve the authoritative parameter shapes in this fixture.
    let tensors = arrays
        .iter()
        .map(|(name, array)| {
            let shape = if ["input_min", "input_max", "output_min", "output_max"]
                .iter()
                .any(|suffix| name.ends_with(suffix))
            {
                Vec::new()
            } else {
                array
                    .shape()
                    .iter()
                    .map(|dimension| usize::try_from(*dimension).unwrap())
                    .collect()
            };
            (name.as_str(), shape, 0.0)
        })
        .collect::<Vec<_>>();
    write_f32_shard(&directory.join("model.safetensors"), &tensors);
}

fn write_muse_glimmer_tensor_parallel_fixture(directory: &Path) {
    let config = serde_json::json!({
        "architectures": ["MuseGlimmerForConditionalGeneration"],
        "model_type": "muse_glimmer",
        "image_token_id": 22,
        "video_token_id": 23,
        "out_hidden_size": 32,
        "projector_hidden_size": 16,
        "text_config": {
            "model_type": "muse_glimmer_text",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 16,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "rms_norm_eps": 0.00001,
            "post_norm_eps": 0.00001,
            "vocab_size": 32,
            "max_position_embeddings": 64,
            "rope_theta": 10000.0,
            "layer_types": ["sliding_attention", "full_attention"],
            "layer_rope_theta": [10000.0, 0.0],
            "sliding_window": 8,
            "tie_word_embeddings": false,
            "hidden_act": "silu",
            "attention_dropout": 0.0,
            "qk_scale_factor": 1.0,
            "output_multiplier": 1.0,
            "final_logit_softcapping": 30.0
        },
        "vision_config": {
            "model_type": "muse_glimmer_vision",
            "hidden_size": 8,
            "intermediate_size": 8,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "patch_size": 2,
            "patch_temporal": 1,
            "merge_size": 2,
            "pos_emb_height": 2,
            "pos_emb_width": 2,
            "max_position_embeddings": 4,
            "layer_norm_eps": 0.00001,
            "hidden_act": "gelu",
            "layer_types": ["full_attention"],
            "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}
        }
    });
    std::fs::create_dir_all(directory).unwrap();
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let args = eredu_architectures::muse_glimmer::DecoderConfig::from_hf_json(
        &serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    type Architecture = eredu_architectures::muse_glimmer::LayeredModel<
        crate::backend::nn::shared::MlxNeuralBackend,
    >;
    type State = crate::backend::runtime::cache::state::MlxKeyValueState;
    let architecture = Architecture::new(args, stream).unwrap();
    let mut arrays = Vec::<(String, Array)>::new();
    struct Collector<'a> {
        stream: &'a Stream,
        arrays: &'a mut Vec<(String, Array)>,
    }
    impl<'tensor> ParameterVisitor<'tensor, MlxTensor> for Collector<'_> {
        fn visit(&mut self, metadata: ParameterMetadata, parameter: &'tensor MlxTensor) {
            let parameter = parameter.as_array();
            self.arrays.push((
                metadata.id.to_string(),
                safemlx::ops::zeros_dtype(parameter.shape(), parameter.dtype(), self.stream)
                    .unwrap(),
            ));
        }
    }
    let mut collector = Collector {
        stream,
        arrays: &mut arrays,
    };
    <Architecture as eredu_runtime::LayeredArchitecture<
        crate::backend::nn::shared::MlxNeuralBackend,
        State,
    >>::static_modules(&architecture)
    .visit_parameters(&mut collector);
    for group in 0..2 {
        let count = <Architecture as eredu_runtime::LayeredArchitecture<
            crate::backend::nn::shared::MlxNeuralBackend,
            State,
        >>::group_unit_count(&architecture, group)
        .unwrap();
        for index in 0..count {
            <Architecture as eredu_runtime::LayeredArchitecture<
                crate::backend::nn::shared::MlxNeuralBackend,
                State,
            >>::build_unit(&architecture, group, index, stream)
            .unwrap()
            .visit_parameters(&mut collector);
        }
    }
    Array::save_safetensors(
        arrays.iter().map(|(name, array)| (name.as_str(), array)),
        None,
        directory.join("model.safetensors"),
    )
    .unwrap();
}

fn write_inkling_quantizable_fixture(directory: &Path) {
    write_inkling_fixture_with_config(directory, inkling_quantizable_config());
}

fn write_inkling_multimodal_fixture(directory: &Path) {
    write_inkling_fixture_with_config(directory, inkling_multimodal_config());
}

fn initialized_inkling_parameters(
    config: &serde_json::Value,
    stream: &Stream,
) -> (
    eredu_architectures::inkling::ModelArgs,
    BTreeMap<String, Array>,
) {
    type Architecture =
        eredu_architectures::inkling::LayeredModel<crate::backend::nn::shared::MlxNeuralBackend>;
    type State = crate::backend::runtime::cache::state::MlxHybridState;

    struct Initializer<'a> {
        stream: &'a Stream,
        parameters: &'a mut BTreeMap<String, Array>,
    }

    impl<'tensor> ParameterVisitorMut<'tensor, MlxTensor> for Initializer<'_> {
        fn visit_mut(&mut self, metadata: ParameterMetadata, parameter: &'tensor mut MlxTensor) {
            let name = metadata.id.to_string();
            let shape = parameter.as_array().shape().to_vec();
            let dtype = parameter.as_array().dtype();
            let value = if name.ends_with("norm.weight")
                || name.ends_with("layernorm.weight")
                || name.ends_with("o_norm.weight")
                || name.ends_with("global_scale")
                || name == "model.norm_f.weight"
            {
                Array::ones::<f32>(&shape, self.stream).unwrap()
            } else if name.ends_with("A_log") {
                Array::full::<f32>(&shape, Array::from_f32(-0.2), self.stream).unwrap()
            } else {
                let ordinal = name.bytes().fold(0u32, |sum, byte| sum + u32::from(byte)) % 29;
                Array::full::<f32>(
                    &shape,
                    Array::from_f32(0.002 + ordinal as f32 * 0.0002),
                    self.stream,
                )
                .unwrap()
            }
            .as_dtype(dtype, self.stream)
            .unwrap();
            *parameter = MlxTensor::from_array(value);
            self.parameters.insert(name, parameter.as_array().clone());
        }
    }

    let args =
        eredu_architectures::inkling::ModelArgs::from_hf_json(&serde_json::to_vec(config).unwrap())
            .unwrap();
    let mut architecture = Architecture::new(args.clone(), stream).unwrap();
    let mut parameters = BTreeMap::new();
    <Architecture as eredu_runtime::LayeredArchitecture<
        crate::backend::nn::shared::MlxNeuralBackend,
        State,
    >>::static_modules_mut(&mut architecture)
    .visit_parameters_mut(&mut Initializer {
        stream,
        parameters: &mut parameters,
    });
    let graph = <Architecture as eredu_runtime::LayeredArchitecture<
        crate::backend::nn::shared::MlxNeuralBackend,
        State,
    >>::execution_graph(&architecture)
    .unwrap();
    for group in 0..graph.groups().len() {
        let count = <Architecture as eredu_runtime::LayeredArchitecture<
            crate::backend::nn::shared::MlxNeuralBackend,
            State,
        >>::group_unit_count(&architecture, group)
        .unwrap();
        for index in 0..count {
            let mut unit = <Architecture as eredu_runtime::LayeredArchitecture<
                crate::backend::nn::shared::MlxNeuralBackend,
                State,
            >>::build_unit(&architecture, group, index, stream)
            .unwrap();
            unit.visit_parameters_mut(&mut Initializer {
                stream,
                parameters: &mut parameters,
            });
        }
    }
    (args, parameters)
}

fn write_inkling_fixture_with_config(directory: &Path, config: serde_json::Value) {
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let (args, parameters) = initialized_inkling_parameters(&config, stream);
    let mut arrays = Vec::<(String, Array)>::new();
    for (name, value) in &parameters {
        let name = name.as_str();
        if name.ends_with(".dense.up_proj.weight") {
            continue;
        }
        if let Some(prefix) = name.strip_suffix(".dense.gate_proj.weight") {
            let up = parameters
                .get(format!("{prefix}.dense.up_proj.weight").as_str())
                .unwrap();
            arrays.push((
                format!("model.llm.{}.mlp.w13_dn.weight", &prefix["model.".len()..]),
                interleave(value, up, 1, stream),
            ));
            continue;
        }
        if let Some(prefix) = name.strip_suffix(".moe.experts.gate_up_proj") {
            let intermediate = args.text_config.moe_intermediate_size.unwrap();
            let gate = value
                .try_index_device((.., ..intermediate, ..), stream)
                .unwrap();
            let up = value
                .try_index_device((.., intermediate.., ..), stream)
                .unwrap();
            arrays.push((
                format!(
                    "model.llm.{}.mlp.experts.w13_weight",
                    &prefix["model.".len()..]
                ),
                interleave(&gate, &up, 2, stream),
            ));
            continue;
        }
        if let Some(prefix) = name.strip_suffix(".moe.shared_experts.gate_up_proj") {
            let intermediate = args.text_config.moe_intermediate_size.unwrap();
            let gate = value
                .try_index_device((.., ..intermediate, ..), stream)
                .unwrap();
            let up = value
                .try_index_device((.., intermediate.., ..), stream)
                .unwrap();
            arrays.push((
                format!(
                    "model.llm.{}.mlp.shared_experts.shared_w13_weight",
                    &prefix["model.".len()..]
                ),
                interleave(&gate, &up, 2, stream),
            ));
            continue;
        }
        let raw = inkling_released_name(name);
        arrays.push((raw, (*value).clone()));
    }
    save_indexed_pipeline_fixture(directory, &arrays, "model.llm.layers.", 3);
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
}

fn inkling_gguf_metadata() -> BTreeMap<String, GgufMetadataValue> {
    BTreeMap::from([
        (
            "general.architecture".into(),
            GgufMetadataValue::String("inkling".into()),
        ),
        ("general.file_type".into(), GgufMetadataValue::Uint32(0)),
        ("inkling.block_count".into(), GgufMetadataValue::Uint32(3)),
        (
            "inkling.embedding_length".into(),
            GgufMetadataValue::Uint32(16),
        ),
        (
            "inkling.attention.head_count".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "inkling.attention.head_count_kv".into(),
            GgufMetadataValue::Array(MetadataArray::Uint32(vec![2, 2, 2])),
        ),
        (
            "inkling.attention.sliding_window_pattern".into(),
            GgufMetadataValue::Array(MetadataArray::Bool(vec![false, true, false])),
        ),
        (
            "inkling.attention.key_length".into(),
            GgufMetadataValue::Uint32(4),
        ),
        ("inkling.vocab_size".into(), GgufMetadataValue::Uint32(32)),
        (
            "inkling.attention.sliding_window".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "inkling.dense_block_count".into(),
            GgufMetadataValue::Uint32(1),
        ),
        (
            "inkling.shortconv_kernel".into(),
            GgufMetadataValue::Uint32(3),
        ),
        ("inkling.rel_extent".into(), GgufMetadataValue::Uint32(8)),
        ("inkling.d_rel".into(), GgufMetadataValue::Uint32(4)),
        (
            "inkling.attention.layer_norm_rms_epsilon".into(),
            GgufMetadataValue::Float32(1e-6),
        ),
        (
            "inkling.unpadded_vocab_size".into(),
            GgufMetadataValue::Uint32(30),
        ),
        (
            "inkling.logit_scale_denom".into(),
            GgufMetadataValue::Float32(2.0),
        ),
        (
            "inkling.expert_feed_forward_length".into(),
            GgufMetadataValue::Uint32(8),
        ),
        (
            "inkling.feed_forward_length".into(),
            GgufMetadataValue::Uint32(16),
        ),
        ("inkling.expert_count".into(), GgufMetadataValue::Uint32(2)),
        (
            "inkling.expert_used_count".into(),
            GgufMetadataValue::Uint32(1),
        ),
        (
            "inkling.expert_shared_count".into(),
            GgufMetadataValue::Uint32(1),
        ),
        (
            "inkling.expert_weights_scale".into(),
            GgufMetadataValue::Float32(1.0),
        ),
        (
            "inkling.context_length".into(),
            GgufMetadataValue::Uint32(64),
        ),
    ])
}

fn inkling_gguf_layer_name(runtime: &str) -> Option<String> {
    for (source, target) in [
        ("model.embed_tokens.weight", "token_embd.weight"),
        ("model.embed_norm.weight", "token_embd_norm.weight"),
        ("model.norm.weight", "output_norm.weight"),
        ("lm_head.weight", "output.weight"),
    ] {
        if runtime == source {
            return Some(target.into());
        }
    }
    let rest = runtime.strip_prefix("model.layers.")?;
    let (layer, parameter) = rest.split_once('.')?;
    let target = match parameter {
        "input_layernorm.weight" => "attn_norm.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.r_proj.weight" => "attn_r.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        "self_attn.rel_proj" => "attn_rel_proj",
        "self_attn.k_sconv.weight" => "shortconv_k.weight",
        "self_attn.v_sconv.weight" => "shortconv_v.weight",
        "attn_sconv.weight" => "shortconv_attn.weight",
        "post_attention_layernorm.weight" => "ffn_norm.weight",
        "dense.gate_proj.weight" => "ffn_gate.weight",
        "dense.up_proj.weight" => "ffn_up.weight",
        "dense.down_proj.weight" => "ffn_down.weight",
        "dense_global_scale" | "moe.router.global_scale" => "ffn_gscale",
        "moe.router.weight" => "ffn_gate_inp.weight",
        "moe.router.bias" => "exp_probs_b.bias",
        "moe.experts.down_proj" => "ffn_down_exps.weight",
        "moe.shared_experts.down_proj" => "ffn_down_shexp.weight",
        "mlp_sconv.weight" => "shortconv_mlp.weight",
        _ => return None,
    };
    Some(format!("blk.{layer}.{target}"))
}

fn gguf_tensor_from_array(name: impl Into<String>, array: &Array) -> GgufFixtureTensor {
    let evaluated = array.evaluated().unwrap();
    let mut dimensions = array
        .shape()
        .iter()
        .rev()
        .map(|&dimension| dimension as u64)
        .collect::<Vec<_>>();
    if dimensions.is_empty() {
        dimensions.push(1);
    }
    f32_gguf_tensor(name, dimensions, evaluated.as_slice::<f32>().to_vec())
}

fn write_inkling_gguf_fixture(path: &Path) {
    let config = inkling_config();
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let (_args, parameters) = initialized_inkling_parameters(&config, stream);
    let mut specs = Vec::new();
    for (runtime, value) in &parameters {
        let runtime = runtime.as_str();
        for (suffix, gate_name, up_name) in [
            (
                ".moe.experts.gate_up_proj",
                "ffn_gate_exps.weight",
                "ffn_up_exps.weight",
            ),
            (
                ".moe.shared_experts.gate_up_proj",
                "ffn_gate_shexp.weight",
                "ffn_up_shexp.weight",
            ),
        ] {
            if let Some(prefix) = runtime.strip_suffix(suffix) {
                let layer = prefix.strip_prefix("model.layers.").unwrap();
                let gate = value.try_index_device((.., ..8, ..), stream).unwrap();
                let up = value.try_index_device((.., 8.., ..), stream).unwrap();
                specs.push(gguf_tensor_from_array(
                    format!("blk.{layer}.{gate_name}"),
                    &gate,
                ));
                specs.push(gguf_tensor_from_array(
                    format!("blk.{layer}.{up_name}"),
                    &up,
                ));
                break;
            }
        }
        if runtime.ends_with(".moe.experts.gate_up_proj")
            || runtime.ends_with(".moe.shared_experts.gate_up_proj")
        {
            continue;
        }
        let name = inkling_gguf_layer_name(runtime)
            .unwrap_or_else(|| panic!("missing Inkling GGUF name for {runtime}"));
        specs.push(gguf_tensor_from_array(name, value));
    }
    let tensors = specs
        .iter()
        .map(|tensor| TensorInput {
            name: &tensor.name,
            dimensions: &tensor.dimensions,
            ggml_type: GgmlType::F32,
            data: &tensor.data,
        })
        .collect::<Vec<_>>();
    Writer::default()
        .write(
            std::fs::File::create(path).unwrap(),
            &inkling_gguf_metadata(),
            &tensors,
        )
        .unwrap();
}

struct GgufFixtureTensor {
    name: String,
    dimensions: Vec<u64>,
    data: Vec<u8>,
}

fn patterned_values(length: usize, scale: f32, phase: usize) -> Vec<f32> {
    (0..length)
        .map(|index| {
            let centered = ((index * 17 + phase * 11) % 29) as f32 - 14.0;
            centered * scale
        })
        .collect()
}

fn f32_gguf_tensor(
    name: impl Into<String>,
    dimensions: Vec<u64>,
    values: Vec<f32>,
) -> GgufFixtureTensor {
    assert_eq!(dimensions.iter().product::<u64>() as usize, values.len());
    GgufFixtureTensor {
        name: name.into(),
        dimensions,
        data: values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    }
}

fn deepseek_gguf_metadata() -> BTreeMap<String, GgufMetadataValue> {
    BTreeMap::from([
        (
            "general.architecture".into(),
            GgufMetadataValue::String("deepseek2".into()),
        ),
        ("general.file_type".into(), GgufMetadataValue::Uint32(0)),
        ("deepseek2.block_count".into(), GgufMetadataValue::Uint32(2)),
        (
            "deepseek2.context_length".into(),
            GgufMetadataValue::Uint32(64),
        ),
        (
            "deepseek2.embedding_length".into(),
            GgufMetadataValue::Uint32(12),
        ),
        (
            "deepseek2.feed_forward_length".into(),
            GgufMetadataValue::Uint32(16),
        ),
        (
            "deepseek2.attention.head_count".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "deepseek2.attention.layer_norm_rms_epsilon".into(),
            GgufMetadataValue::Float32(0.000001),
        ),
        (
            "deepseek2.rope.freq_base".into(),
            GgufMetadataValue::Float32(10_000.0),
        ),
        (
            "deepseek2.rope.dimension_count".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "deepseek2.attention.q_lora_rank".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "deepseek2.attention.kv_lora_rank".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "deepseek2.attention.key_length_mla".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "deepseek2.attention.value_length_mla".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "deepseek2.leading_dense_block_count".into(),
            GgufMetadataValue::Uint32(1),
        ),
        (
            "deepseek2.expert_count".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "deepseek2.expert_shared_count".into(),
            GgufMetadataValue::Uint32(1),
        ),
        (
            "deepseek2.expert_feed_forward_length".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "deepseek2.expert_used_count".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "deepseek2.expert_group_count".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "deepseek2.expert_group_used_count".into(),
            GgufMetadataValue::Uint32(1),
        ),
        (
            "deepseek2.expert_gating_func".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "deepseek2.expert_weights_norm".into(),
            GgufMetadataValue::Bool(true),
        ),
        (
            "deepseek2.expert_weights_scale".into(),
            GgufMetadataValue::Float32(1.0),
        ),
        ("deepseek2.vocab_size".into(), GgufMetadataValue::Uint32(13)),
    ])
}

fn deepseek_gguf_tensors() -> Vec<GgufFixtureTensor> {
    let tensor = |name: &str, row_major_shape: &[u64], phase: usize| {
        let mut dimensions = row_major_shape.to_vec();
        dimensions.reverse();
        let elements = dimensions.iter().product::<u64>() as usize;
        f32_gguf_tensor(name, dimensions, patterned_values(elements, 0.003, phase))
    };
    let norm =
        |name: &str, width: u64| f32_gguf_tensor(name, vec![width], vec![1.0; width as usize]);
    let mut tensors = vec![
        tensor("token_embd.weight", &[13, 12], 1),
        norm("output_norm.weight", 12),
        tensor("output.weight", &[13, 12], 2),
    ];
    for layer in 0..2 {
        let phase = 3 + layer * 9;
        tensors.extend([
            norm(&format!("blk.{layer}.attn_norm.weight"), 12),
            norm(&format!("blk.{layer}.ffn_norm.weight"), 12),
            tensor(&format!("blk.{layer}.attn_q_a.weight"), &[4, 12], phase),
            norm(&format!("blk.{layer}.attn_q_a_norm.weight"), 4),
            tensor(&format!("blk.{layer}.attn_q_b.weight"), &[16, 4], phase + 1),
            tensor(
                &format!("blk.{layer}.attn_kv_a_mqa.weight"),
                &[6, 12],
                phase + 2,
            ),
            norm(&format!("blk.{layer}.attn_kv_a_norm.weight"), 4),
            tensor(
                &format!("blk.{layer}.attn_k_b.weight"),
                &[4, 4, 2],
                phase + 3,
            ),
            tensor(
                &format!("blk.{layer}.attn_v_b.weight"),
                &[4, 2, 4],
                phase + 4,
            ),
            tensor(
                &format!("blk.{layer}.attn_output.weight"),
                &[12, 8],
                phase + 5,
            ),
        ]);
    }
    tensors.extend([
        tensor("blk.0.ffn_gate.weight", &[16, 12], 21),
        tensor("blk.0.ffn_up.weight", &[16, 12], 22),
        tensor("blk.0.ffn_down.weight", &[12, 16], 23),
        tensor("blk.1.ffn_gate_inp.weight", &[4, 12], 24),
        f32_gguf_tensor(
            "blk.1.exp_probs_b.bias",
            vec![4],
            patterned_values(4, 0.001, 25),
        ),
        tensor("blk.1.ffn_gate_shexp.weight", &[4, 12], 26),
        tensor("blk.1.ffn_up_shexp.weight", &[4, 12], 27),
        tensor("blk.1.ffn_down_shexp.weight", &[12, 4], 28),
        tensor("blk.1.ffn_gate_exps.weight", &[4, 4, 12], 29),
        tensor("blk.1.ffn_up_exps.weight", &[4, 4, 12], 30),
        tensor("blk.1.ffn_down_exps.weight", &[4, 12, 4], 31),
    ]);
    tensors
}

fn write_deepseek_gguf_fixture(path: &Path) {
    let tensors = deepseek_gguf_tensors();
    let inputs = tensors
        .iter()
        .map(|tensor| TensorInput {
            name: &tensor.name,
            dimensions: &tensor.dimensions,
            ggml_type: GgmlType::F32,
            data: &tensor.data,
        })
        .collect::<Vec<_>>();
    Writer::default()
        .write(
            std::fs::File::create(path).unwrap(),
            &deepseek_gguf_metadata(),
            &inputs,
        )
        .unwrap();
}

fn kimi_linear_gguf_metadata() -> BTreeMap<String, GgufMetadataValue> {
    BTreeMap::from([
        (
            "general.architecture".into(),
            GgufMetadataValue::String("kimi-linear".into()),
        ),
        ("general.file_type".into(), GgufMetadataValue::Uint32(0)),
        (
            "kimi-linear.block_count".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "kimi-linear.embedding_length".into(),
            GgufMetadataValue::Uint32(12),
        ),
        (
            "kimi-linear.attention.head_count".into(),
            GgufMetadataValue::Uint32(3),
        ),
        (
            "kimi-linear.attention.head_count_kv".into(),
            GgufMetadataValue::Array(MetadataArray::Uint32(vec![0, 1])),
        ),
        (
            "kimi-linear.rope.dimension_count".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "kimi-linear.attention.key_length_mla".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "kimi-linear.vocab_size".into(),
            GgufMetadataValue::Uint32(13),
        ),
        (
            "kimi-linear.feed_forward_length".into(),
            GgufMetadataValue::Uint32(17),
        ),
        (
            "kimi-linear.context_length".into(),
            GgufMetadataValue::Uint32(64),
        ),
        (
            "kimi-linear.attention.layer_norm_rms_epsilon".into(),
            GgufMetadataValue::Float32(0.00001),
        ),
        (
            "kimi-linear.kda.head_dim".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "kimi-linear.ssm.conv_kernel".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "kimi-linear.expert_count".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "kimi-linear.expert_feed_forward_length".into(),
            GgufMetadataValue::Uint32(9),
        ),
        (
            "kimi-linear.attention.kv_lora_rank".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "kimi-linear.attention.value_length_mla".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "kimi-linear.leading_dense_block_count".into(),
            GgufMetadataValue::Uint32(1),
        ),
        (
            "kimi-linear.expert_used_count".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "kimi-linear.expert_shared_count".into(),
            GgufMetadataValue::Uint32(1),
        ),
    ])
}

fn kimi_linear_gguf_specs() -> Vec<GgufFixtureTensor> {
    let tensor = |name: &str, row_major_shape: &[u64], phase: usize| {
        let mut dimensions = row_major_shape.to_vec();
        dimensions.reverse();
        let elements = dimensions.iter().product::<u64>() as usize;
        f32_gguf_tensor(name, dimensions, patterned_values(elements, 0.003, phase))
    };
    let norm =
        |name: &str, width: u64| f32_gguf_tensor(name, vec![width], vec![1.0; width as usize]);
    let mut specs = vec![
        tensor("token_embd.weight", &[13, 12], 1),
        norm("output_norm.weight", 12),
        tensor("output.weight", &[13, 12], 2),
    ];
    for layer in 0..2 {
        specs.push(norm(&format!("blk.{layer}.attn_norm.weight"), 12));
        specs.push(norm(&format!("blk.{layer}.ffn_norm.weight"), 12));
    }
    specs.extend([
        tensor("blk.0.attn_q.weight", &[12, 12], 3),
        tensor("blk.0.attn_k.weight", &[12, 12], 4),
        tensor("blk.0.attn_v.weight", &[12, 12], 5),
        tensor("blk.0.ssm_conv1d_q.weight", &[12, 2], 6),
        tensor("blk.0.ssm_conv1d_k.weight", &[12, 2], 7),
        tensor("blk.0.ssm_conv1d_v.weight", &[12, 2], 8),
        tensor("blk.0.ssm_f_a.weight", &[4, 12], 9),
        tensor("blk.0.ssm_f_b.weight", &[12, 4], 10),
        tensor("blk.0.ssm_beta.weight", &[3, 12], 11),
        tensor("blk.0.ssm_g_a.weight", &[4, 12], 12),
        tensor("blk.0.ssm_g_b.weight", &[12, 4], 13),
        f32_gguf_tensor("blk.0.ssm_a", vec![3], vec![-0.7, -0.9, -1.1]),
        f32_gguf_tensor(
            "blk.0.ssm_dt.bias",
            vec![12],
            patterned_values(12, 0.002, 14),
        ),
        norm("blk.0.ssm_norm.weight", 4),
        tensor("blk.0.attn_output.weight", &[12, 12], 15),
        tensor("blk.0.ffn_gate.weight", &[17, 12], 16),
        tensor("blk.0.ffn_up.weight", &[17, 12], 17),
        tensor("blk.0.ffn_down.weight", &[12, 17], 18),
        tensor("blk.1.attn_q.weight", &[12, 12], 19),
        tensor("blk.1.attn_kv_a_mqa.weight", &[6, 12], 20),
        norm("blk.1.attn_kv_a_norm.weight", 4),
        tensor("blk.1.attn_kv_b.weight", &[12, 4], 21),
        tensor("blk.1.attn_output.weight", &[12, 6], 22),
        tensor("blk.1.ffn_gate_inp.weight", &[4, 12], 23),
        f32_gguf_tensor(
            "blk.1.exp_probs_b.bias",
            vec![4],
            patterned_values(4, 0.001, 24),
        ),
        tensor("blk.1.ffn_gate_shexp.weight", &[9, 12], 25),
        tensor("blk.1.ffn_up_shexp.weight", &[9, 12], 26),
        tensor("blk.1.ffn_down_shexp.weight", &[12, 9], 27),
        tensor("blk.1.ffn_gate_exps.weight", &[4, 9, 12], 28),
        tensor("blk.1.ffn_up_exps.weight", &[4, 9, 12], 29),
        tensor("blk.1.ffn_down_exps.weight", &[4, 12, 9], 30),
    ]);
    specs
}

fn write_kimi_linear_gguf_fixture(path: &Path) {
    let specs = kimi_linear_gguf_specs();
    let tensors = specs
        .iter()
        .map(|tensor| TensorInput {
            name: &tensor.name,
            dimensions: &tensor.dimensions,
            ggml_type: GgmlType::F32,
            data: &tensor.data,
        })
        .collect::<Vec<_>>();
    Writer::default()
        .write(
            std::fs::File::create(path).unwrap(),
            &kimi_linear_gguf_metadata(),
            &tensors,
        )
        .unwrap();
}

fn render_failure(rank: usize, output: &Output) -> String {
    format!(
        "pipeline Ring rank {rank} exited with {}\n--- stdout ---\n{}\n--- stderr ---\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    )
}

/// Run with:
/// `cargo test -p eredu-backend-mlx --lib tests::distributed_pipeline_ring::ring_two_process_pipeline -- --ignored --exact --nocapture`
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_pipeline() {
    run_ring_pipeline(false, FixtureFamily::Llama);
}

#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_pipeline_inspection_uses_canonical_paths() {
    run_ring_pipeline_mode(false, FixtureFamily::Llama, WorkerMode::OpaqueInspection);
}

#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_pipeline_final_output_intervention_is_uniform_across_families() {
    for family in [
        FixtureFamily::Llama,
        FixtureFamily::Qwen3,
        FixtureFamily::DeepSeek,
        FixtureFamily::KimiLinear,
    ] {
        run_ring_pipeline_mode(false, family, WorkerMode::FinalOutputIntervention);
    }
}

/// Runs backend-generic text generation across a pipeline whose non-output
/// rank legitimately produces no local logits.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_pipeline_generic_text_generation() {
    run_ring_pipeline_mode(
        false,
        FixtureFamily::Llama,
        WorkerMode::OpaqueTextGeneration,
    );
}

/// Compares the public Llama TP=2 session's prefill and decode logits with a
/// fully resident single-rank reference built from the identical checkpoint.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic Llama fixture"]
fn ring_two_process_llama_tensor_parallel_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Llama,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        Some("tp"),
    );
}

/// Compares a two-rank PP=2 neutral Llama session with the same resident reference.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic Llama fixture"]
fn ring_two_process_llama_pipeline_parallel_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Llama,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        None,
    );
}

/// Compares a four-rank TP=2, PP=2 neutral session with the same resident reference.
#[test]
#[ignore = "requires the MLX Ring backend, four loopback CPU ranks, and the synthetic Llama fixture"]
fn ring_four_process_llama_tensor_pipeline_parallel_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Llama,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        Some("tp-pp"),
    );
}

/// Proves selected affine transformation stays inside the neutral TP session.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic Llama fixture"]
fn ring_two_process_llama_transformed_tensor_parallel_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Llama,
        WorkerMode::OpaqueSessionRequantize,
        checkpoint,
        checkpoint_path,
        Some("tp"),
    );
}

/// Proves a nonzero PP-local source unit is transformed into its exact target unit.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic Llama fixture"]
fn ring_two_process_llama_transformed_pipeline_parallel_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Llama,
        WorkerMode::OpaqueSessionRequantize,
        checkpoint,
        checkpoint_path,
        None,
    );
}

/// Runs the same TP=2 resident-reference oracle through the Mistral
/// specialization of the shared neutral decoder.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic Mistral fixture"]
fn ring_two_process_mistral_tensor_parallel_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_mistral_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Mistral,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        Some("tp"),
    );
}

/// Compares a two-rank PP=2 neutral Mistral session with the same resident reference.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic Mistral fixture"]
fn ring_two_process_mistral_pipeline_parallel_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_mistral_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Mistral,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        None,
    );
}

/// Compares a four-rank TP=2, PP=2 neutral Mistral session with the same resident reference.
#[test]
#[ignore = "requires the MLX Ring backend, four loopback CPU ranks, and the synthetic Mistral fixture"]
fn ring_four_process_mistral_tensor_pipeline_parallel_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_mistral_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Mistral,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        Some("tp-pp"),
    );
}

/// Proves the public neutral TP loader consumes a single unindexed Llama
/// SafeTensors payload without selecting the complete-model bridge.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_llama_unindexed_safetensors_tensor_parallel_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_unindexed_llama_compatible_fixture(checkpoint.path(), "llama");
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Llama,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        Some("tp"),
    );
}

/// Proves the public neutral PP loader consumes a single unindexed Mistral
/// SafeTensors payload and matches the same-artifact resident reference.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_mistral_unindexed_safetensors_pipeline_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_unindexed_llama_compatible_fixture(checkpoint.path(), "mistral");
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Mistral,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        None,
    );
}

/// Proves the public neutral PP loader consumes an admitted Llama GGUF store
/// without instantiating a backend-owned pipeline model.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_llama_gguf_pipeline_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    let checkpoint_path = checkpoint.path().join("model.gguf");
    write_llama_compatible_gguf(&checkpoint_path, "llama");
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Llama,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        None,
    );
}

/// Proves the public neutral TP loader consumes an admitted Mistral GGUF store
/// and matches the same-artifact resident reference.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_mistral_gguf_tensor_parallel_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    let checkpoint_path = checkpoint.path().join("model.gguf");
    write_llama_compatible_gguf(&checkpoint_path, "mistral");
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Mistral,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        Some("tp"),
    );
}

/// Proves the public generic loader and architecture-erased model session own
/// pipeline loading, prefill, repeated decode, cache state, and communication.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_opaque_model_session() {
    run_ring_pipeline_mode(false, FixtureFamily::Llama, WorkerMode::OpaqueSession);
}

/// Verifies fair multi-request scheduling, independent request caches, exact
/// schedule consensus, variable prompt shapes, decode parity, EOS, and cancel.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_pipeline_opaque_session_repeated_decode() {
    run_ring_pipeline_mode(false, FixtureFamily::Llama, WorkerMode::OpaqueSession);
}

/// Verifies the public Llama PP session genuinely selects disk-streamed local
/// layers while preserving numeric output, cache isolation, and neutral ownership.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_dense_stream_opaque_session() {
    run_ring_pipeline_mode(true, FixtureFamily::Llama, WorkerMode::OpaqueSession);
}

/// Verifies public Mistral TP execution genuinely selects host-layerwise
/// traversal while retaining the neutral partitioned constructor.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_mistral_layerwise_host_tensor_parallel_opaque_session() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_mistral_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::LayerwiseHost,
        FixtureFamily::Mistral,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        Some("tp"),
    );
}

/// Verifies that divergent rank-local schedules fail before point-to-point
/// Exercises paged cache selection through the opaque session lifecycle.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_pipeline_opaque_session_cache_policy() {
    run_ring_pipeline_mode(false, FixtureFamily::Llama, WorkerMode::OpaqueSession);
}

/// Run with:
/// `cargo test -p eredu-backend-mlx --lib tests::distributed_pipeline_ring::ring_two_process_dense_stream_pipeline -- --ignored --exact --nocapture`
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_dense_stream_pipeline() {
    run_ring_pipeline(true, FixtureFamily::Llama);
}

/// Verifies stage-local 4-bit materialization precedes TP+PP dense-stream
/// residency and preserves synchronized prefill, decode, and generation.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_qwen3_requantized_dense_stream_tensor_pipeline() {
    run_ring_cartesian_pipeline_mode(true, FixtureFamily::Qwen3, "tp-pp", WorkerMode::Requantize);
}

/// Verifies the same packed stage overlay feeds host-layerwise residency before
/// bounded device promotion under TP+PP.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_qwen3_requantized_layerwise_host_tensor_pipeline() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::Qwen3,
        "tp-pp",
        WorkerMode::Requantize,
    );
}

/// Verifies host-resident stage layers and bounded device promotion compose
/// with TP-sharded attention, cache state, and pipeline transport.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_qwen3_layerwise_host_tensor_pipeline() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::Qwen3,
        "tp-pp",
        WorkerMode::Standard,
    );
}

/// Verifies DeepSeek MLA paged-prefix persistence across two pipeline stages.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_deepseek_pipeline_persistence() {
    run_ring_pipeline(false, FixtureFamily::DeepSeek);
}

/// Proves prediction-free DeepSeek-V3 pure TP uses one neutral routed session.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_deepseek_v3_tensor_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::DeepSeek,
        "tp",
        WorkerMode::OpaqueSession,
    );
}

/// Proves a public DeepSeek-V3 artifact containing embedded MTP weights builds
/// one neutral ordinary TP target and retains its typed prediction extension,
/// without constructing a complete-TP or pipeline target shell.
#[test]
#[ignore = "spawns two local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_deepseek_v3_mtp_target_tensor_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::DeepSeek,
        "tp",
        WorkerMode::OpaqueDeepSeekMtpTarget,
    );
}

/// Proves prediction-free DeepSeek-V3 pure PP uses typed MLA boundaries.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_deepseek_v3_pipeline_opaque_session() {
    run_ring_pipeline_mode(false, FixtureFamily::DeepSeek, WorkerMode::OpaqueSession);
}

/// Proves prediction-free DeepSeek-V3 pure EP uses exact compact expert banks.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_deepseek_v3_expert_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::DeepSeek,
        "ep",
        WorkerMode::OpaqueSession,
    );
}

/// Proves prediction-free DeepSeek-V3 TP x PP remains on the neutral driver.
#[test]
#[ignore = "spawns four local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_deepseek_v3_tensor_pipeline_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::DeepSeek,
        "tp-pp",
        WorkerMode::OpaqueSession,
    );
}

/// Proves prediction-free DeepSeek-V3 TP x EP consumes compound local banks.
#[test]
#[ignore = "spawns four local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_deepseek_v3_tensor_expert_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::DeepSeek,
        "tp-ep",
        WorkerMode::OpaqueSession,
    );
}

/// Proves prediction-free DeepSeek-V3 PP x EP uses all-stage expert waves.
#[test]
#[ignore = "spawns four local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_deepseek_v3_pipeline_expert_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::DeepSeek,
        "pp-ep",
        WorkerMode::OpaqueSession,
    );
}

/// Proves prediction-free DeepSeek-V3 TP x PP x EP uses one neutral session.
#[test]
#[ignore = "spawns eight local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_deepseek_v3_triple_axis_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::DeepSeek,
        "tp-pp-ep",
        WorkerMode::OpaqueSession,
    );
}

/// Verifies DeepSeek TP=2 + PP=2 across dense and routed-MoE stages with
/// tensor-sharded MLA, compressed caches, bounded reads, and generation.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_deepseek_tensor_pipeline() {
    run_ring_cartesian_pipeline(true, FixtureFamily::DeepSeek, "tp-pp");
}

/// Verifies DeepSeek PP=2 + EP=2 across a dense-to-MoE stage boundary with
/// stage-local expert ownership, compressed cache persistence, and generation.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_deepseek_pipeline_expert() {
    run_ring_cartesian_pipeline(true, FixtureFamily::DeepSeek, "pp-ep");
}

/// Proves resident DeepSeek TP=2 x PP=2 x EP=2 execution across compressed
/// MLA state, TP-sharded shared projections, and EP-owned routed experts.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_deepseek_triple_axis() {
    run_ring_cartesian_pipeline(false, FixtureFamily::DeepSeek, "tp-pp-ep");
}

/// Verifies DeepSeek V4 local/compressed attention, hyper-connections, hash
/// routing, prompt persistence, and embedded MTP across two pipeline stages.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_deepseek_v4_pipeline_persistence_and_mtp() {
    run_ring_pipeline(false, FixtureFamily::DeepSeekV4);
}

/// Ensures the backend-generic speculative capability query delegates to the
/// distributed session instead of assuming a replicated complete model.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_deepseek_v4_prepared_speculative_capability() {
    run_ring_pipeline_mode(
        false,
        FixtureFamily::DeepSeekV4,
        WorkerMode::OpaquePreparedSpeculativeCapability,
    );
}

/// Proves sequential DeepSeek-V4 MTP uses the neutral pooling-state target and
/// extension-only MLX units without a complete-TP or pipeline target shell.
#[test]
#[ignore = "spawns two local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_deepseek_v4_mtp_target_tensor_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::DeepSeekV4,
        "tp",
        WorkerMode::OpaqueDeepSeekMtpTarget,
    );
}

/// Proves admitted DeepSeek-V4 DSpark executes fused proposals, target
/// verification, and transactional commit through the public neutral scheduler.
#[test]
#[ignore = "spawns two local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_deepseek_v4_dspark_tensor_parallel_neutral_scheduler() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::DeepSeekV4,
        "tp",
        WorkerMode::OpaqueDeepSeekDsparkTarget,
    );
}

/// Proves a prediction-bearing V4 artifact reuses its neutral pure-EP target
/// when the prepared speculative capability is queried.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_deepseek_v4_expert_prepared_speculative_capability() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::DeepSeekV4,
        "ep",
        WorkerMode::OpaquePreparedSpeculativeCapability,
    );
}

/// Verifies V4 output-group/head sharding and pipeline transport together.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_deepseek_v4_tensor_pipeline() {
    run_ring_cartesian_pipeline(true, FixtureFamily::DeepSeekV4, "tp-pp");
}

/// Verifies V4 token/learned routing through stage-local expert ownership.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_deepseek_v4_pipeline_expert() {
    run_ring_cartesian_pipeline(true, FixtureFamily::DeepSeekV4, "pp-ep");
}

/// Proves the prediction-free V4 target takes the neutral pooling-state route
/// through exact TP and EP manifest groups without constructing a duplicate model.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_deepseek_v4_prediction_free_tensor_expert_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::DeepSeekV4,
        "tp-ep",
        WorkerMode::OpaqueSessionPredictionFree,
    );
}

/// Exercises V4 TP, PP, EP, streamed non-experts, and independent expert
/// caching in the full admitted Cartesian topology.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_deepseek_v4_streamed_triple_axis_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::DeepSeekV4,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Keeps DeepSeek non-expert stage units resident while routed experts remain
/// independently cached across TP, PP, and EP.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_deepseek_resident_nonexpert_triple_axis_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::DeepSeek,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Exercises dense-streamed DeepSeek non-experts and independent expert
/// caching across all three Cartesian axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_deepseek_streamed_triple_axis_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::DeepSeek,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Compares the cache-backed neutral DeepSeek V3 session under TP=2 x EP=2
/// with the replicated model, covering rank-local expert geometry and exact-once TP.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_deepseek_cached_tensor_expert_model_parity() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::DeepSeek,
        "tp-ep",
        WorkerMode::OpaqueSessionAddressableParameterBank,
    );
}

/// Applies the same model-level TP+EP parity check to DeepSeek V4, including
/// its hyper-connection block and independently cached routed expert bank.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_deepseek_v4_cached_tensor_expert_model_parity() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::DeepSeekV4,
        "tp-ep",
        WorkerMode::OpaqueSessionAddressableParameterBank,
    );
}

/// Exercises host-layerwise DeepSeek MLA blocks and independent expert caches
/// across all three axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_deepseek_layerwise_host_triple_axis_parameter_bank() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::DeepSeek,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Covers DeepSeek2 GGUF recipes, bounded reads, and independent expert
/// caching across TP, PP, and EP.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_deepseek_gguf_triple_axis_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::DeepSeekGguf,
        "tp-pp-ep",
        WorkerMode::OpaqueSessionAddressableParameterBank,
    );
}

/// Covers independent DeepSeek expert caching for TP+PP with EP inactive.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_deepseek_tensor_pipeline_parameter_bank_without_ep() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::DeepSeek,
        "tp-pp",
        WorkerMode::AddressableParameterBank,
    );
}

/// Verifies cached DeepSeek schedule failure reaches consensus without leaving
/// compressed MLA state reusable.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_deepseek_parameter_bank_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::DeepSeek,
        "pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Verifies dependency-safe Gemma placement, auxiliary-state transport, shared
/// KV decode state, and prompt-cache restoration across two pipeline stages.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_gemma_pipeline() {
    run_ring_pipeline(false, FixtureFamily::Gemma);
}

/// Exercises Gemma 4's canonical vision, audio, and decoder unit traversal
/// through bounded checkpoint streaming across two pipeline stages.
#[test]
#[ignore = "spawns local processes, opens loopback sockets, and initializes MLX; run explicitly"]
fn ring_two_process_gemma4_dense_stream_pipeline() {
    run_ring_pipeline(true, FixtureFamily::Gemma);
}

/// Verifies biased GQA, mixed full/sliding cache persistence, and two-stage
/// execution for a Qwen2 decoder.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_qwen2_pipeline() {
    run_ring_pipeline(false, FixtureFamily::Qwen2);
}

/// Compares Qwen2 TP=2 prefill and decode logits with the fully resident
/// single-rank public-loader result.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic Qwen2 fixture"]
fn ring_two_process_qwen2_tensor_parallel_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_qwen_fixture(checkpoint.path(), "qwen2");
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Qwen2,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        Some("tp"),
    );
}

/// Proves indexed SafeTensors follows the same neutral Qwen2 TP constructor.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_qwen2_indexed_tensor_parallel_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_indexed_qwen_fixture(checkpoint.path(), "qwen2");
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Qwen2,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        Some("tp"),
    );
}

/// Proves Qwen2 pure PP uses one neutral manifest and the public cache lifecycle.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_qwen2_pipeline_resident_reference() {
    run_ring_pipeline_mode(false, FixtureFamily::Qwen2, WorkerMode::OpaqueSession);
}

/// Compares dense Qwen3 TP=2 prefill and decode logits with the fully
/// resident single-rank public-loader result.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic dense Qwen3 fixture"]
fn ring_two_process_qwen3_tensor_parallel_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_qwen_fixture(checkpoint.path(), "qwen3");
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Qwen3,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        Some("tp"),
    );
}

/// Proves one-rank MLX cache preparation failure rolls back the peer shard,
/// propagates its causal phase, and fences retry before further native work.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_qwen3_prompt_cache_prepare_failure_rolls_back_and_fences() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_qwen_fixture(checkpoint.path(), "qwen3");
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Qwen3,
        WorkerMode::PromptCachePrepareFailure,
        checkpoint,
        checkpoint_path,
        Some("tp"),
    );
}

/// Proves Qwen3 pure PP uses the family-blind neutral resident driver.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_qwen3_pipeline_resident_reference() {
    run_ring_pipeline_mode(false, FixtureFamily::Qwen3, WorkerMode::OpaqueSession);
}

/// Proves dense Qwen2 GGUF enters the neutral TP path without a complete shell.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_qwen2_gguf_tensor_parallel_resident_reference() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen2Gguf,
        "tp",
        WorkerMode::OpaqueSession,
    );
}

/// Proves dense Qwen3 GGUF enters the neutral pure-PP path.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_qwen3_gguf_pipeline_resident_reference() {
    run_ring_pipeline_mode(false, FixtureFamily::Qwen3Gguf, WorkerMode::OpaqueSession);
}

/// Compares Qwen2 TP=2 x PP=2 prefill, decode, and prompt-cache continuity
/// with the fully resident single-rank public-loader result.
#[test]
#[ignore = "requires the MLX Ring backend, four loopback CPU ranks, and the synthetic Qwen2 fixture"]
fn ring_four_process_qwen2_tensor_pipeline_resident_reference() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen2,
        "tp-pp",
        WorkerMode::OpaqueSession,
    );
}

/// Proves Qwen3 TP=2 x PP=2 numeric/cache parity through neutral construction.
#[test]
#[ignore = "requires the MLX Ring backend and four loopback CPU ranks"]
fn ring_four_process_qwen3_tensor_pipeline_resident_reference() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3,
        "tp-pp",
        WorkerMode::OpaqueSession,
    );
}

/// Proves bounded host-local Qwen2 units retain neutral TP execution.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_qwen2_layerwise_host_tensor_parallel_reference() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::Qwen2,
        "tp",
        WorkerMode::OpaqueSession,
    );
}

/// Proves disk-streamed Qwen3 units retain neutral pure-PP execution.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_qwen3_dense_stream_pipeline_reference() {
    run_ring_pipeline_mode(true, FixtureFamily::Qwen3, WorkerMode::OpaqueSession);
}

/// Proves the architecture-selected affine transform materializes once for TP.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_qwen3_transformed_tensor_parallel_reference() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3,
        "tp",
        WorkerMode::OpaqueSessionRequantize,
    );
}

/// Proves the architecture-selected affine transform materializes once for PP.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_qwen3_transformed_pipeline_reference() {
    run_ring_pipeline_mode(
        false,
        FixtureFamily::Qwen3,
        WorkerMode::OpaqueSessionRequantize,
    );
}

/// Proves transformed Qwen3 TP=2 x PP=2 uses one neutral construction.
#[test]
#[ignore = "requires the MLX Ring backend and four loopback CPU ranks"]
fn ring_four_process_qwen3_transformed_tensor_pipeline_reference() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3,
        "tp-pp",
        WorkerMode::OpaqueSessionRequantize,
    );
}

/// Proves routed Qwen3-MoE GGUF uses the neutral partitioned session.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_qwen3_moe_gguf_tensor_parallel_neutral_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3MoeGguf,
        "tp",
        WorkerMode::OpaqueSession,
    );
}

/// Verifies Q/K-normalized Qwen3 execution through streamed local layers.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_qwen3_dense_stream_pipeline() {
    run_ring_pipeline(true, FixtureFamily::Qwen3);
}

/// Verifies Qwen3 routed-expert ownership, paged cache persistence, and
/// rank-synchronized two-stage execution through the shared Qwen stage.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_qwen3_moe_pipeline() {
    run_ring_pipeline(false, FixtureFamily::Qwen3Moe);
}

/// Verifies GPT-OSS native MXFP4 experts and mixed full/sliding state across
/// two pipeline ranks.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_gpt_oss_pipeline() {
    run_ring_pipeline(false, FixtureFamily::GptOss);
}

/// Proves public Qwen3-MoE TP uses the typed neutral partition constructor.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_qwen3_moe_tensor_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3Moe,
        "tp",
        WorkerMode::OpaqueSession,
    );
}

/// Proves Qwen3-MoE pure PP uses routed local units and typed boundaries.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_qwen3_moe_pipeline_opaque_session() {
    run_ring_pipeline_mode(false, FixtureFamily::Qwen3Moe, WorkerMode::OpaqueSession);
}

/// Proves Qwen3-MoE TP=2 x PP=2 stays on the neutral routed driver.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_qwen3_moe_tensor_pipeline_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3Moe,
        "tp-pp",
        WorkerMode::OpaqueSession,
    );
}

/// Proves public GPT-OSS TP uses the typed neutral partition constructor.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_gpt_oss_tensor_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::GptOss,
        "tp",
        WorkerMode::OpaqueSession,
    );
}

/// Proves GPT-OSS pure PP uses the neutral routed pipeline strategy.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_gpt_oss_pipeline_opaque_session() {
    run_ring_pipeline_mode(false, FixtureFamily::GptOss, WorkerMode::OpaqueSession);
}

/// Proves GPT-OSS TP=2 x PP=2 uses the neutral routed pipeline strategy.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_gpt_oss_tensor_pipeline_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::GptOss,
        "tp-pp",
        WorkerMode::OpaqueSession,
    );
}

/// Proves Qwen3-MoE PP=2 x EP=2 follows the architecture-selected collective
/// wave on inactive pipeline stages through one neutral public session.
#[test]
#[ignore = "spawns four local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_qwen3_moe_pipeline_expert_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3Moe,
        "pp-ep",
        WorkerMode::OpaqueSession,
    );
}

/// Proves GPT-OSS PP=2 x EP=2 preserves its biased reverse wave through one
/// neutral public session.
#[test]
#[ignore = "spawns four local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_gpt_oss_pipeline_expert_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::GptOss,
        "pp-ep",
        WorkerMode::OpaqueSession,
    );
}

/// Proves GPT-OSS TP=2 x PP=2 x EP=2 executes the architecture-selected
/// world-wide expert waves through the neutral public session.
#[test]
#[ignore = "spawns eight local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_gpt_oss_triple_axis_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::GptOss,
        "tp-pp-ep",
        WorkerMode::OpaqueSession,
    );
}

/// Proves public Qwen3-MoE EP uses neutral owner/count exchange.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_qwen3_moe_expert_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3Moe,
        "ep",
        WorkerMode::OpaqueSession,
    );
}

/// Proves public GPT-OSS EP uses neutral owner/count exchange.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_gpt_oss_expert_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::GptOss,
        "ep",
        WorkerMode::OpaqueSession,
    );
}

/// Proves public Qwen3-MoE TP=2 x EP=2 uses one neutral manifest and the
/// consensus-proven overlapping logical-subgroup exchange wave.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_qwen3_moe_tensor_expert_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3Moe,
        "tp-ep",
        WorkerMode::OpaqueSession,
    );
}

/// Proves public GPT-OSS TP=2 x EP=2 retains its biased expert semantics over
/// one neutral manifest and the consensus-proven logical-subgroup wave.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_gpt_oss_tensor_expert_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::GptOss,
        "tp-ep",
        WorkerMode::OpaqueSession,
    );
}

/// Proves TP+EP uses the selected rank-local addressable Qwen bank and exposes
/// eviction/reload telemetry through the neutral public session.
#[test]
#[ignore = "spawns four local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_qwen3_moe_tensor_expert_addressable_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3Moe,
        "tp-ep",
        WorkerMode::OpaqueSessionEvictingAddressableParameterBank,
    );
}

/// Proves PP+EP uses one neutral session with independent Qwen expert banks.
#[test]
#[ignore = "spawns four local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_qwen3_moe_pipeline_expert_addressable_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3Moe,
        "pp-ep",
        WorkerMode::OpaqueSessionAddressableParameterBank,
    );
}

/// Proves GPT-OSS TP+PP+EP composes bounded ordinary storage with independent
/// expert banks while preserving post-reduction bias exactly once.
#[test]
#[ignore = "spawns eight local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_gpt_oss_streamed_triple_axis_addressable_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::GptOss,
        "tp-pp-ep",
        WorkerMode::OpaqueSessionAddressableParameterBank,
    );
}

/// Proves the ReLU-squared routed equation uses the same exact addressable
/// session while inactive PP stages retain an empty local bank catalog.
#[test]
#[ignore = "spawns eight local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_nemotron_h_streamed_triple_axis_addressable_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::NemotronH,
        "tp-pp-ep",
        WorkerMode::OpaqueSessionAddressableParameterBank,
    );
}

/// Proves GPT-OSS resident TP+PP+EP execution against the single-rank model.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_gpt_oss_triple_axis() {
    run_ring_cartesian_pipeline(false, FixtureFamily::GptOss, "tp-pp-ep");
}

/// Exercises GPT-OSS triple-axis dense streaming with independent expert caches.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_gpt_oss_streamed_triple_axis_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::GptOss,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Exercises GPT-OSS host-backed non-expert layers with independent expert
/// caching across TP=2 x PP=2 x EP=2.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_gpt_oss_layerwise_host_triple_axis_parameter_bank() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::GptOss,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Exercises canonical type-39 GPT-OSS GGUF across all three axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_gpt_oss_gguf_triple_axis_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::GptOssGguf,
        "tp-pp-ep",
        WorkerMode::OpaqueSessionAddressableParameterBank,
    );
}

/// Covers independent GPT-OSS expert caching when PP is active without EP.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_gpt_oss_pipeline_parameter_bank() {
    run_ring_pipeline_mode(
        false,
        FixtureFamily::GptOss,
        WorkerMode::AddressableParameterBank,
    );
}

/// Verifies opaque-session execution with GPT-OSS cached experts.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_gpt_oss_pipeline_parameter_bank_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::GptOss,
        "pp-ep",
        WorkerMode::OpaqueSessionAddressableParameterBank,
    );
}

/// Verifies descriptor-backed convolution state, paged KV state, and persisted
/// replay across two LFM2 pipeline ranks.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_lfm2_pipeline() {
    run_ring_pipeline(false, FixtureFamily::Lfm2);
}

/// Compares dense indexed LFM2 TP=2 prefill, repeated decode, and prompt-cache
/// restoration with the fully resident single-rank public-loader result.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic LFM2 fixture"]
fn ring_two_process_lfm2_tensor_parallel_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_lfm2_pipeline_fixture(checkpoint.path(), false);
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Lfm2,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        Some("tp"),
    );
}

/// Compares dense indexed LFM2 PP=2 through the neutral resident runtime with
/// the fully resident single-rank public-loader result.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic LFM2 fixture"]
fn ring_two_process_lfm2_pipeline_parallel_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_lfm2_pipeline_fixture(checkpoint.path(), false);
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Lfm2,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        None,
    );
}

/// Compares dense indexed LFM2 TP=2 x PP=2 through exact heterogeneous local
/// state and architecture-owned publication with the resident reference.
#[test]
#[ignore = "requires the MLX Ring backend, four loopback CPU ranks, and the synthetic LFM2 fixture"]
fn ring_four_process_lfm2_tensor_pipeline_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_lfm2_pipeline_fixture(checkpoint.path(), false);
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Lfm2,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        Some("tp-pp"),
    );
}

/// Proves the public opaque preparation retains LFM2's bounded disk-resident
/// parameter policy while using the neutral heterogeneous-state partition.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_lfm2_neutral_bounded_pipeline() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_lfm2_pipeline_fixture(checkpoint.path(), false);
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::DenseDiskStream,
        FixtureFamily::Lfm2,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        None,
    );
}

/// Verifies LFM2's heterogeneous operators through bounded local-layer reads.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_lfm2_dense_stream_pipeline() {
    run_ring_pipeline(true, FixtureFamily::Lfm2);
}

/// Verifies that LFM2-MoE uses the same heterogeneous pipeline-state runtime.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_lfm2_moe_pipeline() {
    run_ring_pipeline(false, FixtureFamily::Lfm2Moe);
}

/// Verifies LFM2 TP=2 + PP=2 with tensor-sharded convolution/attention state,
/// corresponding-coordinate stage transport, persistence, and generation.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_lfm2_tensor_pipeline() {
    run_ring_cartesian_pipeline(true, FixtureFamily::Lfm2, "tp-pp");
}

/// Verifies LFM2-MoE PP=2 + EP=2 across a dense-to-sparse stage boundary,
/// including stage-local expert exchange, persistence, and generation.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_lfm2_pipeline_expert() {
    run_ring_cartesian_pipeline(true, FixtureFamily::Lfm2Moe, "pp-ep");
}

/// Proves resident LFM2-MoE TP=2 x PP=2 x EP=2 execution, including
/// convolution/KV state, expert ownership, persistence, and generation.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_lfm2_moe_triple_axis() {
    run_ring_cartesian_pipeline(false, FixtureFamily::Lfm2Moe, "tp-pp-ep");
}

/// Exercises dense-streamed non-experts and independently cached LFM2 experts
/// across all three Cartesian axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_lfm2_moe_streamed_triple_axis_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::Lfm2Moe,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Exercises host-layerwise non-experts and independently cached LFM2 experts
/// across all three Cartesian axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_lfm2_moe_layerwise_host_triple_axis_parameter_bank() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::Lfm2Moe,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Exercises representative LFM2-MoE GGUF bindings, bounded non-expert reads,
/// and independent expert caching across all three axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_lfm2_moe_gguf_triple_axis_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::Lfm2MoeGguf,
        "tp-pp-ep",
        WorkerMode::OpaqueSessionAddressableParameterBank,
    );
}

/// Covers independent LFM2 expert caching when PP is active without EP.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_lfm2_moe_pipeline_parameter_bank() {
    run_ring_pipeline_mode(
        false,
        FixtureFamily::Lfm2Moe,
        WorkerMode::AddressableParameterBank,
    );
}

/// Verifies opaque-session execution with LFM2 cached experts.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_lfm2_moe_pipeline_parameter_bank_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Lfm2Moe,
        "pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Verifies Kimi's KDA and compressed-latent stages against resident prefill
/// and decode while keeping each rank's layer reads bounded.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_kimi_linear_dense_stream_pipeline() {
    run_ring_pipeline(true, FixtureFamily::KimiLinear);
}

/// Compares dense indexed Kimi Linear TP=2 prefill, repeated decode, and the
/// exact mixed KDA/MLA prompt-cache round trip with a resident single-rank run.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic Kimi Linear fixture"]
fn ring_two_process_kimi_linear_tensor_parallel_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_kimi_linear_dense_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::KimiLinear,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        Some("tp"),
    );
}

/// Compares dense indexed Kimi Linear PP=2 through the neutral resident
/// boundary and mixed-state persistence with a resident single-rank run.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic Kimi Linear fixture"]
fn ring_two_process_kimi_linear_pipeline_parallel_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_kimi_linear_dense_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::KimiLinear,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        None,
    );
}

/// Compares dense indexed Kimi Linear TP=2 x PP=2 with TP-local KDA state,
/// head-independent MLA state, publication authority, and cache restoration.
#[test]
#[ignore = "requires the MLX Ring backend, four loopback CPU ranks, and the synthetic Kimi Linear fixture"]
fn ring_four_process_kimi_linear_tensor_pipeline_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_kimi_linear_dense_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::KimiLinear,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        Some("tp-pp"),
    );
}

/// Compares dense prediction-free Nemotron-H TP=2 with exact TP-local Mamba
/// state and owner-only publication against a resident reference.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_nemotron_h_tensor_parallel_neutral_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_nemotron_dense_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::NemotronH,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        Some("tp"),
    );
}

/// Compares dense prediction-free Nemotron-H PP=2 with role-exact hidden,
/// token, and embedded boundary provenance plus prompt-cache restoration.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_nemotron_h_pipeline_neutral_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_nemotron_dense_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::NemotronH,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        None,
    );
}

/// Compares dense prediction-free Nemotron-H TP=2 x PP=2 with TP-local mixed
/// state, role-exact auxiliary transport, publication authority, and cache reload.
#[test]
#[ignore = "requires the MLX Ring backend and four loopback CPU ranks"]
fn ring_four_process_nemotron_h_tensor_pipeline_neutral_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_nemotron_dense_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::NemotronH,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        Some("tp-pp"),
    );
}

/// Proves the public opaque preparation retains bounded parameter reads for
/// the neutral mixed Mamba/attention Nemotron-H pipeline.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_nemotron_h_neutral_bounded_pipeline() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_nemotron_dense_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::DenseDiskStream,
        FixtureFamily::NemotronH,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        None,
    );
}

/// Proves load-time MXFP4 conversion consumes the architecture-retained source
/// partition before constructing the neutral dense Nemotron-H target.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_nemotron_h_neutral_transformed_pipeline() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_nemotron_dense_quantizable_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::NemotronH,
        WorkerMode::OpaqueSessionRequantize,
        checkpoint,
        checkpoint_path,
        None,
    );
}

/// Exercises the same Kimi stage adapter and heterogeneous cache contract from
/// a real GGUF artifact rather than a SafeTensors directory.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_kimi_linear_gguf_pipeline() {
    run_ring_pipeline(true, FixtureFamily::KimiLinearGguf);
}

/// Proves the public opaque preparation routes a bounded dense Kimi Linear
/// pipeline through the neutral partitioned session without a backend-owned
/// family stage adapter.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_kimi_linear_neutral_bounded_pipeline() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_kimi_linear_dense_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::DenseDiskStream,
        FixtureFamily::KimiLinear,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        None,
    );
}

/// Verifies Kimi Linear TP=2 + PP=2 across KDA and MLA stages with TP-local
/// recurrent state, shared/routed projections, persistence, and generation.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_kimi_linear_tensor_pipeline() {
    run_ring_cartesian_pipeline(true, FixtureFamily::KimiLinear, "tp-pp");
}

/// Exercises Kimi Linear TP+PP rank-local recipes from a representative GGUF
/// artifact, including bounded streaming reads.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_kimi_linear_gguf_tensor_pipeline() {
    run_ring_cartesian_pipeline(true, FixtureFamily::KimiLinearGguf, "tp-pp");
}

/// Verifies Kimi Linear PP=2 + EP=2 across the dense KDA to sparse MLA stage
/// transition with stage-local expert ownership, persistence, and generation.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_kimi_linear_pipeline_expert() {
    run_ring_cartesian_pipeline(true, FixtureFamily::KimiLinear, "pp-ep");
}

/// Exercises Kimi Linear PP+EP stage-local expert selection and bounded reads
/// from the representative GGUF fixture.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_kimi_linear_gguf_pipeline_expert() {
    run_ring_cartesian_pipeline(true, FixtureFamily::KimiLinearGguf, "pp-ep");
}

/// Proves resident Kimi Linear TP=2 x PP=2 x EP=2 execution across KDA and
/// MLA state, TP-sharded shared experts, and EP-owned routed experts.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_kimi_linear_triple_axis() {
    run_ring_cartesian_pipeline(false, FixtureFamily::KimiLinear, "tp-pp-ep");
}

/// Exercises dense-streamed Kimi non-experts and a stage-local independent
/// expert cache across all three Cartesian axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_kimi_linear_streamed_triple_axis_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::KimiLinear,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Exercises host-layerwise Kimi KDA/MLA state with independently cached
/// routed experts across TP, PP, and EP.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_kimi_linear_layerwise_host_triple_axis_parameter_bank() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::KimiLinear,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Exercises representative Kimi Linear GGUF recipes, bounded reads, and an
/// independent expert cache across all three axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_kimi_linear_gguf_triple_axis_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::KimiLinearGguf,
        "tp-pp-ep",
        WorkerMode::OpaqueSessionAddressableParameterBank,
    );
}

/// Covers independent Kimi expert caching for TP+PP with EP inactive.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_kimi_linear_tensor_pipeline_parameter_bank_without_ep() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::KimiLinear,
        "tp-pp",
        WorkerMode::AddressableParameterBank,
    );
}

/// Verifies cached-expert schedule failure reaches consensus without leaving
/// Kimi's recurrent or compressed-latent state reusable.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_kimi_linear_parameter_bank_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::KimiLinear,
        "pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Verifies Mamba, dense, sparse, and sliding-attention Nemotron operators over
/// the two balanced stage ranges.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_nemotron_h_dense_stream_pipeline() {
    run_ring_pipeline(true, FixtureFamily::NemotronH);
}

/// Verifies Nemotron-H TP=2 + PP=2 across Mamba, dense, sparse-MoE, and
/// sliding-attention stages with rank-local cache geometry and persistence.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_nemotron_h_tensor_pipeline() {
    run_ring_cartesian_pipeline(true, FixtureFamily::NemotronH, "tp-pp");
}

/// Verifies Nemotron-H-MoE PP=2 + EP=2 with stage-local routed experts,
/// matching-EP pipeline transport, cached decode, persistence, and generation.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_nemotron_h_pipeline_expert() {
    run_ring_cartesian_pipeline(true, FixtureFamily::NemotronH, "pp-ep");
}

/// Proves resident Nemotron-H-MoE TP=2 x PP=2 x EP=2 execution across Mamba,
/// dense, sparse, and attention layers with rank-local state and experts.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_nemotron_h_moe_triple_axis() {
    run_ring_cartesian_pipeline(false, FixtureFamily::NemotronH, "tp-pp-ep");
}

/// Exercises bounded MXFP4 dense-streamed non-experts and independently cached
/// Nemotron-H routed experts across all three Cartesian axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_nemotron_h_moe_streamed_triple_axis_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::NemotronH,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBankRequantize,
    );
}

/// Exercises host-layerwise non-experts and independently cached
/// Nemotron-H routed experts across all three Cartesian axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_nemotron_h_moe_layerwise_host_triple_axis_parameter_bank() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::NemotronH,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Exercises canonical Nemotron-H-MoE GGUF bindings, bounded non-expert
/// reads, and independent expert caching across all three axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_nemotron_h_moe_gguf_triple_axis_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::NemotronHGguf,
        "tp-pp-ep",
        WorkerMode::OpaqueSessionAddressableParameterBank,
    );
}

/// Covers independent Nemotron-H expert caching when PP is active without EP.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_nemotron_h_moe_pipeline_parameter_bank() {
    run_ring_pipeline_mode(
        false,
        FixtureFamily::NemotronH,
        WorkerMode::AddressableParameterBank,
    );
}

/// Verifies cached-expert session execution for Nemotron-H stateful stages.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_nemotron_h_moe_pipeline_parameter_bank_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::NemotronH,
        "pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Proves bounded MXFP4 materialization feeds stage-local Nemotron-H expert
/// caches under PP+EP, including persistence and synchronized decode.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_nemotron_h_quantized_pipeline_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::NemotronH,
        "pp-ep",
        WorkerMode::AddressableParameterBankRequantize,
    );
}

/// Proves a PP+EP stage can bounded-MXFP4-quantize and pin its complete rank-local
/// Nemotron-H expert banks without introducing an independent expert cache.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_nemotron_h_fully_resident_load_time_quantization() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::NemotronH,
        "pp-ep",
        WorkerMode::Requantize,
    );
}

/// Verifies Qwen3-Next linear and full-attention state through distributed
/// prefill, decode, persistence, and bounded streaming.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_qwen3_next_dense_stream_pipeline() {
    run_ring_pipeline(true, FixtureFamily::Qwen3Next);
}

/// Verifies that Qwen3.5 uses the same hybrid pipeline contract without being
/// treated as a Qwen3-Next checkpoint alias.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_qwen35_dense_stream_pipeline() {
    run_ring_pipeline(true, FixtureFamily::Qwen35);
}

/// Verifies output-owned Qwen Hybrid prediction state is persisted and restored
/// with the target prompt cache instead of being truncated to the decoder range.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_qwen35_prompt_cache_round_trip_includes_mtp() {
    run_ring_pipeline_mode(
        false,
        FixtureFamily::Qwen35,
        WorkerMode::QwenHybridPromptCache,
    );
}

/// Verifies Qwen3-Next TP=2 + PP=2 across recurrent and full-attention stages,
/// including rank-local state, persistence, and synchronized generation.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_qwen3_next_tensor_pipeline() {
    run_ring_cartesian_pipeline(true, FixtureFamily::Qwen3Next, "tp-pp");
}

/// Proves the unsupported direct Qwen3-Next prediction route is rejected at
/// neutral TP admission; embedded prediction requires its selected adapter.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic Qwen3-Next fixture"]
fn ring_two_process_qwen3_next_tensor_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3Next,
        "tp",
        WorkerMode::OpaqueUnsupportedDirectPartition,
    );
}

/// Proves the unsupported direct Qwen3.5 prediction route is rejected at
/// neutral TP admission; embedded prediction requires its selected adapter.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic Qwen3.5 fixture"]
fn ring_two_process_qwen35_tensor_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen35,
        "tp",
        WorkerMode::OpaqueUnsupportedDirectPartition,
    );
}

/// Covers pure TP loading and text execution through the conditional Qwen3.5
/// graph, including its vision-aware static parameter topology.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic multimodal Qwen3.5 fixture"]
fn ring_two_process_qwen35_multimodal_tensor_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen35Multimodal,
        "tp",
        WorkerMode::OpaqueSession,
    );
}

/// Covers the prediction-free conditional Qwen graph through the generic
/// composite partition binder. Prediction-bearing variants are classified as
/// separate neutral prediction targets before composite admission.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic conditional Qwen fixture"]
fn ring_two_process_qwen35_zero_prediction_tensor_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen35ZeroPrediction,
        "tp",
        WorkerMode::OpaqueQwenConditionalMedia,
    );
}

/// Covers the prediction-free conditional Qwen media graph across two neutral
/// pipeline owners without constructing a family pipeline shell.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic conditional Qwen fixture"]
fn ring_two_process_qwen35_zero_prediction_pipeline_neutral_composite_session() {
    run_ring_pipeline_mode(
        false,
        FixtureFamily::Qwen35ZeroPrediction,
        WorkerMode::OpaqueQwenConditionalMedia,
    );
}

/// Covers pure TP loading and multimodal execution for Qwen3-VL through the
/// neutral composite partition binder.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic Qwen3-VL fixture"]
fn ring_two_process_qwen3_vl_multimodal_tensor_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3Vl,
        "tp",
        WorkerMode::OpaqueQwen3VlMedia,
    );
}

/// Proves Qwen3-VL media continuation traverses the ordinary neutral PP session.
#[test]
#[ignore = "requires the MLX Ring backend and two loopback CPU ranks"]
fn ring_two_process_qwen3_vl_pipeline_neutral_composite_session() {
    run_ring_pipeline_mode(
        false,
        FixtureFamily::Qwen3Vl,
        WorkerMode::OpaqueQwen3VlMedia,
    );
}

/// Verifies Qwen3.5-MoE TP=2 + PP=2 with tensor-sharded routed/shared
/// intermediates and corresponding-coordinate pipeline transport.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_qwen35_moe_tensor_pipeline() {
    run_ring_cartesian_pipeline(true, FixtureFamily::Qwen35Moe, "tp-pp");
}

/// Verifies scheduled typed image ingress, TP-sharded vision/text blocks,
/// corresponding-coordinate PP transport, cached decode, and persistence.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_qwen35_multimodal_tensor_pipeline() {
    run_ring_cartesian_pipeline(false, FixtureFamily::Qwen35Multimodal, "tp-pp");
}

/// Verifies Qwen3-VL DeepStack vision, mRoPE text, and position-delta state
/// through tensor- and pipeline-parallel prefill and decode.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_qwen3_vl_tensor_pipeline() {
    run_ring_cartesian_pipeline(false, FixtureFamily::Qwen3Vl, "tp-pp");
}

/// Exercises Qwen3-VL's vision unit and decoder layers through bounded
/// checkpoint streaming under TP=2 x PP=2, with resident-reference logits.
#[test]
#[ignore = "requires the MLX Ring backend, four loopback CPU ranks, and the synthetic Qwen3-VL media fixture"]
fn ring_four_process_qwen3_vl_streamed_tensor_pipeline() {
    run_ring_cartesian_pipeline(true, FixtureFamily::Qwen3Vl, "tp-pp");
}

/// Exercises the same Qwen3-VL TP+PP graph with host-resident media/decoder
/// units and a one-unit device window.
#[test]
#[ignore = "requires the MLX Ring backend, four loopback CPU ranks, and the synthetic Qwen3-VL host-layerwise media fixture"]
fn ring_four_process_qwen3_vl_layerwise_host_tensor_pipeline() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::Qwen3Vl,
        "tp-pp",
        WorkerMode::Standard,
    );
}

/// Exercises routed Qwen3-VL across TP=2 x PP=2 x EP=2 with the neutral
/// shared vision tower and stage-local expert ownership.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_qwen3_vl_moe_triple_axis() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3VlMoe,
        "tp-pp-ep",
        WorkerMode::OpaqueQwen3VlMedia,
    );
}

/// Combines bounded Qwen3-VL media/decoder streaming with independent cached
/// routed experts across TP=2 x PP=2 x EP=2.
#[test]
#[ignore = "requires the MLX Ring backend, eight loopback CPU ranks, and the synthetic Qwen3-VL-MoE media fixture"]
fn ring_eight_process_qwen3_vl_moe_streamed_triple_axis_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::Qwen3VlMoe,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Combines host-layerwise Qwen3-VL media/decoder units with independently
/// cached routed experts across all three parallel axes.
#[test]
#[ignore = "requires the MLX Ring backend, eight loopback CPU ranks, and the synthetic Qwen3-VL-MoE host-layerwise media fixture"]
fn ring_eight_process_qwen3_vl_moe_layerwise_host_triple_axis_parameter_bank() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::Qwen3VlMoe,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Verifies multimodal Qwen3.5-MoE across all Cartesian axes with bounded
/// media/decoder reads and independently cached routed experts.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_qwen35_moe_multimodal_streamed_triple_axis() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::Qwen35MoeMultimodal,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Verifies the same typed ingress and expert-storage semantics with host-backed
/// vision and decoder windows.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_qwen35_moe_multimodal_layerwise_host_triple_axis() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::Qwen35MoeMultimodal,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Verifies Qwen3.5-MoE PP=2 + EP=2 with stage-local packed experts,
/// recurrent/full-attention state, persistence, and generation.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_qwen35_pipeline_expert() {
    run_ring_cartesian_pipeline(true, FixtureFamily::Qwen35Moe, "pp-ep");
}

/// Verifies the Qwen3-Next checkpoint specialization uses the same PP+EP
/// ownership and transport contract as Qwen3.5-MoE.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_qwen3_next_pipeline_expert() {
    run_ring_cartesian_pipeline(true, FixtureFamily::Qwen3NextMoe, "pp-ep");
}

/// Verifies resident Qwen3-Next-MoE execution across all Cartesian axes,
/// including recurrent state, routed/shared TP projections, and generation.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_qwen3_next_moe_triple_axis() {
    run_ring_cartesian_pipeline(false, FixtureFamily::Qwen3NextMoe, "tp-pp-ep");
}

/// Verifies Qwen3.5-MoE dense streaming composes with stage/EP-local expert
/// caches, bounded reads, prompt-cache reload, and synchronized decode.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_qwen35_moe_streamed_triple_axis_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::Qwen35Moe,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Exercises host-backed hybrid non-expert layers with independent expert
/// caching across TP, PP, and EP.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_qwen3_next_moe_layerwise_host_triple_axis_parameter_bank() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::Qwen3NextMoe,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Covers independent hybrid expert caching when PP is active without EP.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_qwen35_moe_pipeline_parameter_bank() {
    run_ring_pipeline_mode(
        false,
        FixtureFamily::Qwen35Moe,
        WorkerMode::AddressableParameterBank,
    );
}

/// Verifies cached Qwen hybrid expert execution through one session.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_qwen35_moe_pipeline_parameter_bank_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen35Moe,
        "pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Verifies arbitrary Cartesian composition with TP-sharded Qwen3-MoE
/// projections, stage-local EP ownership, corresponding-coordinate pipeline
/// transport, cache persistence, and globally synchronized generation.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_tensor_pipeline_expert() {
    run_ring_cartesian_pipeline(false, FixtureFamily::Qwen3Moe, "tp-pp-ep");
}

/// Verifies triple-axis ownership and generation with a tied output head.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_tied_tensor_pipeline_expert() {
    run_ring_cartesian_pipeline(false, FixtureFamily::Qwen3MoeTied, "tp-pp-ep");
}

/// Exercises the same triple-axis semantics with bounded rank-local layer
/// materialization.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_streamed_tensor_pipeline_expert() {
    run_ring_cartesian_pipeline(true, FixtureFamily::Qwen3Moe, "tp-pp-ep");
}

/// Proves complete Qwen3-MoE expert banks use the shared atomic packed overlay
/// under PP+EP without constructing an independent expert cache.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_qwen3_moe_fully_resident_load_time_quantization() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3Moe,
        "pp-ep",
        WorkerMode::Requantize,
    );
}

/// Verifies resident non-expert parameters plus stage/EP-local independent
/// expert caches across PP=2 x EP=2, including persistence and generation.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_resident_nonexpert_pipeline_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3Moe,
        "pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Verifies TP-sharded cached experts compose with dense-streamed non-experts
/// and corresponding-coordinate pipeline lanes in an eight-rank topology.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_streamed_tensor_pipeline_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::Qwen3Moe,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Exercises host-backed non-expert layers, bounded device windows, and
/// independent expert caching for Qwen3-MoE across all three axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_layerwise_host_tensor_pipeline_parameter_bank() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::Qwen3Moe,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Proves the same host-layerwise path reads canonical GGUF and preserves
/// stage-local ownership in a PP=2 x EP=2 topology.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_gguf_layerwise_host_pipeline_parameter_bank() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::Qwen3MoeGguf,
        "pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Exercises stage-local cached expert selections and bounded reads from GGUF.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_gguf_pipeline_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::Qwen3MoeGguf,
        "pp-ep",
        WorkerMode::OpaqueSessionAddressableParameterBank,
    );
}

/// Verifies one session owns both pipeline communication and expert caches.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_pipeline_parameter_bank_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3Moe,
        "pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Verifies PP-only stages cache all of their local layers' experts without
/// constructing an EP communicator. Prefill, decode, prompt persistence, and
/// synchronized generation are exercised by the shared worker.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_pipeline_parameter_bank_without_ep() {
    run_ring_pipeline_mode(
        false,
        FixtureFamily::Qwen3Moe,
        WorkerMode::AddressableParameterBank,
    );
}

/// Verifies TP-sharded cached experts and dense-streamed non-experts compose
/// across TP=2 x PP=2 while EP remains inactive.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_streamed_tensor_pipeline_parameter_bank_without_ep() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::Qwen3Moe,
        "tp-pp",
        WorkerMode::AddressableParameterBank,
    );
}

/// Exercises PP-only cache ownership and bounded reads from canonical GGUF.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_gguf_pipeline_parameter_bank_without_ep() {
    run_ring_pipeline_mode(
        true,
        FixtureFamily::Qwen3MoeGguf,
        WorkerMode::AddressableParameterBank,
    );
}

/// Executes the Qwen3-MoE triple-axis path from a canonical GGUF and verifies
/// rank-local ownership, bounded reads, cache persistence, and parity.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_gguf_tensor_pipeline_expert() {
    run_ring_cartesian_pipeline(false, FixtureFamily::Qwen3MoeGguf, "tp-pp-ep");
}

/// Exercises the canonical GGUF path with rank-local dense disk streaming.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_gguf_streamed_tensor_pipeline_expert() {
    run_ring_cartesian_pipeline(true, FixtureFamily::Qwen3MoeGguf, "tp-pp-ep");
}

/// Verifies opaque model-session execution across every triple-axis rank.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_tensor_pipeline_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3Moe,
        "tp-pp-ep",
        WorkerMode::OpaqueSession,
    );
}

/// Exercises the public complete-model loader and opaque session through the
/// explicit Inkling tensor-parallel bridge without realizing a neutral manifest.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_inkling_tensor_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Inkling,
        "tp",
        WorkerMode::OpaqueSession,
    );
}

/// Exercises an admitted dense, prediction-free Inkling through the generic
/// composite partition binder without constructing a duplicate model shell.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_inkling_dense_tensor_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::InklingDense,
        "tp",
        WorkerMode::OpaqueSession,
    );
}

/// Proves dense Inkling's active image and audio roots traverse the same
/// neutral composite TP session used by routed Inkling variants.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_inkling_dense_multimodal_tensor_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::InklingDenseMultimodal,
        "tp",
        WorkerMode::OpaqueInklingMedia,
    );
}

/// Exercises the public complete-model loader and sparse neutral Inkling
/// decoder on a pure two-rank expert-parallel topology without PP or TP.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_inkling_expert_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Inkling,
        "ep",
        WorkerMode::OpaqueSession,
    );
}

/// Exercises ordered projected image embeddings plus the native dMel tower
/// through the public neutral Inkling TP session and prompt-cache lifecycle.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_inkling_multimodal_tensor_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::InklingMultimodal,
        "tp",
        WorkerMode::OpaqueInklingMedia,
    );
}

/// Exercises the neutral embedded predictor through the complete public TP
/// loader, rank-synchronized scheduler, sharded vocabulary, and MTP state.
#[test]
#[ignore = "spawns local processes, opens loopback sockets, and initializes MLX; run explicitly"]
fn ring_two_process_inkling_mtp_tensor_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Inkling,
        "tp",
        WorkerMode::OpaqueInklingMtp,
    );
}

/// Exercises routed prediction units through authoritative expert providers
/// and banks on a pure expert-parallel target, without TP or PP ownership.
#[test]
#[ignore = "spawns local processes, opens loopback sockets, and initializes MLX; run explicitly"]
fn ring_two_process_inkling_mtp_expert_parallel_neutral_visitor() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Inkling,
        "ep",
        WorkerMode::OpaqueInklingMtp,
    );
}

/// Proves routed prediction reuses the target's addressable expert bank while
/// extension weights remain a separately materialized resident component.
#[test]
#[ignore = "spawns local processes, opens loopback sockets, and initializes an addressable expert cache"]
fn ring_two_process_inkling_mtp_expert_parallel_addressable_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Inkling,
        "ep",
        WorkerMode::OpaqueInklingMtpAddressableParameterBank,
    );
}

/// Exercises pipeline MTP through the same neutral visitor lifecycle used by
/// facade prepared-chat generation on every pipeline rank.
#[test]
#[ignore = "spawns local processes, opens loopback sockets, and initializes MLX; run explicitly"]
fn ring_two_process_inkling_mtp_pipeline_neutral_visitor() {
    run_ring_pipeline_mode(false, FixtureFamily::Inkling, WorkerMode::OpaqueInklingMtp);
}

/// Exercises conditional Qwen hybrid MTP over one neutral TP target and extension-only state.
#[test]
#[ignore = "spawns local processes, opens loopback sockets, and initializes MLX; run explicitly"]
fn ring_two_process_qwen35_mtp_tensor_parallel_neutral_visitor() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen35Multimodal,
        "tp",
        WorkerMode::OpaqueQwenHybridMtp,
    );
}

/// Exercises conditional Qwen hybrid MTP over the neutral two-stage target session.
#[test]
#[ignore = "spawns local processes, opens loopback sockets, and initializes MLX; run explicitly"]
fn ring_two_process_qwen35_mtp_pipeline_neutral_visitor() {
    run_ring_pipeline_mode(
        false,
        FixtureFamily::Qwen35Multimodal,
        WorkerMode::OpaqueQwenHybridMtp,
    );
}

/// Exercises composite Qwen hybrid prediction over the admitted Cartesian
/// tensor-by-pipeline target using the same public speculative scheduler.
#[test]
#[ignore = "spawns local processes, opens loopback sockets, and initializes MLX; run explicitly"]
fn ring_four_process_qwen35_mtp_tensor_pipeline_neutral_visitor() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen35Multimodal,
        "tp-pp",
        WorkerMode::OpaqueQwenHybridMtp,
    );
}

/// Exercises patterned Nemotron-H MTP over one neutral TP target and
/// extension-only state, with no family model/session fallback.
#[test]
#[ignore = "spawns local processes, opens loopback sockets, and initializes MLX; run explicitly"]
fn ring_two_process_nemotron_h_mtp_tensor_parallel_neutral_visitor() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::NemotronH,
        "tp",
        WorkerMode::OpaqueNemotronHMtp,
    );
}

/// Exercises the architecture-owned Gemma 4 composite TP partition through
/// the public loader and neutral manifest runtime.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic untied Gemma 4 text fixture"]
fn ring_two_process_gemma4_tensor_parallel_opaque_session() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_gemma4_tensor_parallel_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Gemma,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        Some("tp"),
    );
}

/// Exercises the architecture-owned Gemma 4 composite across two pipeline
/// stages with public-session cache continuation and publication.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic untied Gemma 4 text fixture"]
fn ring_two_process_gemma4_pipeline_neutral_composite_session() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    // This exact PP proof uses independent per-layer KV state. Gemma 4
    // checkpoints whose later layers consume a pass-local shared KV
    // publication still require that typed publication in the neutral wire
    // boundary and therefore remain outside this first production slice.
    write_gemma4_tensor_parallel_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Gemma,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        None,
    );
}

/// Exercises Gemma 4 Unified's neutral image and audio towers, ordered media
/// assembly, per-layer inputs, TP decoder, and prompt-cache continuation.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic untied Gemma 4 Unified media fixture"]
fn ring_two_process_gemma4_multimodal_tensor_parallel_opaque_session() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_gemma4_multimodal_tensor_parallel_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Gemma,
        WorkerMode::OpaqueGemma4Media,
        checkpoint,
        checkpoint_path,
        Some("tp"),
    );
}

/// Exercises Gemma 4 Unified media ingress, optional roots, merge, and decoder
/// continuation across two neutral pipeline owners.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic untied Gemma 4 Unified media fixture"]
fn ring_two_process_gemma4_multimodal_pipeline_neutral_composite_session() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_gemma4_multimodal_tensor_parallel_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Gemma,
        WorkerMode::OpaqueGemma4Media,
        checkpoint,
        checkpoint_path,
        None,
    );
}

#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic untied Gemma 4 Unified media fixture"]
fn ring_two_process_gemma4_multimodal_inspection_uses_canonical_paths() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_gemma4_multimodal_tensor_parallel_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Gemma,
        WorkerMode::OpaqueGemma4MediaInspection,
        checkpoint,
        checkpoint_path,
        None,
    );
}

/// Compares tied Gemma 4 Unified image/audio prefill and decode with the
/// single-rank resident model while the TP path projects through embeddings.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic tied Gemma 4 Unified media fixture"]
fn ring_two_process_gemma4_tied_multimodal_tensor_parallel_resident_reference() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_gemma4_tied_multimodal_tensor_parallel_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Gemma,
        WorkerMode::OpaqueGemma4Media,
        checkpoint,
        checkpoint_path,
        Some("tp"),
    );
}

/// Verifies that a tied Gemma 4 checkpoint without an `lm_head` is bound and
/// projected through the rank-local vocabulary embedding by the public path.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic tied Gemma 4 text fixture"]
fn ring_two_process_gemma4_tied_tensor_parallel_opaque_session() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_gemma4_tensor_parallel_fixture_with_tied_embeddings(checkpoint.path(), true);
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::Gemma,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        Some("tp"),
    );
}

/// Exercises the neutral Muse-Glimmer text binder through the public loader
/// and opaque session on a pure two-rank tensor-parallel topology.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic Muse-Glimmer text fixture"]
fn ring_two_process_muse_glimmer_tensor_parallel_opaque_session() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_muse_glimmer_tensor_parallel_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::MuseGlimmer,
        WorkerMode::OpaqueSession,
        checkpoint,
        checkpoint_path,
        Some("tp"),
    );
}

/// Exercises Muse-Glimmer's neutral vision tower and media assembly through
/// the public loader on a pure two-rank tensor-parallel topology, including a
/// paged prompt-cache save/open/continue round trip.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic Muse-Glimmer image fixture"]
fn ring_two_process_muse_glimmer_image_tensor_parallel_opaque_session() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_muse_glimmer_tensor_parallel_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::MuseGlimmer,
        WorkerMode::OpaqueMuseImage,
        checkpoint,
        checkpoint_path,
        Some("tp"),
    );
}

/// Exercises Muse-Glimmer's image root and decoder continuation across two
/// neutral pipeline owners without constructing a family pipeline shell.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic Muse-Glimmer image fixture"]
fn ring_two_process_muse_glimmer_image_pipeline_neutral_composite_session() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_muse_glimmer_tensor_parallel_fixture(checkpoint.path());
    let checkpoint_path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::FullyResident,
        FixtureFamily::MuseGlimmer,
        WorkerMode::OpaqueMuseImage,
        checkpoint,
        checkpoint_path,
        None,
    );
}

/// Exercises Muse-Glimmer's canonical vision and decoder unit traversal
/// through bounded checkpoint streaming across two pipeline stages.
#[test]
#[ignore = "spawns local processes, opens loopback sockets, and initializes MLX; run explicitly"]
fn ring_two_process_muse_glimmer_dense_stream_pipeline() {
    run_ring_pipeline(true, FixtureFamily::MuseGlimmer);
}

/// Verifies Inkling's uneven 2+1 stage placement and combined KV/convolution
/// state against the resident text decoder.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_inkling_dense_stream_pipeline() {
    run_ring_pipeline(true, FixtureFamily::Inkling);
}

/// Verifies Inkling TP=2 + PP=2 across full/sliding attention, dense/sparse
/// transitions, rank-local KV/convolution state, persistence, and generation.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_inkling_tensor_pipeline() {
    run_ring_cartesian_pipeline(true, FixtureFamily::Inkling, "tp-pp");
}

/// Exercises Inkling TP+PP rank-local recipes and bounded reads from a
/// representative canonical text GGUF.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_inkling_gguf_tensor_pipeline() {
    run_ring_cartesian_pipeline(true, FixtureFamily::InklingGguf, "tp-pp");
}

/// Verifies Inkling PP=2 + EP=2 with stage-local routed experts, shared
/// experts, matching-EP transport, persistence, and bounded layer reads.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_inkling_pipeline_expert() {
    run_ring_cartesian_pipeline(true, FixtureFamily::Inkling, "pp-ep");
}

/// Exercises Inkling PP+EP expert selection and stage-local state from a
/// representative canonical text GGUF.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_inkling_gguf_pipeline_expert() {
    run_ring_cartesian_pipeline(true, FixtureFamily::InklingGguf, "pp-ep");
}

/// Proves resident Inkling TP=2 x PP=2 x EP=2 execution across uneven stage
/// placement, full/sliding attention, short-convolution state, and routed/shared
/// expert ownership.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_inkling_triple_axis() {
    run_ring_cartesian_pipeline(false, FixtureFamily::Inkling, "tp-pp-ep");
}

/// Exercises dense-streamed Inkling non-experts and stage-local independent
/// expert caches across all three Cartesian axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_inkling_streamed_triple_axis_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::Inkling,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Exercises host-layerwise Inkling attention/convolution state with
/// independently cached routed experts across TP, PP, and EP.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_inkling_layerwise_host_triple_axis_parameter_bank() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::Inkling,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Exercises canonical Inkling GGUF recipes, bounded reads, and independent
/// expert caching across all three axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_inkling_gguf_triple_axis_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::InklingGguf,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Proves TP+PP can independently cache every stage-local Inkling expert bank
/// without constructing an EP communicator.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_inkling_tensor_pipeline_parameter_bank_without_ep() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::Inkling,
        "tp-pp",
        WorkerMode::AddressableParameterBank,
    );
}

/// Verifies each Inkling stage's expert cache remains session-owned.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_inkling_parameter_bank_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Inkling,
        "pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Proves bounded affine materialization feeds stage-local Inkling expert
/// caches under PP+EP, including persistence and synchronized decode.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_inkling_quantized_pipeline_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::Inkling,
        "pp-ep",
        WorkerMode::AddressableParameterBankRequantize,
    );
}

/// Proves a PP+EP stage can bounded-quantize and pin its complete rank-local
/// Inkling routed/shared banks without introducing an independent expert cache.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_inkling_fully_resident_load_time_quantization() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Inkling,
        "pp-ep",
        WorkerMode::Requantize,
    );
}

/// Runs scheduled Inkling audio/image ingress through TP-sharded placed groups,
/// matching-TP pipeline transport, persistence, decode, and generation.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_inkling_multimodal_tensor_pipeline() {
    run_ring_cartesian_pipeline(true, FixtureFamily::InklingMultimodal, "tp-pp");
}

/// Runs scheduled Inkling audio/image ingress through stage-local expert
/// ownership and matching-EP pipeline lanes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_inkling_multimodal_pipeline_expert() {
    run_ring_cartesian_pipeline(false, FixtureFamily::InklingMultimodal, "pp-ep");
}

/// Proves scheduled multimodal ingress composes with all three Cartesian axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_inkling_multimodal_triple_axis() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::InklingMultimodal,
        "tp-pp-ep",
        WorkerMode::OpaqueInklingMedia,
    );
}

/// Proves multimodal ingress composes with streamed non-experts and stage-local
/// independent expert caches across all three Cartesian axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_inkling_multimodal_streamed_triple_axis_parameter_bank() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::InklingMultimodal,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

/// Proves multimodal ingress composes with host-layerwise non-experts and
/// independent expert caches across all three Cartesian axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_inkling_multimodal_layerwise_host_triple_axis_parameter_bank() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::InklingMultimodal,
        "tp-pp-ep",
        WorkerMode::AddressableParameterBank,
    );
}

fn run_ring_pipeline(dense_stream: bool, family: FixtureFamily) {
    run_ring_pipeline_mode(dense_stream, family, WorkerMode::Standard);
}

fn run_ring_cartesian_pipeline(dense_stream: bool, family: FixtureFamily, axes: &'static str) {
    run_ring_cartesian_pipeline_mode(dense_stream, family, axes, WorkerMode::Standard);
}

fn run_ring_cartesian_pipeline_mode(
    dense_stream: bool,
    family: FixtureFamily,
    axes: &'static str,
    mode: WorkerMode,
) {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    let checkpoint_path = if family == FixtureFamily::DeepSeekGguf {
        let path = checkpoint.path().join("model.gguf");
        write_deepseek_gguf_fixture(&path);
        path
    } else if matches!(family, FixtureFamily::Qwen2Gguf | FixtureFamily::Qwen3Gguf) {
        let path = checkpoint.path().join("model.gguf");
        write_qwen_gguf_fixture(
            &path,
            if family == FixtureFamily::Qwen2Gguf {
                "qwen2"
            } else {
                "qwen3"
            },
        );
        path
    } else if family == FixtureFamily::Qwen3MoeGguf {
        let path = checkpoint.path().join("model.gguf");
        write_qwen3_moe_gguf_fixture(&path);
        path
    } else if family == FixtureFamily::GptOssGguf {
        let path = checkpoint.path().join("model.gguf");
        write_gpt_oss_gguf_fixture(&path);
        path
    } else if family == FixtureFamily::Lfm2MoeGguf {
        let path = checkpoint.path().join("model.gguf");
        write_lfm2_moe_gguf_fixture(&path);
        path
    } else if family == FixtureFamily::NemotronHGguf {
        let path = checkpoint.path().join("model.gguf");
        write_nemotron_h_moe_gguf_fixture(&path);
        path
    } else if family == FixtureFamily::KimiLinearGguf {
        let path = checkpoint.path().join("model.gguf");
        write_kimi_linear_gguf_fixture(&path);
        path
    } else if family == FixtureFamily::InklingGguf {
        let path = checkpoint.path().join("model.gguf");
        write_inkling_gguf_fixture(&path);
        path
    } else {
        match family {
            FixtureFamily::Qwen2 => write_qwen_fixture(checkpoint.path(), "qwen2"),
            FixtureFamily::Qwen3
                if matches!(
                    mode,
                    WorkerMode::Requantize | WorkerMode::OpaqueSessionRequantize
                ) =>
            {
                write_qwen_requantized_tp_fixture(checkpoint.path())
            }
            FixtureFamily::Qwen3 => write_qwen_fixture(checkpoint.path(), "qwen3"),
            FixtureFamily::Qwen3Moe => write_qwen_fixture(checkpoint.path(), "qwen3_moe"),
            FixtureFamily::Qwen3MoeTied => {
                write_qwen_fixture_with_tied_head(checkpoint.path(), "qwen3_moe", true)
            }
            FixtureFamily::DeepSeek if mode == WorkerMode::OpaqueDeepSeekMtpTarget => {
                write_deepseek_fixture_with_prediction(checkpoint.path(), 2, 1)
            }
            FixtureFamily::DeepSeek => write_deepseek_fixture(checkpoint.path(), 2),
            FixtureFamily::DeepSeekV4 if mode == WorkerMode::OpaqueDeepSeekDsparkTarget => {
                write_deepseek_v4_dspark_fixture(checkpoint.path())
            }
            FixtureFamily::DeepSeekV4 => write_deepseek_v4_fixture(
                checkpoint.path(),
                if matches!(
                    mode,
                    WorkerMode::OpaqueSessionPredictionFree
                        | WorkerMode::OpaqueSessionAddressableParameterBank
                ) {
                    0
                } else {
                    1
                },
            ),
            FixtureFamily::Lfm2 => write_lfm2_pipeline_fixture(checkpoint.path(), false),
            FixtureFamily::Lfm2Moe => write_lfm2_pipeline_fixture(checkpoint.path(), true),
            FixtureFamily::KimiLinear => write_kimi_linear_fixture(checkpoint.path()),
            FixtureFamily::NemotronH
                if matches!(
                    mode,
                    WorkerMode::AddressableParameterBankRequantize | WorkerMode::Requantize
                ) =>
            {
                write_nemotron_quantizable_fixture(checkpoint.path())
            }
            FixtureFamily::NemotronH if mode == WorkerMode::OpaqueNemotronHMtp => {
                write_nemotron_mtp_fixture(checkpoint.path())
            }
            FixtureFamily::NemotronH => write_nemotron_fixture(checkpoint.path()),
            FixtureFamily::Qwen3Next => write_qwen_hybrid_fixture(checkpoint.path(), "qwen3_next"),
            FixtureFamily::Qwen3NextMoe => {
                write_qwen_hybrid_moe_fixture(checkpoint.path(), "qwen3_next")
            }
            FixtureFamily::Qwen35 => write_qwen_hybrid_fixture(checkpoint.path(), "qwen3_5_text"),
            FixtureFamily::Qwen35Moe => {
                write_qwen_hybrid_moe_fixture(checkpoint.path(), "qwen3_5_moe_text")
            }
            FixtureFamily::Qwen35Multimodal => {
                write_qwen35_multimodal_fixture(checkpoint.path(), false)
            }
            FixtureFamily::Qwen35ZeroPrediction => {
                write_qwen35_zero_prediction_fixture(checkpoint.path())
            }
            FixtureFamily::Qwen35MoeMultimodal => {
                write_qwen35_multimodal_fixture(checkpoint.path(), true)
            }
            FixtureFamily::Qwen3Vl => write_qwen3_vl_fixture(checkpoint.path(), false),
            FixtureFamily::Qwen3VlMoe => write_qwen3_vl_fixture(checkpoint.path(), true),
            FixtureFamily::Inkling
                if matches!(
                    mode,
                    WorkerMode::AddressableParameterBankRequantize | WorkerMode::Requantize
                ) =>
            {
                write_inkling_quantizable_fixture(checkpoint.path())
            }
            FixtureFamily::Inkling
                if matches!(
                    mode,
                    WorkerMode::OpaqueInklingMtp
                        | WorkerMode::OpaqueInklingMtpAddressableParameterBank
                ) =>
            {
                write_inkling_mtp_fixture(checkpoint.path())
            }
            FixtureFamily::Inkling => write_inkling_fixture(checkpoint.path()),
            FixtureFamily::InklingDense => write_inkling_dense_fixture(checkpoint.path()),
            FixtureFamily::InklingDenseMultimodal => {
                write_inkling_dense_multimodal_fixture(checkpoint.path())
            }
            FixtureFamily::InklingMultimodal => write_inkling_multimodal_fixture(checkpoint.path()),
            FixtureFamily::GptOss => write_gpt_oss_fixture(checkpoint.path()),
            _ => panic!("Cartesian pipeline helper received unsupported {family:?}"),
        }
        checkpoint.path().to_path_buf()
    };
    run_ring_pipeline_processes(
        WorkerResidency::from_dense_stream(dense_stream),
        family,
        mode,
        checkpoint,
        checkpoint_path,
        Some(axes),
    );
}

fn run_ring_layerwise_host_cartesian_pipeline_mode(
    family: FixtureFamily,
    axes: &'static str,
    mode: WorkerMode,
) {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    let checkpoint_path = if family == FixtureFamily::DeepSeekGguf {
        let path = checkpoint.path().join("model.gguf");
        write_deepseek_gguf_fixture(&path);
        path
    } else if matches!(family, FixtureFamily::Qwen2Gguf | FixtureFamily::Qwen3Gguf) {
        let path = checkpoint.path().join("model.gguf");
        write_qwen_gguf_fixture(
            &path,
            if family == FixtureFamily::Qwen2Gguf {
                "qwen2"
            } else {
                "qwen3"
            },
        );
        path
    } else if family == FixtureFamily::Qwen3MoeGguf {
        let path = checkpoint.path().join("model.gguf");
        write_qwen3_moe_gguf_fixture(&path);
        path
    } else {
        match family {
            FixtureFamily::Qwen2 => write_qwen_fixture(checkpoint.path(), "qwen2"),
            FixtureFamily::Qwen3
                if matches!(
                    mode,
                    WorkerMode::Requantize | WorkerMode::OpaqueSessionRequantize
                ) =>
            {
                write_qwen_requantized_tp_fixture(checkpoint.path())
            }
            FixtureFamily::Qwen3 => write_qwen_fixture(checkpoint.path(), "qwen3"),
            FixtureFamily::DeepSeek if mode == WorkerMode::OpaqueDeepSeekMtpTarget => {
                write_deepseek_fixture_with_prediction(checkpoint.path(), 2, 1)
            }
            FixtureFamily::DeepSeek => write_deepseek_fixture(checkpoint.path(), 2),
            FixtureFamily::DeepSeekV4 => write_deepseek_v4_fixture(checkpoint.path(), 1),
            FixtureFamily::Qwen3Moe => write_qwen_fixture(checkpoint.path(), "qwen3_moe"),
            FixtureFamily::GptOss => write_gpt_oss_fixture(checkpoint.path()),
            FixtureFamily::Lfm2Moe => write_lfm2_pipeline_fixture(checkpoint.path(), true),
            FixtureFamily::KimiLinear => write_kimi_linear_fixture(checkpoint.path()),
            FixtureFamily::NemotronH => write_nemotron_fixture(checkpoint.path()),
            FixtureFamily::Inkling => write_inkling_fixture(checkpoint.path()),
            FixtureFamily::InklingDense => write_inkling_dense_fixture(checkpoint.path()),
            FixtureFamily::InklingDenseMultimodal => {
                write_inkling_dense_multimodal_fixture(checkpoint.path())
            }
            FixtureFamily::InklingMultimodal => write_inkling_multimodal_fixture(checkpoint.path()),
            FixtureFamily::Qwen3NextMoe => {
                write_qwen_hybrid_moe_fixture(checkpoint.path(), "qwen3_next")
            }
            FixtureFamily::Qwen35Moe => {
                write_qwen_hybrid_moe_fixture(checkpoint.path(), "qwen3_5_moe_text")
            }
            FixtureFamily::Qwen35Multimodal => {
                write_qwen35_multimodal_fixture(checkpoint.path(), false)
            }
            FixtureFamily::Qwen35ZeroPrediction => {
                write_qwen35_zero_prediction_fixture(checkpoint.path())
            }
            FixtureFamily::Qwen35MoeMultimodal => {
                write_qwen35_multimodal_fixture(checkpoint.path(), true)
            }
            FixtureFamily::Qwen3Vl => write_qwen3_vl_fixture(checkpoint.path(), false),
            FixtureFamily::Qwen3VlMoe => write_qwen3_vl_fixture(checkpoint.path(), true),
            _ => panic!("host-layerwise Cartesian helper received unsupported {family:?}"),
        }
        checkpoint.path().to_path_buf()
    };
    run_ring_pipeline_processes(
        WorkerResidency::LayerwiseHost,
        family,
        mode,
        checkpoint,
        checkpoint_path,
        Some(axes),
    );
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum WorkerMode {
    Standard,
    FinalOutputIntervention,
    AddressableParameterBank,
    AddressableParameterBankRequantize,
    Requantize,
    OpaqueSession,
    OpaqueSessionPredictionFree,
    OpaquePreparedSpeculativeCapability,
    OpaqueUnsupportedDirectPartition,
    OpaqueSessionRequantize,
    OpaqueInspection,
    OpaqueTextGeneration,
    OpaqueSessionAddressableParameterBank,
    OpaqueSessionEvictingAddressableParameterBank,
    OpaqueMuseImage,
    OpaqueInklingMedia,
    OpaqueInklingMtp,
    OpaqueInklingMtpAddressableParameterBank,
    OpaqueQwenHybridMtp,
    OpaqueNemotronHMtp,
    OpaqueDeepSeekMtpTarget,
    OpaqueDeepSeekDsparkTarget,
    OpaqueGemma4Media,
    OpaqueGemma4MediaInspection,
    OpaqueQwen3VlMedia,
    OpaqueQwenConditionalMedia,
    QwenHybridPromptCache,
    PromptCachePrepareFailure,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum WorkerResidency {
    FullyResident,
    LayerwiseHost,
    DenseDiskStream,
}

impl WorkerResidency {
    const fn from_dense_stream(enabled: bool) -> Self {
        if enabled {
            Self::DenseDiskStream
        } else {
            Self::FullyResident
        }
    }
}

fn run_ring_pipeline_mode(dense_stream: bool, family: FixtureFamily, mode: WorkerMode) {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    let checkpoint_path = if family == FixtureFamily::DeepSeekGguf {
        let path = checkpoint.path().join("model.gguf");
        write_deepseek_gguf_fixture(&path);
        path
    } else if matches!(family, FixtureFamily::Qwen2Gguf | FixtureFamily::Qwen3Gguf) {
        let path = checkpoint.path().join("model.gguf");
        write_qwen_gguf_fixture(
            &path,
            if family == FixtureFamily::Qwen2Gguf {
                "qwen2"
            } else {
                "qwen3"
            },
        );
        path
    } else if family == FixtureFamily::Qwen3MoeGguf {
        let path = checkpoint.path().join("model.gguf");
        write_qwen3_moe_gguf_fixture(&path);
        path
    } else if family == FixtureFamily::GptOssGguf {
        let path = checkpoint.path().join("model.gguf");
        write_gpt_oss_gguf_fixture(&path);
        path
    } else if family == FixtureFamily::Lfm2MoeGguf {
        let path = checkpoint.path().join("model.gguf");
        write_lfm2_moe_gguf_fixture(&path);
        path
    } else if family == FixtureFamily::NemotronHGguf {
        let path = checkpoint.path().join("model.gguf");
        write_nemotron_h_moe_gguf_fixture(&path);
        path
    } else if family == FixtureFamily::KimiLinearGguf {
        let path = checkpoint.path().join("model.gguf");
        write_kimi_linear_gguf_fixture(&path);
        path
    } else if family == FixtureFamily::InklingGguf {
        let path = checkpoint.path().join("model.gguf");
        write_inkling_gguf_fixture(&path);
        path
    } else {
        match family {
            FixtureFamily::Llama => write_fixture(checkpoint.path()),
            FixtureFamily::Mistral => write_mistral_fixture(checkpoint.path()),
            FixtureFamily::DeepSeek if mode == WorkerMode::OpaqueDeepSeekMtpTarget => {
                write_deepseek_fixture_with_prediction(checkpoint.path(), 2, 1)
            }
            FixtureFamily::DeepSeek => write_deepseek_fixture(checkpoint.path(), 2),
            FixtureFamily::DeepSeekV4 if mode == WorkerMode::OpaqueDeepSeekDsparkTarget => {
                write_deepseek_v4_dspark_fixture(checkpoint.path())
            }
            FixtureFamily::DeepSeekV4 => write_deepseek_v4_fixture(checkpoint.path(), 1),
            FixtureFamily::Gemma => write_gemma_fixture(checkpoint.path()),
            FixtureFamily::Qwen2 => write_qwen_fixture(checkpoint.path(), "qwen2"),
            FixtureFamily::Qwen3
                if matches!(
                    mode,
                    WorkerMode::Requantize | WorkerMode::OpaqueSessionRequantize
                ) =>
            {
                write_qwen_requantized_tp_fixture(checkpoint.path())
            }
            FixtureFamily::Qwen3 => write_qwen_fixture(checkpoint.path(), "qwen3"),
            FixtureFamily::Qwen3Moe => write_qwen_fixture(checkpoint.path(), "qwen3_moe"),
            FixtureFamily::Qwen3MoeTied => {
                write_qwen_fixture_with_tied_head(checkpoint.path(), "qwen3_moe", true)
            }
            FixtureFamily::GptOss => write_gpt_oss_fixture(checkpoint.path()),
            FixtureFamily::Lfm2 => write_lfm2_pipeline_fixture(checkpoint.path(), false),
            FixtureFamily::Lfm2Moe => write_lfm2_pipeline_fixture(checkpoint.path(), true),
            FixtureFamily::KimiLinear => write_kimi_linear_fixture(checkpoint.path()),
            FixtureFamily::NemotronH => write_nemotron_fixture(checkpoint.path()),
            FixtureFamily::Qwen3Next => write_qwen_hybrid_fixture(checkpoint.path(), "qwen3_next"),
            FixtureFamily::Qwen3NextMoe => {
                write_qwen_hybrid_moe_fixture(checkpoint.path(), "qwen3_next")
            }
            FixtureFamily::Qwen35 => write_qwen_hybrid_fixture(checkpoint.path(), "qwen3_5_text"),
            FixtureFamily::Qwen35Moe => {
                write_qwen_hybrid_moe_fixture(checkpoint.path(), "qwen3_5_moe_text")
            }
            FixtureFamily::Qwen35Multimodal => {
                write_qwen35_multimodal_fixture(checkpoint.path(), false)
            }
            FixtureFamily::Qwen35ZeroPrediction => {
                write_qwen35_zero_prediction_fixture(checkpoint.path())
            }
            FixtureFamily::Qwen35MoeMultimodal => {
                write_qwen35_multimodal_fixture(checkpoint.path(), true)
            }
            FixtureFamily::Qwen3Vl => write_qwen3_vl_fixture(checkpoint.path(), false),
            FixtureFamily::Qwen3VlMoe => write_qwen3_vl_fixture(checkpoint.path(), true),
            FixtureFamily::Inkling if mode == WorkerMode::OpaqueInklingMtp => {
                write_inkling_pipeline_mtp_fixture(checkpoint.path())
            }
            FixtureFamily::Inkling => write_inkling_fixture(checkpoint.path()),
            FixtureFamily::InklingDense => write_inkling_dense_fixture(checkpoint.path()),
            FixtureFamily::InklingDenseMultimodal => {
                write_inkling_dense_multimodal_fixture(checkpoint.path())
            }
            FixtureFamily::InklingMultimodal => write_inkling_multimodal_fixture(checkpoint.path()),
            FixtureFamily::MuseGlimmer => {
                write_muse_glimmer_tensor_parallel_fixture(checkpoint.path())
            }
            FixtureFamily::DeepSeekGguf
            | FixtureFamily::Qwen2Gguf
            | FixtureFamily::Qwen3Gguf
            | FixtureFamily::Qwen3MoeGguf
            | FixtureFamily::GptOssGguf
            | FixtureFamily::Lfm2MoeGguf
            | FixtureFamily::NemotronHGguf
            | FixtureFamily::KimiLinearGguf
            | FixtureFamily::InklingGguf => unreachable!(),
        }
        checkpoint.path().to_path_buf()
    };
    run_ring_pipeline_processes(
        WorkerResidency::from_dense_stream(dense_stream),
        family,
        mode,
        checkpoint,
        checkpoint_path,
        None,
    );
}

fn run_ring_pipeline_processes(
    residency: WorkerResidency,
    family: FixtureFamily,
    mode: WorkerMode,
    _checkpoint: tempfile::TempDir,
    checkpoint_path: PathBuf,
    cartesian_axes: Option<&'static str>,
) {
    let prompt_cache = tempfile::tempdir().unwrap();
    let world_size = match cartesian_axes {
        Some("tp-pp-ep") => 8,
        Some("tp" | "ep") => 2,
        Some(_) => 4,
        None => 2,
    };
    let sockets = (0..world_size)
        .map(|_| TcpListener::bind(("127.0.0.1", 0)).unwrap())
        .collect::<Vec<_>>();
    let hosts = sockets
        .iter()
        .map(|socket| vec![format!("127.0.0.1:{}", socket.local_addr().unwrap().port())])
        .collect::<Vec<_>>();
    let ring = tempfile::tempdir().unwrap();
    let hostfile = ring.path().join("ring-hosts.json");
    std::fs::write(&hostfile, serde_json::to_vec(&hosts).unwrap()).unwrap();
    drop(sockets);
    let executable = std::env::current_exe().unwrap();
    let mut children = ChildGuard {
        children: Vec::with_capacity(world_size),
    };
    for rank in 0..world_size {
        let mut command = Command::new(&executable);
        command
            .args([
                "--exact",
                "tests::distributed_pipeline_ring::pipeline_ring_worker",
                "--nocapture",
            ])
            .env(WORKER_RANK, rank.to_string())
            .env(CHECKPOINT_DIR, &checkpoint_path)
            .env(FIXTURE_FAMILY, family.name())
            .env(PROMPT_CACHE_ROOT, prompt_cache.path())
            .env("MLX_RANK", rank.to_string())
            .env("MLX_HOSTFILE", &hostfile)
            .env_remove("MLX_RING_VERBOSE")
            .stdout(Stdio::piped());
        if family == FixtureFamily::Qwen3VlMoe && cartesian_axes == Some("tp-pp-ep") {
            command.env("EREDU_TEST_PARTITION_COLLECTIVE_TRACE", "1");
        }
        command.stderr(Stdio::piped());
        if let Some(axes) = cartesian_axes {
            command.env(CARTESIAN_AXES, axes);
        }
        match residency {
            WorkerResidency::FullyResident => {}
            WorkerResidency::LayerwiseHost => {
                command.env(LAYERWISE_HOST, "1");
            }
            WorkerResidency::DenseDiskStream => {
                command.env(DENSE_STREAM, "1");
            }
        }
        match mode {
            WorkerMode::Standard => {}
            WorkerMode::FinalOutputIntervention => {
                command.env(FINAL_OUTPUT_INTERVENTION, "1");
            }
            WorkerMode::AddressableParameterBank => {
                command.env(EXPERT_CACHE, "1");
            }
            WorkerMode::AddressableParameterBankRequantize => {
                command.env(EXPERT_CACHE, "1");
                command.env(REQUANTIZE, "1");
            }
            WorkerMode::Requantize => {
                command.env(REQUANTIZE, "1");
            }
            WorkerMode::OpaqueSession => {
                command.env(OPAQUE_SESSION, "1");
            }
            WorkerMode::OpaqueSessionPredictionFree => {
                command.env(OPAQUE_SESSION, "1");
                command.env(PREDICTION_FREE_TARGET, "1");
            }
            WorkerMode::OpaquePreparedSpeculativeCapability => {
                command.env(OPAQUE_SESSION, "1");
                command.env(PREPARED_SPECULATIVE_CAPABILITY, "1");
            }
            WorkerMode::OpaqueUnsupportedDirectPartition => {
                command.env(OPAQUE_SESSION, "1");
                command.env(EXPECTED_UNSUPPORTED_DIRECT_PARTITION, "1");
            }
            WorkerMode::OpaqueSessionRequantize => {
                command.env(OPAQUE_SESSION, "1");
                command.env(REQUANTIZE, "1");
            }
            WorkerMode::OpaqueInspection => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_INSPECTION, "1");
            }
            WorkerMode::OpaqueTextGeneration => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_TEXT_GENERATION, "1");
            }
            WorkerMode::OpaqueSessionAddressableParameterBank => {
                command.env(OPAQUE_SESSION, "1");
                command.env(EXPERT_CACHE, "1");
            }
            WorkerMode::OpaqueSessionEvictingAddressableParameterBank => {
                command.env(OPAQUE_SESSION, "1");
                command.env(EXPERT_CACHE, "1");
                command.env(EXPERT_CACHE_EVICTION, "1");
            }
            WorkerMode::OpaqueMuseImage => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_MUSE_IMAGE, "1");
            }
            WorkerMode::OpaqueInklingMedia => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_INKLING_MEDIA, "1");
            }
            WorkerMode::OpaqueInklingMtp => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_INKLING_MTP, "1");
            }
            WorkerMode::OpaqueInklingMtpAddressableParameterBank => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_INKLING_MTP, "1");
                command.env(EXPERT_CACHE, "1");
            }
            WorkerMode::OpaqueQwenHybridMtp => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_QWEN_HYBRID_MTP, "1");
            }
            WorkerMode::OpaqueNemotronHMtp => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_NEMOTRON_H_MTP, "1");
            }
            WorkerMode::OpaqueDeepSeekMtpTarget => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_DEEPSEEK_MTP_TARGET, "1");
            }
            WorkerMode::OpaqueDeepSeekDsparkTarget => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_DEEPSEEK_MTP_TARGET, "1");
                command.env(OPAQUE_DEEPSEEK_DSPARK_TARGET, "1");
            }
            WorkerMode::OpaqueGemma4Media => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_GEMMA4_MEDIA, "1");
            }
            WorkerMode::OpaqueGemma4MediaInspection => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_GEMMA4_MEDIA, "1");
                command.env(OPAQUE_INSPECTION, "1");
            }
            WorkerMode::OpaqueQwen3VlMedia => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_QWEN3_VL_MEDIA, "1");
            }
            WorkerMode::OpaqueQwenConditionalMedia => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_QWEN_CONDITIONAL_MEDIA, "1");
            }
            WorkerMode::QwenHybridPromptCache => {
                command.env(QWEN_HYBRID_PROMPT_CACHE, "1");
            }
            WorkerMode::PromptCachePrepareFailure => {
                command.env(OPAQUE_SESSION, "1");
                command.env(PROMPT_CACHE_PREPARE_FAILURE, "1");
            }
        }
        children.children.push(command.spawn().unwrap());
    }
    let deadline = Instant::now()
        + Duration::from_secs(match world_size {
            8 => 180,
            4 => 90,
            _ => 45,
        });
    let mut timed_out = false;
    loop {
        let statuses = children
            .children
            .iter_mut()
            .map(|child| child.try_wait().unwrap())
            .collect::<Vec<_>>();
        if statuses.iter().all(Option::is_some) {
            break;
        }
        timed_out = Instant::now() >= deadline;
        let peer_failed = statuses.iter().flatten().any(|status| !status.success());
        if timed_out || peer_failed {
            // A peer often reports the global failure agreement before the
            // originating worker has flushed its local architecture error.
            // Preserve that causal diagnostic without waiting for the full Ring
            // deadline or allowing a failed process set to run unbounded.
            if peer_failed && !timed_out {
                thread::sleep(Duration::from_millis(500));
            }
            for child in &mut children.children {
                if child.try_wait().unwrap().is_none() {
                    let _ = child.kill();
                }
            }
            break;
        }
        thread::sleep(Duration::from_millis(20));
    }
    let outputs = children.finish();
    let failures = outputs
        .iter()
        .enumerate()
        .filter(|(_, output)| !output.status.success())
        .map(|(rank, output)| render_failure(rank, output))
        .collect::<Vec<_>>();
    assert!(
        failures.is_empty() && !timed_out,
        "{world_size}-process pipeline Ring test failed:\n{}",
        if timed_out {
            format!(
                "timed out waiting for Ring workers\n\n{}",
                failures.join("\n\n")
            )
        } else {
            failures.join("\n\n")
        }
    );
}
