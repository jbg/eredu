#![cfg(unix)]

use std::{
    collections::BTreeMap,
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Child, Command, Output, Stdio},
    thread,
    time::{Duration, Instant},
};

use crate::native::{
    distributed::{self, Backend},
    module::ModuleParameters,
    ops::{indexing::TryIndexOp, stack_axis},
    Array, Device, DeviceType, Dtype as MlxDtype, ExecutionContext, Stream,
};
use crate::MlxTensor;
use crate::{
    backend::runtime::{
        execution::layerwise::open_safetensors_weight_store,
        media::{input::InputPayload, PreparedModelInput},
    },
    backend::{
        error::Error as MlxError,
        nn::shared::{MlxModule, MlxNeuralBackend},
        DeviceAssignment, MlxBackend, MlxDistributedSession, MlxParallelContext, ModelLoadOptions,
    },
    composition::mlx::distributed::pipeline::{
        load_pipeline_model_with_options, PipelineLayerCache, PipelineModel, PipelineStep,
    },
    composition::{kimi_linear as neutral_kimi_linear, lfm2, nemotron_h as neutral_nemotron_h},
};
use eredu_architectures::gpt_oss;
use eredu_architectures::qwen::hybrid as qwen_hybrid;
use eredu_architectures::ModelKind;
use eredu_checkpoint::{AffineQuantization, WeightQuantization};
use eredu_core::cache::{PromptCacheDescriptor, PromptCacheOptions, PromptCacheTopology};
use eredu_core::{
    load_model, residency::OffloadConfig, BackendSession as _, FinishReason,
    GenerationCancellationToken, InputExtent, InputMetadataKey, InputModality, ModelRuntime,
    MtpCapability, MtpCheckpointKind, MtpConfig, ObservationRequest, SemanticEvent,
    SpeculativeDraft, SpeculativeGenerationBackend, SpeculativeGenerationBatchRequest,
    SpeculativeGenerationLane, SpeculativeOutputError, SpeculativeSemanticState,
    SpeculativeTokenFilterController, TextGenerationConfig, TokenFilter, TokenFilterController,
    TokenOutput as _,
};
use eredu_gguf::{
    GgmlType, MetadataArray, MetadataValue as GgufMetadataValue, TensorInput, Writer,
};
use eredu_nn::{ParameterMetadata, ParameterVisitor, ParameterVisitorMut, Parameterized};
use eredu_runtime::{
    CacheResidencyPolicy, DefaultSampler, DenseDiskStreamLoadOptions, ExpertCacheLoadOptions,
    LayerwiseLoadOptions, NonExpertWeightResidency, PagedCacheOptions, WeightResidency,
};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

const WORKER_RANK: &str = "EREDU_PIPELINE_RING_WORKER";
const CHECKPOINT_DIR: &str = "EREDU_PIPELINE_CHECKPOINT";
const FIXTURE_FAMILY: &str = "EREDU_PIPELINE_FIXTURE_FAMILY";
const DENSE_STREAM: &str = "EREDU_PIPELINE_DENSE_STREAM";
const LAYERWISE_HOST: &str = "EREDU_PIPELINE_LAYERWISE_HOST";
const PROMPT_CACHE_ROOT: &str = "EREDU_PIPELINE_PROMPT_CACHE";
const CARTESIAN_AXES: &str = "EREDU_PIPELINE_CARTESIAN_AXES";
const EXPERT_CACHE: &str = "EREDU_PIPELINE_EXPERT_CACHE";
const REQUANTIZE: &str = "EREDU_PIPELINE_REQUANTIZE";
const OPAQUE_SESSION: &str = "EREDU_PIPELINE_OPAQUE_SESSION";
const OPAQUE_INSPECTION: &str = "EREDU_PIPELINE_OPAQUE_INSPECTION";
const OPAQUE_TEXT_GENERATION: &str = "EREDU_PIPELINE_OPAQUE_TEXT_GENERATION";
const OPAQUE_MUSE_IMAGE: &str = "EREDU_PIPELINE_OPAQUE_MUSE_IMAGE";
const OPAQUE_INKLING_MEDIA: &str = "EREDU_PIPELINE_OPAQUE_INKLING_MEDIA";
const OPAQUE_INKLING_MTP: &str = "EREDU_PIPELINE_OPAQUE_INKLING_MTP";
const OPAQUE_GEMMA4_MEDIA: &str = "EREDU_PIPELINE_OPAQUE_GEMMA4_MEDIA";

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
    config: MtpConfig,
) -> eredu_core::SpeculativeGenerationOutput {
    let sampling = eredu_core::resolve_generation_config(
        None,
        eredu_core::GenerationConfigOverrides {
            max_new_tokens: Some(config.max_tokens),
            temperature: Some(config.temperature),
            ..Default::default()
        },
    )
    .unwrap();
    let output = <MlxBackend<'world> as SpeculativeGenerationBackend>::with_speculative_execution(
        runtime,
        SpeculativeGenerationBatchRequest {
            drafting: SpeculativeDraft::Embedded,
            lanes: vec![SpeculativeGenerationLane {
                prompt,
                generation: TextGenerationConfig::new(sampling),
                config,
                constraint: AllowAllTokens,
                semantic: Box::<TokenOnlySemanticState>::default(),
                cancellation: GenerationCancellationToken::new(),
                on_event: Box::new(|_| {}),
            }],
            tokenizer_fingerprint: [0; 32],
        },
        eredu_runtime::RunSpeculativeGeneration::default(),
    )
    .unwrap();
    output.requests.into_iter().next().unwrap()
}

fn load_prepared_pipeline_model(
    checkpoint: &Path,
    options: ModelLoadOptions,
    stream: &Stream,
) -> PipelineModel {
    let inspection = eredu_architectures::configuration::inspect_artifact(checkpoint).unwrap();
    let plan = eredu_core::plan_model_preparation(
        inspection,
        options.preparation_policy().unwrap(),
        eredu_core::SessionCapabilities {
            persistent_cache: true,
            output_observation: true,
            activation_inspection: true,
        },
    )
    .unwrap();
    load_pipeline_model_with_options(plan, options, stream, stream).unwrap()
}

#[test]
fn distributed_materialization_uses_the_planned_configuration() {
    let checkpoint = tempfile::tempdir().unwrap();
    write_deepseek_fixture(checkpoint.path(), 2);
    let topology =
        MlxParallelContext::for_rank(0, 1, 2, 1, DeviceAssignment::new(DeviceType::Cpu, 0))
            .unwrap();
    let layerwise = LayerwiseLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap());
    let options = ModelLoadOptions::with_parallel(
        topology,
        eredu_runtime::PipelineWireContract::new(eredu_runtime::PipelineActivationDtype::Float32),
    )
    .with_weight_residency(WeightResidency::layerwise_host(layerwise));
    let inspection =
        eredu_architectures::configuration::inspect_artifact(checkpoint.path()).unwrap();
    let plan = eredu_core::plan_model_preparation(
        inspection,
        options.preparation_policy().unwrap(),
        eredu_core::SessionCapabilities {
            persistent_cache: true,
            output_observation: true,
            activation_inspection: true,
        },
    )
    .unwrap();

    std::fs::remove_file(checkpoint.path().join("config.json")).unwrap();

    let stream = Stream::new_with_device(&topology.device.device().unwrap());
    let error = match load_pipeline_model_with_options(plan, options, &stream, &stream) {
        Ok(model) => {
            assert_eq!(model.stage_info().global_layer_range, 0..1);
            return;
        }
        Err(error) => error,
    };
    // This compact fixture intentionally retains two tensors outside the
    // stage contract. Reaching binding validation after config.json is gone
    // proves the distributed loader consumed the plan-owned JSON instead of
    // reopening the artifact configuration.
    assert!(
        matches!(error, MlxError::StrictLoadValidation { .. }),
        "expected checkpoint binding validation after planned configuration parsing, got {error}"
    );
}

#[test]
fn pipeline_identity_preserves_family_and_effective_wrapper_type() {
    let checkpoint = tempfile::tempdir().unwrap();
    write_qwen_fixture(checkpoint.path(), "qwen3_moe");
    let topology =
        MlxParallelContext::for_rank(0, 1, 2, 1, DeviceAssignment::new(DeviceType::Cpu, 0))
            .unwrap();
    let stream = Stream::new_with_device(&topology.device.device().unwrap());
    let model = load_prepared_pipeline_model(
        checkpoint.path(),
        ModelLoadOptions::with_parallel(
            topology,
            eredu_runtime::PipelineWireContract::new(
                eredu_runtime::PipelineActivationDtype::Float32,
            ),
        ),
        &stream,
    );

    assert_eq!(model.model_family(), ModelKind::Qwen3);
    assert_eq!(model.effective_model_type(), "qwen3_moe");

    let checkpoint = tempfile::tempdir().unwrap();
    write_qwen35_multimodal_fixture(checkpoint.path(), true);
    let topology =
        MlxParallelContext::for_rank(0, 1, 2, 1, DeviceAssignment::new(DeviceType::Cpu, 0))
            .unwrap();
    let stream = Stream::new_with_device(&topology.device.device().unwrap());
    let model = load_prepared_pipeline_model(
        checkpoint.path(),
        ModelLoadOptions::with_parallel(
            topology,
            eredu_runtime::PipelineWireContract::new(
                eredu_runtime::PipelineActivationDtype::Float32,
            ),
        ),
        &stream,
    );

    assert_eq!(model.model_family(), ModelKind::Qwen35);
    assert_eq!(model.effective_model_type(), "qwen3_5_moe_text");
}

#[test]
fn pipeline_activation_dtype_comes_from_wire_contract_not_weights() {
    let checkpoint = tempfile::tempdir().unwrap();
    write_fixture(checkpoint.path());
    let topology =
        MlxParallelContext::for_rank(0, 1, 2, 1, DeviceAssignment::new(DeviceType::Cpu, 0))
            .unwrap();
    let wire_contract =
        eredu_runtime::PipelineWireContract::new(eredu_runtime::PipelineActivationDtype::Bfloat16);
    let stream = Stream::new_with_device(&topology.device.device().unwrap());
    let model = load_prepared_pipeline_model(
        checkpoint.path(),
        ModelLoadOptions::with_parallel(topology, wire_contract),
        &stream,
    );

    assert_eq!(model.stage_info().wire_contract, wire_contract);
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
    Qwen3,
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
    Qwen35MoeMultimodal,
    Qwen3Vl,
    Qwen3VlMoe,
    Inkling,
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
            Self::Qwen3 => "qwen3",
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
            Self::Qwen35MoeMultimodal => "qwen3.5-moe-multimodal",
            Self::Qwen3Vl => "qwen3-vl",
            Self::Qwen3VlMoe => "qwen3-vl-moe",
            Self::Inkling => "inkling",
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
            Self::Qwen3,
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
            Self::Qwen35MoeMultimodal,
            Self::Qwen3Vl,
            Self::Qwen3VlMoe,
            Self::Inkling,
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
            | Self::Qwen3
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
            Self::Qwen35MoeMultimodal | Self::Qwen3Vl | Self::Qwen3VlMoe | Self::MuseGlimmer => 2,
            Self::Gemma | Self::NemotronH | Self::NemotronHGguf => 4,
            Self::Inkling | Self::InklingMultimodal | Self::InklingGguf => 3,
        }
    }

    fn stage_range(self, rank: usize) -> std::ops::Range<usize> {
        match (self, rank) {
            (Self::Gemma, 0) => 0..1,
            (Self::Gemma, 1) => 1..4,
            (Self::NemotronH | Self::NemotronHGguf, 0) => 0..2,
            (Self::NemotronH | Self::NemotronHGguf, 1) => 2..4,
            (Self::Inkling | Self::InklingMultimodal | Self::InklingGguf, 0) => 0..2,
            (Self::Inkling | Self::InklingMultimodal | Self::InklingGguf, 1) => 2..3,
            (_, rank) => rank..rank + 1,
        }
    }

    fn expert_layer_count(self, range: std::ops::Range<usize>) -> usize {
        match self {
            Self::DeepSeek | Self::DeepSeekGguf | Self::KimiLinear | Self::KimiLinearGguf => {
                range.filter(|index| *index == 1).count()
            }
            Self::DeepSeekV4 => range.len(),
            Self::Inkling | Self::InklingMultimodal | Self::InklingGguf => {
                range.filter(|index| matches!(*index, 1 | 2)).count()
            }
            Self::NemotronH => range.filter(|index| *index == 2).count(),
            Self::NemotronHGguf => range.filter(|index| matches!(*index, 1 | 2)).count(),
            _ => range.len(),
        }
    }

    fn descriptor_names(self) -> (&'static str, &'static str) {
        match self {
            Self::Llama => ("llama", "llama"),
            Self::Mistral => ("llama", "mistral"),
            Self::DeepSeek | Self::DeepSeekGguf => ("deepseek_v3", "deepseek_v3"),
            Self::DeepSeekV4 => ("deepseek_v4", "deepseek_v4"),
            Self::Gemma => ("gemma4", "gemma4"),
            Self::MuseGlimmer => ("muse_glimmer", "muse_glimmer_text"),
            Self::Qwen2 => ("qwen", "qwen2"),
            Self::Qwen3 => ("qwen", "qwen3"),
            Self::Qwen3Moe | Self::Qwen3MoeTied | Self::Qwen3MoeGguf => ("qwen", "qwen3_moe"),
            Self::GptOss | Self::GptOssGguf => ("gpt_oss", "gpt_oss"),
            Self::Lfm2 => ("lfm2", "lfm2"),
            Self::Lfm2Moe | Self::Lfm2MoeGguf => ("lfm2", "lfm2_moe"),
            Self::KimiLinear | Self::KimiLinearGguf => ("kimi_linear", "kimi_linear"),
            Self::NemotronH | Self::NemotronHGguf => ("nemotron_h", "nemotron_h"),
            Self::Qwen3Next => ("qwen_hybrid", "qwen3_next"),
            Self::Qwen3NextMoe => ("qwen_hybrid", "qwen3_next"),
            Self::Qwen35 => ("qwen_hybrid", "qwen3_5_text"),
            Self::Qwen35Moe => ("qwen_hybrid", "qwen3_5_moe_text"),
            Self::Qwen35Multimodal => ("qwen_hybrid", "qwen3_5_text"),
            Self::Qwen35MoeMultimodal => ("qwen_hybrid", "qwen3_5_moe_text"),
            Self::Qwen3Vl => ("qwen3_vl", "qwen3_vl"),
            Self::Qwen3VlMoe => ("qwen3_vl", "qwen3_vl_moe"),
            Self::Inkling | Self::InklingMultimodal | Self::InklingGguf => {
                ("inkling", "inkling_mm_model")
            }
        }
    }

    const fn layer_prefix(self) -> &'static str {
        match self {
            Self::Gemma | Self::MuseGlimmer | Self::Qwen3Vl | Self::Qwen3VlMoe => {
                "model.language_model.layers."
            }
            Self::DeepSeekV4 => "layers.",
            Self::NemotronH | Self::NemotronHGguf => "backbone.layers.",
            Self::Inkling | Self::InklingMultimodal | Self::InklingGguf => "model.llm.layers.",
            _ => "model.layers.",
        }
    }

    const fn has_gguf_source(self) -> bool {
        matches!(
            self,
            Self::DeepSeekGguf
                | Self::KimiLinearGguf
                | Self::InklingGguf
                | Self::Qwen3MoeGguf
                | Self::GptOssGguf
                | Self::Lfm2MoeGguf
                | Self::NemotronHGguf
        )
    }

    const fn needs_resident_reference(self) -> bool {
        matches!(
            self,
            Self::Llama
                | Self::Mistral
                | Self::Gemma
                | Self::MuseGlimmer
                | Self::Qwen2
                | Self::Qwen3
                | Self::DeepSeek
                | Self::DeepSeekV4
                | Self::DeepSeekGguf
                | Self::KimiLinear
                | Self::KimiLinearGguf
                | Self::NemotronH
                | Self::NemotronHGguf
                | Self::Qwen3Next
                | Self::Qwen3Moe
                | Self::Qwen3MoeTied
                | Self::Qwen3MoeGguf
                | Self::GptOss
                | Self::GptOssGguf
                | Self::Lfm2MoeGguf
                | Self::Qwen3NextMoe
                | Self::Qwen35
                | Self::Qwen35Moe
                | Self::Qwen35Multimodal
                | Self::Qwen35MoeMultimodal
                | Self::Qwen3Vl
                | Self::Qwen3VlMoe
                | Self::Inkling
                | Self::InklingMultimodal
                | Self::InklingGguf
        )
    }

    const fn needs_tp2_opaque_reference(self) -> bool {
        matches!(
            self,
            Self::Llama
                | Self::Mistral
                | Self::Gemma
                | Self::MuseGlimmer
                | Self::Qwen2
                | Self::Qwen3
                | Self::Qwen3Next
                | Self::Qwen35
                | Self::Qwen35Multimodal
                | Self::Qwen3Vl
                | Self::DeepSeek
                | Self::DeepSeekV4
        )
    }

    const fn is_multimodal(self) -> bool {
        matches!(
            self,
            Self::InklingMultimodal
                | Self::Qwen35Multimodal
                | Self::Qwen35MoeMultimodal
                | Self::Qwen3Vl
                | Self::Qwen3VlMoe
        )
    }

    const fn has_streamed_media_unit(self) -> bool {
        matches!(
            self,
            Self::Qwen35Multimodal | Self::Qwen35MoeMultimodal | Self::Qwen3Vl | Self::Qwen3VlMoe
        )
    }

    const fn comparison_tolerance(self) -> f32 {
        if matches!(self, Self::DeepSeekV4) {
            1e-3
        } else if self.is_multimodal() {
            5e-4
        } else {
            1e-4
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
    let group = distributed::init(true, Backend::Ring).unwrap();
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
    let topology = MlxParallelContext::for_group(
        &group,
        tensor_parallel_size,
        pipeline_parallel_size,
        expert_parallel_size,
        DeviceAssignment::new(DeviceType::Cpu, 0),
    )
    .unwrap();
    assert_eq!(topology.global_rank, expected_rank);
    let pipeline_rank = topology.pipeline_parallel_rank;
    let stream = Stream::new_with_device(&topology.device.device().unwrap());
    if std::env::var_os(OPAQUE_SESSION).is_some() {
        let backend = crate::native::distributed_backend(&stream, &stream, &group);
        let load_options = if std::env::var_os(EXPERT_CACHE).is_some() {
            ModelLoadOptions::with_parallel(
                topology,
                eredu_runtime::PipelineWireContract::new(
                    eredu_runtime::PipelineActivationDtype::Float32,
                ),
            )
            .with_weight_residency(WeightResidency::with_expert_cache(
                NonExpertWeightResidency::FullyResident,
                ExpertCacheLoadOptions::default(),
            ))
        } else {
            ModelLoadOptions::with_parallel(
                topology,
                eredu_runtime::PipelineWireContract::new(
                    eredu_runtime::PipelineActivationDtype::Float32,
                ),
            )
        };
        let model = load_model(&backend, &checkpoint, load_options).unwrap();
        let expected_effective_model_type = if family == FixtureFamily::Gemma {
            let config: serde_json::Value =
                serde_json::from_slice(&std::fs::read(checkpoint.join("config.json")).unwrap())
                    .unwrap();
            match config["model_type"].as_str().unwrap() {
                "gemma4_unified" => "gemma4_unified",
                _ => family.descriptor_names().1,
            }
        } else {
            family.descriptor_names().1
        };
        let expected_model_family =
            ModelKind::resolve_model_type(expected_effective_model_type).unwrap();
        assert_eq!(model.model_family(), expected_model_family);
        assert_eq!(model.effective_model_type(), expected_effective_model_type);
        let expected_mtp_capability = model.mtp_capability_for_test();
        let mut runtime = eredu_core::ModelRuntime::from_prepared(backend, model).unwrap();
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
        assert_eq!(runtime.session().model_family(), expected_model_family);
        assert_eq!(
            runtime.session().effective_model_type(),
            expected_effective_model_type
        );
        assert_eq!(
            <MlxBackend<'_> as eredu_core::SpeculativeGenerationBackend>::mtp_capability(&runtime),
            expected_mtp_capability
        );
        use crate::backend::runtime::media::input::{InputPayload, ModelInput};
        let capability_tokens = Array::from_slice(&[1u32, 2], &[1, 2]);
        let capability_parts = [text_input_part(&capability_tokens)];
        let capability_input = ModelInput::new(&capability_parts).into();
        let capabilities =
            <MlxBackend<'_> as eredu_core::ModelCapabilityBackend>::model_capabilities(&runtime)
                .unwrap();
        assert_eq!(capabilities.model_type, expected_effective_model_type);
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
        let paged = PagedCacheOptions::new(1, 32768, 32768, 1)
            .unwrap()
            .with_full_attention(true);
        runtime
            .session_mut()
            .configure_cache(CacheResidencyPolicy::Paged(paged))
            .unwrap();
        let image_mode = std::env::var_os(OPAQUE_MUSE_IMAGE).is_some();
        let inkling_media_mode = std::env::var_os(OPAQUE_INKLING_MEDIA).is_some();
        let inkling_mtp_mode = std::env::var_os(OPAQUE_INKLING_MTP).is_some();
        let gemma4_media_mode = std::env::var_os(OPAQUE_GEMMA4_MEDIA).is_some();
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
        } else {
            vec![text_input_part(&prompt)]
        };
        let prefix_tokens = if image_mode {
            vec![1, 22, 2]
        } else if inkling_media_mode {
            vec![1, 2, 21, 20, 20, 20]
        } else if gemma4_media_mode {
            vec![1, 2, 30, 31]
        } else {
            vec![1, 2]
        };
        let reference_input =
            PreparedModelInput::from_model_input(ModelInput::new(&parts)).unwrap();
        let reference = (tensor_parallel_size == 2
            && pipeline_parallel_size == 1
            && (expert_parallel_size == 1
                || matches!(family, FixtureFamily::DeepSeek | FixtureFamily::DeepSeekV4))
            && family.needs_tp2_opaque_reference())
        .then(|| resident_reference_for_prepared(&checkpoint, &reference_input, &stream));
        let reference_tolerance = if image_mode || gemma4_media_mode {
            5e-4
        } else {
            family.comparison_tolerance()
        };
        if std::env::var_os(OPAQUE_INSPECTION).is_some() {
            let local_layers = runtime.session().prompt_cache_global_layer_range().unwrap();
            let layer_root = if family == FixtureFamily::Gemma {
                "model.language_model.layers"
            } else {
                "model.layers"
            };
            let expected = format!("{layer_root}.{}.output", local_layers.start);
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
                pipeline_rank + 1 == pipeline_parallel_size
            );
            return;
        }
        if inkling_mtp_mode {
            let layer_prefix_offsets = runtime
                .session()
                .prompt_cache_layer_prefix_offsets()
                .unwrap();
            assert_eq!(
                layer_prefix_offsets.contains(&-1),
                pipeline_rank + 1 == pipeline_parallel_size
            );
            let max_tokens = 3;
            let output = run_neutral_embedded_mtp(
                &mut runtime,
                ModelInput::new(&parts).into(),
                MtpConfig {
                    max_tokens,
                    max_draft_tokens: 2,
                    temperature: 0.0,
                    eos_token_ids: Vec::new(),
                },
            );
            assert_eq!(output.token_ids.len(), max_tokens);
            assert_eq!(output.stats.emitted_tokens, max_tokens);
            assert!(output.stats.draft_tokens > 0);
            return;
        }
        let (backend, session) = runtime.parts_mut();
        let mut output = session
            .prefill(backend, ModelInput::new(&parts).into())
            .unwrap()
            .wait()
            .unwrap();
        if let (Some(actual), Some((expected, _))) = (output.logits(), &reference) {
            assert_final_logits_close(actual.as_array(), expected, reference_tolerance);
        }
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(checkpoint.join("config.json")).unwrap())
                .unwrap();
        let outer_model_type = config["model_type"].as_str().unwrap();
        let effective_model_type =
            if matches!(outer_model_type, "muse_glimmer" | "qwen3_5" | "qwen3_5_moe") {
                config["text_config"]["model_type"].as_str().unwrap()
            } else {
                outer_model_type
            };
        let model_family = family.descriptor_names().0;
        let layer_layout = session.prompt_cache_layer_layout().unwrap();
        let layer_prefix_offsets = session.prompt_cache_layer_prefix_offsets().unwrap();
        let state_segments = session.prompt_cache_state_segments().unwrap();
        let layer_count = session.prompt_cache_layer_count().unwrap();
        let global_layer_range = session.prompt_cache_global_layer_range().unwrap();
        let descriptor = PromptCacheDescriptor {
            model_family: model_family.into(),
            effective_model_type: effective_model_type.into(),
            checkpoint_fingerprint: "opaque-ring-fixture".into(),
            prefix_content_fingerprint: format!("tokens:{prefix_tokens:?}"),
            architecture_fingerprint: session.prompt_cache_architecture_fingerprint().unwrap(),
            layer_count,
            global_layer_start: global_layer_range.start,
            global_layer_end: global_layer_range.end,
            batch_size: 1,
            layer_prefix_offsets,
            state_segments,
            layer_layout,
            sink_tokens: 0,
            topology: PromptCacheTopology {
                pipeline: (pipeline_parallel_size > 1)
                    .then_some((pipeline_parallel_size, pipeline_rank)),
                tensor_parallel: (tensor_parallel_size > 1)
                    .then_some((tensor_parallel_size, topology.tensor_parallel_rank)),
                expert_parallel: (expert_parallel_size > 1)
                    .then_some((expert_parallel_size, topology.expert_parallel_rank)),
                expert_parallel_cache_replicated: true,
            },
        };
        let rank_prompt_cache = prompt_cache_root.join(format!("rank-{expected_rank}"));
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
        session
            .load_prompt_cache(
                backend,
                &rank_prompt_cache,
                &descriptor,
                &prefix_tokens,
                PagedCacheOptions::new(1, 32768, 32768, 1)
                    .unwrap()
                    .with_full_attention(true),
            )
            .unwrap();
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
            assert_eq!(
                output.logits().is_some(),
                pipeline_rank + 1 == pipeline_parallel_size
            );
            let token = session
                .sample_and_synchronize(output.logits(), 1, &mut DefaultSampler, 0.0, None, false)
                .unwrap()
                .token;
            output = session.decode(backend, token).unwrap().wait().unwrap();
        }
        assert_eq!(
            output.logits().is_some(),
            pipeline_rank + 1 == pipeline_parallel_size
        );
        return;
    }
    let execution = crate::native::backend(&stream, &stream)
        .communication_for_topology(topology, &group)
        .unwrap();
    let reference = (pipeline_rank + 1 == pipeline_parallel_size
        && (family.needs_resident_reference()
            || matches!(family, FixtureFamily::Lfm2 | FixtureFamily::Lfm2Moe)))
    .then(|| {
        if family.is_multimodal() {
            multimodal_resident_reference(family, &checkpoint, &stream)
        } else if family == FixtureFamily::NemotronH && std::env::var_os(REQUANTIZE).is_some() {
            resident_reference_quantized(&checkpoint, Some(WeightQuantization::MxFp4), &stream)
        } else {
            resident_reference(&checkpoint, &stream)
        }
    });
    let dense_stream = std::env::var_os(DENSE_STREAM).is_some();
    let layerwise_host = std::env::var_os(LAYERWISE_HOST).is_some();
    assert!(!(dense_stream && layerwise_host));
    let expert_cache = std::env::var_os(EXPERT_CACHE).is_some();
    let requantize = std::env::var_os(REQUANTIZE).is_some();
    let (requested_quantization, requested_weight_quantization) =
        if family == FixtureFamily::NemotronH {
            (
                eredu_core::QuantizationRequest::MxFp4,
                WeightQuantization::MxFp4,
            )
        } else {
            (
                eredu_core::QuantizationRequest::Affine {
                    group_size: 32,
                    bits: 4,
                },
                WeightQuantization::Affine(AffineQuantization::new(32, 4).unwrap()),
            )
        };
    let base_options = || {
        if requantize {
            ModelLoadOptions::with_quantization(requested_quantization).with_parallel_topology(
                topology,
                eredu_runtime::PipelineWireContract::new(
                    eredu_runtime::PipelineActivationDtype::Float32,
                ),
            )
        } else {
            ModelLoadOptions::with_parallel(
                topology,
                eredu_runtime::PipelineWireContract::new(
                    eredu_runtime::PipelineActivationDtype::Float32,
                ),
            )
        }
    };
    let layerwise_options =
        || LayerwiseLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap());
    let mut model = if expert_cache {
        let non_experts = if dense_stream {
            NonExpertWeightResidency::DenseDiskStream(
                DenseDiskStreamLoadOptions::new(u64::MAX, u64::MAX, 1, 1).unwrap(),
            )
        } else if layerwise_host {
            NonExpertWeightResidency::LayerwiseHost(layerwise_options())
        } else {
            NonExpertWeightResidency::FullyResident
        };
        load_prepared_pipeline_model(
            &checkpoint,
            base_options().with_weight_residency(WeightResidency::with_expert_cache(
                non_experts,
                ExpertCacheLoadOptions::default(),
            )),
            &stream,
        )
    } else if layerwise_host {
        load_prepared_pipeline_model(
            &checkpoint,
            base_options()
                .with_weight_residency(WeightResidency::layerwise_host(layerwise_options())),
            &stream,
        )
    } else if dense_stream {
        let dense = DenseDiskStreamLoadOptions::new(u64::MAX, u64::MAX, 1, 1).unwrap();
        load_prepared_pipeline_model(
            &checkpoint,
            base_options().with_weight_residency(WeightResidency::dense_disk_stream(dense)),
            &stream,
        )
    } else {
        load_prepared_pipeline_model(&checkpoint, base_options(), &stream)
    };
    let expected_effective_model_type = family.descriptor_names().1;
    assert_eq!(
        model.model_family(),
        ModelKind::resolve_model_type(expected_effective_model_type).unwrap()
    );
    assert_eq!(model.effective_model_type(), expected_effective_model_type);
    let info = model.stage_info();
    let expected_range = family.stage_range(pipeline_rank);
    assert_eq!(info.global_layer_range, expected_range);
    if !family.has_gguf_source() {
        let prefix = family.layer_prefix();
        assert_eq!(
            info.owned_tensors.iter().any(|name| expected_range
                .clone()
                .any(|layer| name.starts_with(&format!("{prefix}{layer}.")))),
            !dense_stream && !layerwise_host
        );
        assert!(!info.owned_tensors.iter().any(|name| {
            (0..family.layer_count()).any(|layer| {
                !expected_range.contains(&layer) && name.starts_with(&format!("{prefix}{layer}."))
            })
        }));
    }
    if family == FixtureFamily::Llama {
        assert!(info.local_parameter_bytes < 1_616);
    }
    let opened = info
        .opened_checkpoint_shards
        .iter()
        .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    if family == FixtureFamily::Llama {
        assert_eq!(
            opened.contains(&format!("layer-{expected_rank}.safetensors")),
            !dense_stream
        );
        assert!(!opened.contains(&format!("layer-{}.safetensors", 1 - expected_rank)));
        assert_eq!(
            opened.contains(&"input.safetensors".into()),
            expected_rank == 0
        );
    }
    if family.needs_resident_reference()
        && !family.has_gguf_source()
        && !matches!(
            family,
            FixtureFamily::DeepSeek
                | FixtureFamily::DeepSeekV4
                | FixtureFamily::Qwen3Moe
                | FixtureFamily::Qwen3MoeTied
                | FixtureFamily::GptOss
        )
    {
        for layer in 0..family.layer_count() {
            assert_eq!(
                opened.contains(&format!("layer-{layer}.safetensors")),
                (requantize
                    || (!dense_stream && !layerwise_host)
                    || (expert_cache
                        && !matches!(
                            family,
                            FixtureFamily::KimiLinear
                                | FixtureFamily::Inkling
                                | FixtureFamily::InklingMultimodal
                        ))
                    || matches!(
                        family,
                        FixtureFamily::Qwen3Next
                            | FixtureFamily::Qwen3NextMoe
                            | FixtureFamily::Qwen35
                            | FixtureFamily::Qwen35Moe
                            | FixtureFamily::Qwen35Multimodal
                            | FixtureFamily::Qwen35MoeMultimodal
                            | FixtureFamily::Qwen3Vl
                            | FixtureFamily::Qwen3VlMoe
                    ))
                    && expected_range.contains(&layer),
                "rank {expected_rank} opened the wrong SafeTensors layer shard {layer} for {family:?}: {opened:?}; owned={:?}",
                info.owned_tensors
            );
        }
    }
    if dense_stream {
        let report = model.dense_stream_report().unwrap().unwrap();
        let expected_units = expected_range.len() + usize::from(family.has_streamed_media_unit());
        assert_eq!(report.planned_layer_count(), expected_units);
        assert!(report
            .residency()
            .units()
            .iter()
            .all(|unit| !unit.host_resident() && !unit.device_resident()));
        if requantize {
            let materialization = report.residency().materialization().unwrap();
            assert!(materialization.transformed_weights > 0);
            assert!(materialization.output_bytes < materialization.source_bytes_read);
            assert!(info.materialization.is_some());
        }
        if family.has_gguf_source() {
            let diagnostics = model.checkpoint_diagnostics().unwrap().unwrap();
            let global_payload = std::fs::metadata(&checkpoint).unwrap().len();
            assert!(diagnostics.physical_reads > 0);
            assert!(
                diagnostics.physical_read_bytes < global_payload,
                "rank {expected_rank} read {} GGUF bytes while loading static modules from a {global_payload}-byte global tensor payload",
                diagnostics.physical_read_bytes
            );
        }
    }
    if layerwise_host {
        assert!(model.dense_stream_report().unwrap().is_none());
        let report = model.parameter_residency_report().unwrap().unwrap();
        assert!(report.initialized());
        let expected_units = expected_range.len() + usize::from(family.has_streamed_media_unit());
        assert_eq!(report.units().len(), expected_units);
        assert!(report
            .units()
            .iter()
            .all(|unit| unit.host_resident() && !unit.device_resident()));
        if requantize {
            let materialization = report.materialization().unwrap();
            assert!(materialization.transformed_weights > 0);
            assert!(materialization.output_bytes < materialization.source_bytes_read);
            assert!(info.materialization.is_some());
        }
        if family.has_gguf_source() {
            let diagnostics = model.checkpoint_diagnostics().unwrap().unwrap();
            let global_payload = std::fs::metadata(&checkpoint).unwrap().len();
            assert!(diagnostics.physical_reads > 0);
            assert!(
                diagnostics.physical_read_bytes < global_payload,
                "rank {expected_rank} read {} GGUF bytes while loading a host-layerwise stage from a {global_payload}-byte global tensor payload",
                diagnostics.physical_read_bytes
            );
        }
    }
    if requantize && !dense_stream && !layerwise_host && !expert_cache {
        let materialization = info
            .materialization
            .as_ref()
            .expect("fully resident requantization must report its packed overlay");
        assert!(materialization.transformed_weights > 0);
        assert!(materialization.source_tiles > 0);
        assert!(materialization.output_bytes < materialization.source_bytes_read);
        assert!(
            materialization.peak_planned_working_set_bytes
                <= materialization.admitted_working_set_bytes,
            "rank {expected_rank} exceeded its admitted conversion bound: {materialization:?}"
        );
        assert_eq!(
            materialization.admitted_working_set_bytes,
            materialization
                .output_bytes
                .max(materialization.peak_planned_working_set_bytes),
            "rank {expected_rank} admitted slack beyond its packed stage or smallest legal row tile"
        );
    }
    if expert_cache {
        let report = model.expert_cache_report().unwrap();
        let predictor_expert_layers = usize::from(
            info.owns_output
                && matches!(
                    family,
                    FixtureFamily::DeepSeekV4
                        | FixtureFamily::Qwen3NextMoe
                        | FixtureFamily::Qwen35Moe
                        | FixtureFamily::Qwen35MoeMultimodal
                ),
        ) * info.embedded_mtp_layers;
        let expert_layers = family.expert_layer_count(expected_range.clone());
        let shared_inkling_experts = usize::from(matches!(
            family,
            FixtureFamily::Inkling | FixtureFamily::InklingMultimodal | FixtureFamily::InklingGguf
        )) * expert_layers;
        let expected_experts = (expert_layers + predictor_expert_layers)
            * info.local_expert_ids.len()
            + shared_inkling_experts;
        assert_eq!(report.is_some(), expected_experts > 0);
        if let Some(report) = report {
            assert_eq!(report.owned_experts, expected_experts);
            assert!(report.owned_bytes > 0);
            assert_eq!(report.device_resident_experts, 0);
            if requantize {
                assert_eq!(
                    report.weight_quantization,
                    Some(requested_weight_quantization)
                );
                let materialization = report.materialization.as_ref().unwrap();
                assert!(materialization.transformed_weights > 0);
                assert!(materialization.source_tiles > 0);
                assert!(materialization.output_bytes < materialization.source_bytes_read);
            }
        }
    }
    if family == FixtureFamily::Llama {
        assert_eq!(
            opened.contains(&"output.safetensors".into()),
            expected_rank == 1
        );
    }

    let paged = PagedCacheOptions::new(1, 32768, 32768, 1)
        .unwrap()
        .with_full_attention(true);
    let mut cache = model
        .new_cache_with_options(CacheResidencyPolicy::Paged(paged.clone()))
        .unwrap();
    assert_eq!(
        cache.global_layers(),
        family.stage_range(pipeline_rank).collect::<Vec<_>>()
    );
    assert_family_cache(family, pipeline_rank, &cache, 0);
    let prefix_ids = match family {
        FixtureFamily::InklingMultimodal => vec![1, 2, 21, 20, 20],
        FixtureFamily::Qwen35Multimodal | FixtureFamily::Qwen35MoeMultimodal => {
            vec![1, 2, 42, 42]
        }
        FixtureFamily::Qwen3Vl | FixtureFamily::Qwen3VlMoe => vec![1, 2, 42, 42],
        _ => vec![1, 2],
    };
    let prompt_length = prefix_ids.len() as i32;
    let mut logits = if family.is_multimodal() {
        let prepared = multimodal_prepared_input(family);
        prepared
            .with_model_input(|input| {
                model.prefill_distributed(
                    model.stage_info().owns_input.then_some(input),
                    PipelineStep::new(1, prompt_length).unwrap(),
                    None,
                    &mut cache,
                    &execution,
                )
            })
            .unwrap()
            .into_logits()
            .unwrap()
    } else {
        let prompt = crate::native::Array::from_slice(&prefix_ids, &[1, prompt_length]);
        forward_pipeline_model(
            &mut model,
            (pipeline_rank == 0).then_some(&prompt),
            PipelineStep::new(1, prompt_length).unwrap(),
            &mut cache,
            &execution,
        )
    };
    assert_eq!(logits.is_some(), pipeline_rank == 1);
    if let (Some(actual), Some((expected, _))) = (&logits, &reference) {
        assert_final_logits_close(actual, expected, family.comparison_tolerance());
    }
    assert_family_cache(family, pipeline_rank, &cache, prompt_length);
    let (model_family, effective_model_type) = family.descriptor_names();
    let layer_layout = model.prompt_cache_layer_layout().unwrap();
    let state_segments = model.prompt_cache_state_segments().unwrap();
    let target_layer_count = family.stage_range(pipeline_rank).len();
    let mut layer_prefix_offsets = vec![0; layer_layout.len()];
    layer_prefix_offsets[target_layer_count..].fill(-1);
    let descriptor = PromptCacheDescriptor {
        model_family: model_family.into(),
        effective_model_type: effective_model_type.into(),
        checkpoint_fingerprint: "pipeline-ring-fixture".into(),
        prefix_content_fingerprint: format!("tokens:{prefix_ids:?}"),
        architecture_fingerprint: model.prompt_cache_architecture_fingerprint().unwrap(),
        layer_count: model.prompt_cache_model_identity().unwrap().layer_count,
        global_layer_start: family.stage_range(pipeline_rank).start,
        global_layer_end: family.stage_range(pipeline_rank).start + layer_layout.len(),
        batch_size: 1,
        layer_prefix_offsets,
        state_segments,
        layer_layout,
        sink_tokens: 0,
        topology: PromptCacheTopology {
            pipeline: Some((2, pipeline_rank)),
            tensor_parallel: (tensor_parallel_size > 1)
                .then_some((tensor_parallel_size, topology.tensor_parallel_rank)),
            expert_parallel: (expert_parallel_size > 1)
                .then_some((expert_parallel_size, topology.expert_parallel_rank)),
            expert_parallel_cache_replicated: true,
        },
    };
    model
        .save_prompt_cache(
            &mut cache,
            &prompt_cache_root,
            descriptor.clone(),
            &prefix_ids,
            &PromptCacheOptions::default(),
            &stream,
        )
        .unwrap();
    let token = crate::native::Array::from_slice(&[0u32], &[1, 1]);
    let uninterrupted = forward_pipeline_model(
        &mut model,
        (pipeline_rank == 0).then_some(&token),
        PipelineStep::new(1, 1).unwrap(),
        &mut cache,
        &execution,
    );
    let uninterrupted_values = uninterrupted.as_ref().map(|value| {
        let value = value.evaluated().unwrap();
        value.as_slice::<f32>().to_vec()
    });
    if let (Some(actual), Some((_, expected))) = (&uninterrupted, &reference) {
        assert_final_logits_close(actual, expected, family.comparison_tolerance());
    }
    let (mut cache, manifest) = model
        .load_prompt_cache(&prompt_cache_root, &descriptor, &prefix_ids, paged, &stream)
        .unwrap();
    assert_eq!(manifest.topology, descriptor.topology);
    let restored = forward_pipeline_model(
        &mut model,
        (pipeline_rank == 0).then_some(&token),
        PipelineStep::new(1, 1).unwrap(),
        &mut cache,
        &execution,
    );
    match (&uninterrupted_values, &restored) {
        (Some(uninterrupted), Some(restored)) => {
            let restored = restored.evaluated().unwrap();
            if family == FixtureFamily::DeepSeekV4 {
                let restored = restored.as_slice::<f32>();
                assert_eq!(uninterrupted.len(), restored.len());
                assert!(
                    uninterrupted
                        .iter()
                        .zip(restored)
                        .all(|(left, right)| (left - right).abs() <= 1e-5),
                    "V4 prompt-cache restoration diverged: uninterrupted={uninterrupted:?}, restored={restored:?}"
                );
            } else {
                assert_eq!(uninterrupted, restored.as_slice::<f32>());
            }
        }
        (None, None) => {}
        _ => panic!("pipeline prompt-cache restoration changed stage output ownership"),
    }
    logits = restored;

    let mut sampler = DefaultSampler;
    for sample_index in 0..2 {
        let synchronized = model
            .sample_and_synchronize(
                logits.as_ref(),
                PipelineStep::new(1, 1).unwrap(),
                &mut sampler,
                0.0,
                None,
                false,
                &execution,
            )
            .unwrap();
        let token = synchronized.token.evaluated().unwrap();
        assert_eq!(token.as_array().shape(), &[1, 1]);
        if sample_index == 0 {
            if let Some((_, expected)) = &reference {
                let expected_token = expected
                    .iter()
                    .enumerate()
                    .fold((0usize, f32::NEG_INFINITY), |best, (index, &value)| {
                        if value > best.1 {
                            (index, value)
                        } else {
                            best
                        }
                    })
                    .0 as u32;
                assert_eq!(token.as_slice::<u32>(), &[expected_token]);
            }
        }
        drop(token);
        logits = forward_pipeline_model(
            &mut model,
            (pipeline_rank == 0).then_some(&synchronized.token),
            PipelineStep::new(1, 1).unwrap(),
            &mut cache,
            &execution,
        );
    }
    if dense_stream {
        let report = model.dense_stream_report().unwrap().unwrap();
        assert!(report.prefill_forwards() >= 1);
        assert!(report.decode_forwards() >= 2);
    }
    if layerwise_host {
        let report = model.parameter_residency_report().unwrap().unwrap();
        assert!(report.units().iter().all(|unit| unit.host_resident()));
        assert!(
            report
                .units()
                .iter()
                .filter(|unit| unit.device_resident())
                .count()
                <= 1
        );
    }
    if expert_cache {
        if let Some(report) = model.expert_cache_report().unwrap() {
            let requests = report.prefill.device.requests + report.decode.device.requests;
            if requests > 0 {
                assert!(report.device_resident_experts > 0);
            } else {
                assert_eq!(report.device_resident_experts, 0);
            }
        }
    }

    if matches!(
        family,
        FixtureFamily::DeepSeekV4
            | FixtureFamily::Qwen3Next
            | FixtureFamily::Qwen3NextMoe
            | FixtureFamily::Qwen35
            | FixtureFamily::Qwen35Moe
            | FixtureFamily::Qwen35Multimodal
            | FixtureFamily::Qwen35MoeMultimodal
    ) {
        assert_eq!(
            model.mtp_capability(),
            MtpCapability::Ready {
                checkpoint: MtpCheckpointKind::Embedded
            }
        );
        assert_eq!(model.stage_info().owns_embedded_mtp, pipeline_rank == 1);
        assert_eq!(
            model.stage_info().embedded_mtp_layers,
            usize::from(pipeline_rank == 1)
        );
        assert_eq!(model.stage_info().global_embedded_mtp_layers, 1);
    } else if matches!(family, FixtureFamily::Gemma | FixtureFamily::MuseGlimmer) {
        assert_eq!(
            model.mtp_capability(),
            MtpCapability::Unsupported {
                checkpoint: MtpCheckpointKind::Separate,
                architecture: model.model_family().canonical_name().into(),
            }
        );
    }
}

#[test]
fn complete_qwen3_vl_variants_accept_paged_cache() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    for (expected_family, moe) in [(ModelKind::Qwen3Vl, false), (ModelKind::Qwen3VlMoe, true)] {
        let checkpoint = tempfile::tempdir().unwrap();
        write_qwen3_vl_fixture(checkpoint.path(), moe);
        let backend = crate::native::backend(&stream, &stream);
        let model = load_model(&backend, checkpoint.path(), ModelLoadOptions::default()).unwrap();
        let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
        assert_eq!(runtime.session().model_family(), expected_family);

        let paged = PagedCacheOptions::new(1, 32768, 32768, 1)
            .unwrap()
            .with_full_attention(true);
        runtime
            .session_mut()
            .configure_cache(CacheResidencyPolicy::Paged(paged))
            .unwrap();
        assert!(runtime
            .session()
            .cache_residency_report()
            .unwrap()
            .is_some());
    }
}

fn resident_reference(checkpoint: &Path, stream: &Stream) -> (Vec<f32>, Vec<f32>) {
    resident_reference_quantized(checkpoint, None, stream)
}

fn resident_reference_quantized(
    checkpoint: &Path,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
) -> (Vec<f32>, Vec<f32>) {
    let options = quantization
        .map(|quantization| match quantization {
            WeightQuantization::Affine(config) => {
                ModelLoadOptions::with_quantization(eredu_core::QuantizationRequest::Affine {
                    group_size: u32::try_from(config.group_size).unwrap(),
                    bits: u8::try_from(config.bits).unwrap(),
                })
            }
            WeightQuantization::MxFp4 => {
                ModelLoadOptions::with_quantization(eredu_core::QuantizationRequest::MxFp4)
            }
            WeightQuantization::GgufIQuant { .. } => {
                panic!("checkpoint-native GGUF quantization is not a load-time transform")
            }
        })
        .unwrap_or_default();
    let backend = crate::native::backend(stream, stream);
    let mut model = eredu_core::load_model(&backend, checkpoint, options)
        .unwrap()
        .into_inner()
        .into_complete()
        .unwrap();
    let prompt = Array::from_slice(&[1u32, 2], &[1, 2]);
    let parts = [text_input_part(&prompt)];
    let prefill = model
        .submit_prefill(
            crate::backend::runtime::media::input::ModelInput::new(&parts),
            stream,
        )
        .unwrap()
        .wait()
        .unwrap()
        .evaluated()
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    let token = Array::from_slice(&[0u32], &[1, 1]);
    let decode = model
        .submit_decode(token, stream)
        .unwrap()
        .wait()
        .unwrap()
        .evaluated()
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    (prefill, decode)
}

fn resident_reference_for_prepared(
    checkpoint: &Path,
    prepared: &PreparedModelInput,
    stream: &Stream,
) -> (Vec<f32>, Vec<f32>) {
    let backend = crate::native::backend(stream, stream);
    let mut model = eredu_core::load_model(&backend, checkpoint, ModelLoadOptions::default())
        .unwrap()
        .into_inner()
        .into_complete()
        .unwrap();
    let parts = prepared.input_parts();
    let prefill = model
        .submit_prefill(
            crate::backend::runtime::media::input::ModelInput::new(parts),
            stream,
        )
        .unwrap()
        .wait()
        .unwrap()
        .evaluated()
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    let token = Array::from_slice(&[0u32], &[1, 1]);
    let decode = model
        .submit_decode(token, stream)
        .unwrap()
        .wait()
        .unwrap()
        .evaluated()
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    (prefill, decode)
}

fn inkling_multimodal_prepared_input() -> PreparedModelInput {
    use crate::backend::runtime::media::input::ModelInput;

    let text = Array::from_slice(&[1u32, 2], &[1, 2]);
    let image = Array::from_slice(&[0.01f32; 16], &[1, 1, 16]);
    let audio = Array::from_slice(&[0u32, 1, 2, 3, 4, 5], &[1, 3, 2]);
    let audio_mask = Array::from_slice(&[true, true, false], &[1, 3]);
    let parts = [
        text_input_part(&text),
        input_part(
            InputModality::Image,
            InputPayload::Embeddings(image),
            [],
            [],
        ),
        input_part(
            InputModality::Audio,
            InputPayload::Tensor(audio),
            [(InputMetadataKey::AudioMask, audio_mask)],
            [InputExtent::AudioValidFrames(2)],
        ),
    ];
    PreparedModelInput::from_model_input(ModelInput::new(&parts)).unwrap()
}

fn qwen35_multimodal_prepared_input() -> PreparedModelInput {
    use crate::backend::runtime::media::input::ModelInput;

    let text = Array::from_slice(&[1u32, 2], &[1, 2]);
    let grid = Array::from_slice(&[1i32, 2, 4], &[1, 3]);
    let pixels = Array::from_slice(&[0.01f32; 96], &[8, 12]);
    let parts = [
        text_input_part(&text),
        input_part(
            InputModality::Image,
            InputPayload::Tensor(pixels),
            [(InputMetadataKey::PatchGrid, grid)],
            [],
        ),
    ];
    PreparedModelInput::from_model_input(ModelInput::new(&parts)).unwrap()
}

fn qwen3_vl_prepared_input() -> PreparedModelInput {
    use crate::backend::runtime::media::input::ModelInput;

    let text = Array::from_slice(&[1u32, 2], &[1, 2]);
    let grid = Array::from_slice(&[1i32, 2, 4], &[1, 3]);
    let pixels = Array::from_slice(&[0.01f32; 96], &[8, 12]);
    let parts = [
        text_input_part(&text),
        input_part(
            InputModality::Image,
            InputPayload::Tensor(pixels),
            [(InputMetadataKey::PatchGrid, grid)],
            [],
        ),
    ];
    PreparedModelInput::from_model_input(ModelInput::new(&parts)).unwrap()
}

fn multimodal_prepared_input(family: FixtureFamily) -> PreparedModelInput {
    match family {
        FixtureFamily::InklingMultimodal => inkling_multimodal_prepared_input(),
        FixtureFamily::Qwen35Multimodal | FixtureFamily::Qwen35MoeMultimodal => {
            qwen35_multimodal_prepared_input()
        }
        FixtureFamily::Qwen3Vl | FixtureFamily::Qwen3VlMoe => qwen3_vl_prepared_input(),
        _ => panic!("{family:?} is not a multimodal fixture"),
    }
}

fn multimodal_resident_reference(
    family: FixtureFamily,
    checkpoint: &Path,
    stream: &Stream,
) -> (Vec<f32>, Vec<f32>) {
    let backend = crate::native::backend(stream, stream);
    let mut model = eredu_core::load_model(&backend, checkpoint, ModelLoadOptions::default())
        .unwrap()
        .into_inner()
        .into_complete()
        .unwrap();
    let prepared = multimodal_prepared_input(family);
    let parts = prepared.input_parts();
    let prefill = model
        .submit_prefill(
            crate::backend::runtime::media::input::ModelInput::new(parts),
            stream,
        )
        .unwrap()
        .wait()
        .unwrap()
        .evaluated()
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    let token = Array::from_slice(&[0u32], &[1, 1]);
    let decode = model
        .submit_decode(token, stream)
        .unwrap()
        .wait()
        .unwrap()
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

fn forward_pipeline_model(
    model: &mut crate::composition::mlx::distributed::pipeline::PipelineModel,
    tokens: Option<&Array>,
    step: PipelineStep,
    cache: &mut crate::composition::mlx::distributed::pipeline::PipelineCache,
    execution: &MlxDistributedSession<'_>,
) -> Option<Array> {
    model
        .forward_distributed(tokens, step, None, cache, execution)
        .unwrap()
        .into_logits()
        .unwrap()
}

fn assert_family_cache(
    family: FixtureFamily,
    rank: usize,
    cache: &crate::composition::mlx::distributed::pipeline::PipelineCache,
    expected_offset: i32,
) {
    let populated = expected_offset > 0;
    let assert_slots =
        |slots: &[crate::composition::mlx::distributed::pipeline::PipelineStateSlot], count| {
            assert_eq!(slots.len(), count);
            for slot in slots {
                assert_eq!(slot.value().is_some(), populated);
                assert_eq!(slot.offset(), expected_offset);
            }
        };
    match family {
        FixtureFamily::KimiLinear | FixtureFamily::KimiLinearGguf if rank == 0 => {
            let PipelineLayerCache::StateSlots { slots, .. } = &cache.layers()[0] else {
                panic!("Kimi KDA layer did not materialize fixed state")
            };
            assert_slots(slots, 4);
        }
        FixtureFamily::KimiLinear | FixtureFamily::KimiLinearGguf => {
            assert!(matches!(
                &cache.layers()[0],
                PipelineLayerCache::CompressedLatent { .. }
            ));
        }
        FixtureFamily::NemotronH if rank == 0 => {
            let PipelineLayerCache::StateSlots { slots, .. } = &cache.layers()[0] else {
                panic!("Nemotron Mamba layer did not materialize fixed state")
            };
            assert_slots(slots, 2);
            assert!(matches!(
                &cache.layers()[1],
                PipelineLayerCache::StateSlots { slots, .. } if slots.is_empty()
            ));
        }
        FixtureFamily::NemotronH => {
            assert!(matches!(
                &cache.layers()[0],
                PipelineLayerCache::StateSlots { slots, .. } if slots.is_empty()
            ));
            assert!(matches!(
                &cache.layers()[1],
                PipelineLayerCache::KeyValue { slots, .. } if slots.is_empty()
            ));
        }
        FixtureFamily::Qwen3Next
        | FixtureFamily::Qwen3NextMoe
        | FixtureFamily::Qwen35
        | FixtureFamily::Qwen35Moe
        | FixtureFamily::Qwen35Multimodal
        | FixtureFamily::Qwen35MoeMultimodal
            if rank == 0 =>
        {
            let PipelineLayerCache::StateSlots { slots, .. } = &cache.layers()[0] else {
                panic!("Qwen linear-attention layer did not materialize recurrent state")
            };
            assert_slots(slots, 2);
        }
        FixtureFamily::Qwen3Next
        | FixtureFamily::Qwen3NextMoe
        | FixtureFamily::Qwen35
        | FixtureFamily::Qwen35Moe
        | FixtureFamily::Qwen35Multimodal
        | FixtureFamily::Qwen35MoeMultimodal => assert!(matches!(
            &cache.layers()[0],
            PipelineLayerCache::KeyValue { slots, .. } if slots.is_empty()
        )),
        FixtureFamily::Inkling | FixtureFamily::InklingMultimodal | FixtureFamily::InklingGguf => {
            for layer in cache.layers() {
                let PipelineLayerCache::KeyValue { slots, .. } = layer else {
                    panic!("Inkling layer did not materialize KV plus convolution state")
                };
                assert_slots(slots, 4);
            }
        }
        _ => {}
    }
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

fn write_llama_compatible_fixture(directory: &Path, model_type: &str) {
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "model_type": model_type,
            "hidden_size": 4,
            "num_hidden_layers": 2,
            "intermediate_size": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 2,
            "rms_norm_eps": 0.00001,
            "vocab_size": 8,
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
        &[("model.embed_tokens.weight", vec![8, 4], 0.01)],
    );
    for layer in 0..2 {
        let prefix = format!("model.layers.{layer}");
        let names = [
            (
                format!("{prefix}.self_attn.q_proj.weight"),
                vec![4, 4],
                0.01,
            ),
            (
                format!("{prefix}.self_attn.k_proj.weight"),
                vec![4, 4],
                0.01,
            ),
            (
                format!("{prefix}.self_attn.v_proj.weight"),
                vec![4, 4],
                0.01,
            ),
            (
                format!("{prefix}.self_attn.o_proj.weight"),
                vec![4, 4],
                0.01,
            ),
            (format!("{prefix}.mlp.gate_proj.weight"), vec![8, 4], 0.01),
            (format!("{prefix}.mlp.up_proj.weight"), vec![8, 4], 0.01),
            (format!("{prefix}.mlp.down_proj.weight"), vec![4, 8], 0.01),
            (format!("{prefix}.input_layernorm.weight"), vec![4], 1.0),
            (
                format!("{prefix}.post_attention_layernorm.weight"),
                vec![4],
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
            ("model.norm.weight", vec![4], 1.0),
            ("lm_head.weight", vec![8, 4], 0.01),
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
        "num_nextn_predict_layers": 0,
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
    for layer in 0..layers as usize {
        eredu_architectures::deepseek::block::V3Block::<Backend>::new(&args, layer, stream)
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

fn write_deepseek_v4_fixture(directory: &Path) {
    let config = serde_json::json!({
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
        "compress_ratios": [0, 4, 0],
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
        "num_nextn_predict_layers": 1
    });
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
            "model_type": "gemma4",
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
    let config = qwen_config("qwen3_moe");
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
            .replace("model.embed_tokens", "token_embd")
            .replace("model.norm", "output_norm")
            .replace("lm_head", "output");
        specs.push(gguf_tensor_from_array(name, value));
    }
    let key = |suffix: &str| format!("qwen3moe.{suffix}");
    let metadata = BTreeMap::from([
        (
            "general.architecture".into(),
            GgufMetadataValue::String("qwen3moe".into()),
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
            key("expert_feed_forward_length"),
            GgufMetadataValue::Uint32(args.moe_intermediate_size as u32),
        ),
        (
            key("expert_count"),
            GgufMetadataValue::Uint32(args.num_experts as u32),
        ),
        (
            key("expert_used_count"),
            GgufMetadataValue::Uint32(args.num_experts_per_tok as u32),
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
        options: ModelLoadOptions,
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
    }

    inspect(
        |directory| {
            write_gpt_oss_fixture(directory);
            directory.to_path_buf()
        },
        ModelLoadOptions::default().with_weight_residency(WeightResidency::with_expert_cache(
            NonExpertWeightResidency::FullyResident,
            ExpertCacheLoadOptions::default(),
        )),
        "model.layers.0.output",
    );
    inspect(
        |directory| {
            write_nemotron_fixture(directory);
            directory.to_path_buf()
        },
        ModelLoadOptions::default(),
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

fn write_gpt_oss_gguf_fixture(path: &Path) {
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
    let mut model = MlxModule::new(lfm2::Lfm2CheckpointTemplate::new(args, stream).unwrap());
    for (name, parameter) in model.parameters_mut().flatten() {
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
    let arrays = model
        .parameters()
        .flatten()
        .into_iter()
        .map(|(name, value)| (name.to_string(), value.clone()))
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
    let mut model =
        MlxModule::new(lfm2::Lfm2CheckpointTemplate::new(args.clone(), stream).unwrap());
    initialize_fixture(&mut model, stream);
    let mut specs = Vec::new();
    for (runtime_name, value) in model.parameters().flatten() {
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

fn initialize_fixture(model: &mut impl ModuleParameters, stream: &Stream) {
    for (name, parameter) in model.parameters_mut().flatten() {
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
    model: &impl ModuleParameters,
) {
    let arrays = model
        .parameters()
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
    let config = kimi_linear_config();
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let args = eredu_architectures::kimi_linear::model_args_from_config_value(&config).unwrap();
    let mut model = MlxModule::new(
        neutral_kimi_linear::KimiLinearCheckpointTemplate::new(args.clone(), stream).unwrap(),
    );
    initialize_fixture(&mut model, stream);
    let mut arrays = Vec::<(String, Array)>::new();
    for (name, value) in model.parameters().flatten() {
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
        let checkpoint_name = if name.starts_with("model.layers.1.mlp.") {
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

fn write_nemotron_quantizable_fixture(directory: &Path) {
    write_nemotron_fixture_with_config(directory, nemotron_quantizable_config());
}

fn write_nemotron_fixture_with_config(directory: &Path, config: serde_json::Value) {
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let args = eredu_architectures::nemotron_h::model_args_from_config_value(&config).unwrap();
    let mut model = MlxModule::new(
        neutral_nemotron_h::NemotronHCheckpointTemplate::new(args.clone(), stream).unwrap(),
    );
    initialize_fixture(&mut model, stream);
    let mut arrays = Vec::<(String, Array)>::new();
    for (name, value) in model.parameters().flatten() {
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
        neutral_nemotron_h::NemotronHCheckpointTemplate::new(args.clone(), stream).unwrap(),
    );
    initialize_fixture(&mut model, stream);
    let mut specs = Vec::new();
    for (runtime_name, value) in model.parameters().flatten() {
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
        "vocab_size": 32,
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
        crate::composition::qwen::hybrid::QwenHybridCheckpointTemplate::new(parsed.text, stream)
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
        crate::composition::qwen::hybrid::QwenHybridCheckpointTemplate::new(parsed.text, stream)
            .unwrap(),
    );
    initialize_fixture(&mut model, stream);
    save_parameter_fixture(directory, &config, &model);
}

fn write_qwen35_multimodal_fixture(directory: &Path, moe: bool) {
    let text_config = if moe {
        qwen_hybrid_moe_config("qwen3_5_moe_text")
    } else {
        qwen_hybrid_config("qwen3_5_text")
    };
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
        crate::composition::qwen::hybrid::QwenConditionalCheckpointTemplate::new(parsed, stream)
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
        crate::composition::qwen::vl::QwenVlCheckpointTemplate::new(args, stream).unwrap(),
    );
    initialize_fixture(&mut model, stream);
    let arrays = model
        .parameters()
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
                crate::native::ops::zeros_dtype(parameter.shape(), parameter.dtype(), self.stream)
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
                crate::native::ops::zeros_dtype(parameter.shape(), parameter.dtype(), self.stream)
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
                crate::native::ops::zeros_dtype(parameter.shape(), parameter.dtype(), self.stream)
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

/// Verifies the same scheduler and cache isolation over bounded local layers.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_dense_stream_opaque_session() {
    run_ring_pipeline_mode(true, FixtureFamily::Llama, WorkerMode::OpaqueSession);
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
    run_ring_pipeline_mode(false, FixtureFamily::DeepSeekV4, WorkerMode::OpaqueSession);
}

#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_deepseek_v4_expert_prepared_speculative_capability() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::DeepSeekV4,
        "ep",
        WorkerMode::OpaqueSession,
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

/// Exercises V4 TP, PP, EP, streamed non-experts, and independent expert
/// caching in the full admitted Cartesian topology.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_deepseek_v4_streamed_triple_axis_expert_cache() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::DeepSeekV4,
        "tp-pp-ep",
        WorkerMode::ExpertCache,
    );
}

/// Keeps DeepSeek non-expert stage units resident while routed experts remain
/// independently cached across TP, PP, and EP.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_deepseek_resident_nonexpert_triple_axis_expert_cache() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::DeepSeek,
        "tp-pp-ep",
        WorkerMode::ExpertCache,
    );
}

/// Exercises dense-streamed DeepSeek non-experts and independent expert
/// caching across all three Cartesian axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_deepseek_streamed_triple_axis_expert_cache() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::DeepSeek,
        "tp-pp-ep",
        WorkerMode::ExpertCache,
    );
}

/// Compares the cache-backed DeepSeek V3 model facade under TP=2 x EP=2 with
/// the replicated model, covering rank-local expert geometry and exact-once TP.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_deepseek_cached_tensor_expert_model_parity() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::DeepSeek,
        "tp-ep",
        WorkerMode::OpaqueSessionExpertCache,
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
        WorkerMode::OpaqueSessionExpertCache,
    );
}

/// Exercises host-layerwise DeepSeek MLA blocks and independent expert caches
/// across all three axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_deepseek_layerwise_host_triple_axis_expert_cache() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::DeepSeek,
        "tp-pp-ep",
        WorkerMode::ExpertCache,
    );
}

/// Covers DeepSeek2 GGUF recipes, bounded reads, and independent expert
/// caching across TP, PP, and EP.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_deepseek_gguf_triple_axis_expert_cache() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::DeepSeekGguf,
        "tp-pp-ep",
        WorkerMode::ExpertCache,
    );
}

/// Covers independent DeepSeek expert caching for TP+PP with EP inactive.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_deepseek_tensor_pipeline_expert_cache_without_ep() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::DeepSeek,
        "tp-pp",
        WorkerMode::ExpertCache,
    );
}

/// Verifies cached DeepSeek schedule failure reaches consensus without leaving
/// compressed MLA state reusable.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_deepseek_expert_cache_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::DeepSeek,
        "pp-ep",
        WorkerMode::ExpertCache,
    );
}

/// Verifies dependency-safe Gemma placement, auxiliary-state transport, shared
/// KV decode state, and prompt-cache restoration across two pipeline stages.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_gemma_pipeline() {
    run_ring_pipeline(false, FixtureFamily::Gemma);
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

/// Proves GPT-OSS resident TP+PP+EP execution against the single-rank model.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_gpt_oss_triple_axis() {
    run_ring_cartesian_pipeline(false, FixtureFamily::GptOss, "tp-pp-ep");
}

/// Exercises GPT-OSS triple-axis dense streaming with independent expert caches.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_gpt_oss_streamed_triple_axis_expert_cache() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::GptOss,
        "tp-pp-ep",
        WorkerMode::ExpertCache,
    );
}

/// Exercises GPT-OSS host-backed non-expert layers with independent expert
/// caching across TP=2 x PP=2 x EP=2.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_gpt_oss_layerwise_host_triple_axis_expert_cache() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::GptOss,
        "tp-pp-ep",
        WorkerMode::ExpertCache,
    );
}

/// Exercises canonical type-39 GPT-OSS GGUF across all three axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_gpt_oss_gguf_triple_axis_expert_cache() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::GptOssGguf,
        "tp-pp-ep",
        WorkerMode::ExpertCache,
    );
}

/// Covers independent GPT-OSS expert caching when PP is active without EP.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_gpt_oss_pipeline_expert_cache() {
    run_ring_pipeline_mode(false, FixtureFamily::GptOss, WorkerMode::ExpertCache);
}

/// Verifies opaque-session execution with GPT-OSS cached experts.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_gpt_oss_pipeline_expert_cache_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::GptOss,
        "pp-ep",
        WorkerMode::OpaqueSessionExpertCache,
    );
}

/// Verifies descriptor-backed convolution state, paged KV state, and persisted
/// replay across two LFM2 pipeline ranks.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_lfm2_pipeline() {
    run_ring_pipeline(false, FixtureFamily::Lfm2);
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
fn ring_eight_process_lfm2_moe_streamed_triple_axis_expert_cache() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::Lfm2Moe,
        "tp-pp-ep",
        WorkerMode::ExpertCache,
    );
}

/// Exercises host-layerwise non-experts and independently cached LFM2 experts
/// across all three Cartesian axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_lfm2_moe_layerwise_host_triple_axis_expert_cache() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::Lfm2Moe,
        "tp-pp-ep",
        WorkerMode::ExpertCache,
    );
}

/// Exercises representative LFM2-MoE GGUF bindings, bounded non-expert reads,
/// and independent expert caching across all three axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_lfm2_moe_gguf_triple_axis_expert_cache() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::Lfm2MoeGguf,
        "tp-pp-ep",
        WorkerMode::ExpertCache,
    );
}

/// Covers independent LFM2 expert caching when PP is active without EP.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_lfm2_moe_pipeline_expert_cache() {
    run_ring_pipeline_mode(false, FixtureFamily::Lfm2Moe, WorkerMode::ExpertCache);
}

/// Verifies opaque-session execution with LFM2 cached experts.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_lfm2_moe_pipeline_expert_cache_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Lfm2Moe,
        "pp-ep",
        WorkerMode::ExpertCache,
    );
}

/// Verifies Kimi's KDA and compressed-latent stages against resident prefill
/// and decode while keeping each rank's layer reads bounded.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_kimi_linear_dense_stream_pipeline() {
    run_ring_pipeline(true, FixtureFamily::KimiLinear);
}

/// Exercises the same Kimi stage adapter and heterogeneous cache contract from
/// a real GGUF artifact rather than a SafeTensors directory.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_kimi_linear_gguf_pipeline() {
    run_ring_pipeline(true, FixtureFamily::KimiLinearGguf);
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
fn ring_eight_process_kimi_linear_streamed_triple_axis_expert_cache() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::KimiLinear,
        "tp-pp-ep",
        WorkerMode::ExpertCache,
    );
}

/// Exercises host-layerwise Kimi KDA/MLA state with independently cached
/// routed experts across TP, PP, and EP.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_kimi_linear_layerwise_host_triple_axis_expert_cache() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::KimiLinear,
        "tp-pp-ep",
        WorkerMode::ExpertCache,
    );
}

/// Exercises representative Kimi Linear GGUF recipes, bounded reads, and an
/// independent expert cache across all three axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_kimi_linear_gguf_triple_axis_expert_cache() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::KimiLinearGguf,
        "tp-pp-ep",
        WorkerMode::ExpertCache,
    );
}

/// Covers independent Kimi expert caching for TP+PP with EP inactive.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_kimi_linear_tensor_pipeline_expert_cache_without_ep() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::KimiLinear,
        "tp-pp",
        WorkerMode::ExpertCache,
    );
}

/// Verifies cached-expert schedule failure reaches consensus without leaving
/// Kimi's recurrent or compressed-latent state reusable.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_kimi_linear_expert_cache_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::KimiLinear,
        "pp-ep",
        WorkerMode::ExpertCache,
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
fn ring_eight_process_nemotron_h_moe_streamed_triple_axis_expert_cache() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::NemotronH,
        "tp-pp-ep",
        WorkerMode::ExpertCacheRequantize,
    );
}

/// Exercises host-layerwise non-experts and independently cached
/// Nemotron-H routed experts across all three Cartesian axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_nemotron_h_moe_layerwise_host_triple_axis_expert_cache() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::NemotronH,
        "tp-pp-ep",
        WorkerMode::ExpertCache,
    );
}

/// Exercises canonical Nemotron-H-MoE GGUF bindings, bounded non-expert
/// reads, and independent expert caching across all three axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_nemotron_h_moe_gguf_triple_axis_expert_cache() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::NemotronHGguf,
        "tp-pp-ep",
        WorkerMode::ExpertCache,
    );
}

/// Covers independent Nemotron-H expert caching when PP is active without EP.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_nemotron_h_moe_pipeline_expert_cache() {
    run_ring_pipeline_mode(false, FixtureFamily::NemotronH, WorkerMode::ExpertCache);
}

/// Verifies cached-expert session execution for Nemotron-H stateful stages.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_nemotron_h_moe_pipeline_expert_cache_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::NemotronH,
        "pp-ep",
        WorkerMode::ExpertCache,
    );
}

/// Proves bounded MXFP4 materialization feeds stage-local Nemotron-H expert
/// caches under PP+EP, including persistence and synchronized decode.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_nemotron_h_quantized_pipeline_expert_cache() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::NemotronH,
        "pp-ep",
        WorkerMode::ExpertCacheRequantize,
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

/// Verifies Qwen3-Next TP=2 + PP=2 across recurrent and full-attention stages,
/// including rank-local state, persistence, and synchronized generation.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_qwen3_next_tensor_pipeline() {
    run_ring_cartesian_pipeline(true, FixtureFamily::Qwen3Next, "tp-pp");
}

/// Compares Qwen3-Next TP=2 prefill and decode with the replicated public
/// loader, covering the placed-stage route without a pipeline or expert axis.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic Qwen3-Next fixture"]
fn ring_two_process_qwen3_next_tensor_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3Next,
        "tp",
        WorkerMode::OpaqueSession,
    );
}

/// Covers pure TP loading and execution for the Qwen3.5 hybrid architecture.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic Qwen3.5 fixture"]
fn ring_two_process_qwen35_tensor_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen35,
        "tp",
        WorkerMode::OpaqueSession,
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

/// Covers pure TP loading and text execution for the Qwen3-VL architecture.
#[test]
#[ignore = "requires the MLX Ring backend, two loopback CPU ranks, and the synthetic Qwen3-VL fixture"]
fn ring_two_process_qwen3_vl_tensor_parallel_opaque_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3Vl,
        "tp",
        WorkerMode::OpaqueSession,
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
    run_ring_cartesian_pipeline(false, FixtureFamily::Qwen3VlMoe, "tp-pp-ep");
}

/// Combines bounded Qwen3-VL media/decoder streaming with independent cached
/// routed experts across TP=2 x PP=2 x EP=2.
#[test]
#[ignore = "requires the MLX Ring backend, eight loopback CPU ranks, and the synthetic Qwen3-VL-MoE media fixture"]
fn ring_eight_process_qwen3_vl_moe_streamed_triple_axis_expert_cache() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::Qwen3VlMoe,
        "tp-pp-ep",
        WorkerMode::ExpertCache,
    );
}

/// Combines host-layerwise Qwen3-VL media/decoder units with independently
/// cached routed experts across all three parallel axes.
#[test]
#[ignore = "requires the MLX Ring backend, eight loopback CPU ranks, and the synthetic Qwen3-VL-MoE host-layerwise media fixture"]
fn ring_eight_process_qwen3_vl_moe_layerwise_host_triple_axis_expert_cache() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::Qwen3VlMoe,
        "tp-pp-ep",
        WorkerMode::ExpertCache,
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
        WorkerMode::ExpertCache,
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
        WorkerMode::ExpertCache,
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
fn ring_eight_process_qwen35_moe_streamed_triple_axis_expert_cache() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::Qwen35Moe,
        "tp-pp-ep",
        WorkerMode::ExpertCache,
    );
}

/// Exercises host-backed hybrid non-expert layers with independent expert
/// caching across TP, PP, and EP.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_qwen3_next_moe_layerwise_host_triple_axis_expert_cache() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::Qwen3NextMoe,
        "tp-pp-ep",
        WorkerMode::ExpertCache,
    );
}

/// Covers independent hybrid expert caching when PP is active without EP.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_qwen35_moe_pipeline_expert_cache() {
    run_ring_pipeline_mode(false, FixtureFamily::Qwen35Moe, WorkerMode::ExpertCache);
}

/// Verifies cached Qwen hybrid expert execution through one session.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_qwen35_moe_pipeline_expert_cache_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen35Moe,
        "pp-ep",
        WorkerMode::ExpertCache,
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
fn ring_qwen3_moe_resident_nonexpert_pipeline_expert_cache() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3Moe,
        "pp-ep",
        WorkerMode::ExpertCache,
    );
}

/// Verifies TP-sharded cached experts compose with dense-streamed non-experts
/// and corresponding-coordinate pipeline lanes in an eight-rank topology.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_streamed_tensor_pipeline_expert_cache() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::Qwen3Moe,
        "tp-pp-ep",
        WorkerMode::ExpertCache,
    );
}

/// Exercises host-backed non-expert layers, bounded device windows, and
/// independent expert caching for Qwen3-MoE across all three axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_layerwise_host_tensor_pipeline_expert_cache() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::Qwen3Moe,
        "tp-pp-ep",
        WorkerMode::ExpertCache,
    );
}

/// Proves the same host-layerwise path reads canonical GGUF and preserves
/// stage-local ownership in a PP=2 x EP=2 topology.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_gguf_layerwise_host_pipeline_expert_cache() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::Qwen3MoeGguf,
        "pp-ep",
        WorkerMode::ExpertCache,
    );
}

/// Exercises stage-local cached expert selections and bounded reads from GGUF.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_gguf_pipeline_expert_cache() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::Qwen3MoeGguf,
        "pp-ep",
        WorkerMode::ExpertCache,
    );
}

/// Verifies one session owns both pipeline communication and expert caches.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_pipeline_expert_cache_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3Moe,
        "pp-ep",
        WorkerMode::ExpertCache,
    );
}

/// Verifies PP-only stages cache all of their local layers' experts without
/// constructing an EP communicator. Prefill, decode, prompt persistence, and
/// synchronized generation are exercised by the shared worker.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_pipeline_expert_cache_without_ep() {
    run_ring_pipeline_mode(false, FixtureFamily::Qwen3Moe, WorkerMode::ExpertCache);
}

/// Verifies TP-sharded cached experts and dense-streamed non-experts compose
/// across TP=2 x PP=2 while EP remains inactive.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_streamed_tensor_pipeline_expert_cache_without_ep() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::Qwen3Moe,
        "tp-pp",
        WorkerMode::ExpertCache,
    );
}

/// Exercises PP-only cache ownership and bounded reads from canonical GGUF.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_moe_gguf_pipeline_expert_cache_without_ep() {
    run_ring_pipeline_mode(true, FixtureFamily::Qwen3MoeGguf, WorkerMode::ExpertCache);
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
/// neutral Inkling tensor-parallel composition without pipeline partitioning.
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

/// Exercises pipeline MTP through the same neutral visitor lifecycle used by
/// facade prepared-chat generation on every pipeline rank.
#[test]
#[ignore = "spawns local processes, opens loopback sockets, and initializes MLX; run explicitly"]
fn ring_two_process_inkling_mtp_pipeline_neutral_visitor() {
    run_ring_pipeline_mode(false, FixtureFamily::Inkling, WorkerMode::OpaqueInklingMtp);
}

/// Exercises the neutral Gemma 4 text binder through the public loader and
/// opaque session on a pure two-rank tensor-parallel topology.
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
fn ring_eight_process_inkling_streamed_triple_axis_expert_cache() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::Inkling,
        "tp-pp-ep",
        WorkerMode::ExpertCache,
    );
}

/// Exercises host-layerwise Inkling attention/convolution state with
/// independently cached routed experts across TP, PP, and EP.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_inkling_layerwise_host_triple_axis_expert_cache() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::Inkling,
        "tp-pp-ep",
        WorkerMode::ExpertCache,
    );
}

/// Exercises canonical Inkling GGUF recipes, bounded reads, and independent
/// expert caching across all three axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_inkling_gguf_triple_axis_expert_cache() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::InklingGguf,
        "tp-pp-ep",
        WorkerMode::ExpertCache,
    );
}

/// Proves TP+PP can independently cache every stage-local Inkling expert bank
/// without constructing an EP communicator.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_inkling_tensor_pipeline_expert_cache_without_ep() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::Inkling,
        "tp-pp",
        WorkerMode::ExpertCache,
    );
}

/// Verifies each Inkling stage's expert cache remains session-owned.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_inkling_expert_cache_session() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Inkling,
        "pp-ep",
        WorkerMode::ExpertCache,
    );
}

/// Proves bounded affine materialization feeds stage-local Inkling expert
/// caches under PP+EP, including persistence and synchronized decode.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_four_process_inkling_quantized_pipeline_expert_cache() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::Inkling,
        "pp-ep",
        WorkerMode::ExpertCacheRequantize,
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
    run_ring_cartesian_pipeline(false, FixtureFamily::InklingMultimodal, "tp-pp-ep");
}

/// Proves multimodal ingress composes with streamed non-experts and stage-local
/// independent expert caches across all three Cartesian axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_inkling_multimodal_streamed_triple_axis_expert_cache() {
    run_ring_cartesian_pipeline_mode(
        true,
        FixtureFamily::InklingMultimodal,
        "tp-pp-ep",
        WorkerMode::ExpertCache,
    );
}

/// Proves multimodal ingress composes with host-layerwise non-experts and
/// independent expert caches across all three Cartesian axes.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_eight_process_inkling_multimodal_layerwise_host_triple_axis_expert_cache() {
    run_ring_layerwise_host_cartesian_pipeline_mode(
        FixtureFamily::InklingMultimodal,
        "tp-pp-ep",
        WorkerMode::ExpertCache,
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
            FixtureFamily::Qwen3 if mode == WorkerMode::Requantize => {
                write_qwen_requantized_tp_fixture(checkpoint.path())
            }
            FixtureFamily::Qwen3 => write_qwen_fixture(checkpoint.path(), "qwen3"),
            FixtureFamily::Qwen3Moe => write_qwen_fixture(checkpoint.path(), "qwen3_moe"),
            FixtureFamily::Qwen3MoeTied => {
                write_qwen_fixture_with_tied_head(checkpoint.path(), "qwen3_moe", true)
            }
            FixtureFamily::DeepSeek => write_deepseek_fixture(checkpoint.path(), 2),
            FixtureFamily::DeepSeekV4 => write_deepseek_v4_fixture(checkpoint.path()),
            FixtureFamily::Lfm2 => write_lfm2_pipeline_fixture(checkpoint.path(), false),
            FixtureFamily::Lfm2Moe => write_lfm2_pipeline_fixture(checkpoint.path(), true),
            FixtureFamily::KimiLinear => write_kimi_linear_fixture(checkpoint.path()),
            FixtureFamily::NemotronH
                if matches!(
                    mode,
                    WorkerMode::ExpertCacheRequantize | WorkerMode::Requantize
                ) =>
            {
                write_nemotron_quantizable_fixture(checkpoint.path())
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
            FixtureFamily::Qwen35MoeMultimodal => {
                write_qwen35_multimodal_fixture(checkpoint.path(), true)
            }
            FixtureFamily::Qwen3Vl => write_qwen3_vl_fixture(checkpoint.path(), false),
            FixtureFamily::Qwen3VlMoe => write_qwen3_vl_fixture(checkpoint.path(), true),
            FixtureFamily::Inkling
                if matches!(
                    mode,
                    WorkerMode::ExpertCacheRequantize | WorkerMode::Requantize
                ) =>
            {
                write_inkling_quantizable_fixture(checkpoint.path())
            }
            FixtureFamily::Inkling if mode == WorkerMode::OpaqueInklingMtp => {
                write_inkling_mtp_fixture(checkpoint.path())
            }
            FixtureFamily::Inkling => write_inkling_fixture(checkpoint.path()),
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
    } else if family == FixtureFamily::Qwen3MoeGguf {
        let path = checkpoint.path().join("model.gguf");
        write_qwen3_moe_gguf_fixture(&path);
        path
    } else {
        match family {
            FixtureFamily::Qwen3 if mode == WorkerMode::Requantize => {
                write_qwen_requantized_tp_fixture(checkpoint.path())
            }
            FixtureFamily::Qwen3 => write_qwen_fixture(checkpoint.path(), "qwen3"),
            FixtureFamily::DeepSeek => write_deepseek_fixture(checkpoint.path(), 2),
            FixtureFamily::DeepSeekV4 => write_deepseek_v4_fixture(checkpoint.path()),
            FixtureFamily::Qwen3Moe => write_qwen_fixture(checkpoint.path(), "qwen3_moe"),
            FixtureFamily::GptOss => write_gpt_oss_fixture(checkpoint.path()),
            FixtureFamily::Lfm2Moe => write_lfm2_pipeline_fixture(checkpoint.path(), true),
            FixtureFamily::KimiLinear => write_kimi_linear_fixture(checkpoint.path()),
            FixtureFamily::NemotronH => write_nemotron_fixture(checkpoint.path()),
            FixtureFamily::Inkling => write_inkling_fixture(checkpoint.path()),
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
    ExpertCache,
    ExpertCacheRequantize,
    Requantize,
    OpaqueSession,
    OpaqueInspection,
    OpaqueTextGeneration,
    OpaqueSessionExpertCache,
    OpaqueMuseImage,
    OpaqueInklingMedia,
    OpaqueInklingMtp,
    OpaqueGemma4Media,
    OpaqueGemma4MediaInspection,
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
            FixtureFamily::DeepSeek => write_deepseek_fixture(checkpoint.path(), 2),
            FixtureFamily::DeepSeekV4 => write_deepseek_v4_fixture(checkpoint.path()),
            FixtureFamily::Gemma => write_gemma_fixture(checkpoint.path()),
            FixtureFamily::Qwen2 => write_qwen_fixture(checkpoint.path(), "qwen2"),
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
            FixtureFamily::Qwen35MoeMultimodal => {
                write_qwen35_multimodal_fixture(checkpoint.path(), true)
            }
            FixtureFamily::Qwen3Vl => write_qwen3_vl_fixture(checkpoint.path(), false),
            FixtureFamily::Qwen3VlMoe => write_qwen3_vl_fixture(checkpoint.path(), true),
            FixtureFamily::Inkling if mode == WorkerMode::OpaqueInklingMtp => {
                write_inkling_pipeline_mtp_fixture(checkpoint.path())
            }
            FixtureFamily::Inkling => write_inkling_fixture(checkpoint.path()),
            FixtureFamily::InklingMultimodal => write_inkling_multimodal_fixture(checkpoint.path()),
            FixtureFamily::DeepSeekGguf
            | FixtureFamily::MuseGlimmer
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
            WorkerMode::ExpertCache => {
                command.env(EXPERT_CACHE, "1");
            }
            WorkerMode::ExpertCacheRequantize => {
                command.env(EXPERT_CACHE, "1");
                command.env(REQUANTIZE, "1");
            }
            WorkerMode::Requantize => {
                command.env(REQUANTIZE, "1");
            }
            WorkerMode::OpaqueSession => {
                command.env(OPAQUE_SESSION, "1");
            }
            WorkerMode::OpaqueInspection => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_INSPECTION, "1");
            }
            WorkerMode::OpaqueTextGeneration => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_TEXT_GENERATION, "1");
            }
            WorkerMode::OpaqueSessionExpertCache => {
                command.env(OPAQUE_SESSION, "1");
                command.env(EXPERT_CACHE, "1");
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
            WorkerMode::OpaqueGemma4Media => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_GEMMA4_MEDIA, "1");
            }
            WorkerMode::OpaqueGemma4MediaInspection => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_GEMMA4_MEDIA, "1");
                command.env(OPAQUE_INSPECTION, "1");
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
        if timed_out || statuses.iter().flatten().any(|status| !status.success()) {
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
