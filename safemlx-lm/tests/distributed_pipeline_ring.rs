#![cfg(unix)]

use std::{
    collections::BTreeMap,
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Child, Command, Output, Stdio},
    thread,
    time::{Duration, Instant},
};

use safemlx::{
    distributed::{self, Backend},
    module::ModuleParameters,
    ops::{indexing::TryIndexOp, stack_axis, GgufMetadataValue},
    Array, Device, DeviceType, Dtype as MlxDtype, ExecutionContext, Stream,
};
use safemlx_gguf::{GgmlType, TensorInput, Writer};
use safemlx_lm::{
    architectures::distributed::pipeline::{
        load_pipeline_model_with_options, PipelineLayerCache, PipelineStep,
    },
    architectures::{
        deepseek_v3::model as deepseek_v3,
        gemma4,
        gpt_oss::model as gpt_oss,
        inkling::model as inkling,
        kimi_linear::model as kimi_model,
        lfm2::model as lfm2,
        nemotron_h::model as nemotron_model,
        qwen::{dense as dense_qwen, hybrid::qwen3_5 as qwen_hybrid},
    },
    backend::mlx::ModelLoadOptions,
    core::{residency::OffloadConfig, Backend as _, BackendSession as _},
    load_model,
    runtime::generation::sampler::DefaultSampler,
    runtime::{
        checkpoint::binding::canonical_checkpoint_name,
        checkpoint::quantization::{AffineQuantization, WeightQuantization},
        media::{input::InputPayload, PreparedModelInput},
    },
    CacheResidencyPolicy, DenseDiskStreamLoadOptions, DeviceAssignment, ExpertCacheLoadOptions,
    LayerwiseLoadOptions, MlxBackend, MlxDistributedSession, MlxParallelContext, MtpCapability,
    MtpCheckpointKind, MtpConfig, NonExpertWeightResidency, PagedCacheOptions,
    PromptCacheDescriptor, PromptCacheOptions, PromptCacheTopology, WeightResidency,
};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

const WORKER_RANK: &str = "SAFEMLX_LM_PIPELINE_RING_WORKER";
const CHECKPOINT_DIR: &str = "SAFEMLX_LM_PIPELINE_CHECKPOINT";
const FIXTURE_FAMILY: &str = "SAFEMLX_LM_PIPELINE_FIXTURE_FAMILY";
const DENSE_STREAM: &str = "SAFEMLX_LM_PIPELINE_DENSE_STREAM";
const LAYERWISE_HOST: &str = "SAFEMLX_LM_PIPELINE_LAYERWISE_HOST";
const PROMPT_CACHE_ROOT: &str = "SAFEMLX_LM_PIPELINE_PROMPT_CACHE";
const CARTESIAN_AXES: &str = "SAFEMLX_LM_PIPELINE_CARTESIAN_AXES";
const EXPERT_CACHE: &str = "SAFEMLX_LM_PIPELINE_EXPERT_CACHE";
const REQUANTIZE: &str = "SAFEMLX_LM_PIPELINE_REQUANTIZE";
const OPAQUE_SESSION: &str = "SAFEMLX_LM_PIPELINE_OPAQUE_SESSION";

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum FixtureFamily {
    Llama,
    DeepSeek,
    DeepSeekGguf,
    Gemma,
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
    Inkling,
    InklingMultimodal,
    InklingGguf,
}

impl FixtureFamily {
    const fn name(self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::DeepSeek => "deepseek",
            Self::DeepSeekGguf => "deepseek-gguf",
            Self::Gemma => "gemma",
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
            Self::Inkling => "inkling",
            Self::InklingMultimodal => "inkling-multimodal",
            Self::InklingGguf => "inkling-gguf",
        }
    }

    fn parse(value: &str) -> Self {
        for family in [
            Self::Llama,
            Self::DeepSeek,
            Self::DeepSeekGguf,
            Self::Gemma,
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
            | Self::DeepSeek
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
            Self::Qwen35MoeMultimodal => 2,
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
            Self::DeepSeek | Self::DeepSeekGguf => ("deepseek_v3", "deepseek_v3"),
            Self::Gemma => ("gemma4", "gemma4"),
            Self::Qwen2 => ("dense_qwen", "qwen2"),
            Self::Qwen3 => ("dense_qwen", "qwen3"),
            Self::Qwen3Moe | Self::Qwen3MoeTied | Self::Qwen3MoeGguf => ("dense_qwen", "qwen3_moe"),
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
            Self::Inkling | Self::InklingMultimodal | Self::InklingGguf => {
                ("inkling", "inkling_mm_model")
            }
        }
    }

    const fn layer_prefix(self) -> &'static str {
        match self {
            Self::Gemma => "model.language_model.layers.",
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
            Self::DeepSeek
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
                | Self::Inkling
                | Self::InklingMultimodal
                | Self::InklingGguf
        )
    }

    const fn is_multimodal(self) -> bool {
        matches!(
            self,
            Self::InklingMultimodal | Self::Qwen35Multimodal | Self::Qwen35MoeMultimodal
        )
    }

    const fn has_streamed_media_unit(self) -> bool {
        matches!(self, Self::Qwen35Multimodal | Self::Qwen35MoeMultimodal)
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
    let (tensor_parallel_size, expert_parallel_size) = match cartesian_axes.as_deref() {
        None => (1, 1),
        Some("tp-pp") => (2, 1),
        Some("pp-ep") => (1, 2),
        Some("tp-pp-ep") => (2, 2),
        Some(other) => panic!("unexpected Cartesian pipeline axes {other:?}"),
    };
    let topology = MlxParallelContext::for_group(
        &group,
        tensor_parallel_size,
        2,
        expert_parallel_size,
        DeviceAssignment::new(DeviceType::Cpu, 0),
    )
    .unwrap();
    assert_eq!(topology.global_rank, expected_rank);
    let pipeline_rank = topology.pipeline_parallel_rank;
    let stream = Stream::new_with_device(&topology.device.device().unwrap());
    if std::env::var_os(OPAQUE_SESSION).is_some() {
        let backend = MlxBackend::with_distributed_world(&stream, &stream, &group);
        let model = load_model(
            &backend,
            &checkpoint,
            ModelLoadOptions::with_parallel(topology),
        )
        .unwrap();
        let mut session = backend.create_session(model).unwrap();
        let paged = PagedCacheOptions::new(1, 32768, 32768, 1)
            .unwrap()
            .with_full_attention(true);
        session
            .configure_cache(CacheResidencyPolicy::Paged(paged))
            .unwrap();
        let prompt = Array::from_slice(&[1u32, 2], &[1, 2]);
        let parts = [safemlx_lm::runtime::media::input::InputPart::text_token_ids(&prompt)];
        let mut output = session
            .prefill(
                &backend,
                safemlx_lm::runtime::media::input::ModelInput::new(&parts).into(),
            )
            .unwrap()
            .wait()
            .unwrap();
        for _ in 0..2 {
            assert_eq!(output.logits().is_some(), pipeline_rank == 1);
            let token = session
                .sample_and_synchronize(output.logits(), 1, &mut DefaultSampler, 0.0, None, false)
                .unwrap()
                .token;
            output = session.decode(&backend, token).unwrap().wait().unwrap();
        }
        assert_eq!(output.logits().is_some(), pipeline_rank == 1);
        return;
    }
    let execution = MlxBackend::new(&stream, &stream)
        .communication_for_topology(topology, &group)
        .unwrap();
    let reference = (pipeline_rank == 1
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
    let requested_quantization = if family == FixtureFamily::NemotronH {
        WeightQuantization::MxFp4
    } else {
        AffineQuantization::new(32, 4).unwrap().into()
    };
    let base_options = || {
        if requantize {
            ModelLoadOptions::with_quantization(requested_quantization)
                .with_parallel_topology(topology)
        } else {
            ModelLoadOptions::with_parallel(topology)
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
        load_pipeline_model_with_options(
            &checkpoint,
            base_options().with_weight_residency(WeightResidency::with_expert_cache(
                non_experts,
                ExpertCacheLoadOptions::default(),
            )),
            &stream,
            &stream,
        )
        .unwrap()
    } else if layerwise_host {
        load_pipeline_model_with_options(
            &checkpoint,
            base_options()
                .with_weight_residency(WeightResidency::layerwise_host(layerwise_options())),
            &stream,
            &stream,
        )
        .unwrap()
    } else if dense_stream {
        let dense = DenseDiskStreamLoadOptions::new(u64::MAX, u64::MAX, 1, 1).unwrap();
        load_pipeline_model_with_options(
            &checkpoint,
            base_options().with_weight_residency(WeightResidency::dense_disk_stream(dense)),
            &stream,
            &stream,
        )
        .unwrap()
    } else {
        load_pipeline_model_with_options(&checkpoint, base_options(), &stream, &stream).unwrap()
    };
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
                    ))
                    && expected_range.contains(&layer),
                "rank {expected_rank} opened the wrong SafeTensors layer shard {layer} for {family:?}: {opened:?}"
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
            info.is_last
                && matches!(
                    family,
                    FixtureFamily::Qwen3NextMoe
                        | FixtureFamily::Qwen35Moe
                        | FixtureFamily::Qwen35MoeMultimodal
                ),
        ) * info.embedded_mtp_layers;
        let expected_experts = (family.expert_layer_count(expected_range.clone())
            + predictor_expert_layers)
            * info.local_expert_ids.len();
        assert_eq!(report.is_some(), expected_experts > 0);
        if let Some(report) = report {
            assert_eq!(report.owned_experts, expected_experts);
            assert!(report.owned_bytes > 0);
            assert_eq!(report.device_resident_experts, 0);
            if requantize {
                assert_eq!(report.weight_quantization, Some(requested_quantization));
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
        _ => vec![1, 2],
    };
    let prompt_length = prefix_ids.len() as i32;
    let mut logits = if family.is_multimodal() {
        let prepared = multimodal_prepared_input(family);
        prepared
            .with_model_input(|input| {
                model.prefill_distributed(
                    model.stage_info().is_first.then_some(input),
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
        let prompt = safemlx::Array::from_slice(&prefix_ids, &[1, prompt_length]);
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
        let tolerance = if family.is_multimodal() { 5e-4 } else { 1e-4 };
        assert_final_logits_close(actual, expected, tolerance);
    }
    assert_family_cache(family, pipeline_rank, &cache, prompt_length);
    let (model_family, effective_model_type) = family.descriptor_names();
    let descriptor = PromptCacheDescriptor {
        model_family: model_family.into(),
        effective_model_type: effective_model_type.into(),
        checkpoint_fingerprint: "pipeline-ring-fixture".into(),
        prefix_content_fingerprint: format!("tokens:{prefix_ids:?}"),
        architecture_fingerprint: model.prompt_cache_architecture_fingerprint().unwrap(),
        layer_count: family.layer_count(),
        global_layer_start: family.stage_range(pipeline_rank).start,
        global_layer_end: family.stage_range(pipeline_rank).end,
        batch_size: 1,
        layer_prefix_offsets: vec![0; family.stage_range(pipeline_rank).len()],
        layer_layout: model.prompt_cache_layer_layout().unwrap(),
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
    let token = safemlx::Array::from_slice(&[0u32], &[1, 1]);
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
        let tolerance = if family.is_multimodal() { 5e-4 } else { 1e-4 };
        assert_final_logits_close(actual, expected, tolerance);
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
            assert_eq!(uninterrupted, restored.as_slice::<f32>());
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
        FixtureFamily::Qwen3Next
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
        let prompt = Array::from_slice(&[1u32, 2], &[1, 2]);
        let parts = [safemlx_lm::runtime::media::input::InputPart::text_token_ids(&prompt)];
        let mut mtp_cache = model.new_cache().unwrap();
        let (generated, stats) = model
            .generate_embedded_mtp_distributed(
                &mut mtp_cache,
                safemlx_lm::runtime::media::input::ModelInput::new(&parts),
                &MtpConfig {
                    max_tokens: 3,
                    max_draft_tokens: 1,
                    temperature: 0.0,
                    eos_token_ids: Vec::new(),
                },
                None,
                &mut DefaultSampler,
                &execution,
            )
            .unwrap();
        assert_eq!(generated.len(), 3);
        assert_eq!(stats.emitted_tokens, 3);
        assert!(stats.draft_tokens > 0);
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
        .map(ModelLoadOptions::with_quantization)
        .unwrap_or_default();
    let backend = MlxBackend::new(stream, stream);
    let mut model = safemlx_lm::load_model(&backend, checkpoint, options)
        .unwrap()
        .into_inner()
        .into_complete()
        .unwrap();
    let mut cache = model.new_cache();
    let prompt = Array::from_slice(&[1u32, 2], &[1, 2]);
    let parts = [safemlx_lm::runtime::media::input::InputPart::text_token_ids(&prompt)];
    let prefill = model
        .submit_prefill(
            safemlx_lm::runtime::media::input::ModelInput::new(&parts),
            &mut cache,
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
        .submit_decode(token, &mut cache, stream)
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
    use safemlx_lm::runtime::media::input::{InputMetadata, InputPart, Modality, ModelInput};

    let text = Array::from_slice(&[1u32, 2], &[1, 2]);
    let image = Array::from_slice(&[0.01f32; 16], &[1, 1, 16]);
    let audio = Array::from_slice(&[0u32, 1, 2, 3, 4, 5], &[3, 2]);
    let audio_mask = Array::from_slice(&[true, true, false], &[1, 3]);
    let parts = [
        InputPart::text_token_ids(&text),
        InputPart {
            modality: Modality::Image,
            payload: InputPayload::Embeddings(&image),
            metadata: InputMetadata::empty(),
        },
        InputPart::audio_tensor(&audio, InputMetadata::audio_mask(&audio_mask)),
    ];
    PreparedModelInput::from_model_input(ModelInput::new(&parts)).unwrap()
}

fn qwen35_multimodal_prepared_input() -> PreparedModelInput {
    use safemlx_lm::runtime::media::input::{InputMetadata, InputPart, ModelInput};

    let text = Array::from_slice(&[1u32, 2], &[1, 2]);
    let grid = Array::from_slice(&[1i32, 2, 4], &[1, 3]);
    let pixels = Array::from_slice(&[0.01f32; 96], &[8, 12]);
    let parts = [
        InputPart::text_token_ids(&text),
        InputPart::image_tensor(&pixels, InputMetadata::qwen_grid_thw(&grid)),
    ];
    PreparedModelInput::from_model_input(ModelInput::new(&parts)).unwrap()
}

fn multimodal_prepared_input(family: FixtureFamily) -> PreparedModelInput {
    match family {
        FixtureFamily::InklingMultimodal => inkling_multimodal_prepared_input(),
        FixtureFamily::Qwen35Multimodal | FixtureFamily::Qwen35MoeMultimodal => {
            qwen35_multimodal_prepared_input()
        }
        _ => panic!("{family:?} is not a multimodal fixture"),
    }
}

fn multimodal_resident_reference(
    family: FixtureFamily,
    checkpoint: &Path,
    stream: &Stream,
) -> (Vec<f32>, Vec<f32>) {
    let backend = MlxBackend::new(stream, stream);
    let mut model = safemlx_lm::load_model(&backend, checkpoint, ModelLoadOptions::default())
        .unwrap()
        .into_inner()
        .into_complete()
        .unwrap();
    let mut cache = model.new_cache();
    let prepared = multimodal_prepared_input(family);
    let parts = prepared.input_parts();
    let prefill = model
        .submit_prefill(
            safemlx_lm::runtime::media::input::ModelInput::new(&parts),
            &mut cache,
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
        .submit_decode(token, &mut cache, stream)
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
    model: &mut safemlx_lm::architectures::distributed::pipeline::PipelineModel,
    tokens: Option<&Array>,
    step: PipelineStep,
    cache: &mut safemlx_lm::architectures::distributed::pipeline::PipelineCache,
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
    cache: &safemlx_lm::architectures::distributed::pipeline::PipelineCache,
    expected_offset: i32,
) {
    let populated = expected_offset > 0;
    let assert_slots =
        |slots: &[safemlx_lm::architectures::distributed::pipeline::PipelineStateSlot], count| {
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
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "model_type": "llama",
            "hidden_size": 4,
            "num_hidden_layers": 2,
            "intermediate_size": 8,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 4,
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
    let args = deepseek_v3::model_args_from_config_value(&config).unwrap();
    let mut model = deepseek_v3::Model::new(args.clone(), stream).unwrap();
    for layer in &mut model.model.layers {
        let deepseek_v3::FeedForward::Moe(moe) = &mut layer.mlp else {
            continue;
        };
        moe.experts.gate_proj = safemlx::module::Param::new(Some(
            Array::full::<f32>(
                &[
                    args.n_routed_experts,
                    args.moe_intermediate_size,
                    args.hidden_size,
                ],
                Array::from_f32(0.01),
                stream,
            )
            .unwrap(),
        ));
        moe.experts.up_proj = safemlx::module::Param::new(Some(
            Array::full::<f32>(
                &[
                    args.n_routed_experts,
                    args.moe_intermediate_size,
                    args.hidden_size,
                ],
                Array::from_f32(0.01),
                stream,
            )
            .unwrap(),
        ));
        moe.experts.down_proj = safemlx::module::Param::new(Some(
            Array::full::<f32>(
                &[
                    args.n_routed_experts,
                    args.hidden_size,
                    args.moe_intermediate_size,
                ],
                Array::from_f32(0.01),
                stream,
            )
            .unwrap(),
        ));
    }
    for (name, parameter) in model.parameters_mut().flatten() {
        let shape = parameter.shape().to_vec();
        *parameter = if name.ends_with("norm.weight") {
            Array::ones::<f32>(&shape, stream).unwrap()
        } else {
            Array::full::<f32>(&shape, Array::from_f32(0.01), stream).unwrap()
        };
    }
    let mut arrays = Vec::new();
    for (name, value) in model.parameters().flatten() {
        let projection = ["gate_proj", "up_proj", "down_proj"]
            .into_iter()
            .find(|projection| name.ends_with(&format!(".mlp.experts.{projection}")));
        if let Some(projection) = projection {
            let prefix = name.strip_suffix(&format!(".{projection}")).unwrap();
            for expert in 0..args.n_routed_experts {
                arrays.push((
                    format!("{prefix}.{expert}.{projection}.weight"),
                    value.try_index_device(expert, stream).unwrap(),
                ));
            }
        } else {
            arrays.push((canonical_checkpoint_name(&name), value.clone()));
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
    let mut args = gemma4::model::model_args_from_config_value(&config["text_config"]).unwrap();
    args.model_type = "gemma4".into();
    args.tie_word_embeddings = true;
    let mut model = gemma4::model::Model::new(args, stream).unwrap();
    for (name, parameter) in model.parameters_mut().flatten() {
        let shape = parameter.shape().to_vec();
        *parameter = if name.ends_with("norm.weight") {
            Array::ones::<f32>(&shape, stream).unwrap()
        } else {
            Array::full::<f32>(&shape, Array::from_f32(0.01), stream).unwrap()
        };
    }
    let arrays = model
        .parameters()
        .flatten()
        .into_iter()
        .map(|(name, value)| (canonical_checkpoint_name(&name), value.clone()))
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
            _ => panic!("unsupported dense-Qwen pipeline fixture model type {model_type}"),
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

fn write_qwen_config_fixture(directory: &Path, config: serde_json::Value) {
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let args = dense_qwen::load_config(directory).unwrap();
    let mut model = dense_qwen::Model::new(args, stream).unwrap();
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
        .map(|(name, value)| (canonical_checkpoint_name(&name), value.clone()))
        .collect::<Vec<_>>();
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
    let args = dense_qwen::load_config(path.parent().unwrap()).unwrap();
    let mut model = dense_qwen::Model::new(args.clone(), stream).unwrap();
    initialize_fixture(&mut model, stream);
    let mut specs = Vec::new();
    for (runtime_name, value) in model.parameters().flatten() {
        let runtime_name = canonical_checkpoint_name(&runtime_name);
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
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let args = gpt_oss::model_args_from_config_value(&config).unwrap();
    let mut model = gpt_oss::Model::new(args, stream).unwrap();
    for (name, parameter) in model.parameters_mut().flatten() {
        let shape = parameter.shape().to_vec();
        *parameter = if name.ends_with("_scales") {
            Array::full::<u8>(&shape, Array::from_slice(&[127u8], &[]), stream).unwrap()
        } else if name.ends_with("_blocks") {
            Array::full::<u8>(&shape, Array::from_slice(&[0x11u8], &[]), stream).unwrap()
        } else if name.ends_with("norm.weight") {
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
        .map(|(name, value)| (canonical_checkpoint_name(&name), value.clone()))
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
    let args = lfm2::model_args_from_config_value(&config).unwrap();
    let mut model = lfm2::Model::new(args, stream).unwrap();
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
        .map(|(name, value)| (canonical_checkpoint_name(&name), value.clone()))
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
    let args = lfm2::model_args_from_config_value(&config).unwrap();
    let mut model = lfm2::Model::new(args.clone(), stream).unwrap();
    initialize_fixture(&mut model, stream);
    let mut specs = Vec::new();
    for (runtime_name, value) in model.parameters().flatten() {
        let runtime_name = canonical_checkpoint_name(&runtime_name);
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
            GgufMetadataValue::Array(safemlx::ops::GgufMetadataArray::Uint32(vec![0, 3])),
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
        .map(|(name, value)| (canonical_checkpoint_name(&name), value.clone()))
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
    let args = kimi_model::model_args_from_config_value(&config).unwrap();
    let mut model = kimi_model::Model::new(args, stream).unwrap();
    initialize_fixture(&mut model, stream);
    let mut arrays = Vec::<(String, Array)>::new();
    for (name, value) in model.parameters().flatten() {
        if name.as_ref() == "model.layers.1.mlp.experts.gate_up_proj" {
            for expert in 0..model.args.num_experts {
                arrays.push((
                    format!("model.layers.1.block_sparse_moe.experts.{expert}.w1.weight"),
                    value
                        .try_index_device((expert, ..model.args.moe_intermediate_size, ..), stream)
                        .unwrap(),
                ));
                arrays.push((
                    format!("model.layers.1.block_sparse_moe.experts.{expert}.w3.weight"),
                    value
                        .try_index_device((expert, model.args.moe_intermediate_size.., ..), stream)
                        .unwrap(),
                ));
            }
            continue;
        }
        if name.as_ref() == "model.layers.1.mlp.experts.down_proj" {
            for expert in 0..model.args.num_experts {
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
                        model.args.kda_config.num_heads * model.args.kda_config.head_dim,
                        model.args.kda_config.short_conv_kernel_size,
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

fn nemotron_public_name(runtime: &str, args: &nemotron_model::ModelArgs) -> String {
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
            nemotron_model::LayerPolicy::Mamba => "mamba",
            nemotron_model::LayerPolicy::SelfAttention(_) => "attention",
            nemotron_model::LayerPolicy::DenseMlp => "mlp",
            nemotron_model::LayerPolicy::SparseMoe => "moe",
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
    let args = nemotron_model::model_args_from_config_value(&config).unwrap();
    let mut model = nemotron_model::Model::new(args, stream).unwrap();
    initialize_fixture(&mut model, stream);
    let mut arrays = Vec::<(String, Array)>::new();
    for (name, value) in model.parameters().flatten() {
        let runtime = canonical_checkpoint_name(&name);
        if runtime.ends_with("moe.experts.up_proj") {
            let prefix = nemotron_public_name(runtime.trim_end_matches(".up_proj"), &model.args);
            for expert in 0..model.args.n_routed_experts {
                arrays.push((
                    format!("{prefix}.{expert}.up_proj.weight"),
                    value.try_index_device((expert, .., ..), stream).unwrap(),
                ));
            }
        } else if runtime.ends_with("moe.experts.down_proj") {
            let prefix = nemotron_public_name(runtime.trim_end_matches(".down_proj"), &model.args);
            for expert in 0..model.args.n_routed_experts {
                arrays.push((
                    format!("{prefix}.{expert}.down_proj.weight"),
                    value.try_index_device((expert, .., ..), stream).unwrap(),
                ));
            }
        } else {
            arrays.push((nemotron_public_name(&runtime, &model.args), value.clone()));
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
    let args = nemotron_model::model_args_from_config_value(&config).unwrap();
    let mut model = nemotron_model::Model::new(args, stream).unwrap();
    initialize_fixture(&mut model, stream);
    let mut specs = Vec::new();
    for (runtime_name, value) in model.parameters().flatten() {
        let runtime_name = canonical_checkpoint_name(&runtime_name);
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
            GgufMetadataValue::Array(safemlx::ops::GgufMetadataArray::Uint32(vec![0, 17, 17, 0])),
        ),
        (
            key("attention.head_count_kv"),
            GgufMetadataValue::Array(safemlx::ops::GgufMetadataArray::Uint32(vec![0, 0, 0, 3])),
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
    let args = qwen_hybrid::model_args_from_config_value(&config).unwrap();
    let mut model = qwen_hybrid::Model::new(args, None, None, None, stream).unwrap();
    initialize_fixture(&mut model, stream);
    save_parameter_fixture(directory, &config, &model);
}

fn write_qwen_hybrid_moe_fixture(directory: &Path, model_type: &str) {
    let config = qwen_hybrid_moe_config(model_type);
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let args = qwen_hybrid::model_args_from_config_value(&config).unwrap();
    let mut model = qwen_hybrid::Model::new(args, None, None, None, stream).unwrap();
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
    let (args, image_token_id, video_token_id, vision) =
        qwen_hybrid::get_qwen3_5_model_args(directory).unwrap();
    let mut model =
        qwen_hybrid::Model::new(args, image_token_id, video_token_id, vision, stream).unwrap();
    initialize_fixture(&mut model, stream);
    save_parameter_fixture(directory, &config, &model);
}

fn inkling_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "inkling_mm_model",
        "eos_token_id": 1,
        "text_config": {
            "torch_dtype": "float32",
            "hidden_size": 16,
            "num_hidden_layers": 3,
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

fn write_inkling_quantizable_fixture(directory: &Path) {
    write_inkling_fixture_with_config(directory, inkling_quantizable_config());
}

fn write_inkling_multimodal_fixture(directory: &Path) {
    write_inkling_fixture_with_config(directory, inkling_multimodal_config());
}

fn write_inkling_fixture_with_config(directory: &Path, config: serde_json::Value) {
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let args = inkling::model_args_from_config_value(&config).unwrap();
    let mut model = inkling::Model::new(args, stream).unwrap();
    initialize_fixture(&mut model, stream);
    let parameters = model.parameters().flatten();
    let mut arrays = Vec::<(String, Array)>::new();
    for (name, value) in &parameters {
        let name = name.as_ref();
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
            let intermediate = model.args.text_config.moe_intermediate_size.unwrap();
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
            let intermediate = model.args.text_config.moe_intermediate_size.unwrap();
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
    use safemlx::ops::GgufMetadataArray;
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
            GgufMetadataValue::Array(GgufMetadataArray::Uint32(vec![2, 2, 2])),
        ),
        (
            "inkling.attention.sliding_window_pattern".into(),
            GgufMetadataValue::Array(GgufMetadataArray::Bool(vec![false, true, false])),
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
    let args = inkling::model_args_from_config_value(&config).unwrap();
    let mut model = inkling::Model::new(args, stream).unwrap();
    initialize_fixture(&mut model, stream);
    let parameters = model.parameters().flatten();
    let mut specs = Vec::new();
    for (runtime, value) in &parameters {
        let runtime = runtime.as_ref();
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
            GgufMetadataValue::Uint32(17),
        ),
        (
            "deepseek2.attention.head_count".into(),
            GgufMetadataValue::Uint32(3),
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
            GgufMetadataValue::Uint32(5),
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
    let tensor = |name: &str, mlx_shape: &[u64], phase: usize| {
        let mut dimensions = mlx_shape.to_vec();
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
            tensor(&format!("blk.{layer}.attn_q_b.weight"), &[12, 4], phase + 1),
            tensor(
                &format!("blk.{layer}.attn_kv_a_mqa.weight"),
                &[6, 12],
                phase + 2,
            ),
            norm(&format!("blk.{layer}.attn_kv_a_norm.weight"), 4),
            tensor(
                &format!("blk.{layer}.attn_k_b.weight"),
                &[3, 4, 2],
                phase + 3,
            ),
            tensor(
                &format!("blk.{layer}.attn_v_b.weight"),
                &[3, 2, 4],
                phase + 4,
            ),
            tensor(
                &format!("blk.{layer}.attn_output.weight"),
                &[12, 6],
                phase + 5,
            ),
        ]);
    }
    tensors.extend([
        tensor("blk.0.ffn_gate.weight", &[17, 12], 21),
        tensor("blk.0.ffn_up.weight", &[17, 12], 22),
        tensor("blk.0.ffn_down.weight", &[12, 17], 23),
        tensor("blk.1.ffn_gate_inp.weight", &[4, 12], 24),
        f32_gguf_tensor(
            "blk.1.exp_probs_b.bias",
            vec![4],
            patterned_values(4, 0.001, 25),
        ),
        tensor("blk.1.ffn_gate_shexp.weight", &[5, 12], 26),
        tensor("blk.1.ffn_up_shexp.weight", &[5, 12], 27),
        tensor("blk.1.ffn_down_shexp.weight", &[12, 5], 28),
        tensor("blk.1.ffn_gate_exps.weight", &[4, 5, 12], 29),
        tensor("blk.1.ffn_up_exps.weight", &[4, 5, 12], 30),
        tensor("blk.1.ffn_down_exps.weight", &[4, 12, 5], 31),
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
            GgufMetadataValue::Array(safemlx::ops::GgufMetadataArray::Uint32(vec![0, 1])),
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
    let tensor = |name: &str, mlx_shape: &[u64], phase: usize| {
        let mut dimensions = mlx_shape.to_vec();
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
/// `cargo test -p safemlx-lm --test distributed_pipeline_ring ring_two_process_pipeline -- --ignored --exact --nocapture`
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_pipeline() {
    run_ring_pipeline(false, FixtureFamily::Llama);
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
/// `cargo test -p safemlx-lm --test distributed_pipeline_ring ring_two_process_dense_stream_pipeline -- --ignored --exact --nocapture`
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
/// execution for a dense Qwen2 decoder.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_qwen2_pipeline() {
    run_ring_pipeline(false, FixtureFamily::Qwen2);
}

/// Verifies Q/K-normalized dense Qwen3 execution through streamed local layers.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_qwen3_dense_stream_pipeline() {
    run_ring_pipeline(true, FixtureFamily::Qwen3);
}

/// Verifies Qwen3 routed-expert ownership, paged cache persistence, and
/// rank-synchronized two-stage execution through the shared dense-Qwen stage.
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
        WorkerMode::ExpertCache,
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
            FixtureFamily::Inkling
                if matches!(
                    mode,
                    WorkerMode::ExpertCacheRequantize | WorkerMode::Requantize
                ) =>
            {
                write_inkling_quantizable_fixture(checkpoint.path())
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
            FixtureFamily::DeepSeek => write_deepseek_fixture(checkpoint.path(), 2),
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
            FixtureFamily::Inkling => write_inkling_fixture(checkpoint.path()),
            FixtureFamily::InklingMultimodal => write_inkling_multimodal_fixture(checkpoint.path()),
            FixtureFamily::DeepSeekGguf
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
                "distributed_pipeline_ring::pipeline_ring_worker",
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
