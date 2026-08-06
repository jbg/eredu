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
        load_pipeline_model, load_pipeline_model_with_options, PipelineInferencePhase,
        PipelineInferenceScheduler, PipelineLayerCache, PipelineMicrobatchInput, PipelineStep,
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
    runtime::generation::sampler::DefaultSampler,
    runtime::{
        checkpoint::binding::canonical_checkpoint_name,
        scheduler::{RequestId, RequestStatus, SchedulerLimits},
    },
    CacheResidencyPolicy, DenseDiskStreamLoadOptions, DeviceAssignment, ModelLoadOptions,
    PagedCacheOptions, ParallelTopology, PromptCacheDescriptor, PromptCacheOptions,
    PromptCacheTopology, WeightResidency,
};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

const WORKER_RANK: &str = "SAFEMLX_LM_PIPELINE_RING_WORKER";
const CHECKPOINT_DIR: &str = "SAFEMLX_LM_PIPELINE_CHECKPOINT";
const FIXTURE_FAMILY: &str = "SAFEMLX_LM_PIPELINE_FIXTURE_FAMILY";
const DENSE_STREAM: &str = "SAFEMLX_LM_PIPELINE_DENSE_STREAM";
const PROMPT_CACHE_ROOT: &str = "SAFEMLX_LM_PIPELINE_PROMPT_CACHE";
const MICROBATCH: &str = "SAFEMLX_LM_PIPELINE_MICROBATCH";
const SCHEDULE_MISMATCH: &str = "SAFEMLX_LM_PIPELINE_SCHEDULE_MISMATCH";

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum FixtureFamily {
    Llama,
    DeepSeek,
    Gemma,
    Qwen2,
    Qwen3,
    Qwen3Moe,
    GptOss,
    Lfm2,
    Lfm2Moe,
    KimiLinear,
    KimiLinearGguf,
    NemotronH,
    Qwen3Next,
    Qwen35,
    Inkling,
}

impl FixtureFamily {
    const fn name(self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::DeepSeek => "deepseek",
            Self::Gemma => "gemma",
            Self::Qwen2 => "qwen2",
            Self::Qwen3 => "qwen3",
            Self::Qwen3Moe => "qwen3-moe",
            Self::GptOss => "gpt-oss",
            Self::Lfm2 => "lfm2",
            Self::Lfm2Moe => "lfm2-moe",
            Self::KimiLinear => "kimi-linear",
            Self::KimiLinearGguf => "kimi-linear-gguf",
            Self::NemotronH => "nemotron-h",
            Self::Qwen3Next => "qwen3-next",
            Self::Qwen35 => "qwen3.5",
            Self::Inkling => "inkling",
        }
    }

    fn parse(value: &str) -> Self {
        for family in [
            Self::Llama,
            Self::DeepSeek,
            Self::Gemma,
            Self::Qwen2,
            Self::Qwen3,
            Self::Qwen3Moe,
            Self::GptOss,
            Self::Lfm2,
            Self::Lfm2Moe,
            Self::KimiLinear,
            Self::KimiLinearGguf,
            Self::NemotronH,
            Self::Qwen3Next,
            Self::Qwen35,
            Self::Inkling,
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
            | Self::Qwen2
            | Self::Qwen3
            | Self::Qwen3Moe
            | Self::GptOss
            | Self::Lfm2
            | Self::Lfm2Moe
            | Self::KimiLinear
            | Self::KimiLinearGguf
            | Self::Qwen3Next
            | Self::Qwen35 => 2,
            Self::Gemma | Self::NemotronH => 4,
            Self::Inkling => 3,
        }
    }

    fn stage_range(self, rank: usize) -> std::ops::Range<usize> {
        match (self, rank) {
            (Self::Gemma, 0) => 0..1,
            (Self::Gemma, 1) => 1..4,
            (Self::NemotronH, 0) => 0..2,
            (Self::NemotronH, 1) => 2..4,
            (Self::Inkling, 0) => 0..2,
            (Self::Inkling, 1) => 2..3,
            (_, rank) => rank..rank + 1,
        }
    }

    fn descriptor_names(self) -> (&'static str, &'static str) {
        match self {
            Self::Llama => ("llama", "llama"),
            Self::DeepSeek => ("deepseek_v3", "deepseek_v3"),
            Self::Gemma => ("gemma4", "gemma4"),
            Self::Qwen2 => ("dense_qwen", "qwen2"),
            Self::Qwen3 => ("dense_qwen", "qwen3"),
            Self::Qwen3Moe => ("dense_qwen", "qwen3_moe"),
            Self::GptOss => ("gpt_oss", "gpt_oss"),
            Self::Lfm2 => ("lfm2", "lfm2"),
            Self::Lfm2Moe => ("lfm2", "lfm2_moe"),
            Self::KimiLinear | Self::KimiLinearGguf => ("kimi_linear", "kimi_linear"),
            Self::NemotronH => ("nemotron_h", "nemotron_h"),
            Self::Qwen3Next => ("qwen_hybrid", "qwen3_next"),
            Self::Qwen35 => ("qwen_hybrid", "qwen3_5_text"),
            Self::Inkling => ("inkling", "inkling_mm_model"),
        }
    }

    const fn layer_prefix(self) -> &'static str {
        match self {
            Self::Gemma => "model.language_model.layers.",
            Self::NemotronH => "backbone.layers.",
            Self::Inkling => "model.llm.layers.",
            _ => "model.layers.",
        }
    }

    const fn has_gguf_source(self) -> bool {
        matches!(self, Self::KimiLinearGguf)
    }

    const fn needs_resident_reference(self) -> bool {
        matches!(
            self,
            Self::KimiLinear
                | Self::KimiLinearGguf
                | Self::NemotronH
                | Self::Qwen3Next
                | Self::Qwen35
                | Self::Inkling
        )
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
    let topology =
        ParallelTopology::from_group(&group, 1, 2, 1, DeviceAssignment::new(DeviceType::Cpu, 0))
            .unwrap();
    assert_eq!(topology.global_rank, expected_rank);
    let stream = Stream::new_with_device(&topology.device.device().unwrap());
    let reference = (expected_rank == 1 && family.needs_resident_reference())
        .then(|| resident_reference(&checkpoint, &stream));
    let dense_stream = std::env::var_os(DENSE_STREAM).is_some();
    let mut model = if dense_stream {
        let dense = DenseDiskStreamLoadOptions::new(u64::MAX, u64::MAX, 1, 1, 1).unwrap();
        load_pipeline_model_with_options(
            &checkpoint,
            ModelLoadOptions::with_parallel(topology)
                .with_weight_residency(WeightResidency::DenseDiskStream(dense)),
            &stream,
            &stream,
        )
        .unwrap()
    } else {
        load_pipeline_model(&checkpoint, topology, &stream, &stream).unwrap()
    };
    let info = model.stage_info();
    let expected_range = family.stage_range(expected_rank);
    assert_eq!(info.global_layer_range, expected_range);
    if !family.has_gguf_source() {
        let prefix = family.layer_prefix();
        assert_eq!(
            info.owned_tensors.iter().any(|name| expected_range
                .clone()
                .any(|layer| name.starts_with(&format!("{prefix}{layer}.")))),
            !dense_stream
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
    if family.needs_resident_reference() && !family.has_gguf_source() {
        for layer in 0..family.layer_count() {
            assert_eq!(
                opened.contains(&format!("layer-{layer}.safetensors")),
                !dense_stream && expected_range.contains(&layer),
                "rank {expected_rank} opened the wrong SafeTensors layer shard for {family:?}"
            );
        }
    }
    if dense_stream {
        let report = model.dense_stream_report().unwrap().unwrap();
        assert_eq!(report.planned_layer_count(), expected_range.len());
        assert!(report
            .residency()
            .units()
            .iter()
            .all(|unit| !unit.host_resident() && !unit.device_resident()));
        if family.has_gguf_source() {
            let diagnostics = model.checkpoint_diagnostics().unwrap().unwrap();
            let global_payload = kimi_linear_gguf_payload_bytes();
            assert!(diagnostics.physical_reads > 0);
            assert!(
                diagnostics.physical_read_bytes < global_payload,
                "rank {expected_rank} read {} GGUF bytes while loading static modules from a {global_payload}-byte global tensor payload",
                diagnostics.physical_read_bytes
            );
        }
    }
    if family == FixtureFamily::Llama {
        assert_eq!(
            opened.contains(&"output.safetensors".into()),
            expected_rank == 1
        );
    }

    if std::env::var_os(MICROBATCH).is_some() {
        run_microbatch_worker(&mut model, expected_rank, &group, &stream);
        return;
    }
    if std::env::var_os(SCHEDULE_MISMATCH).is_some() {
        run_schedule_mismatch_worker(&mut model, expected_rank, &group, &stream);
        return;
    }

    let paged = PagedCacheOptions::new(1, 4096, 4096, 1)
        .unwrap()
        .with_full_attention(true);
    let mut cache = model
        .new_cache_with_options(CacheResidencyPolicy::Paged(paged.clone()))
        .unwrap();
    assert_eq!(
        cache.global_layers(),
        family.stage_range(expected_rank).collect::<Vec<_>>()
    );
    assert_family_cache(family, expected_rank, &cache, false);
    let prompt = safemlx::Array::from_slice(&[1u32, 2], &[1, 2]);
    let mut logits = model
        .forward_pipeline(
            (expected_rank == 0).then_some(&prompt),
            PipelineStep::new(1, 2).unwrap(),
            None,
            &mut cache,
            &group,
            &stream,
        )
        .unwrap();
    assert_eq!(logits.is_some(), expected_rank == 1);
    if let (Some(actual), Some((expected, _))) = (&logits, &reference) {
        assert_final_logits_close(actual, expected, 1e-4);
    }
    assert_family_cache(family, expected_rank, &cache, true);
    let (model_family, effective_model_type) = family.descriptor_names();
    let descriptor = PromptCacheDescriptor {
        model_family: model_family.into(),
        effective_model_type: effective_model_type.into(),
        checkpoint_fingerprint: "pipeline-ring-fixture".into(),
        prefix_content_fingerprint: "tokens:1,2".into(),
        architecture_fingerprint: model.prompt_cache_architecture_fingerprint().unwrap(),
        layer_count: family.layer_count(),
        global_layer_start: family.stage_range(expected_rank).start,
        global_layer_end: family.stage_range(expected_rank).end,
        batch_size: 1,
        layer_layout: model.prompt_cache_layer_layout().unwrap(),
        sink_tokens: 0,
        topology: PromptCacheTopology {
            pipeline: Some((2, expected_rank)),
            tensor_parallel: None,
            expert_parallel: None,
            expert_parallel_cache_replicated: true,
        },
    };
    model
        .save_prompt_cache(
            &mut cache,
            &prompt_cache_root,
            descriptor.clone(),
            &[1, 2],
            &PromptCacheOptions::default(),
        )
        .unwrap();
    let token = safemlx::Array::from_slice(&[0u32], &[1, 1]);
    let uninterrupted = model
        .forward_pipeline(
            (expected_rank == 0).then_some(&token),
            PipelineStep::new(1, 1).unwrap(),
            None,
            &mut cache,
            &group,
            &stream,
        )
        .unwrap();
    let uninterrupted_values = uninterrupted.as_ref().map(|value| {
        let value = value.evaluated().unwrap();
        value.as_slice::<f32>().to_vec()
    });
    if let (Some(actual), Some((_, expected))) = (&uninterrupted, &reference) {
        assert_final_logits_close(actual, expected, 1e-4);
    }
    let (mut cache, manifest) = model
        .load_prompt_cache(&prompt_cache_root, &descriptor, &[1, 2], paged, &stream)
        .unwrap();
    assert_eq!(manifest.topology, descriptor.topology);
    let restored = model
        .forward_pipeline(
            (expected_rank == 0).then_some(&token),
            PipelineStep::new(1, 1).unwrap(),
            None,
            &mut cache,
            &group,
            &stream,
        )
        .unwrap();
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
                &group,
                &stream,
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
        logits = model
            .forward_pipeline(
                (expected_rank == 0).then_some(&synchronized.token),
                PipelineStep::new(1, 1).unwrap(),
                None,
                &mut cache,
                &group,
                &stream,
            )
            .unwrap();
    }
    if dense_stream {
        let report = model.dense_stream_report().unwrap().unwrap();
        assert!(report.prefill_forwards() >= 1);
        assert!(report.decode_forwards() >= 2);
    }
}

fn resident_reference(checkpoint: &Path, stream: &Stream) -> (Vec<f32>, Vec<f32>) {
    let mut model = safemlx_lm::api::load_model(checkpoint, stream, stream).unwrap();
    let mut cache = model.new_cache();
    let prompt = Array::from_slice(&[1u32, 2], &[1, 2]);
    let parts = [safemlx_lm::runtime::media::input::InputPart::text_token_ids(&prompt)];
    let prefill = model
        .prefill_input_with_cache(
            safemlx_lm::runtime::media::input::ModelInput::new(&parts),
            &mut cache,
            stream,
        )
        .unwrap()
        .evaluated()
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    let token = Array::from_slice(&[0u32], &[1, 1]);
    let parts = [safemlx_lm::runtime::media::input::InputPart::text_token_ids(&token)];
    let decode = model
        .prefill_input_with_cache(
            safemlx_lm::runtime::media::input::ModelInput::new(&parts),
            &mut cache,
            stream,
        )
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

fn assert_family_cache(
    family: FixtureFamily,
    rank: usize,
    cache: &safemlx_lm::architectures::distributed::pipeline::PipelineCache,
    populated: bool,
) {
    let assert_slots =
        |slots: &[safemlx_lm::architectures::distributed::pipeline::PipelineStateSlot], count| {
            assert_eq!(slots.len(), count);
            for slot in slots {
                assert_eq!(slot.value().is_some(), populated);
                assert_eq!(slot.offset(), if populated { 2 } else { 0 });
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
        FixtureFamily::Qwen3Next | FixtureFamily::Qwen35 if rank == 0 => {
            let PipelineLayerCache::StateSlots { slots, .. } = &cache.layers()[0] else {
                panic!("Qwen linear-attention layer did not materialize recurrent state")
            };
            assert_slots(slots, 2);
        }
        FixtureFamily::Qwen3Next | FixtureFamily::Qwen35 => assert!(matches!(
            &cache.layers()[0],
            PipelineLayerCache::KeyValue { slots, .. } if slots.is_empty()
        )),
        FixtureFamily::Inkling => {
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

fn run_microbatch_worker(
    model: &mut safemlx_lm::architectures::distributed::pipeline::PipelineModel,
    rank: usize,
    group: &safemlx::distributed::Group,
    stream: &Stream,
) {
    let first_request = RequestId::new(101);
    let second_request = RequestId::new(202);
    let first_prompt = Array::from_slice(&[1u32, 2], &[1, 2]);
    let second_prompt = Array::from_slice(&[3u32, 4, 5], &[1, 3]);
    let first_decode = Array::from_slice(&[6u32], &[1, 1]);
    let second_decode = Array::from_slice(&[7u32], &[1, 1]);
    let work = [
        (
            first_request,
            PipelineInferencePhase::Prefill,
            PipelineStep::new(1, 2).unwrap(),
            &first_prompt,
        ),
        (
            second_request,
            PipelineInferencePhase::Prefill,
            PipelineStep::new(1, 3).unwrap(),
            &second_prompt,
        ),
        (
            first_request,
            PipelineInferencePhase::Decode,
            PipelineStep::new(1, 1).unwrap(),
            &first_decode,
        ),
        (
            second_request,
            PipelineInferencePhase::Decode,
            PipelineStep::new(1, 1).unwrap(),
            &second_decode,
        ),
    ];

    let mut first_reference_cache = model.new_cache().unwrap();
    let mut second_reference_cache = model.new_cache().unwrap();
    let mut reference = Vec::with_capacity(work.len());
    for (request, _, step, tokens) in &work {
        let cache = if *request == first_request {
            &mut first_reference_cache
        } else {
            &mut second_reference_cache
        };
        let logits = model
            .forward_pipeline(
                (rank == 0).then_some(*tokens),
                *step,
                None,
                cache,
                group,
                stream,
            )
            .unwrap();
        reference.push(logits.map(|logits| {
            let logits = logits.evaluated().unwrap();
            logits.as_slice::<f32>().to_vec()
        }));
    }

    let mut scheduler =
        PipelineInferenceScheduler::new(model, SchedulerLimits::new(2, 4).unwrap()).unwrap();
    let paged = PagedCacheOptions::new(1, 4096, 4096, 1)
        .unwrap()
        .with_full_attention(true);
    scheduler
        .register_request_with_options(
            model,
            first_request,
            CacheResidencyPolicy::Paged(paged.clone()),
        )
        .unwrap();
    scheduler
        .register_request_with_options(model, second_request, CacheResidencyPolicy::Paged(paged))
        .unwrap();
    // Enqueue two transitions for each request. Round-robin draining must
    // produce A0, B0, A1, B1 while retaining independent caches.
    for (request, phase, step, tokens) in [work[0], work[2], work[1], work[3]] {
        let input = PipelineMicrobatchInput::new(request, phase, step);
        let input = if rank == 0 {
            input.with_tokens(tokens.clone())
        } else {
            input
        };
        scheduler.enqueue(input).unwrap();
    }
    let output = scheduler.run_queued(model, group, stream).unwrap();
    assert_eq!(
        output
            .iter()
            .map(|output| (output.work().request().value(), output.work().sequence()))
            .collect::<Vec<_>>(),
        vec![(101, 0), (202, 0), (101, 1), (202, 1)]
    );
    for (expected, actual) in reference.iter().zip(&output) {
        match (expected, actual.logits()) {
            (Some(expected), Some(actual)) => {
                let actual = actual.evaluated().unwrap();
                let actual = actual.as_slice::<f32>();
                assert_eq!(expected.len(), actual.len());
                assert!(expected
                    .iter()
                    .zip(actual)
                    .all(|(left, right)| (left - right).abs() <= 1e-5));
            }
            (None, None) => {}
            _ => panic!("microbatch scheduling changed final-stage output ownership"),
        }
    }
    let report = scheduler.report();
    assert_eq!(report.completed_work, 4);
    assert_eq!(report.drain_cycles, 1);
    assert_eq!(report.peak_queued_work, 4);
    assert_eq!(report.active_requests, 2);
    assert!(!report.poisoned);

    scheduler.finish_request(second_request).unwrap();
    assert_eq!(
        scheduler.request_status(second_request),
        Some(RequestStatus::Finished)
    );
    let input = PipelineMicrobatchInput::new(
        first_request,
        PipelineInferencePhase::Decode,
        PipelineStep::new(1, 1).unwrap(),
    );
    let input = if rank == 0 {
        input.with_tokens(Array::from_slice(&[0u32], &[1, 1]))
    } else {
        input
    };
    scheduler.enqueue(input).unwrap();
    scheduler.cancel_request(first_request).unwrap();
    assert_eq!(
        scheduler.request_status(first_request),
        Some(RequestStatus::Cancelled)
    );
    let report = scheduler.report();
    assert_eq!(report.active_requests, 0);
    assert_eq!(report.queued_work, 0);
    assert_eq!(report.discarded_work, 1);
}

fn run_schedule_mismatch_worker(
    model: &mut safemlx_lm::architectures::distributed::pipeline::PipelineModel,
    rank: usize,
    group: &safemlx::distributed::Group,
    stream: &Stream,
) {
    let request = RequestId::new(if rank == 0 { 101 } else { 999 });
    let mut scheduler =
        PipelineInferenceScheduler::new(model, SchedulerLimits::new(1, 1).unwrap()).unwrap();
    scheduler.register_request(model, request).unwrap();
    let input = PipelineMicrobatchInput::new(
        request,
        PipelineInferencePhase::Prefill,
        PipelineStep::new(1, 2).unwrap(),
    );
    let input = if rank == 0 {
        input.with_tokens(Array::from_slice(&[1u32, 2], &[1, 2]))
    } else {
        input
    };
    scheduler.enqueue(input).unwrap();
    let error = scheduler.run_queued(model, group, stream).unwrap_err();
    assert!(error.to_string().contains("work descriptors differ"));
    assert!(scheduler.report().poisoned);
    assert_eq!(
        scheduler.request_status(request),
        Some(RequestStatus::Failed)
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
        "first_k_dense_replace": layers,
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
    let mut model = deepseek_v3::Model::new(args, stream).unwrap();
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
    let config = qwen_config(model_type);
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
    let config = nemotron_config();
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

fn qwen_hybrid_config(model_type: &str) -> serde_json::Value {
    serde_json::json!({
        "architectures": [if model_type == "qwen3_next" { "Qwen3NextForCausalLM" } else { "Qwen3_5ForCausalLM" }],
        "model_type": model_type,
        "vocab_size": 32,
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "mtp_num_hidden_layers": 0,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
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

fn write_qwen_hybrid_fixture(directory: &Path, model_type: &str) {
    let config = qwen_hybrid_config(model_type);
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let args = qwen_hybrid::model_args_from_config_value(&config).unwrap();
    let mut model = qwen_hybrid::Model::new(args, None, None, None, stream).unwrap();
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
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "swa_num_attention_heads": 2,
            "swa_num_key_value_heads": 1,
            "swa_head_dim": 8,
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

fn inkling_released_name(runtime: &str) -> String {
    if runtime == "lm_head.weight" {
        return "model.llm.unembed.weight".into();
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
    let config = inkling_config();
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

fn kimi_linear_gguf_payload_bytes() -> u64 {
    kimi_linear_gguf_specs()
        .iter()
        .map(|tensor| tensor.data.len() as u64)
        .sum()
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

fn reserve_two_ports() -> (TcpListener, TcpListener, u16, u16) {
    let first = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let second = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let first_port = first.local_addr().unwrap().port();
    let second_port = second.local_addr().unwrap().port();
    (first, second, first_port, second_port)
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

/// Verifies fair multi-request scheduling, independent request caches, exact
/// schedule consensus, variable prompt shapes, decode parity, EOS, and cancel.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_pipeline_microbatch_scheduler() {
    run_ring_pipeline_mode(false, FixtureFamily::Llama, WorkerMode::Microbatch);
}

/// Verifies the same scheduler and cache isolation over bounded local layers.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_dense_stream_pipeline_microbatch_scheduler() {
    run_ring_pipeline_mode(true, FixtureFamily::Llama, WorkerMode::Microbatch);
}

/// Verifies that divergent rank-local schedules fail before point-to-point
/// traffic and poison every local request cache rather than risking reuse.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_pipeline_schedule_mismatch_fails_closed() {
    run_ring_pipeline_mode(false, FixtureFamily::Llama, WorkerMode::ScheduleMismatch);
}

/// Run with:
/// `cargo test -p safemlx-lm --test distributed_pipeline_ring ring_two_process_dense_stream_pipeline -- --ignored --exact --nocapture`
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_dense_stream_pipeline() {
    run_ring_pipeline(true, FixtureFamily::Llama);
}

/// Verifies DeepSeek MLA paged-prefix persistence across two pipeline stages.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_deepseek_pipeline_persistence() {
    run_ring_pipeline(false, FixtureFamily::DeepSeek);
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

/// Verifies Mamba, dense, sparse, and sliding-attention Nemotron operators over
/// the two balanced stage ranges.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_nemotron_h_dense_stream_pipeline() {
    run_ring_pipeline(true, FixtureFamily::NemotronH);
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

/// Verifies Inkling's uneven 2+1 stage placement and combined KV/convolution
/// state against the resident text decoder.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_inkling_dense_stream_pipeline() {
    run_ring_pipeline(true, FixtureFamily::Inkling);
}

fn run_ring_pipeline(dense_stream: bool, family: FixtureFamily) {
    run_ring_pipeline_mode(dense_stream, family, WorkerMode::Standard);
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum WorkerMode {
    Standard,
    Microbatch,
    ScheduleMismatch,
}

fn run_ring_pipeline_mode(dense_stream: bool, family: FixtureFamily, mode: WorkerMode) {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    let checkpoint_path = if family == FixtureFamily::KimiLinearGguf {
        let path = checkpoint.path().join("model.gguf");
        write_kimi_linear_gguf_fixture(&path);
        path
    } else {
        match family {
            FixtureFamily::Llama => write_fixture(checkpoint.path()),
            FixtureFamily::DeepSeek => write_deepseek_fixture(checkpoint.path(), 2),
            FixtureFamily::Gemma => write_gemma_fixture(checkpoint.path()),
            FixtureFamily::Qwen2 => write_qwen_fixture(checkpoint.path(), "qwen2"),
            FixtureFamily::Qwen3 => write_qwen_fixture(checkpoint.path(), "qwen3"),
            FixtureFamily::Qwen3Moe => write_qwen_fixture(checkpoint.path(), "qwen3_moe"),
            FixtureFamily::GptOss => write_gpt_oss_fixture(checkpoint.path()),
            FixtureFamily::Lfm2 => write_lfm2_pipeline_fixture(checkpoint.path(), false),
            FixtureFamily::Lfm2Moe => write_lfm2_pipeline_fixture(checkpoint.path(), true),
            FixtureFamily::KimiLinear => write_kimi_linear_fixture(checkpoint.path()),
            FixtureFamily::NemotronH => write_nemotron_fixture(checkpoint.path()),
            FixtureFamily::Qwen3Next => write_qwen_hybrid_fixture(checkpoint.path(), "qwen3_next"),
            FixtureFamily::Qwen35 => write_qwen_hybrid_fixture(checkpoint.path(), "qwen3_5_text"),
            FixtureFamily::Inkling => write_inkling_fixture(checkpoint.path()),
            FixtureFamily::KimiLinearGguf => unreachable!(),
        }
        checkpoint.path().to_path_buf()
    };
    run_ring_pipeline_processes(dense_stream, family, mode, checkpoint, checkpoint_path);
}

fn run_ring_pipeline_processes(
    dense_stream: bool,
    family: FixtureFamily,
    mode: WorkerMode,
    _checkpoint: tempfile::TempDir,
    checkpoint_path: PathBuf,
) {
    let prompt_cache = tempfile::tempdir().unwrap();
    let (first_socket, second_socket, first_port, second_port) = reserve_two_ports();
    let ring = tempfile::tempdir().unwrap();
    let hostfile = ring.path().join("ring-hosts.json");
    std::fs::write(
        &hostfile,
        format!("[[\"127.0.0.1:{first_port}\"],[\"127.0.0.1:{second_port}\"]]"),
    )
    .unwrap();
    let executable = std::env::current_exe().unwrap();
    let mut children = ChildGuard {
        children: Vec::with_capacity(2),
    };
    let mut reservations = [Some(first_socket), Some(second_socket)];
    for (rank, reservation) in reservations.iter_mut().enumerate() {
        drop(reservation.take());
        let mut command = Command::new(&executable);
        command
            .args(["--exact", "pipeline_ring_worker", "--nocapture"])
            .env(WORKER_RANK, rank.to_string())
            .env(CHECKPOINT_DIR, &checkpoint_path)
            .env(FIXTURE_FAMILY, family.name())
            .env(PROMPT_CACHE_ROOT, prompt_cache.path())
            .env("MLX_RANK", rank.to_string())
            .env("MLX_HOSTFILE", &hostfile)
            .env_remove("MLX_RING_VERBOSE")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if dense_stream {
            command.env(DENSE_STREAM, "1");
        }
        match mode {
            WorkerMode::Standard => {}
            WorkerMode::Microbatch => {
                command.env(MICROBATCH, "1");
            }
            WorkerMode::ScheduleMismatch => {
                command.env(SCHEDULE_MISMATCH, "1");
            }
        }
        children.children.push(command.spawn().unwrap());
    }
    let deadline = Instant::now() + Duration::from_secs(45);
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
        "two-process pipeline Ring test failed:\n{}",
        if timed_out {
            format!("timed out after 45 seconds\n\n{}", failures.join("\n\n"))
        } else {
            failures.join("\n\n")
        }
    );
}
