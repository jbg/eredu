//! Application composition of distributed native execution and public control.
use eredu::{api::*, runtime::chat::ChatTemplateRequest};
use eredu_backend_mlx::{native, MlxBackendFactory, MlxLoadRequest};
use eredu_core::{
    capture::*, execution_control::SnapshotLimits, residency::OffloadConfig, DraftingPlan,
    ExecutionPlan, GenerationConfigOverrides, ParallelRankTopology, ParallelTopology,
    PrefillChunkPolicy,
};
use eredu_evaluation::qwen4_exp::{add_prediction_weights, metadata, Fixture, PreparedFixtures};
use eredu_gguf::MetadataValue;
use eredu_runtime::{
    CommunicationCompletionPolicy, DraftingLoadRequest, LayerWeightResidency, LayerwiseLoadOptions,
    NormalizedLoadRequest, PipelineActivationDtype, PipelineWireContract, WeightResidency,
};
use safemlx::{distributed, Device, DeviceType, Stream};
use serde::{Deserialize, Serialize};
use std::{
    net::TcpListener,
    num::NonZeroUsize,
    path::PathBuf,
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};
use tokenizers::{models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace, Tokenizer};

const WORKER: &str = "EREDU_FACADE_QWEN4_RING_WORKER";
const CHECKPOINT: &str = "EREDU_FACADE_QWEN4_CHECKPOINT";
const CASE: &str = "EREDU_FACADE_QWEN4_CASE";
const COMPANION: &str = "EREDU_FACADE_QWEN4_COMPANION";
const TEMPLATE: &str = "{% for m in messages %}{{ m.content }}{% endfor %}";
const PROMPT: [u32; 19] = [3, 4, 0, 5, 8, 9, 11, 12, 14, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
enum Source {
    SafeTensors,
    Gguf,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
enum Residency {
    Resident,
    Host,
    Disk,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct Case {
    source: Source,
    axes: [usize; 3],
    residency: Residency,
}
impl Case {
    fn topology(self) -> ParallelTopology {
        ParallelTopology::new(self.axes[0], self.axes[1], self.axes[2], 1).unwrap()
    }
    fn layers(self) -> LayerWeightResidency {
        match self.residency {
            Residency::Resident => LayerWeightResidency::FullyResident,
            Residency::Host => LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
                OffloadConfig::new(Some(1 << 25), Some(1 << 26), 1).unwrap(),
            )),
            Residency::Disk => LayerWeightResidency::DenseDiskStream(
                eredu_runtime::DenseDiskStreamLoadOptions::new(1 << 25, 1 << 26, 1, 1).unwrap(),
            ),
        }
    }
}

const CAPTURE_PATHS: [&str; 5] = [
    "model.layers.0.input",
    "model.layers.1.output",
    "mtp.layers.0.prediction.fusion",
    "mtp.layers.0.prediction.capture",
    "mtp.layers.0.prediction.attention.channels",
];

fn capture_plan(world_size: usize) -> SpeculativeActivationPlan {
    let receiver_allowance = if world_size > 4 { 256 << 20 } else { 128 << 20 };
    let allowance = CaptureUsage {
        captures: 32,
        // Existing distributed receipt admission reserves parser storage at
        // 128 times the encoded bound on every receiving rank. Rank-4
        // residuals (at most 19 * 2 * 32 values each), across multiple ranks,
        // therefore need more host and retention allowance than their tensor payloads.
        // Eight receivers double the existing four-rank reservation envelope.
        retained_bytes: receiver_allowance,
        host_bytes: receiver_allowance,
        encoded_bytes: 8 << 20,
    };
    SpeculativeActivationPlan {
        schema_version: SPECULATIVE_ACTIVATION_SCHEMA_VERSION,
        bounds: CaptureInvocationBounds {
            batch: 1,
            max_sequence: PROMPT.len() as u64,
            max_context: None,
            max_predictions: 8,
        },
        captures: CapturePlan {
            schema_version: CAPTURE_SCHEMA_VERSION,
            selections: CAPTURE_PATHS
                .iter()
                .map(|path| CaptureSelection {
                    id: (*path).into(),
                    path: (*path).into(),
                    schedule: Default::default(),
                    slices: vec![],
                    transform: CaptureTransform::Preview { max_elements: 2048 },
                })
                .collect(),
            limits: CaptureLimits {
                per_step: allowance,
                cumulative: allowance.checked_mul(64).unwrap(),
                physical_native_bytes: None,
                on_limit: CaptureLimitPolicy::Fail,
            },
        },
        interventions: eredu_core::intervention::InterventionPlan {
            schema_version: eredu_core::intervention::INTERVENTION_SCHEMA_VERSION,
            operations: vec![],
        },
    }
}

struct Workers(Vec<Child>);
impl Drop for Workers {
    fn drop(&mut self) {
        for child in &mut self.0 {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn run(case: Case) {
    let world = case.topology().world_size();
    eprintln!("public capture case: {case:?}");
    assert!(distributed::is_available(distributed::Backend::Ring));
    let directory = tempfile::tempdir().unwrap();
    let fixture = PreparedFixtures::write(directory.path()).unwrap();
    add_prediction_weights(&fixture.safetensors_path).unwrap();
    let vocabulary = (0..16).map(|id| (format!("word{id}"), id)).collect();
    let mut tokenizer = Tokenizer::new(
        WordLevel::builder()
            .vocab(vocabulary)
            .unk_token("word0".into())
            .build()
            .unwrap(),
    );
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer
        .save(fixture.safetensors_path.join("tokenizer.json"), false)
        .unwrap();
    std::fs::write(
        fixture.safetensors_path.join("chat_template.jinja"),
        TEMPLATE,
    )
    .unwrap();
    let checkpoint = match case.source {
        Source::SafeTensors => fixture.safetensors_path.clone(),
        Source::Gguf => {
            let mut values = metadata();
            values.insert(
                "tokenizer.huggingface.json".into(),
                MetadataValue::String(tokenizer.to_string(false).unwrap()),
            );
            values.insert(
                "tokenizer.chat_template".into(),
                MetadataValue::String(TEMPLATE.into()),
            );
            values.insert(
                "tokenizer.ggml.eos_token_id".into(),
                MetadataValue::Uint32(0),
            );
            // Keep the fixture's retained original source provenance unchanged.
            let path = directory.path().join("target-with-tokenizer.gguf");
            Fixture::new().write_metadata(&path, &values);
            path
        }
    };
    let sockets: Vec<_> = (0..world)
        .map(|_| TcpListener::bind(("127.0.0.1", 0)).unwrap())
        .collect();
    let hosts: Vec<_> = sockets
        .iter()
        .map(|socket| vec![socket.local_addr().unwrap().to_string()])
        .collect();
    let hostfile = directory.path().join("ring-hosts.json");
    std::fs::write(&hostfile, serde_json::to_vec(&hosts).unwrap()).unwrap();
    drop(sockets);
    let mut workers = Workers(Vec::new());
    for rank in 0..world {
        workers.0.push(
            Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "ring_worker", "--nocapture"])
                .env(WORKER, rank.to_string())
                .env(CHECKPOINT, &checkpoint)
                .env(COMPANION, &fixture.safetensors_path)
                .env(CASE, serde_json::to_string(&case).unwrap())
                .env("MLX_RANK", rank.to_string())
                .env("MLX_HOSTFILE", &hostfile)
                .env("RUST_MIN_STACK", "33554432")
                .env_remove("MLX_RING_VERBOSE")
                .stdout(Stdio::inherit())
                .stderr(Stdio::inherit())
                .spawn()
                .unwrap(),
        );
    }
    let started = Instant::now();
    loop {
        let statuses: Vec<_> = workers
            .0
            .iter_mut()
            .map(|child| child.try_wait().unwrap())
            .collect();
        assert!(
            statuses.iter().flatten().all(|status| status.success()),
            "Ring worker failed: {statuses:?}"
        );
        if statuses.iter().all(Option::is_some) {
            break;
        }
        assert!(
            started.elapsed() < Duration::from_secs(180),
            "Ring facade workers timed out"
        );
        std::thread::sleep(Duration::from_millis(25));
    }
}

#[test]
#[ignore = "requires two CPU MLX Ring ranks and loopback sockets"]
fn qwen4_facade_controlled_tp2() {
    run(Case {
        source: Source::SafeTensors,
        axes: [2, 1, 1],
        residency: Residency::Resident,
    });
}

#[test]
#[ignore = "requires two CPU MLX Ring ranks and loopback sockets"]
fn qwen4_facade_controlled_pp2_host() {
    run(Case {
        source: Source::SafeTensors,
        axes: [1, 2, 1],
        residency: Residency::Host,
    });
}

#[test]
#[ignore = "requires four CPU MLX Ring ranks and loopback sockets"]
fn qwen4_facade_controlled_tp2_pp2_disk() {
    run(Case {
        source: Source::SafeTensors,
        axes: [2, 2, 1],
        residency: Residency::Disk,
    });
}

#[test]
#[ignore = "requires two CPU MLX Ring ranks and loopback sockets"]
fn qwen4_facade_controlled_gguf_tp2() {
    run(Case {
        source: Source::Gguf,
        axes: [2, 1, 1],
        residency: Residency::Resident,
    });
}

#[test]
#[ignore = "requires two CPU MLX Ring ranks and loopback sockets"]
fn qwen4_facade_controlled_gguf_pp2_host() {
    run(Case {
        source: Source::Gguf,
        axes: [1, 2, 1],
        residency: Residency::Host,
    });
}

#[test]
#[ignore = "requires four CPU MLX Ring ranks and loopback sockets"]
fn qwen4_facade_controlled_gguf_tp2_pp2_disk() {
    run(Case {
        source: Source::Gguf,
        axes: [2, 2, 1],
        residency: Residency::Disk,
    });
}

#[test]
#[ignore = "requires two CPU MLX Ring ranks and loopback sockets"]
fn qwen4_facade_controlled_ep2() {
    run(Case {
        source: Source::SafeTensors,
        axes: [1, 1, 2],
        residency: Residency::Resident,
    });
}

#[test]
#[ignore = "requires four CPU MLX Ring ranks and loopback sockets"]
fn qwen4_facade_controlled_gguf_tp2_ep2_disk() {
    run(Case {
        source: Source::Gguf,
        axes: [2, 1, 2],
        residency: Residency::Disk,
    });
}

#[test]
#[ignore = "requires four CPU MLX Ring ranks and loopback sockets"]
fn qwen4_facade_controlled_pp2_ep2_host() {
    run(Case {
        source: Source::SafeTensors,
        axes: [1, 2, 2],
        residency: Residency::Host,
    });
}

#[test]
#[ignore = "requires eight CPU MLX Ring ranks and loopback sockets"]
fn qwen4_facade_controlled_gguf_tp2_pp2_ep2_host() {
    run(Case {
        source: Source::Gguf,
        axes: [2, 2, 2],
        residency: Residency::Host,
    });
}

#[test]
fn ring_worker() {
    let Ok(rank) = std::env::var(WORKER) else {
        return;
    };
    let checkpoint = PathBuf::from(std::env::var_os(CHECKPOINT).unwrap());
    let case: Case = serde_json::from_str(&std::env::var(CASE).unwrap()).unwrap();
    let companion = matches!(case.source, Source::Gguf)
        .then(|| PathBuf::from(std::env::var_os(COMPANION).unwrap()));
    let topology = ParallelRankTopology::new(case.topology(), rank.parse().unwrap()).unwrap();
    let group = distributed::init(true, distributed::Backend::Ring).unwrap();
    let mut plan = ExecutionPlan::fully_resident(local_device_plan(LocalDevice::Cpu).unwrap())
        .with_drafting(DraftingPlan::Embedded {
            max_draft_tokens: 1,
            lookahead: false,
            adaptive_lookahead: false,
        });
    if let Some(path) = &companion {
        plan = plan.with_prediction_source(path);
    }
    let loaded =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &checkpoint, &plan)
            .unwrap();
    let options = loaded.speculative_generation_options().unwrap().unwrap();
    let (mut baseline, _) = loaded.into_parts();
    let chat_request = || ChatTemplateRequest {
        messages: vec![serde_json::json!({"role":"user", "content":"word3 word4"})],
        add_generation_prompt: false,
        ..Default::default()
    };
    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            do_sample: Some(false),
            max_new_tokens: Some(8),
            ..Default::default()
        },
        prefill: PrefillChunkPolicy::Bounded(NonZeroUsize::new(3).unwrap()),
        ..Default::default()
    };
    let chat = baseline.prepare_chat(chat_request()).unwrap();
    let expected = baseline
        .generate_prepared_text_speculative(PreparedChatSpeculativeGenerationRequest {
            input: PreparedChatInput::token_ids(&chat, PROMPT.to_vec()),
            drafting: eredu_core::SpeculativeDraft::Embedded,
            settings,
            options,
            caller_stop_sequences: &[],
            cancellation: Default::default(),
            on_event: |_| {},
        })
        .unwrap();
    assert!(expected.token_ids().len() >= 2);
    baseline.reset().unwrap();
    let activations = baseline
        .prepare_speculative_activations(capture_plan(case.topology().world_size()))
        .unwrap();
    let mut baseline_captures = Vec::new();
    let captured_baseline = baseline
        .with_controlled_text_speculative(
            PreparedChatSpeculativeGenerationRequest {
                input: PreparedChatInput::token_ids(&chat, PROMPT.to_vec()),
                drafting: eredu_core::SpeculativeDraft::Embedded,
                settings,
                options,
                caller_stop_sequences: &[],
                cancellation: Default::default(),
                on_event: |_| {},
            },
            ControlledSpeculativeOptions {
                activations: Some(activations),
                ..Default::default()
            },
            |session| {
                while let Some(step) = session.step()? {
                    baseline_captures.extend(step.activations);
                }
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(captured_baseline.token_ids(), expected.token_ids());
    drop(baseline);
    let mut load = NormalizedLoadRequest::default()
        .with_bounded_execution(eredu_evaluation::qwen4_exp::bounded_policy())
        .with_weight_residency(WeightResidency::with_layers(case.layers()))
        .with_drafting(DraftingLoadRequest::embedded(1).unwrap())
        .with_required_session_capabilities(eredu_core::SessionCapabilities::new(true, true, true));
    if let Some(path) = companion {
        load = load.with_prediction_source(path);
    }
    let invocation = load.bounded_execution().unwrap().invocation();
    let device = native::DeviceAssignment::new(DeviceType::Cpu, 0);
    let load = MlxLoadRequest::from_normalized(load)
        .with_parallel_topology(
            topology,
            device,
            PipelineWireContract::new(PipelineActivationDtype::Float32),
            invocation.batch(),
            invocation.chunk_tokens(),
            CommunicationCompletionPolicy::new(
                Duration::from_secs(30),
                eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
            )
            .unwrap(),
        )
        .unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let weights = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    // The factory handles local device realization. Distributed applications
    // explicitly own their group and rank request, then use the same facade load.
    let mut model = LoadedModel::load(
        native::distributed_backend(&stream, &weights, &group),
        &checkpoint,
        load,
    )
    .unwrap();
    let chat = model.prepare_chat(chat_request()).unwrap();
    let request = || PreparedChatSpeculativeGenerationRequest {
        input: PreparedChatInput::token_ids(&chat, PROMPT.to_vec()),
        drafting: eredu_core::SpeculativeDraft::Embedded,
        settings,
        options,
        caller_stop_sequences: &[],
        cancellation: Default::default(),
        on_event: |_| {},
    };
    let ordinary = model.generate_prepared_text_speculative(request()).unwrap();
    assert_eq!(ordinary.token_ids(), expected.token_ids());
    assert_eq!(ordinary.finish_reason(), expected.finish_reason());
    model.reset().unwrap();
    let snapshot_limits = || SnapshotLimits {
        max_snapshots: 1,
        max_branches: 1,
        retained_bytes: 32 << 20,
        cumulative_copy_bytes: 128 << 20,
    };
    // Establish controller parity independently of optional capture admission.
    let controlled = model
        .with_controlled_text_speculative(
            request(),
            ControlledSpeculativeOptions {
                snapshots: Some(snapshot_limits()),
                ..Default::default()
            },
            |session| {
                assert!(session.step()?.is_some());
                assert!(session.can_snapshot(), "{:?}", session.snapshot_support());
                let saved = session.snapshot()?;
                while session.step()?.is_some() {}
                let completed = session.token_ids().to_vec();
                let copies = session.snapshot_usage().cumulative_copy_bytes;
                session.restore(&saved)?;
                assert!(session.snapshot_usage().cumulative_copy_bytes > copies);
                while session.step()?.is_some() {}
                assert_eq!(session.token_ids(), completed);
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(controlled.token_ids(), expected.token_ids());
    assert_eq!(controlled.finish_reason(), expected.finish_reason());
    assert!(controlled.stats().draft_tokens() > 0);
    eprintln!("rank {rank}: public controlled MTP snapshot replay passed");
    model.reset().unwrap();
    // Target boundaries span PP stages. Prediction fusion/capture preserve all
    // residual streams; prediction attention channels assemble the TP shards.
    let activations = model
        .prepare_speculative_activations(capture_plan(case.topology().world_size()))
        .unwrap();
    let mut captured = Vec::new();
    let controlled = model
        .with_controlled_text_speculative(
            request(),
            ControlledSpeculativeOptions {
                activations: Some(activations),
                snapshots: Some(snapshot_limits()),
                ..Default::default()
            },
            |session| {
                captured.extend(session.step()?.unwrap().activations);
                assert!(session.can_snapshot(), "{:?}", session.snapshot_support());
                let saved = session.snapshot()?;
                let start = captured.len();
                while let Some(step) = session.step()? {
                    captured.extend(step.activations);
                }
                let completed = session.token_ids().to_vec();
                let copies = session.snapshot_usage().cumulative_copy_bytes;
                session.restore(&saved)?;
                assert!(session.snapshot_usage().cumulative_copy_bytes > copies);
                let mut replay = Vec::new();
                while let Some(step) = session.step()? {
                    replay.extend(step.activations);
                }
                assert_eq!(replay.len(), captured.len() - start);
                for (actual, expected) in replay.iter().zip(&captured[start..]) {
                    assert_eq!(actual.captures.records, expected.captures.records);
                    assert_eq!(actual.admission_identity, expected.admission_identity);
                }
                assert_eq!(session.token_ids(), completed);
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(controlled.token_ids(), expected.token_ids());
    assert_eq!(controlled.finish_reason(), expected.finish_reason());
    assert!(controlled.stats().draft_tokens() > 0);
    assert_eq!(captured.len(), baseline_captures.len());
    for (actual, reference) in captured.iter().zip(&baseline_captures) {
        assert_eq!(actual.phase, reference.phase);
        assert_eq!(actual.completed, reference.completed);
        assert_eq!(
            actual.captures.records.len(),
            reference.captures.records.len()
        );
        for (actual, reference) in actual
            .captures
            .records
            .iter()
            .zip(&reference.captures.records)
        {
            assert_eq!(actual.path, reference.path);
            assert_eq!(actual.source_shape, reference.source_shape);
            assert_eq!(actual.selected_shape, reference.selected_shape);
            assert_eq!(actual.source_dtype, reference.source_dtype);
            assert_eq!(actual.outcome, reference.outcome);
            match (&actual.payload, &reference.payload) {
                (Some(CapturePayload::Tensor(actual)), Some(CapturePayload::Tensor(reference))) => {
                    assert_eq!(actual.shape(), reference.shape());
                    let (
                        eredu_core::TensorObservationData::F32(actual),
                        eredu_core::TensorObservationData::F32(reference),
                    ) = (actual.data(), reference.data())
                    else {
                        panic!("activation capture must be floating point")
                    };
                    assert_eq!(actual.len(), reference.len());
                    for (actual, reference) in actual.iter().zip(reference) {
                        assert!(actual.is_finite() && reference.is_finite());
                        assert!(
                            (actual - reference).abs() <= 2e-4,
                            "distributed capture {actual} differs from single-rank {reference}"
                        );
                    }
                }
                (None, None) => {}
                _ => panic!("unexpected activation capture payload"),
            }
        }
    }
    for path in CAPTURE_PATHS {
        assert!(
            captured
                .iter()
                .flat_map(|event| &event.captures.records)
                .any(|record| {
                    record.path == path
                        && record.outcome == CaptureOutcome::Captured
                        && record.payload.is_some()
                }),
            "no completed capture for {path}: {captured:?}"
        );
    }
    // A catalog entry or a skipped prediction-prefill record is insufficient:
    // each internal point must be reached by an actual MTP proposal forward.
    for (index, path) in CAPTURE_PATHS[2..].iter().enumerate() {
        let record = captured
            .iter()
            .filter(|event| {
                event.completed && event.phase == SpeculativeActivationPhase::Proposal { depth: 0 }
            })
            .flat_map(|event| &event.captures.records)
            .find(|record| record.path == *path && record.outcome == CaptureOutcome::Captured)
            .unwrap_or_else(|| panic!("no completed MTP proposal capture for {path}"));
        let Some(CapturePayload::Tensor(tensor)) = &record.payload else {
            panic!("MTP proposal capture {path} has no tensor");
        };
        let expected_shape: &[u64] = if index < 2 {
            &[1, 1, 2, 32]
        } else {
            &[1, 1, 32]
        };
        assert_eq!(
            record.source_shape.as_deref(),
            Some(expected_shape),
            "global MTP source geometry for {path}"
        );
        assert_eq!(
            record.selected_shape.as_deref(),
            Some(expected_shape),
            "global MTP selected geometry for {path}"
        );
        let eredu_core::TensorObservationData::F32(values) = tensor.data() else {
            panic!("MTP proposal capture {path} is not floating point");
        };
        // Preview transport flattens its bounded payload while preserving the
        // original and selected tensor geometry in the record above.
        assert_eq!(values.len() as u64, expected_shape.iter().product::<u64>());
        assert!(
            values.iter().any(|value| value.abs() > 1e-8),
            "zero fixture capture for {path}"
        );
    }
    model.synchronize().unwrap();
}
