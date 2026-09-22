//! Application-level validation of the same facade driver on every native rank.
#![cfg(feature = "mlx")]
use eredu::api::*;
use eredu::runtime::chat::ChatTemplateRequest;
use eredu_core::{capture::*, execution_control::*, GenerationConfigOverrides};
use safemlx::{distributed, DeviceType};
use std::{
    io::Write,
    ops::ControlFlow,
    path::Path,
    process::{Command, Stdio},
    time::{Duration, Instant},
};

fn fixture(root: &Path, family: &str) {
    let fixtures: serde_json::Value = serde_json::from_str(include_str!(
        "../../eredu-architectures/tests/fixtures/k2_horizon/reference.json"
    ))
    .unwrap();
    let mut config = fixtures[family]["config"].clone();
    config["num_hidden_layers"] = 3.into();
    config["vocab_size"] = 64.into();
    config.as_object_mut().unwrap().remove("eos_token_id");
    std::fs::write(
        root.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let args = eredu_architectures::k2_horizon::model_args_from_config_value(&config).unwrap();
    let plan = eredu_architectures::k2_horizon::safetensors_plan(&args).unwrap();
    let mut data = vec![];
    let mut header = serde_json::Map::new();
    for tensor in plan.common_tensors.iter().chain(
        plan.layout_groups
            .iter()
            .filter_map(|group| group.variants.first())
            .flat_map(|variant| &variant.tensors),
    ) {
        let start = data.len();
        for index in 0..tensor.shape.iter().product::<usize>() {
            let value = if tensor.key.contains("norm") {
                1.0f32
            } else {
                (((index * 17 + tensor.key.len() * 7) % 101) as f32 - 50.0) * 0.003
            };
            data.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(tensor.key.clone(),serde_json::json!({"dtype":"F32","shape":tensor.shape,"data_offsets":[start,data.len()]}));
    }
    let mut header = serde_json::to_vec(&header).unwrap();
    while header.len() % 8 != 0 {
        header.push(b' ');
    }
    let mut file = std::fs::File::create(root.join("model.safetensors")).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    file.write_all(&data).unwrap();
    let vocab = (0..64)
        .map(|id| (format!("word{id}"), serde_json::json!(id)))
        .collect::<serde_json::Map<_, _>>();
    let tokenizer = serde_json::json!({"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":vocab,"unk_token":"word0"}});
    std::fs::write(
        root.join("tokenizer.json"),
        serde_json::to_vec(&tokenizer).unwrap(),
    )
    .unwrap();
    std::fs::write(
        root.join("chat_template.jinja"),
        "{% for m in messages %}{{ m.content }}{% endfor %}",
    )
    .unwrap();
}

fn baseline_failure<T>(error: eredu_core::BackendFailure) -> T {
    let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(&error);
    while let Some(error) = cause {
        eprintln!("{error}");
        cause = error.source();
    }
    panic!("distributed baseline preparation failed; source chain printed above")
}

fn run_control_case(family: &str, residency: &str) {
    let root = tempfile::tempdir().unwrap();
    fixture(root.path(), family);
    let world = if family == "dense" { 4 } else { 8 };
    let sockets = (0..world)
        .map(|_| std::net::TcpListener::bind("127.0.0.1:0").unwrap())
        .collect::<Vec<_>>();
    let hosts = sockets
        .iter()
        .map(|socket| vec![format!("127.0.0.1:{}", socket.local_addr().unwrap().port())])
        .collect::<Vec<_>>();
    let hostfile = root.path().join("hosts.json");
    std::fs::write(&hostfile, serde_json::to_vec(&hosts).unwrap()).unwrap();
    drop(sockets);
    let mut children = vec![];
    for rank in 0..world {
        let log = std::fs::File::create(root.path().join(format!("rank-{rank}.log"))).unwrap();
        children.push(
            Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "k2_distributed_facade_worker", "--nocapture"])
                .env("K2_CONTROL_ARTIFACT", root.path())
                .env("K2_CONTROL_FAMILY", family)
                .env("K2_CONTROL_RESIDENCY", residency)
                .env("MLX_RANK", rank.to_string())
                .env("MLX_HOSTFILE", &hostfile)
                .stdout(Stdio::from(log.try_clone().unwrap()))
                .stderr(Stdio::from(log))
                .spawn()
                .unwrap(),
        );
    }
    // Eight Mova ranks run two complete sampling/snapshot/delivery matrices.
    // Preserve a finite hang deadline while allowing their measured CPU work.
    let deadline = Instant::now() + Duration::from_secs(240);
    while children
        .iter_mut()
        .any(|child| child.try_wait().unwrap().is_none())
        && Instant::now() < deadline
    {
        std::thread::sleep(Duration::from_millis(25));
    }
    let mut failures = vec![];
    for (rank, mut child) in children.into_iter().enumerate() {
        let status = match child.try_wait().unwrap() {
            Some(status) => status,
            None => {
                child.kill().unwrap();
                child.wait().unwrap()
            }
        };
        if !status.success() {
            failures.push(format!(
                "rank {rank}: {}",
                std::fs::read_to_string(root.path().join(format!("rank-{rank}.log"))).unwrap()
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{family}/{residency}: {}",
        failures.join("\n")
    );
}

mod k2_distributed_facade_control_matrix {
    macro_rules! case {
        ($name:ident, $family:literal, $residency:literal) => {
            #[test]
            #[ignore = "spawns local CPU ranks and opens loopback sockets; run explicitly"]
            fn $name() {
                super::run_control_case($family, $residency);
            }
        };
    }

    case!(dense_resident, "dense", "resident");
    case!(dense_host, "dense", "host");
    case!(dense_disk, "dense", "disk");
    case!(mova_resident, "mova", "resident");
    case!(mova_host, "mova", "host");
    case!(mova_disk, "mova", "disk");
}

// A callback refusal also retains the canonical failed session. Both facade
// variants preserve the same typed local capture cause and its original owner.
fn is_cumulative_encoded_limit(error: &ControlledGenerationError) -> bool {
    let capture = match error {
        ControlledGenerationError::Capture(capture)
        | ControlledGenerationError::CaptureFailure { capture, .. } => capture,
        _ => return false,
    };
    matches!(
        capture.cause(),
        CaptureError::Limit {
            budget: CaptureBudget::Encoded,
            cumulative: true,
        }
    )
}

fn reset_admitted(
    model: &mut LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,
    capacity: u64,
) {
    let before = model.text_preparation_usage().unwrap();
    assert!(before.attempts > 0);
    // Every participant refuses the same zero host-domain limit before any
    // readiness producer or vote; no retry can recover spent preparation work.
    let refused = model.prepare_reset_ordinary().unwrap().reset_admitted(
        eredu_core::SessionResetLimits::new(eredu_core::MemoryLimitDeclarations::new([(
            "host".into(),
            eredu_core::MemoryLimit::Finite(0),
        )])),
    );
    assert!(refused.is_err());
    drop(refused);
    assert_eq!(model.text_preparation_usage().unwrap(), before);
    // All stages use this retained coordinator's same two-exchange protocol.
    // Successful reset adds construction and publication votes, preserving all
    // previous spending rather than resetting the coordinator's history.
    assert_eq!(before.retained_bytes % before.attempts, 0);
    assert_eq!(before.host_bytes % before.attempts, 0);
    let retained_per_attempt = before.retained_bytes / before.attempts;
    let host_per_attempt = before.host_bytes / before.attempts;
    assert!(retained_per_attempt > 0 && host_per_attempt > 0);
    model
        .prepare_reset_ordinary()
        .unwrap()
        .reset_admitted(eredu_core::SessionResetLimits::new(
            eredu_core::MemoryLimitDeclarations::new([(
                "host".into(),
                eredu_core::MemoryLimit::Finite(capacity),
            )]),
        ))
        .unwrap();
    model.synchronize().unwrap();
    safemlx::memory::clear_cache().unwrap();
    eredu_backend_mlx::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
    let after = model.text_preparation_usage().unwrap();
    assert_eq!(after.attempts, before.attempts + 2);
    assert_eq!(
        after.retained_bytes,
        before.retained_bytes + 2 * retained_per_attempt
    );
    assert_eq!(after.host_bytes, before.host_bytes + 2 * host_per_attempt);
}

#[track_caller]
fn load_fixture<'a>(
    root: &Path,
    family: &str,
    residency: &str,
    world: &'a distributed::Group,
    rank: usize,
) -> LoadedModel<eredu_backend_mlx::backend::MlxBackend<'a>> {
    // Each model retains its actual admitted stream and communicator sources.
    let backend =
        eredu_backend_mlx::native::prepared_distributed_backend_on(world, DeviceType::Cpu)
            .expect("prepared distributed execution/source stream construction")
            .expect("the selected CPU has a qualified stream constructor");
    let topology = eredu_core::ParallelRankTopology::new(
        eredu_core::ParallelTopology::new(2, 2, if family == "mova" { 2 } else { 1 }, 1).unwrap(),
        rank,
    )
    .unwrap();
    let ordinary = match residency {
        "resident" => eredu_runtime::OrdinaryWeightResidency::FullyResident,
        "host" => eredu_runtime::OrdinaryWeightResidency::LayerwiseHost(
            eredu_runtime::LayerwiseLoadOptions::new(
                eredu_core::residency::OffloadConfig::new(Some(1 << 20), Some(1 << 20), 1).unwrap(),
            ),
        ),
        _ => eredu_runtime::OrdinaryWeightResidency::DenseDiskStream(
            eredu_runtime::DenseDiskStreamLoadOptions::new(1 << 20, 1 << 20, 1, 1).unwrap(),
        ),
    };
    let weights = if family == "mova" {
        eredu_runtime::WeightResidency::with_independent_parameter_banks(
            ordinary,
            eredu_runtime::ParameterBankLoadOptions::new(
                eredu_core::residency::OffloadConfig::new(Some(1152), Some(1 << 20), 1).unwrap(),
                1152,
                1152,
            )
            .unwrap(),
        )
    } else {
        eredu_runtime::WeightResidency::with_layers(ordinary.layers())
    };
    let options = eredu_backend_mlx::MlxLoadRequest::from_normalized(
        eredu_runtime::NormalizedLoadRequest::default()
            .with_weight_residency(weights)
            .with_state_residency(eredu_runtime::CacheResidencyPolicy::Paged(
                eredu_runtime::PagedCacheOptions::new(2, 32768, 1 << 20, 1)
                    .unwrap()
                    .with_full_attention(true),
            )),
    )
    .with_parallel_topology(
        topology,
        eredu_backend_mlx::native::DeviceAssignment::new(DeviceType::Cpu, 0),
        eredu_runtime::PipelineWireContract::new(eredu_runtime::PipelineActivationDtype::Float32),
        1,
        128,
        eredu_runtime::CommunicationCompletionPolicy::new(
            Duration::from_secs(30),
            eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap(),
    )
    .unwrap();
    let prepared = eredu_core::load_model(&backend, root, options).unwrap();
    let runtime = eredu_core::ModelRuntime::from_prepared(backend, prepared).unwrap();
    let tokenizer =
        eredu_text::tokenizer::Tokenizer::from_file(root.join("tokenizer.json")).unwrap();
    LoadedModel::from_runtime(
        runtime,
        tokenizer,
        LoadedTextModelConfig {
            model_family: eredu_architectures::configuration::ModelKind::K2Horizon,
            effective_model_type: "k2_horizon".into(),
            model_id: "k2-control-fixture".into(),
            chat_template: Some("{% for m in messages %}{{ m.content }}{% endfor %}".into()),
            eos_token_ids: vec![],
            checkpoint_generation_config: None,
        },
    )
    .unwrap()
}

#[test]
fn k2_distributed_facade_worker() {
    let Some(root) = std::env::var_os("K2_CONTROL_ARTIFACT") else {
        return;
    };
    let root = std::path::PathBuf::from(root);
    let family = std::env::var("K2_CONTROL_FAMILY").unwrap();
    let residency = std::env::var("K2_CONTROL_RESIDENCY").unwrap();
    let rank = std::env::var("MLX_RANK").unwrap().parse().unwrap();
    let started = Instant::now();
    let stage = |name: &str| {
        eprintln!(
            "{family}/{residency} rank {rank} phase={name} elapsed={:?} memory={:?}",
            started.elapsed(),
            local_memory_snapshot(),
        );
    };
    stage("communicator construction");
    let world = distributed::init(true, distributed::Backend::Ring).unwrap();
    stage("reference model construction");
    let mut model = load_fixture(&root, &family, &residency, &world, rank);
    stage("reference model ready");
    // CPU evaluation quotes each nested frontier against the reachable graph,
    // including conservative container growth and each control's ledger owner.
    // This parity fixture admits those allowances for simultaneous branches;
    // the limit is not an expected allocation volume or a process RSS bound.
    const CAPACITY: u64 = 16 << 40;
    let physical_limits = eredu_core::MemoryLimitDeclarations::new(
        eredu_backend_mlx::memory_topology()
            .unwrap()
            .domains()
            .map(|(_, domain)| {
                (
                    domain.name.clone(),
                    eredu_core::MemoryLimit::Finite(CAPACITY),
                )
            }),
    );
    let cancellation = eredu_core::GenerationCancellationToken::new();
    // Reference outputs and each controlled trial use the same ordinary
    // reservations while retaining their own native completion owners.
    let mut references = Vec::new();
    for temperature in [0.0, 0.8] {
        eprintln!("rank {rank} reference temperature={temperature}");
        stage("reference generation");
        let settings = PreparedChatGenerationSettings {
            overrides: GenerationConfigOverrides {
                temperature: Some(temperature),
                max_new_tokens: Some(8),
                ..Default::default()
            },
            seed: 827,
            inference: TextInferencePolicy {
                memory_limits: physical_limits.clone(),
                ..Default::default()
            },
            ..Default::default()
        };
        let baseline = model
            .generate_tokens(
                vec![1, 2],
                eredu_core::TextGenerationConfig::new(
                    model.resolve_generation_config(settings.overrides).unwrap(),
                )
                .with_seed(settings.seed)
                .with_inference_policy(settings.inference.clone()),
            )
            .unwrap_or_else(baseline_failure)
            .map(|token| token.unwrap_or_else(baseline_failure).token_id().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(baseline.len(), 8);
        stage("reference reset");
        model.reset().unwrap();
        references.push((settings, baseline));
    }
    // Retire the reference model before the controlled trials. Those trials
    // share one model and its cumulative distributed agreement history.
    model.synchronize().unwrap();
    drop(model);
    stage("reference retirement");
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        safemlx::memory::clear_cache().unwrap();
        eredu_backend_mlx::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
        let retired = local_memory_snapshot().unwrap();
        if retired.funding_accounts == 0 && retired.reservations == 0 {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "reference model still retains request custody: {retired:?}"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    stage("controlled model construction");
    let mut model = load_fixture(&root, &family, &residency, &world, rank);
    stage("controlled model ready");
    let tokenizer = model
        .compile_managed_plain_text_source(TokenizerSourceInput::RetainedConfiguration)
        .unwrap();
    let source = model
        .compile_managed_chat_source(
            &tokenizer,
            ChatSourceInput::RetainedConfiguration,
            false,
            &cancellation,
        )
        .unwrap()
        .unwrap();
    for (settings, baseline) in references {
        eprintln!(
            "rank {rank} controlled temperature={:?}",
            settings.overrides.temperature
        );
        stage("chat preparation");
        let chat = model
            .prepare_chat(
                &source,
                &ChatTemplateRequest {
                    messages: vec![serde_json::json!({"role":"user","content":"word1 word2"})],
                    ..Default::default()
                },
                &physical_limits,
                &cancellation,
            )
            .unwrap()
            .unwrap();
        let request = |settings| {
            let mut request = PreparedChatRequest::new(&chat, settings);
            request.output_mode = PreparedChatOutputMode::Text;
            request
        };
        let trace = TraceLimits {
            per_record_bytes: 16384,
            total_bytes: 1 << 20,
        };
        let prepared = request(settings.clone());
        let prepared_trace = trace;
        let emit = |_: ControlledGenerationRecord| ControlFlow::Continue(());
        stage("snapshot trial preparation");
        let mut run = model
            .start_controlled_chat(prepared, prepared_trace, Default::default(), emit)
            .unwrap_or_else(|error| {
                panic!(
                    "{family}/{residency} rank {rank} snapshot trial at temperature {}: {error:?}",
                    settings.overrides.temperature.unwrap(),
                )
            })
            .expect("live control");
        stage("snapshot trial ready");
        run.enable_snapshots(
            SnapshotLimits {
                max_snapshots: 2,
                max_branches: 1,
                retained_bytes: 64 << 20,
                cumulative_copy_bytes: 256 << 20,
            },
            physical_limits.clone(),
            eredu_runtime::working_memory::WorkspaceCopyLimits::new(physical_limits.clone()),
        )
        .unwrap();
        run.step(emit).unwrap();
        run.step(emit).unwrap();
        stage("snapshot creation");
        let saved = run.snapshot(emit).unwrap();
        run.run(emit).unwrap();
        stage("snapshot trial completed");
        assert_eq!(run.token_ids(), baseline);
        let before = run.snapshot_usage().unwrap().cumulative_copy_bytes;
        stage("first snapshot restore");
        run.restore(&saved, emit).unwrap();
        stage("first snapshot restored");
        assert!(run.snapshot_usage().unwrap().cumulative_copy_bytes > before);
        run.run(emit).unwrap();
        assert_eq!(run.token_ids(), baseline);
        stage("snapshot fork");
        let mut child = run
            .fork(
                &saved,
                GenerationBranchOptions {
                    trace_limits: trace,
                    capture_limits: None,
                    sampling: None,
                    intervention: None,
                },
                emit,
            )
            .unwrap();
        stage("snapshot forked");
        // Keep both branches unfinished while exchanging their actual request
        // banks. A live dormant branch retains sources and spent slots, but may
        // not prevent the other branch from activating its next prediction.
        let before = run.snapshot_usage().unwrap().cumulative_copy_bytes;
        stage("second snapshot restore");
        run.restore(&saved, emit).unwrap();
        stage("second snapshot restored");
        assert!(run.snapshot_usage().unwrap().cumulative_copy_bytes > before);
        run.step(emit).unwrap();
        assert_eq!(run.token_ids(), &baseline[..3]);
        run.exchange(&mut child, emit).unwrap();
        stage("first branch exchange");
        run.step(emit).unwrap();
        assert_eq!(run.token_ids(), &baseline[..3]);
        run.exchange(&mut child, emit).unwrap();
        stage("second branch exchange");
        run.step(emit).unwrap();
        assert_eq!(run.token_ids(), &baseline[..4]);
        run.exchange(&mut child, emit).unwrap();
        stage("third branch exchange");
        run.run(emit).unwrap();
        assert_eq!(run.token_ids(), baseline);
        run.exchange(&mut child, emit).unwrap();
        stage("fourth branch exchange");
        run.run(emit).unwrap();
        assert_eq!(run.token_ids(), baseline);
        let consumed = run.snapshot_usage().unwrap().cumulative_copy_bytes;
        drop(child);
        assert_eq!(
            run.snapshot_usage().unwrap().cumulative_copy_bytes,
            consumed
        );
        drop(run);
        drop(saved);
        stage("snapshot trial reset");
        reset_admitted(&mut model, CAPACITY);
        for controlled in [false, true] {
            eprintln!("rank {rank} cancellation controlled={controlled}");
            stage("cancellation trial");
            let prepared = request(settings.clone());
            let prepared_trace = trace;
            let before = model.text_preparation_usage().unwrap();
            let cancelled = if controlled {
                let mut run = model
                    .start_controlled_chat(prepared, prepared_trace, Default::default(), emit)
                    .unwrap()
                    .expect("live control");
                run.run(|record| {
                    if rank == 0
                        && matches!(
                            record.event.progress(),
                            Some(ObservedGenerationEvent::Token { .. })
                        )
                    {
                        ControlFlow::Break(())
                    } else {
                        ControlFlow::Continue(())
                    }
                })
                .unwrap();
                assert_eq!(run.status(), GenerationStatus::Cancelled);
                run.token_ids().to_vec()
            } else {
                (|| -> Result<Vec<u32>, ControlledGenerationError> {
                    let mut emit = |record: ControlledGenerationRecord| {
                        if rank == 0
                            && matches!(
                                record.event.progress(),
                                Some(ObservedGenerationEvent::Token { .. })
                            )
                        {
                            ControlFlow::Break(())
                        } else {
                            ControlFlow::Continue(())
                        }
                    };
                    let mut run = model
                        .start_controlled_chat(
                            prepared,
                            prepared_trace,
                            GenerationControlHandle::new(Default::default()),
                            &mut emit,
                        )?
                        .expect("live control");
                    run.run(&mut emit)?;
                    Ok(run.token_ids().to_vec())
                })()
                .unwrap()
            };
            assert_eq!(
                cancelled,
                baseline[..1],
                "peer cancellation must retain the same committed prefix"
            );
            let after = model.text_preparation_usage().unwrap();
            assert!(after.attempts > before.attempts);
            reset_admitted(&mut model, CAPACITY);

            if controlled {
                let prepared = request(settings.clone());
                let prepared_trace = trace;
                let mut run = model
                    .start_controlled_chat(prepared, prepared_trace, Default::default(), emit)
                    .unwrap()
                    .expect("live control");
                run.run(|record| {
                    if rank == 0
                        && matches!(
                            record.event.progress(),
                            Some(ObservedGenerationEvent::Lifecycle {
                                next_prediction: 1,
                                ..
                            })
                        )
                    {
                        ControlFlow::Break(())
                    } else {
                        ControlFlow::Continue(())
                    }
                })
                .unwrap();
                assert_eq!(run.status(), GenerationStatus::Cancelled);
                assert_eq!(run.token_ids(), &baseline[..1]);
                drop(run);
                reset_admitted(&mut model, CAPACITY);
            }

            for boundary_record in [false, true] {
                eprintln!(
                    "rank {rank} delivery controlled={controlled} boundary={boundary_record}"
                );
                stage("delivery trial");
                // Reject rank zero's first token or its following lifecycle/terminal
                // record. Other participants retain their ordinary quota.
                let trial_settings = if boundary_record && !controlled {
                    let mut trial = settings.clone();
                    trial.overrides.max_new_tokens = Some(1);
                    trial
                } else {
                    settings.clone()
                };
                let prepared = request(trial_settings.clone());
                let prepared_trace = trace;
                let mut initial_bytes = 0;
                if controlled {
                    let mut run = model
                        .start_controlled_chat(
                            prepared,
                            prepared_trace,
                            Default::default(),
                            |record| {
                                initial_bytes += serde_json::to_vec(&record).unwrap().len() as u64;
                                ControlFlow::Continue(())
                            },
                        )
                        .unwrap()
                        .expect("live control");
                    if boundary_record {
                        run.step(|record| {
                            if !matches!(
                                record.event.progress(),
                                Some(ObservedGenerationEvent::Lifecycle { .. })
                            ) {
                                initial_bytes += serde_json::to_vec(&record).unwrap().len() as u64;
                            }
                            ControlFlow::Continue(())
                        })
                        .unwrap();
                    }
                    drop(run);
                } else {
                    (|| -> Result<Vec<u32>, ControlledGenerationError> {
                        let mut emit = |record: ControlledGenerationRecord| {
                            if (boundary_record
                                && !matches!(
                                    record.event.progress(),
                                    Some(ObservedGenerationEvent::Completed { .. })
                                ))
                                || (!boundary_record
                                    && matches!(
                                        record.event.progress(),
                                        Some(ObservedGenerationEvent::Started { .. })
                                    ))
                            {
                                initial_bytes += serde_json::to_vec(&record).unwrap().len() as u64;
                            }
                            ControlFlow::Continue(())
                        };
                        let mut run = model
                            .start_controlled_chat(
                                prepared,
                                prepared_trace,
                                GenerationControlHandle::new(Default::default()),
                                &mut emit,
                            )?
                            .expect("live control");
                        run.run(&mut emit)?;
                        Ok(run.token_ids().to_vec())
                    })()
                    .unwrap();
                }
                reset_admitted(&mut model, CAPACITY);
                let limits = if rank == 0 {
                    TraceLimits {
                        total_bytes: initial_bytes + 64,
                        ..trace
                    }
                } else {
                    trace
                };
                let prepared = request(trial_settings.clone());
                let prepared_trace = limits;
                let before = model.text_preparation_usage().unwrap();
                let (local_limit, error): (bool, Box<dyn std::error::Error>) = if controlled {
                    let mut run = model
                        .start_controlled_chat(prepared, prepared_trace, Default::default(), emit)
                        .unwrap()
                        .expect("live control");
                    let error = run.run(emit).unwrap_err();
                    let local = is_cumulative_encoded_limit(&error);
                    (local, Box::new(error))
                } else {
                    let error = (|| -> Result<Vec<u32>, ControlledGenerationError> {
                        let mut emit = |_| ControlFlow::Continue(());
                        let mut run = model
                            .start_controlled_chat(
                                prepared,
                                prepared_trace,
                                GenerationControlHandle::new(Default::default()),
                                &mut emit,
                            )?
                            .expect("live control");
                        run.run(&mut emit)?;
                        Ok(run.token_ids().to_vec())
                    })()
                    .unwrap_err();
                    let local = is_cumulative_encoded_limit(&error);
                    (local, Box::new(error))
                };
                if rank == 0 {
                    assert!(
                        local_limit,
                        "rank zero lost its original delivery budget error: {error}"
                    );
                } else {
                    let mut cause: Option<&(dyn std::error::Error + 'static)> =
                        Some(error.as_ref());
                    let mut rejected = false;
                    while let Some(error) = cause {
                        rejected |= error
                            .downcast_ref::<eredu_core::run_preparation::TextPreparationRejected>()
                            .is_some_and(|error| {
                                error.rank == 0
                                && error.stage
                                    == eredu_core::run_preparation::TextPreparationStage::Delivery
                            });
                        cause = error.source();
                    }
                    assert!(rejected, "peer lost its typed delivery rejection: {error}");
                }
                let after = model.text_preparation_usage().unwrap();
                assert!(after.attempts > before.attempts);
                reset_admitted(&mut model, CAPACITY);
                let prepared = request(settings.clone());
                let prepared_trace = trace;
                let retry = (|| -> Result<Vec<u32>, ControlledGenerationError> {
                    let mut emit = |_| ControlFlow::Continue(());
                    let mut run = model
                        .start_controlled_chat(
                            prepared,
                            prepared_trace,
                            GenerationControlHandle::new(Default::default()),
                            &mut emit,
                        )?
                        .expect("live control");
                    run.run(&mut emit)?;
                    Ok(run.token_ids().to_vec())
                })()
                .unwrap();
                assert_eq!(
                    retry, baseline,
                    "corrected delivery must reproduce the baseline after reset"
                );
                reset_admitted(&mut model, CAPACITY);
            }
        }
        stage("temperature completed");
    }
    stage("controlled model retirement");
    drop(source);
    drop(tokenizer);
    model.synchronize().unwrap();
    drop(model);
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        safemlx::memory::clear_cache().unwrap();
        eredu_backend_mlx::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
        let retired = local_memory_snapshot().unwrap();
        if retired.funding_accounts == 0 && retired.reservations == 0 {
            assert_eq!(retired.unquoted_owners, 0);
            break;
        }
        assert!(
            Instant::now() < deadline,
            "controlled model still retains request custody: {retired:?}"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    stage("complete");
}
