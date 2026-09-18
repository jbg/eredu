fn provider_failure_cause(mut error: &(dyn std::error::Error + 'static)) {
    let mut chain = Vec::new();
    loop {
        chain.push(error.to_string());
        if let Some(native) = error.downcast_ref::<safemlx::error::Exception>() {
            assert!(native.what().contains("reshape"), "{native}");
            assert_eq!(
                (
                    native.what().to_owned(),
                    native.location().file().to_owned(),
                    native.location().line()
                ),
                crate::tests::support::provider_failure::cause(),
            );
            assert!(native.location().line() > 0);
            return;
        }
        error = error
            .source()
            .unwrap_or_else(|| panic!("original native provider cause missing from {chain:?}"));
    }
}

fn provider_failure_capture(
    runtime: &ModelRuntime<MlxBackend<'_>>,
) -> eredu_core::capture::AdmittedCapturePlan {
    use eredu_core::{capture::*, TextGenerationBackend as _};
    let discovery = MlxBackend::capture_discovery(runtime).unwrap();
    let allowance = CaptureUsage {
        captures: 100,
        // Includes bounded transport envelopes for every producer/receiver.
        // These are work credits, not physical allocations.
        retained_bytes: 512 << 20,
        host_bytes: 4 << 30,
        encoded_bytes: 512 << 20,
    };
    CapturePlan {
        schema_version: 1,
        selections: vec![CaptureSelection {
            id: "input-before-provider".into(),
            path: "readout.embedding".into(),
            schedule: CaptureSchedule::default(),
            slices: vec![],
            transform: CaptureTransform::FullTensor,
        }],
        limits: CaptureLimits {
            per_step: allowance,
            cumulative: allowance.checked_mul(4).unwrap(),
            physical_native_bytes: None,
            on_limit: CaptureLimitPolicy::Fail,
        },
    }
    .admit(
        &discovery.catalog,
        &discovery.support,
        &discovery.support.capture,
        CaptureRequestShape {
            batch: 1,
            prompt_tokens: 2,
            max_predictions: 3,
        },
    )
    .unwrap()
}

fn verify_loaded_provider_failures(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    operator: crate::tests::support::provider_failure::Operator,
) {
    use crate::tests::support::provider_failure;
    let sampling = eredu_core::resolve_generation_config(
        None,
        eredu_core::GenerationConfigOverrides {
            max_new_tokens: Some(3),
            temperature: Some(0.0),
            ..Default::default()
        },
    )
    .unwrap();
    let baseline = {
        let run = eredu_core::TextGeneration::new(
            runtime,
            vec![1, 2],
            TextGenerationConfig::new(sampling),
        )
        .unwrap();
        run.map(|token| token.unwrap().token_id().unwrap())
            .collect::<Vec<_>>()
    };
    assert_eq!(baseline.len(), 3);
    runtime.synchronize().unwrap();
    runtime.reset().unwrap();
    let capture = provider_failure_capture(runtime);
    for controlled in [false, true] {
        for preceding_predictions in [0, 1] {
            eprintln!("provider failure {operator:?}: controlled={controlled}, preceding={preceding_predictions}");
            if controlled {
                let mut run = eredu_core::ControlledTextGeneration::new(
                    runtime,
                    vec![1, 2],
                    TextGenerationConfig::new(sampling),
                    ComponentCaptureController::default(),
                )
                .unwrap();
                run.enable_capture(capture.clone()).unwrap();
                for expected in &baseline[..preceding_predictions] {
                    assert_eq!(run.next().unwrap().unwrap().token_id(), *expected);
                    let step = run.take_captured_delivery().unwrap().unwrap();
                    assert_eq!(
                        step.outcome,
                        eredu_core::capture::CaptureStepOutcome::Committed
                    );
                    assert!(!step.records.is_empty());
                    assert!(step.step_usage.captures > 0);
                }
                let _fault = provider_failure::arm(operator, 0);
                let error = run
                    .next()
                    .expect("failed prediction")
                    .err()
                    .expect("provider error");
                assert_eq!(provider_failure::hits(), 1);
                provider_failure_cause(&error);
                // Earlier captures remain actual measurements, explicitly belonging
                // to an aborted forward; all their reservations remain charged.
                let failed = run.take_captured_delivery().unwrap().unwrap();
                assert_eq!(
                    failed.outcome,
                    eredu_core::capture::CaptureStepOutcome::Aborted
                );
                assert_eq!(failed.prediction_index, preceding_predictions as u64);
                assert!(failed.step_usage.captures > 0);
                assert!(failed.step_usage.host_bytes > 0);
                assert!(failed.partitions.is_empty());
                assert!(run.take_captured_delivery().unwrap().is_none());
                assert!(run.next().is_none());
            } else {
                let mut run = eredu_core::TextGeneration::new(
                    runtime,
                    vec![1, 2],
                    TextGenerationConfig::new(sampling),
                )
                .unwrap();
                for expected in &baseline[..preceding_predictions] {
                    assert_eq!(run.next().unwrap().unwrap().token_id().unwrap(), *expected);
                }
                let _fault = provider_failure::arm(operator, 0);
                let error: eredu_core::BackendFailure =
                    run.next().unwrap().err().expect("provider error");
                assert_eq!(provider_failure::hits(), 1);
                provider_failure_cause(&error);
                assert!(run.next().is_none());
            }
            // Generation drop and native synchronization establish completion;
            // the presence of an error alone does not authorize reuse.
            runtime.synchronize().unwrap();
            runtime.reset().unwrap();
            let replay = eredu_core::TextGeneration::new(
                runtime,
                vec![1, 2],
                TextGenerationConfig::new(sampling),
            )
            .unwrap()
            .map(|token| token.unwrap().token_id().unwrap())
            .collect::<Vec<_>>();
            assert_eq!(replay, baseline);
            runtime.synchronize().unwrap();
            runtime.reset().unwrap();
        }
    }
}

fn verify_provider_failures(device_type: DeviceType) {
    use crate::tests::support::provider_failure::Operator;
    let stream = Stream::new_with_device(&Device::new(device_type, 0));
    let source_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    for operator in [Operator::Gated, Operator::Relu2, Operator::Linear] {
        let checkpoint = tempfile::tempdir().unwrap();
        match operator {
            Operator::Gated => write_qwen_fixture(checkpoint.path(), "qwen3_moe"),
            Operator::Relu2 => write_nemotron_fixture(checkpoint.path()),
            Operator::Linear => write_k2_fixture(checkpoint.path(), true),
        }
        for residency in 0..4 {
            eprintln!("provider fixture {operator:?} residency={residency}");
            let ordinary = match residency {
                0 | 1 => OrdinaryWeightResidency::FullyResident,
                2 => OrdinaryWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
                    OffloadConfig::new(None, None, 1).unwrap(),
                )),
                3 => OrdinaryWeightResidency::DenseDiskStream(
                    DenseDiskStreamLoadOptions::new(u64::MAX, u64::MAX, 1, 1).unwrap(),
                ),
                _ => unreachable!(),
            };
            let request = if residency == 0 {
                MlxLoadRequest::default()
            } else {
                MlxLoadRequest::from_normalized(
                    eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(
                        WeightResidency::with_independent_parameter_banks(
                            ordinary,
                            ParameterBankLoadOptions::default(),
                        ),
                    ),
                )
            };
            let backend = MlxBackend::new(&stream, &source_stream);
            let prepared = load_model(&backend, checkpoint.path(), request).unwrap();
            let mut runtime = ModelRuntime::from_prepared(backend, prepared).unwrap();
            verify_loaded_provider_failures(&mut runtime, operator);
        }
    }
}

#[test]
#[ignore = "requires native MLX CPU execution"]
fn provider_failures_retain_native_causes_cpu() {
    verify_provider_failures(DeviceType::Cpu);
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires native Metal execution"]
fn provider_failures_retain_native_causes_metal() {
    verify_provider_failures(DeviceType::Gpu);
}

fn provider_peer_failure(mut error: &(dyn std::error::Error + 'static)) {
    let mut chain = Vec::new();
    loop {
        chain.push(error.to_string());
        if error.is::<eredu_runtime::expert::ProviderAgreementRejected>()
            || matches!(
                error.downcast_ref::<eredu_architectures::RoutedMechanismExecutionError>(),
                Some(eredu_architectures::RoutedMechanismExecutionError::ProviderRejected)
            )
            || matches!(
                error.downcast_ref::<eredu_runtime::PartitionExecutionError>(),
                Some(eredu_runtime::PartitionExecutionError::RemotePhaseFailure(
                    eredu_runtime::DistributedExecutionPhase::Execution
                ))
            )
        {
            return;
        }
        error = error.source().unwrap_or_else(|| {
            panic!("typed provider peer rejection missing (a timeout is not agreement): {chain:?}")
        });
    }
}

fn verify_partition_provider_failure(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    family: FixtureFamily,
    topology: eredu_core::ParallelRankTopology,
) {
    use crate::tests::support::provider_failure::{self, Operator};
    let operator = match family {
        FixtureFamily::Qwen3Moe => Operator::Gated,
        FixtureFamily::NemotronH => Operator::Relu2,
        FixtureFamily::K2Mova => Operator::Linear,
        _ => panic!("unexpected provider fault fixture"),
    };
    // Nemotron's routed E block and K2's last MoVA block belong to the final
    // fixture stage. Qwen starts with a routed block on the first stage.
    let failure_stage = if family == FixtureFamily::Qwen3Moe {
        0
    } else {
        topology.pipeline_parallel_size() - 1
    };
    let failure_rank = topology.topology().rank_for(eredu_core::ParallelCoordinates::new(
        0, failure_stage, 0, 0,
    )).unwrap();
    let rank = topology.global_rank();
    let sampling = eredu_core::resolve_generation_config(
        None,
        eredu_core::GenerationConfigOverrides {
            max_new_tokens: Some(3),
            temperature: Some(0.0),
            ..Default::default()
        },
    )
    .unwrap();
    let baseline = eredu_core::TextGeneration::new(
        runtime, vec![1, 2], TextGenerationConfig::new(sampling),
    ).unwrap().map(|token| token.unwrap().token_id().unwrap()).collect::<Vec<_>>();
    assert_eq!(baseline.len(), 3);
    runtime.synchronize().unwrap();
    runtime.reset().unwrap();
    let capture = provider_failure_capture(runtime);
    for controlled in [false, true] {
        for preceding_predictions in [0, 1] {
            eprintln!("rank {rank} provider {operator:?}, origin={failure_rank}, controlled={controlled}, preceding={preceding_predictions}");
            // Ordinary execution has no observer. Controlled execution captures
            // embeddings only: neither path installs a provider unit observer.
            if controlled {
                let mut run = eredu_core::ControlledTextGeneration::new(
                    runtime, vec![1, 2], TextGenerationConfig::new(sampling),
                    ComponentCaptureController::default(),
                ).unwrap();
                run.enable_capture(capture.clone()).unwrap();
                for expected in &baseline[..preceding_predictions] {
                    assert_eq!(run.next().unwrap().unwrap().token_id(), *expected);
                    assert_eq!(run.take_captured_delivery().unwrap().unwrap().outcome,
                        eredu_core::capture::CaptureStepOutcome::Committed);
                }
                let _fault = (rank == failure_rank).then(|| provider_failure::arm(operator, 0));
                let error = run.next().expect("failed prediction").err().expect("agreed provider failure");
                if rank == failure_rank {
                    assert_eq!(provider_failure::hits(), 1);
                    provider_failure_cause(&error);
                } else {
                    provider_peer_failure(&error);
                }
                let failed = run.take_captured_delivery().unwrap().unwrap();
                assert_eq!(failed.outcome, eredu_core::capture::CaptureStepOutcome::Aborted);
                assert_eq!(failed.prediction_index, preceding_predictions as u64);
                assert!(run.take_captured_delivery().unwrap().is_none());
                assert!(run.next().is_none());
            } else {
                let mut run = eredu_core::TextGeneration::new(
                    runtime, vec![1, 2], TextGenerationConfig::new(sampling),
                ).unwrap();
                for expected in &baseline[..preceding_predictions] {
                    assert_eq!(run.next().unwrap().unwrap().token_id().unwrap(), *expected);
                }
                let _fault = (rank == failure_rank).then(|| provider_failure::arm(operator, 0));
                let error = run.next().expect("failed prediction").err().expect("agreed provider failure");
                if rank == failure_rank {
                    assert_eq!(provider_failure::hits(), 1);
                    provider_failure_cause(&error);
                } else {
                    provider_peer_failure(&error);
                }
                assert!(run.next().is_none());
            }
            runtime.synchronize().unwrap();
            runtime.reset().unwrap();
            let replay = eredu_core::TextGeneration::new(
                runtime, vec![1, 2], TextGenerationConfig::new(sampling),
            ).unwrap().map(|token| token.unwrap().token_id().unwrap()).collect::<Vec<_>>();
            assert_eq!(replay, baseline);
            runtime.synchronize().unwrap();
            runtime.reset().unwrap();
        }
    }
}

#[test]
#[ignore = "requires native MLX Ring and two loopback processes"]
fn ring_provider_failure_without_unit_observer_expert() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3Moe,
        "ep",
        WorkerMode::OpaqueProviderFailure,
    );
}

#[test]
#[ignore = "requires native MLX Ring and two loopback processes"]
fn ring_provider_failure_without_unit_observer_tensor() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3Moe,
        "tp",
        WorkerMode::OpaqueProviderFailure,
    );
}

macro_rules! provider_failure_matrix {
    ($name:ident, $family:ident, $residency:ident) => {
        #[test]
        #[ignore = "requires native MLX Ring with up to eight loopback processes"]
        fn $name() {
            for axes in [None, Some("tp"), Some("ep"), Some("tp-ep"),
                         Some("tp-pp"), Some("pp-ep"), Some("tp-pp-ep")] {
                let checkpoint = tempfile::tempdir().unwrap();
                let family = FixtureFamily::$family;
                match family {
                    FixtureFamily::Qwen3Moe => write_qwen_fixture(checkpoint.path(), "qwen3_moe"),
                    FixtureFamily::NemotronH => write_nemotron_fixture(checkpoint.path()),
                    FixtureFamily::K2Mova => write_k2_fixture(checkpoint.path(), true),
                    _ => unreachable!(),
                }
                let path = checkpoint.path().to_owned();
                eprintln!("provider matrix {family:?}, {}, axes={axes:?}", stringify!($residency));
                run_ring_pipeline_processes(WorkerResidency::$residency, family,
                    WorkerMode::OpaqueProviderFailure, checkpoint, path, axes);
            }
        }
    };
}
provider_failure_matrix!(ring_provider_failure_gated_resident, Qwen3Moe, FullyResident);
provider_failure_matrix!(ring_provider_failure_gated_host, Qwen3Moe, LayerwiseHost);
provider_failure_matrix!(ring_provider_failure_gated_disk, Qwen3Moe, DenseDiskStream);
provider_failure_matrix!(ring_provider_failure_relu2_resident, NemotronH, FullyResident);
provider_failure_matrix!(ring_provider_failure_relu2_host, NemotronH, LayerwiseHost);
provider_failure_matrix!(ring_provider_failure_relu2_disk, NemotronH, DenseDiskStream);
provider_failure_matrix!(ring_provider_failure_linear_resident, K2Mova, FullyResident);
provider_failure_matrix!(ring_provider_failure_linear_host, K2Mova, LayerwiseHost);
provider_failure_matrix!(ring_provider_failure_linear_disk, K2Mova, DenseDiskStream);

#[test]
#[ignore = "requires native MLX Ring and eight loopback processes"]
fn ring_provider_failure_eight_rank_wave() {
    run_ring_cartesian_pipeline_mode(
        false,
        FixtureFamily::Qwen3Moe,
        "tp-pp-ep",
        WorkerMode::OpaqueProviderFailure,
    );
}
