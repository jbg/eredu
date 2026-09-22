#[cfg(test)]
use crate::memory_fixture::LedgerFixture as _;
#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
#[ignore = "requires native Metal execution"]
fn prepared_workspace_native_routed_and_composite_text_use_retained_routes() {
    use eredu_core::{InferenceGeometry, OutputDemand};
    for case in 0..3 {
        let checkpoint = tempfile::tempdir().unwrap();
        match case {
            0 => write_deepseek_v4_target_component_fixture(checkpoint.path(), false),
            1 => write_muse_glimmer_component_fixture(checkpoint.path(), true, false),
            _ => write_inkling_dense_multimodal_fixture(checkpoint.path()),
        }
        let execution = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = execution.stream();
        let weights = fixture_weights_stream(stream);
        let backend = crate::native::backend(stream, &weights);
        let prepared = load_model(&backend, checkpoint.path(), MlxLoadRequest::default())
            .unwrap()
            .into_inner();
        let (mut executable, _) = prepared.into_execution_parts();
        for chunk in [1, 3, 9] {
            let geometry = InferenceGeometry {
                batch_size: 1,
                cached_positions: 0,
                input_positions: 9,
                max_output_tokens: 3,
                prefill_chunk_positions: chunk,
                output: OutputDemand::LastPosition,
            };
            let before = executable.erased().state_snapshot();
            let report = executable
                .quote_replicated_resident_text(geometry)
                .unwrap_or_else(|error| panic!("case {case}, chunk {chunk}: {error}"));
            assert_eq!(report.completed_spans(), 9u64.div_ceil(chunk) + 3);
            assert_eq!(executable.erased().state_snapshot(), before);
            // Native operators without a certified bound remain explicit gaps.
            assert_eq!(
                report.transient().bytes().is_none(),
                report.first_gap().is_some()
            );
            assert!(
                report.transient().bytes().is_some(),
                "prepared case {case}, chunk {chunk}: {report:?}"
            );
            assert!(report.retained_peak_bytes().is_some());
        }
        let tokens = Array::from_slice(&[1_u32, 3, 2, 4, 5, 2, 1, 3, 4], &[1, 9]);
        let parts = [text_input_part(&tokens)];
        let input = crate::backend::runtime::media::input::ModelInput::new(&parts)
            .with_prefill_chunk_positions(std::num::NonZeroU64::new(3).unwrap());
        let output = executable.prefill(input, stream).unwrap();
        output.evaluated().unwrap();
        for token in [2_u32, 3] {
            let output = executable
                .erased_mut()
                .decode(&Array::from_slice(&[token], &[1, 1]), stream)
                .unwrap();
            output.evaluated().unwrap();
        }
        let geometry = InferenceGeometry {
            batch_size: 1,
            cached_positions: 11,
            input_positions: 5,
            max_output_tokens: 3,
            prefill_chunk_positions: 3,
            output: OutputDemand::LastPosition,
        };
        let before = executable.erased().state_snapshot();
        let report = executable
            .quote_replicated_resident_text(geometry)
            .unwrap_or_else(|error| panic!("case {case}, cached state: {error}"));
        assert_eq!(report.completed_spans(), 5);
        assert_eq!(executable.erased().state_snapshot(), before);
        assert!(
            report.transient().bytes().is_some(),
            "cached prepared case {case}: {report:?}"
        );
        assert!(report.retained_peak_bytes().is_some());
        eprintln!(
            "cached prepared workspace case={case} first_gap={:?}",
            report.first_gap()
        );
        eprintln!(
            "cached prepared workspace case={case} bound={}",
            format!("{:?}", report.transient())
                .chars()
                .take(1500)
                .collect::<String>()
        );
        let continuation = Array::from_slice(&[2_u32, 1, 3, 4, 2], &[1, 5]);
        let parts = [text_input_part(&continuation)];
        let input = crate::backend::runtime::media::input::ModelInput::new(&parts)
            .with_prefill_chunk_positions(std::num::NonZeroU64::new(3).unwrap());
        executable
            .prefill(input, stream)
            .unwrap()
            .evaluated()
            .unwrap();
        executable
            .erased_mut()
            .decode(&Array::from_slice(&[3_u32], &[1, 1]), stream)
            .unwrap()
            .evaluated()
            .unwrap();
    }
}

#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
#[ignore = "requires native Metal execution"]
fn prepared_workspace_native_executable_retains_selection_and_projects_live_state() {
    use eredu_core::{InferenceGeometry, OutputDemand};

    let checkpoint = tempfile::tempdir().unwrap();
    let artifact = checkpoint.path().join("original");
    std::fs::create_dir(&artifact).unwrap();
    write_qwen_hybrid_component_fixture(&artifact, FixtureFamily::Qwen35);
    let execution = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let stream = execution.stream();
    let weights = fixture_weights_stream(stream);
    let backend = crate::native::backend(stream, &weights);
    let prepared = load_model(&backend, &artifact, MlxLoadRequest::default())
        .unwrap()
        .into_inner();
    let (mut executable, _target) = prepared.into_execution_parts();
    // The executable must use its retained sources after the original path no
    // longer exists, including when it constructs a fresh metadata execution.
    std::fs::rename(&artifact, checkpoint.path().join("moved")).unwrap();
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 5,
        max_output_tokens: 3,
        prefill_chunk_positions: 2,
        output: OutputDemand::LastPosition,
    };
    let report = executable.quote_replicated_resident_text(geometry).unwrap();
    assert_eq!(report.completed_spans(), 6);
    assert!(
        matches!(
            report.transient(),
            eredu_core::WorkspaceBound::Bounded { .. }
        ),
        "{report:?}"
    );
    assert!(report.retained_peak_bytes().is_some());
    let sampling = eredu_core::resolve_generation_config(
        None,
        eredu_core::GenerationConfigOverrides {
            temperature: Some(0.7),
            ..Default::default()
        },
    )
    .unwrap();
    for adaptive in [false, true] {
        let config = eredu_core::TextGenerationConfig::new(sampling);
        let config = if adaptive {
            config.with_mirostat_v2(5.0, 0.1).unwrap()
        } else {
            config
        };
        let combined = executable
            .quote_replicated_resident_text_with_sampling(
                geometry,
                config,
                &eredu_core::TokenFilter::Allowed(vec![true, false, true]),
            )
            .unwrap();
        assert_eq!(combined.equations.geometry(), geometry);
        assert_eq!(combined.sampling.steps, 3);
        assert!(combined.sampling.peak.bytes().is_some());
        assert_eq!(combined.sampling.final_history_bytes, 16);
        let prompt = executable
            .quote_text_prompt_workspace(geometry, Some(128))
            .unwrap();
        assert_eq!(prompt.geometry(), combined.equations.geometry());
        let logical = geometry.batch_size * geometry.input_positions * 4;
        let minimal = executable.quote_text_prompt_workspace(geometry, Some(logical)).unwrap();
        let host = prompt.host_peak_bytes().unwrap();
        assert_eq!(host.checked_sub(minimal.host_peak_bytes().unwrap()), Some(128 - logical));
        let identity = eredu_runtime::input::TextInputIdentityPlan::new(
            geometry.batch_size, geometry.input_positions).unwrap();
        assert!(host >= 128 + identity.peak_bytes() + safemlx::physical_backing_control_bytes() as u64);
        assert!(prompt.tensor_peak_bytes().is_some());
        assert_eq!(prompt.peak().bytes(), host.checked_add(prompt.tensor_peak_bytes().unwrap()));
        assert!(prompt.peak().bytes().unwrap() >= 128 + geometry.input_positions * 4);
    }

    // Compose certified native prompt, transaction, sampling and selected-state
    // bounds. This isolated fixture has no capture, snapshots, active prediction
    // or streaming materialization. Its pool tests request ownership; global
    // managed-domain participation remains a separate admission prerequisite.
    let capability = executable.erased().capability_estimate();
    let admission_request = eredu_core::AdmissionRequest {
        input: eredu_core::InputTokenCount::text(5),
        max_output_tokens: 3,
        batch_size: 1,
        additional_headroom: crate::memory_fixture::headroom(0),
        memory_limits: Default::default(),
    };
    let zero = || {
        eredu_core::WorkspaceBound::bounded(0, "fixture has resident parameters, no enclosing capture, snapshots or prediction; all transaction state work included by trace")
    };
    let outside = crate::memory_fixture::workspace(eredu_core::ExecutionWorkspaceEstimate { physical_domains: None,
        geometry,
        activations: zero(),
        attention: zero(),
        vocabulary: zero(),
        state_update: zero(),
        materialization: zero(),
        retained: zero(),
    });
    let sampling = eredu_core::resolve_generation_config(
        None,
        eredu_core::GenerationConfigOverrides {
            max_new_tokens: Some(3),
            temperature: Some(0.7),
            ..Default::default()
        },
    )
    .unwrap();
    let config = eredu_core::TextGenerationConfig::new(sampling);
    let controller = eredu_core::TextControllerWorkspace {
        filter: (&eredu_core::TokenFilter::All).into(),
        additional_host_bytes: 32,
    };
    let state = executable
        .quote_text_preparation_workspace(geometry, Some(128), config.clone(), controller, outside.clone())
        .unwrap();
    assert!(state
        .selected_state_backing
        .as_ref()
        .unwrap()
        .bytes()
        .is_some());
    assert_eq!(
        state.completeness,
        eredu_core::EstimationCompleteness::Conservative
    );
    let no_source = executable
        .quote_text_preparation_workspace(geometry, None, config.clone(), controller, outside.clone())
        .unwrap();
    assert_eq!(
        no_source.completeness,
        eredu_core::EstimationCompleteness::PersistentStateOnly
    );
    let mut missing = outside.clone();
    missing.materialization = eredu_core::WorkspaceBound::Unknown {
        reason: "fixture omitted enclosing cost".into(),
    };
    assert_eq!(
        executable
            .quote_text_preparation_workspace(geometry, Some(128), config.clone(), controller, missing,)
            .unwrap()
            .completeness,
        eredu_core::EstimationCompleteness::PersistentStateOnly
    );
    let larger = executable
        .quote_text_preparation_workspace(
            geometry,
            Some(256),
            config,
            eredu_core::TextControllerWorkspace {
                additional_host_bytes: 96,
                ..controller
            },
            outside,
        )
        .unwrap();
    assert_eq!(
        larger
            .execution_workspace
            .as_ref()
            .unwrap()
            .peak_bytes()
            .unwrap()
            .unwrap()
            - state
                .execution_workspace
                .as_ref()
                .unwrap()
                .peak_bytes()
                .unwrap()
                .unwrap(),
        192
    );
    let eredu_core::AdmissionResult::Admitted(admission) = eredu_core::apply_admission_policy(
        capability.capabilities(),
        admission_request,
        state)
    .unwrap() else {
        panic!("native fixture admission")
    };
    let pool = crate::memory_fixture::ledger(1 << 30, 0).unwrap();
    let request: eredu_runtime::working_memory::InferenceRequest = pool
        .reserve(
            executable.erased().inference_execution_identity(),
            &admission,
        )
        .unwrap()
        .into();
    let charged = request.memory_reservation().requirements().get(crate::memory_fixture::topology().host_domain()).unwrap().total().unwrap();
    let tokens = Array::from_slice(&[1_u32, 3, 2, 4, 5], &[1, 5]);
    let parts = [text_input_part(&tokens)];
    let prompt = crate::composition::mlx::MlxModelInput::from(
        crate::backend::runtime::media::input::ModelInput::new(&parts),
    )
    .with_inference_request(request.clone());
    let before = executable.erased().state_snapshot();
    let foreign = pool
        .reserve(
            &eredu_runtime::working_memory::InferenceExecutionIdentity::default(),
            &admission,
        )
        .unwrap();
    let wrong_target = crate::composition::mlx::MlxModelInput::from(
        crate::backend::runtime::media::input::ModelInput::new(&parts),
    )
    .with_inference_request(foreign.into());
    let error = wrong_target
        .with_borrowed(|input| executable.prefill(input, stream))
        .unwrap_err();
    assert!(error.model_state_preserved());
    assert_eq!(executable.erased().state_snapshot(), before);
    drop(wrong_target);
    let wrong_chunk = prompt
        .clone()
        .with_prefill_chunk_positions(std::num::NonZeroU64::new(1).unwrap());
    let error = wrong_chunk
        .with_borrowed(|input| executable.prefill(input, stream))
        .unwrap_err();
    assert!(error.model_state_preserved());
    assert_eq!(executable.erased().state_snapshot(), before);
    drop(wrong_chunk);
    assert_eq!(pool.fixture_host_charge().unwrap(), charged);
    let scores = prompt
        .with_borrowed(|input| executable.prefill(input, stream))
        .unwrap();
    drop(prompt);
    drop(request);
    assert_eq!(pool.fixture_host_charge().unwrap(), charged);
    scores.evaluated().unwrap();
    let before = executable.erased().state_snapshot();
    assert!(before.iter().all(|(position, _)| *position == 5));
    let continued = InferenceGeometry {
        cached_positions: 5,
        ..geometry
    };
    let report = executable
        .quote_replicated_resident_text(continued)
        .unwrap();
    assert_eq!(report.completed_spans(), 6);
    assert!(
        matches!(
            report.transient(),
            eredu_core::WorkspaceBound::Bounded { .. }
        ),
        "{report:?}"
    );
    assert_eq!(executable.erased().state_snapshot(), before);
    assert!(executable
        .quote_replicated_resident_text(InferenceGeometry {
            cached_positions: 4,
            ..continued
        },)
        .is_err());
    for (step, token) in [6_u32, 2, 1].into_iter().enumerate() {
        let output = executable
            .erased_mut()
            .decode(&Array::from_slice(&[token], &[1, 1]), stream)
            .unwrap();
        let evaluated = output.evaluated().unwrap();
        let values = evaluated.try_as_slice::<f32>().unwrap();
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(values.iter().any(|value| value.abs() > 1e-6));
        assert!(executable
            .erased()
            .state_snapshot()
            .iter()
            .all(|(position, _)| *position == 6 + step as i32));
        assert_eq!(pool.fixture_host_charge().unwrap(), charged);
    }
    let before = executable.erased().state_snapshot();
    assert!(executable
        .erased_mut()
        .decode(&Array::from_slice(&[3_u32], &[1, 1]), stream)
        .is_err());
    assert_eq!(executable.erased().state_snapshot(), before);
    drop(scores);
    executable.reset_cache_distributed().unwrap();
    assert_eq!(pool.fixture_host_charge().unwrap(), 0);
}
