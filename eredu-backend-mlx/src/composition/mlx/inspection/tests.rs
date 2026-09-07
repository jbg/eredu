use super::*;

mod native_control;

struct IndependentMechanisms;

impl eredu_architectures::PreparationMechanismProvider for IndependentMechanisms {
    fn capture_capabilities(&self) -> eredu_core::capture::CaptureCapabilities {
        super::super::session::bounded_capture::capabilities()
    }
    fn observation_mechanisms(&self) -> eredu_core::ObservationMechanisms {
        eredu_core::ObservationMechanisms {
            activation_tensors: true,
            routing_tensors: true,
            floating_to_f32: true,
        }
    }

    fn preparation_capabilities(&self) -> eredu_core::PreparationMechanismCapabilities {
        super::super::structural::preparation_mechanism_capabilities()
    }

    fn supports_grouped_operation(
        &self,
        requirement: eredu_runtime::GroupedOperationRequirement,
    ) -> bool {
        super::super::replicated_text::GROUPED_OPERATION_CAPABILITIES.contains(&requirement)
    }

    fn replicated_text_capabilities(
        &self,
        requirements: &eredu_runtime::ReplicatedTextRequirements,
        request: &eredu_runtime::ReplicatedTextSelectionRequest,
    ) -> eredu_runtime::BackendMechanismCapabilities {
        super::super::replicated_text::capabilities(requirements, request)
    }

    fn processor_capabilities(&self) -> eredu_runtime::MediaPrimitiveCapabilities {
        super::super::processor::capabilities()
    }

    fn speculative_capabilities(&self) -> eredu_runtime::SpeculativeMechanismCapabilities {
        super::super::speculative::speculative_mechanism_capabilities()
    }

    fn communication_capabilities(&self) -> eredu_runtime::CommunicationCapabilities {
        crate::backend::runtime::distributed::topology::mlx_communication_capabilities()
    }
}

fn write_safetensors_fixture(root: &Path) {
    write_safetensors_fixture_with_fill(root, 0.0);
}

fn write_safetensors_fixture_with_fill(root: &Path, fill: f32) {
    write_safetensors_fixture_with_values(root, |_, _| fill);
}

fn write_safetensors_fixture_with_values(root: &Path, value: impl Fn(&str, usize) -> f32) {
    let config = serde_json::json!({
        "model_type": "llama",
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "intermediate_size": 32,
        "num_attention_heads": 4,
        "rms_norm_eps": 0.00001,
        "vocab_size": 64
    });
    write_configured_safetensors_fixture(root, &config, value);
}

fn write_configured_safetensors_fixture(
    root: &Path,
    config: &serde_json::Value,
    value: impl Fn(&str, usize) -> f32,
) {
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
    std::fs::write(
        root.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let resolved = eredu_architectures::configuration::resolve_model_config(&config).unwrap();
    let checkpoint = resolved.architecture.checkpoint();
    let tensors = checkpoint
        .common_tensors
        .iter()
        .chain(
            checkpoint
                .layout_groups
                .iter()
                .filter_map(|group| group.variants.first())
                .flat_map(|variant| variant.tensors.iter()),
        )
        .map(|tensor| {
            let elements = tensor.shape.iter().product::<usize>();
            (
                tensor.key.clone(),
                tensor.shape.clone(),
                (0..elements)
                    .flat_map(|index| value(&tensor.key, index).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    let views = tensors.iter().map(|(name, shape, bytes)| {
        (
            name.as_str(),
            TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
        )
    });
    serialize_to_file(views, None, &root.join("model.safetensors")).unwrap();
}

#[test]
fn valid_inspection_never_opens_payloads_or_realizes_native_resources() {
    let root = tempfile::tempdir().unwrap();
    write_safetensors_fixture(root.path());
    crate::tests::support::path_instrumentation::reset();

    let report = inspect_model(root.path(), MlxInspectionOptions::default()).unwrap();

    assert_eq!(
        report.requested_load,
        eredu_core::InspectionReadiness::Ready
    );
    let counts = crate::tests::support::path_instrumentation::snapshot();
    assert_eq!(counts.payload_opens, 0);
    assert_eq!(counts.architecture_constructions, 0);
    assert_eq!(counts.state_allocations, 0);
    assert_eq!(counts.materializations, 0);
    assert_eq!(
        crate::tests::support::path_instrumentation::target_native_resource_realization_attempts(),
        0
    );
    assert_eq!(
        crate::tests::support::path_instrumentation::communication_realization_attempts(),
        0
    );
}

#[test]
fn retained_inspection_selection_is_consumed_by_source_preparation() {
    let root = tempfile::tempdir().unwrap();
    write_safetensors_fixture(root.path());
    let outcome = inspect_model_preparation(root.path(), MlxInspectionOptions::default()).unwrap();
    let admission = outcome.selected().unwrap().preparation().admission();

    let sources = eredu_architectures::prepare_inspected_model_sources(outcome)
        .unwrap()
        .unwrap();

    assert_eq!(sources.selected().admission(), admission);
    let diagnostics = sources.complete().source_diagnostics().unwrap();
    assert_eq!(diagnostics.physical_reads, 0);
    assert_eq!(diagnostics.physical_read_bytes, 0);
}

#[test]
fn independent_capability_adapters_produce_the_same_report_and_retained_admission() {
    let root = tempfile::tempdir().unwrap();
    write_safetensors_fixture(root.path());
    let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
    let request = MlxLoadRequest::default();
    let (normalized, _rank) = request.checked_normalized().unwrap();

    let first = eredu_architectures::inspect_selected_model(
        inspection.clone(),
        normalized,
        &preparation_mechanisms(),
        media_feature_availability(),
    );
    let second = eredu_architectures::inspect_selected_model(
        inspection,
        normalized,
        &IndependentMechanisms,
        media_feature_availability(),
    );

    assert_eq!(first.report(), second.report());
    let first = first.selected().unwrap().preparation();
    let second = second.selected().unwrap().preparation();
    assert_eq!(first.admission(), second.admission());
    assert_eq!(first.session_capabilities(), second.session_capabilities());
}

#[test]
fn missing_artifact_is_returned_as_a_total_report() {
    let root = tempfile::tempdir().unwrap();
    let report = inspect_model(
        root.path().join("missing.gguf"),
        MlxInspectionOptions::default(),
    )
    .unwrap();

    assert_eq!(report.container, eredu_core::InspectionReadiness::Missing);
    assert_eq!(
        report.requested_load,
        eredu_core::InspectionReadiness::Missing
    );
}

#[test]
fn discovery_supported_paths_match_native_prefill_and_decode_captures() {
    use crate::backend::runtime::media::input::{token_ids_part, ModelInput};
    use eredu_core::{
        ModelRuntime, ObservationRequest, ObservationSelector, ObservationSupportStatus,
    };
    use safemlx::Array;
    let root = tempfile::tempdir().unwrap();
    write_safetensors_fixture(root.path());
    let options = MlxInspectionOptions::default();
    let report = inspect_model(root.path(), options).unwrap();
    let descriptor = report.architecture_descriptor.as_ref().unwrap();
    let support = report.observation_support.as_ref().unwrap();
    let selectors = support
        .points
        .iter()
        .filter(|p| {
            p.prefill == ObservationSupportStatus::Supported
                && p.decode == ObservationSupportStatus::Supported
        })
        .map(|p| ObservationSelector::Exact(p.path.clone()))
        .collect::<Vec<_>>();
    assert_eq!(selectors.len(), descriptor.observations.points.len());
    assert!(!selectors.is_empty());
    let request = ObservationRequest::selected(selectors);
    let stream = crate::test_stream();
    let backend = crate::native::backend(stream, stream);
    let model = eredu_core::load_model(&backend, root.path(), MlxLoadRequest::default()).unwrap();
    let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    let tokens = Array::from_slice(&[1_u32, 2], &[1, 2]);
    let parts = [token_ids_part(&tokens).unwrap()];
    let first = runtime
        .inspect_prefill(ModelInput::new(&parts).into(), &request)
        .unwrap();
    let next = runtime
        .inspect_decode(Array::from_slice(&[3_u32], &[1, 1]), &request)
        .unwrap();
    for observations in [first.observations, next.observations] {
        for point in &descriptor.observations.points {
            assert!(
                observations.get(&point.path).is_some(),
                "missing native capture {}",
                point.path
            );
        }
    }
}

#[test]
fn bounded_capture_preserves_native_generation_tokens_and_rng_progression() {
    use eredu_core::{
        capture::*, ControlledTextGeneration, GenerationConfigOverrides, ModelRuntime,
        TextGenerationBackend, TextGenerationConfig,
    };
    struct Unconstrained;
    impl eredu_core::TokenFilterController for Unconstrained {
        type Error = std::convert::Infallible;
        fn current_filter(&mut self) -> Result<eredu_core::TokenFilter, Self::Error> {
            Ok(eredu_core::TokenFilter::All)
        }
        fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
            Ok(())
        }
        fn is_complete(&mut self) -> Result<bool, Self::Error> {
            Ok(false)
        }
    }
    let root = tempfile::tempdir().unwrap();
    write_safetensors_fixture(root.path());
    let stream = crate::test_stream();
    let config = eredu_core::resolve_generation_config(
        None,
        GenerationConfigOverrides {
            temperature: Some(0.7),
            max_new_tokens: Some(4),
            ..Default::default()
        },
    )
    .unwrap();
    let mut results = Vec::new();
    for captured in [false, true] {
        let backend = crate::native::backend(stream, stream);
        let model =
            eredu_core::load_model(&backend, root.path(), MlxLoadRequest::default()).unwrap();
        let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
        let discovery =
            <crate::backend::MlxBackend as TextGenerationBackend>::capture_discovery(&runtime)
                .unwrap();
        let report = inspect_model(root.path(), MlxInspectionOptions::default()).unwrap();
        assert_eq!(
            discovery.support.capture,
            report.observation_support.unwrap().capture
        );
        let limits = CaptureUsage {
            captures: 1000,
            retained_bytes: 1_000_000_000,
            host_bytes: 10_000_000,
            encoded_bytes: 10_000_000,
        };
        let plan = CapturePlan {
            schema_version: 1,
            selections: discovery
                .catalog
                .points
                .iter()
                .enumerate()
                .map(|(i, p)| CaptureSelection {
                    id: format!("summary-{i}"),
                    path: p.path.clone(),
                    schedule: CaptureSchedule::default(),
                    slices: vec![],
                    transform: CaptureTransform::Summary,
                })
                .collect(),
            limits: CaptureLimits {
                per_step: limits,
                cumulative: limits,
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
                max_predictions: 4,
            },
        )
        .unwrap();
        let mut generator = ControlledTextGeneration::new(
            &mut runtime,
            vec![1, 2],
            TextGenerationConfig::new(config).with_seed(73),
            Unconstrained,
        )
        .unwrap();
        if captured {
            generator.enable_capture(plan.clone()).unwrap();
        }
        let mut tokens = Vec::new();
        while let Some(token) = generator.next() {
            tokens.push(token.unwrap().token_id());
            if captured {
                let step = generator.take_captured_step().unwrap().unwrap();
                assert_eq!(step.prediction_index, tokens.len() as u64 - 1);
                for record in &step.records {
                    assert_eq!(record.outcome, CaptureOutcome::Captured, "{}", record.path);
                    assert!(matches!(record.payload, Some(CapturePayload::Summary(_))));
                }
            }
        }
        drop(generator);
        runtime.parts_mut().1.reset().unwrap();
        let mut following = ControlledTextGeneration::new(
            &mut runtime,
            vec![1, 2],
            TextGenerationConfig::new(config).with_seed(73),
            Unconstrained,
        )
        .unwrap();
        assert_eq!(following.next().unwrap().unwrap().token_id(), tokens[0]);
        drop(following);
        runtime.parts_mut().1.reset().unwrap();
        if captured {
            let mut early_drop = ControlledTextGeneration::new(
                &mut runtime,
                vec![1, 2],
                TextGenerationConfig::new(config).with_seed(73),
                Unconstrained,
            )
            .unwrap();
            early_drop.enable_capture(plan).unwrap();
            assert_eq!(early_drop.next().unwrap().unwrap().token_id(), tokens[0]);
            // Leave a completed host capture batch unconsumed. Dropping the
            // ordinary generator must settle retained native completion authority.
            drop(early_drop);
            runtime.parts_mut().1.reset().unwrap();
        }
        results.push(tokens);
    }
    assert_eq!(results[0], results[1]);
}

#[test]
fn bounded_capture_native_failure_respects_recovery_and_retirement() {
    use eredu_core::{capture::*, ModelRuntime, TextGenerationBackend, TextGenerationConfig};
    let root = tempfile::tempdir().unwrap();
    write_safetensors_fixture_with_fill(root.path(), f32::NAN);
    let stream = crate::test_stream();
    let backend = crate::native::backend(stream, stream);
    let model = eredu_core::load_model(&backend, root.path(), MlxLoadRequest::default()).unwrap();
    let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    let discovery =
        <crate::backend::MlxBackend as TextGenerationBackend>::capture_discovery(&runtime).unwrap();
    let budget = CaptureUsage {
        captures: 8,
        retained_bytes: 1_000_000,
        host_bytes: 1_000_000,
        encoded_bytes: 1_000_000,
    };
    let plan = CapturePlan {
        schema_version: 1,
        selections: vec![CaptureSelection {
            id: "candidates".into(),
            path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
            schedule: CaptureSchedule::default(),
            slices: vec![],
            transform: CaptureTransform::TopCandidates { count: 2 },
        }],
        limits: CaptureLimits {
            per_step: budget,
            cumulative: budget,
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
            max_predictions: 2,
        },
    )
    .unwrap();
    let sampling = eredu_core::resolve_generation_config(
        None,
        eredu_core::GenerationConfigOverrides {
            temperature: Some(0.0),
            ..Default::default()
        },
    )
    .unwrap();
    let mut state = <crate::backend::MlxBackend as TextGenerationBackend>::start_text_generation(
        runtime.backend(),
        TextGenerationConfig::new(sampling),
    )
    .unwrap();
    <crate::backend::MlxBackend as TextGenerationBackend>::configure_text_capture(
        &runtime, &mut state, plan,
    )
    .unwrap();
    let prompt = <crate::backend::MlxBackend as TextGenerationBackend>::prepare_text_prompt(
        runtime.backend(),
        vec![1, 2],
    )
    .unwrap();
    let result = <crate::backend::MlxBackend as TextGenerationBackend>::submit_text_prefill(
        &mut runtime,
        prompt,
        &eredu_core::TokenFilter::All,
        &mut state,
    );
    let error = match result {
        Err(error) => error,
        Ok(_) => panic!("non-finite candidate capture must fail explicitly"),
    };
    assert!(error.to_string().contains("finite"), "{error}");
    let captures =
        <crate::backend::MlxBackend as TextGenerationBackend>::take_text_capture(&mut state)
            .unwrap();
    assert!(matches!(
        captures.records[0].outcome,
        CaptureOutcome::Failed {
            reason: CaptureFailureReason::Native,
            ..
        }
    ));
    // A proven transactional rollback plus settled native work permits reuse.
    // Otherwise the session must stay fenced; an error alone is not completion.
    if runtime.parts_mut().1.reset().is_ok() {
        assert!(error.model_state_preserved());
    } else {
        assert!(runtime.parts_mut().1.reset().is_err());
    }
    // Weak pointers themselves prevent Rc::get_mut, so install this probe only
    // after every attempted native operation, then observe final retirement.
    let payload_retired = runtime.session().test_payload_retirement_probe();
    assert!(!payload_retired());
    drop(state);
    drop(runtime);
    crate::backend::submission_recovery::wait_for_retirement(&payload_retired);
    assert!(payload_retired());
}

#[test]
fn unsupported_container_is_returned_as_a_total_report() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("weights.bin");
    std::fs::write(&path, b"not a supported model container").unwrap();

    let report = inspect_model(&path, MlxInspectionOptions::default()).unwrap();

    assert_eq!(
        report.container,
        eredu_core::InspectionReadiness::Unsupported
    );
    assert_eq!(
        report.requested_load,
        eredu_core::InspectionReadiness::Unsupported
    );
}

#[test]
fn invalid_configuration_is_returned_as_a_total_report() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("config.json"), b"{").unwrap();

    let report = inspect_model(root.path(), MlxInspectionOptions::default()).unwrap();

    assert_ne!(
        report.requested_load,
        eredu_core::InspectionReadiness::Ready
    );
    assert!(!report.issues.is_empty());
}

#[test]
fn native_intervention_loaded_dense_preserves_rng_and_returns_effective_logits() {
    use eredu_core::{
        capture::*, intervention::*, ControlledTextGeneration, GenerationConfigOverrides,
        ModelRuntime, TextGenerationBackend, TextGenerationConfig,
    };
    struct Unconstrained;
    impl eredu_core::TokenFilterController for Unconstrained {
        type Error = std::convert::Infallible;
        fn current_filter(&mut self) -> Result<eredu_core::TokenFilter, Self::Error> {
            Ok(eredu_core::TokenFilter::All)
        }
        fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
            Ok(())
        }
        fn is_complete(&mut self) -> Result<bool, Self::Error> {
            Ok(false)
        }
    }
    let root = tempfile::tempdir().unwrap();
    write_safetensors_fixture_with_fill(root.path(), 0.03);
    let stream = crate::test_stream();
    let backend = crate::native::backend(stream, stream);
    let model = eredu_core::load_model(&backend, root.path(), MlxLoadRequest::default()).unwrap();
    let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    let discovery =
        <crate::backend::MlxBackend as TextGenerationBackend>::intervention_discovery(&runtime)
            .unwrap();
    let captures =
        <crate::backend::MlxBackend as TextGenerationBackend>::capture_discovery(&runtime).unwrap();
    let request = CaptureRequestShape {
        batch: 1,
        prompt_tokens: 2,
        max_predictions: 4,
    };
    let budget = CaptureUsage {
        captures: 1000,
        retained_bytes: 1_000_000_000,
        host_bytes: 10_000_000,
        encoded_bytes: 10_000_000,
    };
    let capture = CapturePlan {
        schema_version: 1,
        selections: vec![],
        limits: CaptureLimits {
            per_step: budget,
            cumulative: budget,
            physical_native_bytes: None,
            on_limit: CaptureLimitPolicy::Fail,
        },
    }
    .admit(
        &captures.catalog,
        &captures.support,
        &captures.support.capture,
        request,
    )
    .unwrap();
    let sampling = eredu_core::resolve_generation_config(
        None,
        GenerationConfigOverrides {
            temperature: Some(0.7),
            max_new_tokens: Some(4),
            ..Default::default()
        },
    )
    .unwrap();
    let mut results = Vec::new();
    for experiment in 0..3 {
        runtime.parts_mut().1.reset().unwrap();
        let operations = if experiment == 1 {
            discovery
                .points
                .iter()
                .filter(|point| point.prefill == eredu_core::ObservationSupportStatus::Supported)
                .enumerate()
                .map(|(i, point)| InterventionOperation {
                    id: format!("identity-{i}"),
                    target: point.path.clone(),
                    schedule: CaptureSchedule::default(),
                    slices: vec![],
                    action: InterventionAction::Scale {
                        dtype: InterventionDtype::Float32,
                        factor: 1.0,
                    },
                    evidence: InterventionEvidence::Preview { max_elements: 2 },
                })
                .collect()
        } else if experiment == 2 {
            let mut values = vec![-1000.0; 64];
            values[7] = 1000.0;
            vec![InterventionOperation {
                id: "force-token".into(),
                target: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
                schedule: CaptureSchedule {
                    decode: false,
                    ..Default::default()
                },
                slices: vec![CaptureSlice {
                    axis: "sequence".into(),
                    start: 1,
                    end: 2,
                    stride: 1,
                }],
                action: InterventionAction::Replace {
                    tensor: InterventionTensor {
                        shape: vec![1, 1, 64],
                        values: InterventionValues::Float32(values),
                    },
                },
                evidence: InterventionEvidence::Preview { max_elements: 8 },
            }]
        } else {
            vec![]
        };
        let plan = InterventionPlan {
            schema_version: 1,
            operations,
        }
        .admit(&discovery, request, "native-test-session")
        .unwrap();
        <crate::backend::MlxBackend as TextGenerationBackend>::validate_text_interventions(
            &runtime, &capture, &plan,
        )
        .unwrap();
        let mut generator = ControlledTextGeneration::new(
            &mut runtime,
            vec![1, 2],
            TextGenerationConfig::new(sampling).with_seed(73),
            Unconstrained,
        )
        .unwrap();
        if experiment != 0 {
            generator
                .enable_interventions(capture.clone(), plan)
                .unwrap();
        }
        let mut tokens = Vec::new();
        while let Some(token) = generator.next() {
            tokens.push(token.unwrap().token_id());
            if experiment != 0 {
                let step = generator.take_captured_step().unwrap().unwrap();
                for record in &step.interventions {
                    assert_eq!(
                        record.outcome,
                        if experiment == 2 && step.prediction_index > 0 {
                            InterventionOutcome::Inactive
                        } else {
                            InterventionOutcome::Applied
                        }
                    );
                    if record.outcome == InterventionOutcome::Applied {
                        assert_eq!(record.evidence.len(), 2);
                        assert!(record.evidence.iter().all(|e| e.payload.is_some()));
                    }
                }
            }
        }
        drop(generator);
        results.push(tokens);
    }
    assert_eq!(
        results[0], results[1],
        "scale-one interventions must preserve the ordinary sampler RNG progression"
    );
    assert_eq!(
        results[2][0], 7,
        "the ordinary sampler must consume the replacement logits"
    );
    for (fail_dtype, prompt_mismatch) in [(false, false), (true, false), (false, true)] {
        runtime.parts_mut().1.reset().unwrap();
        let plan = InterventionPlan {
            schema_version: 1,
            operations: vec![InterventionOperation {
                id: "lifecycle".into(),
                target: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
                schedule: CaptureSchedule::default(),
                slices: vec![],
                action: InterventionAction::Scale {
                    dtype: if fail_dtype {
                        InterventionDtype::Float16
                    } else {
                        InterventionDtype::Float32
                    },
                    factor: 1.0,
                },
                evidence: InterventionEvidence::Preview { max_elements: 2 },
            }],
        }
        .admit(&discovery, request, "native-test-session")
        .unwrap();
        let mut generator = ControlledTextGeneration::new(
            &mut runtime,
            if prompt_mismatch { vec![1] } else { vec![1, 2] },
            TextGenerationConfig::new(sampling).with_seed(73),
            Unconstrained,
        )
        .unwrap();
        generator
            .enable_interventions(capture.clone(), plan.clone())
            .unwrap();
        let result = generator.next().unwrap();
        if prompt_mismatch {
            assert!(result.is_err());
            assert!(
                generator.take_captured_step().unwrap().is_none(),
                "known prompt mismatch must fail before starting the native step"
            );
        } else if fail_dtype {
            assert!(result.is_err());
            let step = generator.take_captured_step().unwrap().unwrap();
            assert!(matches!(
                step.interventions[0].outcome,
                InterventionOutcome::Failed { .. }
            ));
        } else {
            assert_eq!(result.unwrap().token_id(), results[0][0]);
            assert!(
                generator
                    .enable_interventions(capture.clone(), plan)
                    .is_err(),
                "hot replacement after prefill must fail"
            );
        }
        // Drop an unfinished run, including a failed forward that may have written
        // cache state. The native owner must settle work before reset can succeed.
        drop(generator);
        crate::backend::submission_recovery::wait_for_retirement(|| {
            runtime.session().ensure_no_submission_in_flight().is_ok()
        });
        runtime.parts_mut().1.reset().unwrap();
    }
    runtime.parts_mut().1.reset().unwrap();
}
