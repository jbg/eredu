// Native binding conformance for the shared bounded collector. Public facade
// admission remains pending; this fixture supplies explicit hook authority.
include!("public.rs");
fn prove_bounded(device: DeviceType) {
    use crate::composition::mlx::{
        fixture_bounded_capture as bounded_capture, fixture_intervention as intervention,
    };
    use eredu_core::{capture::*, intervention::*, TensorObservationData};
    let checkpoint = tempfile::tempdir().unwrap();
    write_deepseek_fixture_with_values(checkpoint.path(), 2, 2, true);
    let config =
        serde_json::from_slice(&std::fs::read(checkpoint.path().join("config.json")).unwrap())
            .unwrap();
    let resolved = eredu_architectures::configuration::MODEL_CONFIGURATIONS
        .resolve_safetensors(&config)
        .unwrap();
    let architecture = resolved.architecture_plan();
    let graph = architecture.architecture_descriptor();
    let scope = &graph.component_scopes[0];
    let channel = scope
        .components
        .iter()
        .find(|g| {
            matches!(
                g.activation_equation,
                eredu_core::component::ComponentActivation::Attention { .. }
            )
        })
        .unwrap();
    let paths = [
        channel.activation.clone(),
        channel.effective_activation.clone(),
        scope.readout.normalized.clone(),
        scope.readout.logits.clone(),
        eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
    ];
    let mut support = eredu_runtime::inspection::observation_support(
        &graph.observations,
        eredu_runtime::inspection::ObservationExecutionContext {
            activation_inspection: true,
            prediction_inspection: true,
            selected: true,
            partitioned: false,
            mechanisms: eredu_core::ObservationMechanisms {
                activation_tensors: true,
                routing_tensors: true,
                routed_unit_tensors: true,
                floating_to_f32: true,
            },
        },
    );
    // This fixture explicitly invokes the retained prediction groups and gives
    // the collector exact per-depth applicability. Ordinary discovery correctly
    // keeps this condition unresolved for callers without that execution scope.
    for status in &mut support.points {
        let point = graph.observations.get(&status.path).unwrap();
        if point
            .requirements
            .contains(&eredu_core::ObservationRequirement::PredictionExecution)
        {
            assert!(!point
                .requirements
                .contains(&eredu_core::ObservationRequirement::MediaInput));
            eredu_architectures::speculative_execution::speculative_capture_scope(
                &graph,
                &point.node_id,
            )
            .unwrap();
            for phase in [&mut status.prefill, &mut status.decode] {
                if matches!(phase, eredu_core::ObservationSupportStatus::Conditional(_)) {
                    *phase = eredu_core::ObservationSupportStatus::Supported;
                }
            }
        }
    }
    support.capture = bounded_capture::capabilities();
    let bounds = CaptureInvocationBounds {
        batch: 1,
        max_sequence: 3,
        max_context: None,
        max_predictions: 8,
    };
    let run = |instrumented: bool, mask: bool, maximum: u64| {
        let stream = Stream::new_with_device(&Device::new(device, 0));
        let weights = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let backend = crate::native::backend(&stream, &weights);
        let model = load_model(&backend, checkpoint.path(), MlxLoadRequest::default()).unwrap();
        let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
        let outer = Arc::new(Mutex::new(ExternalObservationTrace::default()));
        let mut observers =
            eredu_architectures::speculative_execution::EmbeddedPredictionObservers::new(
                eredu_runtime::NoopObserver,
                EmbeddedLogitsObserver {
                    trace: Arc::clone(&outer),
                    intervention: ExternalTensorIntervention::None,
                },
            );
        if instrumented {
            let per_step = CaptureUsage {
                captures: 128,
                retained_bytes: 8 << 20,
                host_bytes: 8 << 20,
                encoded_bytes: 8 << 20,
            };
            let plan = CapturePlan {
                schema_version: CAPTURE_SCHEMA_VERSION,
                selections: paths
                    .iter()
                    .enumerate()
                    .map(|(index, path)| CaptureSelection {
                        id: index.to_string(),
                        path: path.clone(),
                        schedule: Default::default(),
                        slices: vec![],
                        transform: CaptureTransform::Preview { max_elements: 2048 },
                    })
                    .collect(),
                limits: CaptureLimits {
                    per_step,
                    cumulative: CaptureUsage {
                        captures: maximum,
                        ..per_step.checked_mul(8).unwrap()
                    },
                    physical_native_bytes: None,
                    on_limit: CaptureLimitPolicy::Fail,
                },
            }
            .admit_invocations(&graph.observations, &support, &support.capture, bounds)
            .unwrap();
            let scopes = plan
                .points()
                .iter()
                .map(|p| {
                    eredu_architectures::speculative_execution::speculative_capture_scope(
                        &graph, &p.node_id,
                    )
                    .unwrap()
                })
                .collect();
            let mut session = eredu_runtime::capture::CaptureSession::new(eredu_core::capture::SharedCapturePlan::new(plan));
            let mut edits = Vec::new();
            if mask {
                let mut discovery = eredu_runtime::inspection::intervention_support(
                    architecture.intervention_points(),
                    &CaptureDiscovery {
                        artifact_identity: "native-v3-bounded-fixture".into(),
                        catalog: graph.observations.clone(),
                        support: support.clone(),
                    },
                    &intervention::mechanisms(),
                );
                discovery.session_identity =
                    <MlxBackend<'_> as eredu_core::TextGenerationBackend>::intervention_discovery(
                        &runtime,
                    )
                    .unwrap()
                    .session_identity;
                let edit = InterventionPlan {
                    schema_version: INTERVENTION_SCHEMA_VERSION,
                    operations: vec![InterventionOperation {
                        id: "channel-mask".into(),
                        target: channel.activation.clone(),
                        schedule: Default::default(),
                        slices: vec![],
                        action: InterventionAction::MaskComponents {
                            dtype: InterventionDtype::Float32,
                            indices: vec![0],
                            keep_selected: false,
                        },
                        evidence: InterventionEvidence::Preview { max_elements: 2048 },
                    }],
                }
                .admit_invocations(&discovery, bounds, "native-v3-bounded-fixture")
                .unwrap();
                edits = edit
                    .points()
                    .iter()
                    .map(|p| {
                        eredu_architectures::speculative_execution::speculative_capture_scope(
                            &graph, &p.node_id,
                        )
                        .unwrap()
                    })
                    .collect();
                session
                    .enable_interventions(edit, Arc::new(intervention::NativeInterventionEstimator))
                    .unwrap();
            }
            observers = observers.with_internal(
                bounded_capture::fixture_speculative_capture(
                    session,
                    stream.clone(),
                    scopes,
                    edits,
                )
                .unwrap(),
            );
        }
        runtime
            .session_mut()
            .install_embedded_prediction_observer_set(observers)
            .unwrap();
        let tokens = [1_u32, 2, 3];
        let prompt = Array::from_slice(&tokens, &[1, 3]);
        let parts = [text_input_part(&prompt)];
        let (output, publications) = execute_neutral_embedded_mtp(
            &mut runtime,
            synthetic_prediction_input(&parts, &tokens),
            SpeculativeConfig {
                max_tokens: 5,
                max_draft_tokens: 2,
                temperature: 0.0,
                eos_token_ids: vec![],
            },
        );
        // Failed execution keeps the actual session payload in nonblocking
        // recovery. Drive that existing owner to retirement before borrowing
        // host evidence; an error return alone is not completion authority.
        if runtime.session().test_payload_owner_count() != 1 {
            assert_eq!(
                runtime
                    .session_mut()
                    .take_speculative_activation_capture()
                    .unwrap_err()
                    .kind(),
                eredu_core::BackendFailureKind::Busy,
            );
        }
        crate::backend::submission_recovery::wait_for_retirement(|| {
            runtime.session().test_payload_owner_count() == 1
        });
        let records: Vec<_> = std::iter::from_fn(|| {
            runtime
                .session_mut()
                .take_speculative_activation_capture()
                .unwrap()
        })
        .collect();
        let error = runtime
            .session_mut()
            .take_speculative_activation_error()
            .unwrap();
        (output, publications, records, error, outer)
    };
    let (ordinary, _, disabled, _, ordinary_logits) = run(false, false, 512);
    assert!(disabled.is_empty());
    let (observed, _, records, error, logits) = run(true, false, 512);
    assert!(error.is_none());
    assert_eq!(observed.unwrap().token_ids(), ordinary.unwrap().token_ids());
    assert_eq!(
        logits.lock().unwrap().proposal_logits,
        ordinary_logits.lock().unwrap().proposal_logits
    );
    assert!(records.len() >= 5);
    assert_eq!(
        (
            records[0].phase,
            records[0].captures.as_step().invocation.unwrap().sequence
        ),
        (Phase::TargetPrefill, 3)
    );
    assert_eq!(
        (
            records[1].phase,
            records[1].captures.as_step().invocation.unwrap().sequence
        ),
        (Phase::PredictionPrefill, 2)
    );
    for (index, record) in records.iter().enumerate() {
        assert_eq!(record.invocation, index as u64);
        assert!(record.completed);
        assert_eq!(record.origin.request.index(), 0);
        assert!(record.origin.prediction >= record.origin.committed_tokens);
        assert!(!record.captures.as_step().records.iter().any(|r| matches!(
            r.outcome,
            CaptureOutcome::Missing | CaptureOutcome::Failed { .. }
        )));
        assert!(
            serde_json::to_vec(record).unwrap().len() as u64
                <= record.captures.as_step().step_usage.encoded_bytes
        );
    }
    let (masked, _, records, error, masked_logits) = run(true, true, 512);
    masked.unwrap();
    assert!(error.is_none());
    assert_ne!(
        masked_logits.lock().unwrap().proposal_logits[0],
        logits.lock().unwrap().proposal_logits[0]
    );
    let values = |record: &CaptureRecord| {
        let Some(CapturePayload::Tensor(value)) = &record.payload else {
            panic!("missing native tensor evidence");
        };
        let TensorObservationData::F32(values) = value.data() else {
            panic!("wrong evidence dtype");
        };
        values.clone()
    };
    let mut applied = 0;
    for record in records {
        for edit in &record.captures.as_step().interventions {
            if edit.evidence.iter().any(|r| r.payload.is_some()) {
                let before = values(&edit.evidence[0]);
                let after = values(&edit.evidence[1]);
                assert!(before.iter().any(|v| v.abs() > 1e-5));
                for (index, (a, b)) in before.iter().zip(&after).enumerate() {
                    assert_eq!(*b, if index % channel.count == 0 { 0.0 } else { *a });
                }
                applied += 1;
            }
        }
    }
    assert!(applied >= 2);
    let (failed, publications, records, error, _) = run(true, false, 1);
    assert!(failed.is_err());
    assert_eq!(publications, 0);
    assert!(matches!(
        error,
        Some(eredu_core::speculative::SpeculativeControlError::Capture(
            CaptureError::Limit {
                budget: CaptureBudget::Captures,
                cumulative: true
            }
        ))
    ));
    assert!(!records.last().unwrap().completed);
}

#[test]
fn native_v3_bounded_speculative_collector_cpu() {
    prove_bounded(DeviceType::Cpu);
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires a local MLX Metal device"]
fn native_v3_bounded_speculative_collector_metal() {
    prove_bounded(DeviceType::Gpu);
}
