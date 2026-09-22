use super::*;

fn source<'a, T: std::error::Error + 'static>(
    mut error: &'a (dyn std::error::Error + 'static),
) -> Option<&'a T> {
    loop {
        if let Some(found) = error.downcast_ref::<T>() {
            return Some(found);
        }
        error = error.source()?;
    }
}

fn native_cause(mut error: &(dyn std::error::Error + 'static)) -> (String, String, u32) {
    loop {
        // Native diagnostics are optional downcasts; all operations below use the
        // public facade and still return its backend-neutral failure types.
        if let Some(eredu_backend_mlx::backend::error::Error::Exception(native)) =
            error.downcast_ref::<eredu_backend_mlx::backend::error::Error>()
        {
            return (
                native.what().into(),
                native.location().file().into(),
                native.location().line(),
            );
        }
        error = error
            .source()
            .expect("public failure retains the native cause");
    }
}

fn budgets() -> CaptureUsage {
    CaptureUsage {
        captures: 8,
        retained_bytes: 16 << 20,
        host_bytes: 1 << 20,
        encoded_bytes: 1 << 20,
    }
}
fn capture(scores: bool, cumulative: CaptureUsage) -> CapturePlan {
    CapturePlan {
        schema_version: CAPTURE_SCHEMA_VERSION,
        selections: vec![CaptureSelection {
            id: "selected".into(),
            path: if scores {
                "model.logits"
            } else {
                "model.layers.0.feed_forward.units"
            }
            .into(),
            schedule: CaptureSchedule::default(),
            slices: vec![],
            transform: if scores {
                CaptureTransform::TokenScores { token_ids: vec![1] }
            } else {
                CaptureTransform::FullTensor
            },
        }],
        limits: CaptureLimits {
            per_step: budgets(),
            cumulative,
            on_limit: CaptureLimitPolicy::Fail,
        },
    }
}
fn settings() -> PreparedChatGenerationSettings {
    PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            temperature: Some(0.0),
            max_new_tokens: Some(1),
            ..Default::default()
        },
        seed: 17,
        ..Default::default()
    }
}
fn trace() -> TraceLimits {
    TraceLimits {
        per_record_bytes: 1 << 20,
        total_bytes: 16 << 20,
    }
}
fn chat_request() -> ChatTemplateRequest {
    ChatTemplateRequest {
        messages: vec![serde_json::json!({"role":"user", "content":"left right"})],
        add_generation_prompt: true,
        ..Default::default()
    }
}
fn first_usage(events: &[ControlledGenerationRecord]) -> CaptureUsage {
    usage_from_events(events.iter().filter_map(|event| event.event.progress()))
}
fn usage_from_events<'a>(
    mut events: impl Iterator<Item = &'a ObservedGenerationEvent>,
) -> CaptureUsage {
    events
        .find_map(|event| match event {
            ObservedGenerationEvent::Token {
                captures: Some(step),
                ..
            } => Some(step.cumulative_usage),
            _ => None,
        })
        .expect("committed capture")
}
fn native_failure_fixture() -> Fixture {
    let root = fixture(false);
    let readout = inspect_architecture(&root.0)
        .unwrap()
        .component_readout
        .unwrap()
        .weight
        .clone();
    let path = root.0.join("model.safetensors");
    let mut bytes = std::fs::read(&path).unwrap();
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
    let offset = header[readout]["data_offsets"][0].as_u64().unwrap() as usize + 8 + header_len;
    bytes[offset..offset + 4].copy_from_slice(&f32::NAN.to_le_bytes());
    std::fs::write(path, bytes).unwrap();
    root
}

fn verify_public_capture_failures(device: LocalDevice) {
    let healthy = fixture(false);
    let invalid = native_failure_fixture();
    for residency in [
        eredu_core::ResidencyPlan::FullyResident,
        eredu_core::ResidencyPlan::LayerwiseHost {
            device_layer_window: 1,
            device_budget_bytes: Some(8 << 20),
            host_budget_bytes: Some(8 << 20),
        },
        eredu_core::ResidencyPlan::DenseDiskStream {
            device_budget_bytes: 8 << 20,
            host_budget_bytes: 8 << 20,
            host_lookahead: 1,
            background_queue: 1,
        },
    ] {
        let execution = ExecutionPlan::fully_resident(local_device_plan(device).unwrap())
            .with_residency(residency)
            .with_required_session_capabilities(SessionCapabilities::new(true, true, true));
        let (mut model, _) =
            LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &healthy.0, &execution)
                .unwrap()
                .into_parts();
        let chat = model.source_chat(chat_request()).unwrap();
        // Measure the complete charged step through an ordinary public run, then
        // admit exactly that cumulative allowance for a controlled replay.
        let prepared_prefix = vec![1, 2, 5, 7];
        let prepared_capture = capture(false, budgets());
        let prepared_trace = trace();
        let mut prepared = PreparedChatRequest::new(&chat, original_settings(settings()));
        prepared.input = PreparedChatPrompt::TokenIds(&prepared_prefix);
        prepared.output_mode = PreparedChatOutputMode::Text;
        prepared.capture = Some(&prepared_capture);
        let mut events = Vec::new();
        (|| -> Result<_, ControlledGenerationError> {
            let mut emit = |event| {
                events.push(event);
                ControlFlow::Continue(())
            };
            let mut run = model
                .start_controlled_chat(
                    prepared,
                    prepared_trace,
                    GenerationControlHandle::new(Default::default()),
                    &mut emit,
                )?
                .expect("live fixture control");
            run.run(&mut emit)
        })()
        .unwrap();
        let usage = first_usage(&events);
        let baseline = super::components::tensors(&events);
        assert!(baseline.values().flatten().any(|value| value.abs() > 1e-6));
        model.reset().unwrap();
        let prepared_prefix = vec![1, 2, 5, 7];
        let prepared_capture = capture(false, usage);
        let prepared_trace = trace();
        let mut prepared = PreparedChatRequest::new(&chat, original_settings(settings()));
        prepared.input = PreparedChatPrompt::TokenIds(&prepared_prefix);
        prepared.output_mode = PreparedChatOutputMode::Text;
        prepared.capture = Some(&prepared_capture);
        {
            let mut records = Vec::new();
            let mut run = model
                .start_controlled_chat(
                    prepared,
                    prepared_trace,
                    Default::default(),
                    collect(&mut records),
                )
                .unwrap()
                .unwrap();
            run.enable_snapshots(
                SnapshotLimits {
                    max_snapshots: 2,
                    max_branches: 1,
                    retained_bytes: 64 << 20,
                    cumulative_copy_bytes: 256 << 20,
                },
                native_limits(ORIGINAL_CAPACITY),
                copy_limits(),
            )
            .unwrap();
            let initial = run.snapshot(collect(&mut records)).unwrap();
            run.run(collect(&mut records)).unwrap();
            assert_eq!(
                super::components::tensors_from_events(
                    records.iter().filter_map(|record| record.event.progress())
                ),
                baseline
            );
            assert_eq!(
                usage_from_events(records.iter().filter_map(|record| record.event.progress())),
                usage
            );
            run.restore(&initial, collect(&mut records)).unwrap();
            let before = records.len();
            let error = run.run(collect(&mut records)).unwrap_err();
            assert!(
                matches!(
                    source::<CaptureError>(&error),
                    Some(CaptureError::Limit {
                        cumulative: true,
                        ..
                    })
                ),
                "{error:?}"
            );
            assert!(!records[before..].iter().any(|record| matches!(
                record.event.progress(),
                Some(ObservedGenerationEvent::Token { .. })
            )));
        }
        model.reset().unwrap();

        let (mut model, _) =
            LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &invalid.0, &execution)
                .unwrap()
                .into_parts();
        let chat = model.source_chat(chat_request()).unwrap();
        let mut expected = None;
        for controlled in [false, true] {
            model.reset().unwrap();
            let prepared_prefix = vec![1, 2, 5, 7];
            let prepared_capture = capture(true, budgets());
            let prepared_trace = trace();
            let mut prepared = PreparedChatRequest::new(&chat, original_settings(settings()));
            prepared.input = PreparedChatPrompt::TokenIds(&prepared_prefix);
            prepared.output_mode = PreparedChatOutputMode::Text;
            prepared.capture = Some(&prepared_capture);
            let mut events = Vec::new();
            let error: Box<dyn std::error::Error> = if controlled {
                let mut records = Vec::new();
                let mut run = model
                    .start_controlled_chat(
                        prepared,
                        prepared_trace,
                        Default::default(),
                        collect(&mut records),
                    )
                    .unwrap()
                    .unwrap();
                let error = run.run(collect(&mut records)).unwrap_err();
                events.extend(records);
                Box::new(error)
            } else {
                Box::new(
                    (|| -> Result<_, ControlledGenerationError> {
                        let mut emit = |event| {
                            events.push(event);
                            ControlFlow::Continue(())
                        };
                        let mut run = model
                            .start_controlled_chat(
                                prepared,
                                prepared_trace,
                                GenerationControlHandle::new(Default::default()),
                                &mut emit,
                            )?
                            .expect("live fixture control");
                        run.run(&mut emit)
                    })()
                    .unwrap_err(),
                )
            };
            let original = native_cause(error.as_ref());
            assert_eq!(
                original.0,
                "full-vocabulary scoring requires finite raw logits"
            );
            assert!(original.2 > 0);
            if let Some(expected) = &expected {
                assert_eq!(&original, expected);
            } else {
                expected = Some(original);
            }
            assert!(!events.iter().any(|event| matches!(
                event.event.progress(),
                Some(ObservedGenerationEvent::Token { .. })
            )));
        }
        model.reset().unwrap();
    }
}

#[test]
fn public_capture_failures_retain_causes_and_replay_budgets_cpu() {
    verify_public_capture_failures(LocalDevice::Cpu);
}

#[test]
#[cfg(feature = "metal")]
#[ignore = "requires Metal device"]
fn public_capture_failures_retain_causes_and_replay_budgets_metal() {
    verify_public_capture_failures(LocalDevice::Accelerator(0));
}
