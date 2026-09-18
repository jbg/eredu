//! Real public startup, copied source rebind and two serial captured branches.
use super::*;
fn limits() -> CaptureLimits {
    let usage = CaptureUsage {
        captures: 256,
        retained_bytes: 1 << 30,
        host_bytes: 1 << 30,
        encoded_bytes: 1 << 30,
    };
    CaptureLimits {
        per_step: usage,
        cumulative: usage,
        physical_native_bytes: None,
        on_limit: CaptureLimitPolicy::Fail,
    }
}
fn plan() -> CapturePlan {
    CapturePlan {
        schema_version: 1,
        selections: vec![CaptureSelection {
            id: "prepared scores".into(),
            path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
            schedule: CaptureSchedule::default(),
            slices: vec![],
            transform: CaptureTransform::FullTensor,
        }],
        limits: limits(),
    }
}
fn frames(records: &[ControlledGenerationRecord]) -> Vec<&CapturedStep> {
    records
        .iter()
        .filter_map(|r| match &r.event {
            ControlledGenerationEvent::Progress { event } => {
                let frame = event.captures();
                if frame.is_some() {
                    assert!(
                        event.shared_captures().is_some(),
                        "ordinary prepared frame keeps custody"
                    );
                }
                frame
            }
            _ => None,
        })
        .collect()
}
pub(super) fn equal(actual: &CapturedStep, expected: &CapturedStep) {
    assert_eq!(actual.outcome, CaptureStepOutcome::Committed);
    assert_eq!(actual.records.len(), expected.records.len());
    for (a, b) in actual.records.iter().zip(&expected.records) {
        assert_eq!(
            (&a.outcome, &a.source_shape, &a.selected_shape),
            (&b.outcome, &b.source_shape, &b.selected_shape)
        );
        match (a.payload.as_ref(), b.payload.as_ref()) {
            (None, None) => {}
            (Some(CapturePayload::Summary(a)), Some(CapturePayload::Summary(b))) => {
                assert_eq!(
                    (
                        a.elements,
                        a.finite,
                        a.non_finite,
                        a.nan,
                        a.positive_infinity,
                        a.negative_infinity
                    ),
                    (
                        b.elements,
                        b.finite,
                        b.non_finite,
                        b.nan,
                        b.positive_infinity,
                        b.negative_infinity
                    )
                );
                for (a, b) in [
                    (a.min, b.min),
                    (a.max, b.max),
                    (a.mean, b.mean),
                    (a.rms, b.rms),
                ] {
                    match (a, b) {
                        (Some(a), Some(b)) => assert!((a - b).abs() <= 3e-4 + 3e-4 * b.abs()),
                        (None, None) => {}
                        _ => panic!("summary"),
                    }
                }
            }
            (Some(CapturePayload::Histogram(a)), Some(CapturePayload::Histogram(b))) => {
                assert_eq!(a, b)
            }
            (Some(CapturePayload::Candidates(a)), Some(CapturePayload::Candidates(b))) => {
                assert_eq!(
                    (&a.stage, &a.source, &a.domain),
                    (&b.stage, &b.source, &b.domain)
                );
                assert!(
                    a.domain.is_some(),
                    "actual facade controller supplies tokenizer provenance"
                );
                assert_eq!(a.candidates.len(), b.candidates.len());
                for (a, b) in a.candidates.iter().zip(&b.candidates) {
                    assert_eq!((a.token_id, a.allowed), (b.token_id, b.allowed));
                    assert!((a.score - b.score).abs() <= 3e-4 + 3e-4 * b.score.abs());
                }
            }
            (Some(a), Some(b)) => {
                let (a, b) = (a.as_tensor().unwrap(), b.as_tensor().unwrap());
                assert_eq!(a.shape(), b.shape());
                let (
                    eredu_core::TensorObservationData::F32(a),
                    eredu_core::TensorObservationData::F32(b),
                ) = (a.data(), b.data())
                else {
                    panic!("floating scores")
                };
                assert!(a.iter().all(|v| v.is_finite()));
                assert!(a.is_empty() || a.iter().any(|v| v.abs() > 1e-7));
                for (a, b) in a.iter().zip(b) {
                    assert!((a - b).abs() <= 3e-4 + 3e-4 * b.abs());
                }
            }
            _ => panic!("payload mismatch"),
        }
    }
}
fn global_plan() -> CapturePlan {
    let mut source = plan();
    for (id, transform) in [
        ("summary", CaptureTransform::Summary),
        (
            "histogram",
            CaptureTransform::Histogram {
                edges: vec![-1000., 0., 1000.],
            },
        ),
        ("preview", CaptureTransform::Preview { max_elements: 7 }),
        (
            "empty preview",
            CaptureTransform::Preview { max_elements: 0 },
        ),
        ("candidates", CaptureTransform::TopCandidates { count: 3 }),
    ] {
        source.selections.push(CaptureSelection {
            id: id.into(),
            path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
            schedule: CaptureSchedule {
                decode: false,
                ..Default::default()
            },
            slices: vec![],
            transform,
        });
    }
    source
}

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run CPU fixtures with --no-default-features --features mlx"
)]
fn native_v2_captured_media_snapshot_restore_and_two_serial_branches_keep_one_source_and_nonrefund()
{
    captured_branches(false);
}
#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run CPU fixtures with --no-default-features --features mlx"
)]
fn native_v2_global_media_capture_restore_and_two_branches_keep_domains_values_and_nonrefund() {
    captured_branches(true);
}
fn captured_branches(global: bool) {
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
        let root = components::qwen_vl_component_fixture(false);
        let execution = ExecutionPlan::fully_resident(local_device_plan(LocalDevice::Cpu).unwrap())
            .with_residency(residency)
            .with_required_session_capabilities(SessionCapabilities::new(true, true, true));
        let (mut model, _) =
            LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &execution)
                .unwrap()
                .into_parts();
        let chat = model
            .source_chat(ChatTemplateRequest {
                messages: vec![serde_json::json!({"role":"user","content":"hello"})],
                tools: vec![],
                tool_choice: ToolChoice::None,
                add_generation_prompt: true,
                ..Default::default()
            })
            .unwrap();
        let settings = PreparedChatGenerationSettings {
            overrides: GenerationConfigOverrides {
                temperature: Some(0.7),
                max_new_tokens: Some(4),
                ..Default::default()
            },
            seed: 17,
            ..Default::default()
        };
        let trace = TraceLimits {
            per_record_bytes: 1 << 20,
            total_bytes: 128 << 20,
        };
        let cancellation = eredu_core::GenerationCancellationToken::new();
        let input = prepared_media_copy::with_parts(|parts| {
            model.prepare_chat_input(&chat, parts, &cancellation)
        })
        .unwrap()
        .expect("live media preparation");
        let capture = if global { global_plan() } else { plan() };
        let mut settings = original_settings(settings);
        settings.inference.prefill_chunk_positions = std::num::NonZeroU64::new(2);
        let mut prepared = PreparedChatRequest::new(&chat, settings);
        prepared.input = PreparedChatPrompt::Media(input);
        prepared.output_mode = PreparedChatOutputMode::Text;
        prepared.capture = Some(&capture);
        let mut session = model
            .start_controlled_chat(prepared, trace, Default::default(), |_| {
                ControlFlow::Continue(())
            })
            .unwrap()
            .expect("live control");
        let attribution = session.prompt_attribution().clone();
        assert_eq!(attribution.decoder_positions, 13);
        let snapshot_limits = SnapshotLimits {
            max_snapshots: 1,
            max_branches: 1,
            retained_bytes: 256 << 20,
            cumulative_copy_bytes: 2 << 30,
        };
        session
            .enable_snapshots(snapshot_limits, ORIGINAL_CAPACITY, copy_limits())
            .unwrap();
        let initial = session.snapshot(|_| ControlFlow::Continue(())).unwrap();
        let mut records = Vec::new();
        session
            .run(|r| {
                records.push(r);
                ControlFlow::Continue(())
            })
            .unwrap();
        let expected = session.token_ids().to_vec();
        assert_eq!(expected.len(), 4);
        let first = frames(&records);
        assert_eq!(first.len(), 4);
        assert_eq!(
            first[0].records[0]
                .payload
                .as_ref()
                .unwrap()
                .as_tensor()
                .unwrap()
                .shape()[1],
            13
        );
        if global {
            assert_eq!(first[0].records.len(), 6);
            for row in &first[0].records {
                assert!(row.payload.is_some());
            }
            assert!(matches!(
                first[0].records[3].outcome,
                CaptureOutcome::Truncated {
                    emitted_elements: 7,
                    ..
                }
            ));
            assert!(matches!(
                first[0].records[4].outcome,
                CaptureOutcome::Truncated {
                    emitted_elements: 0,
                    ..
                }
            ));
            let CapturePayload::Candidates(candidates) =
                first[0].records[5].payload.as_ref().unwrap()
            else {
                panic!("candidates")
            };
            assert!(candidates.domain.is_some());
        }
        let mut copy_usage = session.snapshot_usage().unwrap().cumulative_copy_bytes;
        let mut observed = first.last().unwrap().cumulative_usage;
        for _ in 0..2 {
            session
                .restore(&initial, |_| ControlFlow::Continue(()))
                .unwrap();
            assert_eq!(session.prompt_attribution(), &attribution);
            assert_eq!(session.status(), GenerationStatus::Prepared);
            assert!(session.snapshot_usage().unwrap().cumulative_copy_bytes > copy_usage);
            copy_usage = session.snapshot_usage().unwrap().cumulative_copy_bytes;
            let mut restored = Vec::new();
            session
                .run(|r| {
                    restored.push(r);
                    ControlFlow::Continue(())
                })
                .unwrap();
            assert_eq!(session.token_ids(), expected);
            let restored = frames(&restored);
            assert_eq!(restored.len(), 4);
            for (a, b) in restored.iter().zip(&first) {
                equal(a, b);
            }
            assert!(restored.last().unwrap().cumulative_usage.host_bytes > observed.host_bytes);
            observed = restored.last().unwrap().cumulative_usage;
        }
        // Preserve the original oversized child request as a typed negative.
        // Every possible semantic event slot is priced before native copying.
        assert!(
            trace
                .total_bytes
                .checked_mul(std::mem::size_of::<SemanticEvent>() as u64 + 1)
                .unwrap()
                > snapshot_limits.retained_bytes
        );
        let before_fork = session.snapshot_usage().unwrap();
        let before_status = session.status();
        let mut rejected_emission = false;
        let error = session
            .fork(
                &initial,
                GenerationBranchOptions {
                    trace_limits: trace,
                    capture_limits: Some(limits()),
                    sampling: None,
                    intervention: None,
                },
                |_| {
                    rejected_emission = true;
                    ControlFlow::Continue(())
                },
            )
            .err()
            .expect("oversized captured child trace rejects before native copy");
        let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(&error);
        let mut retained_limit = false;
        while let Some(error) = cause {
            if error.downcast_ref::<ControlledGenerationError>().is_some_and(|error| {
            matches!(error, ControlledGenerationError::Snapshot(snapshot)
                if matches!(snapshot.cause(), eredu_runtime::execution_control::TextSnapshotError::Control(
                    ExecutionControlError::Limit("retained bytes"))))
        }) {
            retained_limit = true;
            break;
        }
            cause = error.source();
        }
        assert!(retained_limit, "typed retained-byte refusal");
        assert!(!rejected_emission);
        assert_eq!(
            session.snapshot_usage().unwrap(),
            before_fork,
            "complete reservation and cumulative-copy usage is unchanged"
        );
        assert_eq!(session.status(), before_status);
        assert_eq!(session.token_ids(), expected);
        assert_eq!(session.prompt_attribution(), &attribution);
        drop(error);
        // Only the child trace changes. The parent's trace, snapshot, copy and
        // capture ceilings stay exact. Actual emitted bytes must fit this request.
        let child_trace = TraceLimits {
            per_record_bytes: 64 << 10,
            total_bytes: 64 << 10,
        };
        for _ in 0..2 {
            let before_child = session.snapshot_usage().unwrap();
            let mut branch = session
                .fork(
                    &initial,
                    GenerationBranchOptions {
                        trace_limits: child_trace,
                        capture_limits: Some(limits()),
                        sampling: None,
                        intervention: None,
                    },
                    |_| ControlFlow::Continue(()),
                )
                .unwrap();
            let after_child = session.snapshot_usage().unwrap();
            assert_eq!(after_child.branches, before_child.branches + 1);
            assert!(after_child.retained_bytes > before_child.retained_bytes);
            assert!(session.snapshot_usage().unwrap().cumulative_copy_bytes > copy_usage);
            copy_usage = session.snapshot_usage().unwrap().cumulative_copy_bytes;
            session
                .exchange(&mut branch, |_| ControlFlow::Continue(()))
                .unwrap();
            assert!(session.token_ids().is_empty());
            assert_eq!(session.prompt_attribution(), &attribution);
            let mut child = Vec::new();
            session
                .run(|r| {
                    child.push(r);
                    ControlFlow::Continue(())
                })
                .unwrap();
            assert_eq!(session.token_ids(), expected);
            assert!(
                session.emitted_bytes() > 0 && session.emitted_bytes() <= child_trace.total_bytes
            );
            assert_eq!(session.prompt_attribution(), &attribution);
            let child = frames(&child);
            assert_eq!(child.len(), 4);
            for (a, b) in child.iter().zip(&first) {
                equal(a, b);
            }
            session
                .exchange(&mut branch, |_| ControlFlow::Continue(()))
                .unwrap();
            assert_eq!(session.token_ids(), expected);
            drop(branch);
            assert_eq!(session.snapshot_usage().unwrap().branches, 0);
            assert_eq!(
                session.snapshot_usage().unwrap().cumulative_copy_bytes,
                copy_usage
            );
        }
        drop(initial);
        assert_eq!(session.snapshot_usage().unwrap().snapshots, 0);
        assert_eq!(
            session.snapshot_usage().unwrap().cumulative_copy_bytes,
            copy_usage
        );
    }
}
