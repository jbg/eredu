//! Actual saved capture across fresh native admission and the shared text cursor.
use super::*;
use eredu_core::capture::{CaptureLimitPolicy, SharedCapturedStep};
use eredu_runtime::{execution_control::SnapshotBudget, working_memory::WorkspaceCopyLimits};

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_managed_capture_restore_preserves_spending_and_fork_inherits_saved_usage() {
    check_saved_capture(true);
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_managed_pending_capture_restore_accepts_shorter_output_allowance() {
    check_saved_capture(false);
}

fn check_saved_capture(after_commit: bool) {
    let root = managed_fixture(fixture(false));
    let execution =
        ExecutionPlan::fully_resident(eredu_core::DevicePlan::new("mlx", "metal:0").unwrap());
    let (model, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &execution)
            .unwrap()
            .into_parts();
    check_saved_capture_loaded(model, root, after_commit, "model.logits", 64, false);
}

pub(in super::super) fn check_saved_capture_loaded(
    model: LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>, root: Fixture,
    after_commit: bool, path: &str, width: usize, partitioned: bool,
) -> serde_json::Value {
    check_saved_capture_loaded_with_transform(model, root, after_commit, path, width, partitioned,
        CaptureTransform::FullTensor, Some(&[8, 38, 26, 1]))
}

pub(in super::super) fn check_saved_capture_loaded_with_transform(
    model: LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>, root: Fixture,
    after_commit: bool, path: &str, width: usize, partitioned: bool,
    transform: CaptureTransform, expected_tokens: Option<&[u32]>,
) -> serde_json::Value {
    check_saved_capture_input(model, root, after_commit, path, width, partitioned,
        transform, expected_tokens, None)
}

pub(in super::super) fn check_saved_prepared_capture_loaded(
    model: LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>, root: Fixture,
    input: eredu_runtime::input::OriginalModelInput<eredu_backend_mlx::native::MlxModelInput>,
    after_commit: bool,
) -> serde_json::Value {
    check_saved_capture_input(model, root, after_commit,
        eredu_core::MODEL_LOGITS_OBSERVATION_PATH, 64, false,
        CaptureTransform::FullTensor, None, Some(input))
}

fn check_saved_capture_input(
    model: LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>, root: Fixture,
    after_commit: bool, path: &str, width: usize, partitioned: bool,
    transform: CaptureTransform, expected_tokens: Option<&[u32]>,
    input: Option<eredu_runtime::input::OriginalModelInput<eredu_backend_mlx::native::MlxModelInput>>,
) -> serde_json::Value {
    let sparse = matches!(transform, CaptureTransform::RoutedUnits);
    let mut plan = CapturePlan::none();
    plan.selections.push(CaptureSelection {
        id: "saved logits".into(),
        path: path.into(),
        schedule: CaptureSchedule::default(),
        slices: vec![],
        transform,
    });
    plan.limits.per_step = CaptureUsage {
        captures: 1,
        retained_bytes: 16 << 20,
        host_bytes: 16 << 20,
        encoded_bytes: 16 << 20,
    };
    plan.limits.cumulative = CaptureUsage {
        captures: 2,
        ..plan.limits.per_step
    };
    if partitioned {
        // One complete source record and two receivers use the same canonical
        // bound as the raw public case. Preserve the two-value logical limit;
        // source/delivery bookkeeping has its separate finite byte allowance.
        plan.limits.per_step.host_bytes = 64 << 20;
        plan.limits.per_step.retained_bytes = 8 << 20;
        plan.limits.per_step.encoded_bytes = 1 << 20;
        plan.limits.cumulative.host_bytes = 256 << 20;
        plan.limits.cumulative.retained_bytes = 32 << 20;
        plan.limits.cumulative.encoded_bytes = 4 << 20;
    }
    plan.limits.on_limit = CaptureLimitPolicy::Skip;
    check_saved_capture_run(model, root, after_commit, 8 << 30, 1,
        expected_tokens, input, None, |_| plan, |ids, text, original, _, branch| {
            compare_single_saved_capture(after_commit, width, partitioned, sparse, original, branch);
            serde_json::json!({"ids":ids, "text":text})
        })
}

/// The existing snapshot/restore/fork driver with a caller-selected capture plan.
/// Capture-count credits cover actual transforms; receiver bytes remain separate.
pub(in super::super) fn check_saved_capture_loaded_with_plan(
    model: LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>, root: Fixture,
    after_commit: bool, capacity: u64, captures_per_step: u64,
    make_plan: impl FnOnce(&eredu_core::capture::CaptureDiscovery) -> CapturePlan,
    inspect_escaped: impl FnOnce(&[u32], &str, &[SharedCapturedStep],
        &[SharedCapturedStep], &[SharedCapturedStep]) -> serde_json::Value,
) -> serde_json::Value {
    check_saved_capture_run(model, root, after_commit, capacity, captures_per_step,
        None, None, None, make_plan, inspect_escaped)
}

fn check_saved_capture_run(
    mut model: LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>, root: Fixture,
    after_commit: bool, capacity: u64, captures_per_step: u64,
    expected_tokens: Option<&[u32]>,
    input: Option<eredu_runtime::input::OriginalModelInput<eredu_backend_mlx::native::MlxModelInput>>,
    interventions: Option<eredu_core::intervention::SharedInterventionPlan>,
    make_plan: impl FnOnce(&eredu_core::capture::CaptureDiscovery) -> CapturePlan,
    inspect_escaped: impl FnOnce(&[u32], &str, &[SharedCapturedStep],
        &[SharedCapturedStep], &[SharedCapturedStep]) -> serde_json::Value,
) -> serde_json::Value {
    let source = model.compile_managed_plain_text_source(
        std::fs::File::open(root.0.join("tokenizer.json")).unwrap()).unwrap();
    let discovery = model.capture_discovery().unwrap();
    let plan = make_plan(&discovery);
    let capture_limit = captures_per_step.checked_mul(2).unwrap();
    assert!(captures_per_step > 0);
    assert_eq!(plan.limits.cumulative.captures, capture_limit);
    assert!(matches!(plan.limits.on_limit, CaptureLimitPolicy::Skip));
    let capture = SharedCapturePlan::new(
        plan.admit(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: 5,
                max_predictions: 4,
            },
        )
        .unwrap(),
    );
    let cancellation = GenerationCancellationToken::new();
    let mut original_frames = Vec::new();
    let mut original_observer = |_: Option<u32>, frame: Option<CapturedStepDelivery>, _: f64| {
        let Some(CapturedStepDelivery::Shared(frame)) = frame else {
            panic!("expected the admitted shared capture owner");
        };
        original_frames.push(frame);
    };
    let mut initial = settings(0.0);
    initial.inference.managed_memory_capacity_bytes = Some(capacity);
    let mut session = match (input, interventions) {
        (Some(input), Some(edits)) => model.start_intervened_managed_prepared_input(
            &source, ManagedPreparedInputRequest::from_original(input, initial),
            capture, edits, &cancellation, &mut original_observer),
        (None, Some(edits)) => model.start_intervened_managed_plain_text(
            &source, ManagedPlainTextRequest::new(PROMPT, initial),
            capture, edits, &cancellation, &mut original_observer),
        (Some(input), None) => model.start_observed_managed_prepared_input(
            &source, ManagedPreparedInputRequest::from_original(input, initial),
            capture, &cancellation, &mut original_observer),
        (None, None) => model.start_observed_managed_plain_text(
            &source, ManagedPlainTextRequest::new(PROMPT, initial),
            capture, &cancellation, &mut original_observer),
    }.unwrap_or_else(report_failure).unwrap();
    if after_commit {
        session = session
            .advance(&cancellation, &mut |_| {})
            .unwrap_or_else(report_failure);
    }
    let budget = SnapshotBudget::new(SnapshotLimits {
        max_snapshots: 1,
        max_branches: 1,
        retained_bytes: capacity,
        cumulative_copy_bytes: capacity.checked_mul(4).unwrap(),
    });
    let copy = WorkspaceCopyLimits::new(capacity);
    let saved = session
        .snapshot(&budget, copy.capacity_bytes, copy)
        .unwrap_or_else(report_failure);
    let original = session
        .run(&cancellation, &mut |_| {})
        .unwrap_or_else(report_failure);
    if let Some(expected) = expected_tokens {
        assert_eq!(original.token_ids.as_ref(), expected);
    }
    assert_eq!(original.token_ids.len(), 4);
    drop(original_observer);
    assert_eq!(original_frames.len(), 4);
    assert_eq!(original_frames[3].cumulative_usage().captures, capture_limit);
    assert_eq!(saved.next_prediction(), u64::from(after_commit));

    // Two outputs deliberately shorten the saved allowance. Restoration shares
    // the original cumulative ledger, which the source has now exhausted.
    let mut resume = settings(0.0);
    resume.overrides.max_new_tokens = Some(2);
    resume.inference.managed_memory_capacity_bytes = Some(capacity);
    let mut restored_frames = Vec::new();
    let mut restored_observer = |_: Option<u32>, frame: Option<CapturedStepDelivery>, _: f64| {
        let Some(CapturedStepDelivery::Shared(frame)) = frame else {
            panic!("restoration must deliver its saved capture source");
        };
        restored_frames.push(frame);
    };
    let restored = model
        .restore_managed_plain_text(&saved, resume.clone(), copy.capacity_bytes, &cancellation)
        .unwrap_or_else(report_failure)
        .unwrap()
        .with_capture_observer(&mut restored_observer)
        .run(&cancellation, &mut |_| {})
        .unwrap_or_else(report_failure);
    drop(restored_observer);
    let end = usize::from(after_commit) + 2;
    assert_eq!(
        restored.token_ids.as_ref(),
        &original.token_ids.as_ref()[..end]
    );
    assert_eq!(restored_frames.len(), 2);
    for (i, frame) in restored_frames.iter().enumerate() {
        assert_eq!(frame.prediction_index(), u64::from(after_commit) + i as u64);
        assert_eq!(frame.cumulative_usage().captures, capture_limit);
        let phase = if !after_commit && i == 0 {
            CapturePhase::Prefill
        } else {
            CapturePhase::Decode
        };
        assert_eq!(
            frame.phase(),
            phase,
            "native opening must preserve the logical capture phase"
        );
        assert_eq!(frame.outcome(), CaptureStepOutcome::Committed);
        assert!(frame
            .records()
            .iter()
            .all(|record| record.payload.is_none()));
    }

    // A separately admitted branch starts at the immutable saved usage, so the
    // first resumed value is available even after the parent spent its limit.
    let mut branch_frames = Vec::new();
    let mut branch_observer = |_: Option<u32>, frame: Option<CapturedStepDelivery>, _: f64| {
        let Some(CapturedStepDelivery::Shared(frame)) = frame else {
            panic!("branch must deliver its independent saved capture source");
        };
        branch_frames.push(frame);
    };
    let branch = model
        .fork_managed_plain_text(&saved, resume, copy.capacity_bytes, &cancellation)
        .unwrap_or_else(report_failure)
        .unwrap()
        .with_capture_observer(&mut branch_observer)
        .run(&cancellation, &mut |_| {})
        .unwrap_or_else(report_failure);
    drop(branch_observer);
    assert_eq!(branch.token_ids.as_ref(), restored.token_ids.as_ref());
    assert_eq!(branch_frames.len(), 2);
    assert_eq!(
        branch_frames[0].phase(),
        if after_commit {
            CapturePhase::Decode
        } else {
            CapturePhase::Prefill
        }
    );
    assert_eq!(
        branch_frames[0].cumulative_usage().captures,
        captures_per_step * (1 + u64::from(after_commit))
    );
    assert_eq!(branch_frames[1].cumulative_usage().captures, capture_limit);
    assert_eq!(budget.usage().branches, 1);
    drop((saved, source, model, root));

    // The caller inspects escaped frames after model/source/snapshot teardown.
    assert_eq!(restored.token_ids.as_ref(), &original.token_ids.as_ref()[..end]);
    inspect_escaped(original.token_ids.as_ref(), original.text.as_str(),
        &original_frames, &restored_frames, &branch_frames)
}

fn compare_single_saved_capture(after_commit: bool, width: usize, partitioned: bool, sparse: bool,
    original_frames: &[SharedCapturedStep], branch_frames: &[SharedCapturedStep]) {
    fn logits(frame: &SharedCapturedStep) -> &[f32] {
        let TensorObservationData::F32(values) = frame.records()[0]
            .payload
            .as_ref()
            .unwrap()
            .as_tensor()
            .unwrap()
            .data()
        else {
            panic!("expected f32 capture");
        };
        values
    }
    if partitioned {
        let original_evidence = &original_frames[usize::from(after_commit)].as_step().partitions;
        let branch_evidence = &branch_frames[0].as_step().partitions;
        assert_eq!(original_evidence.len(), 1);
        assert_eq!(branch_evidence.len(), 1);
        for evidence in original_evidence.iter().chain(branch_evidence) {
            evidence.context.validate().unwrap();
            assert_eq!(evidence.producers, [0]);
            assert_eq!(evidence.context.prediction, u64::from(after_commit));
        }
        assert_ne!(original_evidence[0].context.run_identity, branch_evidence[0].context.run_identity,
            "an independent branch retains a distinct prepared run identity");
    }
    if sparse {
        use eredu_core::capture::CapturePayload;
        let Some(CapturePayload::RoutedUnits(expected)) =
            &original_frames[usize::from(after_commit)].records()[0].payload else {
            panic!("original routed payload");
        };
        let Some(CapturePayload::RoutedUnits(actual)) = &branch_frames[0].records()[0].payload else {
            panic!("branched routed payload");
        };
        assert_eq!(expected.geometry, actual.geometry);
        assert_eq!(expected.rows.len(), if after_commit { 4 } else { 20 });
        assert_eq!(actual.rows.len(), expected.rows.len());
        for (a, b) in actual.rows.iter().zip(&expected.rows) {
            assert_eq!((a.token, a.slot, a.expert, a.unit_start, a.unit_stride),
                (b.token, b.slot, b.expert, b.unit_start, b.unit_stride));
            assert!((a.coefficient - b.coefficient).abs() < 5e-5);
            let (TensorObservationData::F32(a), TensorObservationData::F32(b)) =
                (a.values.data(), b.values.data()) else { panic!("float routed units"); };
            assert_eq!(a.len(), width);
            assert_eq!(a.len(), b.len());
            assert!(a.iter().any(|v| v.abs() > 1e-6));
            assert!(a.iter().zip(b).all(|(a,b)| a.is_finite() && (a-b).abs() <= 5e-5));
        }
    } else {
        let expected = logits(&original_frames[usize::from(after_commit)]);
        let actual = logits(&branch_frames[0]);
        assert_eq!(expected.len(), if after_commit { width } else { 5 * width });
        assert_eq!(actual.len(), expected.len());
        assert!(actual.iter().any(|v| v.abs() > 1e-6));
        for (a, b) in actual.iter().zip(expected) {
            assert!(a.is_finite() && (*a - *b).abs() <= 1e-5 + 1e-5 * b.abs());
        }
    }
}

fn check_saved_interventions(after_commit: bool) {
    let root = managed_fixture(fixture(false));
    let execution = ExecutionPlan::fully_resident(
        eredu_core::DevicePlan::new("mlx", "metal:0").unwrap());
    let (model, _) = LoadedModel::load_execution_plan(
        &MlxBackendFactory::default(), &root.0, &execution).unwrap().into_parts();
    let edits = super::interventions::admitted_edits(
        &model.intervention_discovery().unwrap(),
        CaptureRequestShape { batch: 1, prompt_tokens: 5, max_predictions: 4 },
    );
    check_saved_capture_run(
        model, root, after_commit, 8 << 30, 1, None, None, Some(edits),
        |_| {
            let mut plan = CapturePlan::none();
            plan.selections.push(CaptureSelection {
                id: "edited saved scores".into(),
                path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
                schedule: CaptureSchedule::default(), slices: vec![],
                transform: CaptureTransform::FullTensor,
            });
            plan.limits.per_step = CaptureUsage { captures: 1,
                retained_bytes: 16 << 20, host_bytes: 16 << 20, encoded_bytes: 16 << 20 };
            plan.limits.cumulative = CaptureUsage { captures: 2, ..plan.limits.per_step };
            plan.limits.on_limit = CaptureLimitPolicy::Skip;
            plan
        },
        |ids, text, original, restored, branch| {
            use eredu_core::intervention::InterventionOutcome;
            assert_eq!(ids[1], 17);
            assert_eq!(ids[3], 17);
            for frame in original.iter().chain(restored).chain(branch) {
                assert!(frame.clone().same_storage(frame));
                assert_eq!(frame.interventions().len(), 1);
                let record = &frame.interventions()[0];
                assert_eq!(record.operation_id, "alternate-decode");
                assert_eq!(record.prediction_index, frame.prediction_index());
                assert_eq!(record.outcome, if frame.prediction_index() % 2 == 0 {
                    InterventionOutcome::Inactive
                } else { InterventionOutcome::Applied });
            }
            serde_json::json!({ "ids": ids, "text": text })
        },
    );
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_managed_pending_interventions_restore_and_fork_keep_schedule_and_spending() {
    check_saved_interventions(false);
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_managed_committed_interventions_restore_and_fork_keep_schedule_and_spending() {
    check_saved_interventions(true);
}
