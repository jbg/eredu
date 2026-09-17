//! One actual public original snapshot; no direct admission or copied engine.
use super::*;
use eredu_runtime::{
    execution_control::{SnapshotBudget, TextSnapshotError},
    working_memory::WorkspaceCopyLimits,
};

#[test]
#[ignore = "requires an accessible Metal device and complete managed saved-copy qualification"]
fn native_managed_snapshot_captures_after_commit_with_independent_custody() {
    check_snapshot(true, false);
}

#[test]
#[ignore = "requires an accessible Metal device and complete managed saved-copy qualification"]
fn native_managed_snapshot_captures_pending_prefill_with_independent_custody() {
    check_snapshot(false, false);
}

#[test]
#[ignore = "requires an accessible Metal device and complete managed saved-copy qualification"]
fn native_managed_hybrid_snapshot_captures_recurrent_state_with_independent_custody() {
    check_snapshot(true, true);
}

#[test]
#[ignore = "requires an accessible Metal device and complete managed resume qualification"]
fn native_managed_snapshot_restore_and_fork_match_uninterrupted_future() {
    check_resume(true, false);
}

#[test]
#[ignore = "requires an accessible Metal device and complete managed resume qualification"]
fn native_managed_hybrid_snapshot_restore_and_fork_match_uninterrupted_future() {
    check_resume(true, true);
}

#[test]
#[ignore = "requires an accessible Metal device and complete managed resume qualification"]
fn native_managed_pending_snapshot_restore_and_fork_match_uninterrupted_future() {
    check_resume(false, false);
}

#[test]
#[ignore = "requires an accessible Metal device and complete managed resume qualification"]
fn native_managed_hybrid_pending_snapshot_restore_and_fork_match_uninterrupted_future() {
    check_resume(false, true);
}

fn check_resume(after_commit: bool, hybrid: bool) {
    check_resume_with_state(
        after_commit,
        hybrid,
        eredu_runtime::CacheResidencyPolicy::Device,
    )
}

pub(super) fn check_resume_with_state(
    after_commit: bool,
    hybrid: bool,
    state: eredu_runtime::CacheResidencyPolicy,
) {
    check_resume_after_commits(usize::from(after_commit), hybrid, state, None)
}

pub(super) fn check_resume_after_commits(
    committed: usize,
    hybrid: bool,
    state: eredu_runtime::CacheResidencyPolicy,
    before_snapshot: Option<&dyn Fn()>,
) {
    assert!(committed < 4, "snapshot retains a future decode");
    let root = managed_fixture(if hybrid {
        super::super::components::qwen_35_fixture()
    } else {
        fixture(false)
    });
    let execution =
        ExecutionPlan::fully_resident(eredu_core::DevicePlan::new("mlx", "metal:0").unwrap());
    let (model, _) = LoadedModel::load_execution_plan(
        &MlxBackendFactory::default().with_state_residency(state),
        &root.0,
        &execution,
    )
    .unwrap()
    .into_parts();
    let expected: &[u32] = if hybrid { &[63, 32, 32, 32] } else { &[8, 38, 26, 1] };
    check_resume_loaded(model, root, committed, expected, before_snapshot,
        "PUBLIC_HOST_SAVED_PHASE");
}

pub(super) fn check_resume_loaded(
    model: LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,
    root: Fixture, committed: usize, expected: &[u32],
    before_snapshot: Option<&dyn Fn()>, phase_prefix: &str,
) -> serde_json::Value {
    check_resume_loaded_with_settings(model,root,committed,expected,before_snapshot,phase_prefix,settings(0.0))
}
pub(super) fn check_resume_loaded_with_settings(
    mut model: LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,
    root: Fixture, committed: usize, expected: &[u32],
    before_snapshot: Option<&dyn Fn()>, phase_prefix: &str,
    generation:PreparedChatGenerationSettings,
) -> serde_json::Value {
    assert!(committed < expected.len(), "snapshot retains a future decode");
    let source = model
        .compile_managed_plain_text_source(
            std::fs::File::open(root.0.join("tokenizer.json")).unwrap(),
        )
        .unwrap();
    // Every copy account retains its shared-domain ceiling. Keep the positive
    // request ceiling across saved and resumed owners, and express the separate
    // per-copy allowance through application_memory_budget_bytes.
    let host_capacity = generation.inference.managed_memory_capacity_bytes
        .expect("managed saved-state fixture");
    let cancellation = GenerationCancellationToken::new();
    let mut session = model
        .start_managed_plain_text(
            &source,
            ManagedPlainTextRequest::new(PROMPT, generation.clone()),
            &cancellation,
        )
        .unwrap_or_else(report_failure)
        .expect("live request");
    for _ in 0..committed {
        session = session
            .advance(&cancellation, &mut |_| {})
            .unwrap_or_else(report_failure);
    }
    let prefix = &expected[..committed];
    assert_eq!(session.token_ids(), prefix);
    if let Some(inspect) = before_snapshot {
        inspect();
    }
    let budget = SnapshotBudget::new(SnapshotLimits {
        max_snapshots: 1,
        max_branches: 1,
        retained_bytes: 8 * 1024 * 1024 * 1024,
        cumulative_copy_bytes: 32 * 1024 * 1024 * 1024,
    });
    let native = WorkspaceCopyLimits {
        capacity_bytes: host_capacity,
        application_memory_budget_bytes: Some(8 * 1024 * 1024 * 1024),
        safety_reserve_bytes: 0,
    };
    let phase = |name| {
        if before_snapshot.is_some() {
            eprintln!("{phase_prefix} {name}");
        }
    };
    phase("snapshot");
    let saved = session
        .snapshot(&budget, host_capacity, native)
        .unwrap_or_else(report_failure);
    assert_eq!(saved.remaining_tokens(), Some(4 - committed));
    phase("original-continuation");
    let output = session.run(&cancellation, &mut |_| {}).unwrap_or_else(report_failure);
    assert_eq!(output.token_ids.as_ref(), expected);
    let result = serde_json::json!({"ids":output.token_ids.as_ref(),"text":output.text.as_str()});
    drop(output);

    let mut resume = generation;
    resume.overrides.max_new_tokens = None;
    let cancelled = GenerationCancellationToken::new();
    cancelled.cancel();
    phase("cancelled-restore");
    let before = budget.usage();
    assert!(
        model
            .restore_managed_plain_text(&saved, resume.clone(), host_capacity, &cancelled)
            .unwrap()
            .is_none()
    );
    assert_eq!(budget.usage(), before);

    phase("refused-restore");
    let rejected = match model.restore_managed_plain_text(&saved, resume, 1, &cancellation) {
        Err(error) => error,
        Ok(_) => panic!("one byte cannot admit a fresh saved-state destination"),
    };
    assert!(budget_failure(&rejected), "{rejected:?}");
    drop(rejected);
    assert_eq!(budget.usage().snapshots, before.snapshots);
    assert_eq!(budget.usage().branches, before.branches);
    assert_eq!(budget.usage().retained_bytes, before.retained_bytes);
    assert!(budget.usage().cumulative_copy_bytes >= before.cumulative_copy_bytes);
    assert_eq!(saved.token_ids(), prefix);

    phase("restore");
    let restored = model
        .restore_managed_plain_text(&saved, resume.clone(), host_capacity, &cancellation)
        .unwrap_or_else(report_failure)
        .expect("fresh restored run");
    assert_eq!(restored.token_ids(), prefix);
    phase("restored-continuation");
    let output = restored.run(&cancellation, &mut |_| {}).unwrap_or_else(report_failure);
    assert_eq!(output.token_ids.as_ref(), expected);
    assert_eq!(output.text.as_str(), result["text"].as_str().unwrap());
    drop(output);
    let restored_copy = budget.usage().cumulative_copy_bytes;
    assert!(restored_copy > before.cumulative_copy_bytes);
    assert_eq!(saved.token_ids(), prefix);
    assert_eq!(saved.next_prediction(), u64::try_from(committed).unwrap());

    phase("refused-fork");
    let before_fork = budget.usage();
    let rejected = match model.fork_managed_plain_text(&saved, resume.clone(), 1, &cancellation) {
        Err(error) => error,
        Ok(_) => panic!("one byte cannot admit an independent saved-state branch"),
    };
    assert!(budget_failure(&rejected), "{rejected:?}");
    drop(rejected);
    assert_eq!(budget.usage().snapshots, before_fork.snapshots);
    assert_eq!(budget.usage().branches, before_fork.branches);
    assert_eq!(budget.usage().retained_bytes, before_fork.retained_bytes);
    let refused_fork_copy = budget.usage().cumulative_copy_bytes;
    assert!(refused_fork_copy >= before_fork.cumulative_copy_bytes);
    assert_eq!(saved.token_ids(), prefix);

    phase("fork");
    let branch = model
        .fork_managed_plain_text(&saved, resume, host_capacity, &cancellation)
        .unwrap_or_else(report_failure)
        .expect("fresh independent branch");
    assert_eq!(budget.usage().branches, 1);
    phase("fork-continuation");
    let output = branch.run(&cancellation, &mut |_| {}).unwrap_or_else(report_failure);
    assert_eq!(output.token_ids.as_ref(), expected);
    assert_eq!(output.text.as_str(), result["text"].as_str().unwrap());
    drop(output);
    assert_eq!(saved.token_ids(), prefix);
    let copied = budget.usage().cumulative_copy_bytes;
    assert!(copied > refused_fork_copy);
    phase("retirement");
    drop((saved, source, model));
    // Each ordinary reclamation pass takes a queue snapshot. Dropping a saved
    // decoder, native allocation owner or completed Host source can enqueue a
    // later owner, so finish those real deferred retirements before checking the
    // final copied-provider lease. No logical spending is refunded by this loop.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        eredu_backend_mlx::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
        let usage = budget.usage();
        if usage.snapshots == 0 && usage.branches == 0 && usage.retained_bytes == 0 {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "saved owners did not retire after deferred cleanup: {usage:?}"
        );
        std::thread::yield_now();
    }
    assert_eq!(budget.usage().snapshots, 0);
    assert_eq!(budget.usage().branches, 0);
    assert_eq!(budget.usage().retained_bytes, 0);
    assert_eq!(budget.usage().cumulative_copy_bytes, copied);
    result
}

fn check_snapshot(after_commit: bool, hybrid: bool) {
    let root = managed_fixture(if hybrid {
        super::super::components::qwen_35_fixture()
    } else {
        fixture(false)
    });
    let execution =
        ExecutionPlan::fully_resident(eredu_core::DevicePlan::new("mlx", "metal:0").unwrap());
    let (mut model, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &execution)
            .unwrap()
            .into_parts();
    let source = model
        .compile_managed_plain_text_source(
            std::fs::File::open(root.0.join("tokenizer.json")).unwrap(),
        )
        .unwrap();
    let cancellation = GenerationCancellationToken::new();
    let mut session = model
        .start_managed_plain_text(
            &source,
            ManagedPlainTextRequest::new(PROMPT, settings(0.0)),
            &cancellation,
        )
        .unwrap_or_else(|cause| {
            eprintln!("PUBLIC_MANAGED_SNAPSHOT_START_FAILURE: {cause:?}");
            report_failure(cause)
        })
        .expect("live request");
    if after_commit {
        session = session.advance(&cancellation, &mut |_| {}).unwrap();
    }
    let expected: &[u32] = if hybrid {
        &[63, 32, 32, 32]
    } else {
        &[8, 38, 26, 1]
    };
    let prefix = if after_commit { &expected[..1] } else { &[] };
    let prediction = u64::from(after_commit);
    assert_eq!(session.token_ids(), prefix);
    let limits = SnapshotLimits {
        max_snapshots: 1,
        max_branches: 0,
        retained_bytes: 8 * 1024 * 1024 * 1024,
        cumulative_copy_bytes: 16 * 1024 * 1024 * 1024,
    };
    let disabled = SnapshotBudget::new(SnapshotLimits {
        max_snapshots: 0,
        ..limits
    });
    let native = WorkspaceCopyLimits::new(8 * 1024 * 1024 * 1024);
    let error = match session.snapshot(&disabled, native.capacity_bytes, native) {
        Err(e) => e,
        Ok(_) => panic!("logical count must refuse before copying"),
    };
    assert!(matches!(
        error.cause(),
        TextSnapshotError::Control(ExecutionControlError::Limit("snapshot count"))
    ));
    assert_eq!(disabled.usage(), SnapshotUsage::default());
    let budget = SnapshotBudget::new(limits);
    let error = match session.snapshot(&budget, 1, native) {
        Err(e) => e,
        Ok(_) => panic!("one byte cannot hold an independent cursor"),
    };
    assert!(budget_failure(&error), "{error:?}");
    assert_eq!(budget.usage().snapshots, 0);
    assert_eq!(budget.usage().retained_bytes, 0);
    let failed_copy = budget.usage().cumulative_copy_bytes;
    assert!(failed_copy > 0);
    let saved = session
        .snapshot(&budget, native.capacity_bytes, native)
        .unwrap_or_else(|cause| {
            eprintln!("PUBLIC_MANAGED_SNAPSHOT_CAPTURE_FAILURE: {cause:?}");
            report_failure(cause)
        });
    assert_eq!(saved.token_ids(), prefix);
    assert_eq!(saved.next_prediction(), prediction);
    assert_eq!(saved.finish_reason(), None);
    assert_eq!(budget.usage().snapshots, 1);
    assert_eq!(budget.usage().retained_bytes, saved.retained_bytes());
    let output = session.run(&cancellation, &mut |_| {}).unwrap();
    assert_eq!(output.token_ids.as_ref(), expected);
    drop((output, source, model));
    assert_eq!(saved.token_ids(), prefix);
    assert_eq!(saved.next_prediction(), prediction);
    let copied = budget.usage().cumulative_copy_bytes;
    assert!(copied > failed_copy);
    drop(saved);
    assert_eq!(budget.usage().snapshots, 0);
    assert_eq!(budget.usage().retained_bytes, 0);
    assert_eq!(budget.usage().cumulative_copy_bytes, copied);
}
