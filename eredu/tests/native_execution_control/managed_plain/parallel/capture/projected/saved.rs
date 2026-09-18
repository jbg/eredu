//! Projected source owners through the existing snapshot/restore/fork driver.
use super::*;
use eredu_core::capture::{CaptureBudget, CaptureLimitPolicy};

const RECORDS: usize = 4;
// This is the same finite positive capacity as the projected decode fixture.
const CAPACITY: u64 = 16 * 1024 * 1024 * 1024;

fn selected_saved(discovery: &eredu_core::capture::CaptureDiscovery, world: usize, captures: u64) -> CapturePlan {
    let mut plan = selected::<true>(discovery, world);
    // Keep the exact per-source/receiver byte allowance, but only two logical
    // steps may capture. Restore must preserve exhaustion; fork inherits the
    // saved prefix instead of the parent's subsequently spent credits.
    plan.limits.cumulative.captures = 2 * captures;
    plan.limits.on_limit = CaptureLimitPolicy::Skip;
    plan
}

fn loaded<const AFTER_COMMIT: bool, const TENSOR: usize, const PIPELINE: usize>(
    model: LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>, root: Fixture,
) -> serde_json::Value {
    let topology = topology::<TENSOR, PIPELINE>();
    let descriptor = inspect_architecture(&root.0).unwrap();
    let expected = [expected_source(&descriptor, topology, UNITS),
        expected_source(&descriptor, topology, WRITE)];
    // Every declared producer contributes one actual native fragment for this
    // fixed strided geometry. Metadata/assembly/receiver helpers charge bytes,
    // while receipt_work_costs sums those native transform counts once globally.
    let captures = expected[0].producers.len() as u64 + 3 * expected[1].producers.len() as u64;
    super::super::super::super::capture::check_saved_capture_loaded_with_plan(
        model, root, AFTER_COMMIT, CAPACITY, captures,
        |discovery| selected_saved(discovery, topology.world_size(), captures),
        |ids, text, original, restored, branch| {
            inspect_escaped(AFTER_COMMIT, ids, text, original, restored, branch, &expected)
        },
    )
}

fn inspect_escaped(after_commit: bool, ids: &[u32], text: &str,
    original: &[SharedCapturedStep], restored: &[SharedCapturedStep], branch: &[SharedCapturedStep],
    expected: &[ExpectedSource; 2],
) -> serde_json::Value {
    fn check_stream(frames: &[SharedCapturedStep], first: u64, mut usage: CaptureUsage,
        exhausted: bool, transforms: &[u64; RECORDS]) {
        let captures: u64 = transforms.iter().sum();
        for (offset, frame) in frames.iter().enumerate() {
            let prediction = first + offset as u64;
            assert_eq!(frame.prediction_index(), prediction);
            assert_eq!(frame.phase(), if prediction == 0 { CapturePhase::Prefill } else { CapturePhase::Decode });
            assert_eq!(frame.outcome(), CaptureStepOutcome::Committed);
            assert_eq!(frame.records().len(), RECORDS);
            assert!(frame.same_storage(&frame.clone()));
            // Include retention, Host and encoded receiver credits, not just
            // the logical number of observations. No rewind refunds any axis.
            usage = usage.checked_add(frame.as_step().step_usage).unwrap();
            assert_eq!(frame.cumulative_usage(), usage);
            assert_eq!(usage.captures, captures * if exhausted { 2 } else { (prediction + 1).min(2) });
            let captured = !exhausted && prediction < 2;
            for (record, &transforms) in frame.records().iter().zip(transforms) {
                if captured {
                    assert!(matches!(record.outcome, CaptureOutcome::Captured));
                    assert!(record.payload.is_some());
                    assert_eq!(record.charged.captures, transforms);
                } else {
                    assert!(matches!(record.outcome, CaptureOutcome::Skipped {
                        reason: CaptureSkipReason::Limit { budget: CaptureBudget::Captures, cumulative: true }
                    }));
                    assert!(record.payload.is_none());
                }
            }
            if captured {
                assert!(frame.as_step().step_usage.host_bytes > 0);
                assert!(frame.as_step().step_usage.encoded_bytes > 0);
            }
        }
    }
    assert_eq!((original.len(), restored.len(), branch.len()), (4, 2, 2));
    let first = usize::from(after_commit);
    let transforms = [expected[0].producers.len() as u64,
        expected[1].producers.len() as u64, expected[1].producers.len() as u64,
        expected[1].producers.len() as u64];
    check_stream(original, 0, CaptureUsage::default(), false, &transforms);
    check_stream(restored, first as u64, original[3].cumulative_usage(), true, &transforms);
    let inherited = if after_commit { original[0].cumulative_usage() } else { CaptureUsage::default() };
    check_stream(branch, first as u64, inherited, false, &transforms);

    let captured_branch = 2 - first;
    for (offset, frame) in branch[..captured_branch].iter().enumerate() {
        let parent = &original[first + offset];
        assert!(!frame.same_storage(parent), "fork owns its actual independent captured frame");
        assert_eq!(frame.as_step().partitions.len(), RECORDS);
        assert_eq!(frame.as_step().partitions.len(), parent.as_step().partitions.len());
        for (actual, original) in frame.as_step().partitions.iter().zip(&parent.as_step().partitions) {
            assert_eq!(actual.context.artifact_identity, original.context.artifact_identity);
            assert_eq!(actual.context.execution_identity, original.context.execution_identity);
            assert_eq!(actual.context.overlay_identity, original.context.overlay_identity);
            assert_eq!(actual.context.capture_plan_identity, original.context.capture_plan_identity);
            assert_eq!(actual.context.selection_index, original.context.selection_index);
            assert_eq!(actual.context.prediction, original.context.prediction);
            assert_eq!(actual.context.phase, original.context.phase);
            assert_ne!(actual.context.run_identity, original.context.run_identity,
                "independently admitted fork must rebind its projected source run");
            assert_eq!(actual.producers, original.producers);
            assert_eq!(actual.combination, original.combination);
            assert_eq!(actual.contributions.len(), original.contributions.len());
            for (actual, original) in actual.contributions.iter().zip(&original.contributions) {
                assert_eq!(actual.producer_rank, original.producer_rank);
                assert_eq!(actual.local, original.local);
                assert_eq!(actual.destination, original.destination);
            }
        }
    }
    // The same evaluator checks true producer/combination geometry and derives
    // Summary/Histogram from the captured complete write. It accepts suffixes
    // at their retained logical indices; no fake index-zero frame is inserted.
    let original_frames: Vec<_> = original[..2].iter().map(SharedCapturedStep::as_step).collect();
    let branch_frames: Vec<_> = branch[..captured_branch].iter().map(SharedCapturedStep::as_step).collect();
    let parent_frames: Vec<_> = original[first..2].iter().map(SharedCapturedStep::as_step).collect();
    let original_value = evaluate_frames::<true>(ids, text, &original_frames, true, expected);
    let branch_value = evaluate_frames::<true>(ids, text, &branch_frames, true, expected);
    let parent_value = evaluate_frames::<true>(ids, text, &parent_frames, true, expected);
    compare(&branch_value, &parent_value, "escaped projected fork", 0);
    original_value
}

fn compare(actual: &serde_json::Value, expected: &serde_json::Value, mode: &str, rank: usize) {
    super::super::compare(actual, expected, mode, rank);
    for (actual, expected) in actual["frames"].as_array().unwrap().iter()
        .zip(expected["frames"].as_array().unwrap()) {
        assert_eq!(actual["phase"], expected["phase"], "{mode} rank{rank} logical phase");
    }
}

fn run<const AFTER_COMMIT: bool, const TENSOR: usize, const PIPELINE: usize>(mode: &str) -> serde_json::Value {
    if mode == "serial" {
        // Reuse the existing ordinary projected numerical oracle. The saved
        // plan deliberately captures only its first two logical predictions.
        let mut value = super::run::<true, TENSOR, PIPELINE>(mode);
        value["frames"].as_array_mut().unwrap().truncate(2);
        return value;
    }
    assert_eq!(mode, "lifecycle");
    run_partitioned_with_lifecycle(mode, topology::<TENSOR, PIPELINE>(),
        Some(loaded::<AFTER_COMMIT, TENSOR, PIPELINE>))
}

fn check<const AFTER_COMMIT: bool, const TENSOR: usize, const PIPELINE: usize>(
    case: &str, mode: &str, label: &str,
) {
    compare_selected_modes_by(case, mode, "PUBLIC_PROJECTED_SAVED_RESULT:", label,
        TENSOR * PIPELINE, &["lifecycle"], run::<AFTER_COMMIT, TENSOR, PIPELINE>, compare)
}

#[test]
#[ignore = "requires Metal and two local Ring processes"]
fn native_projected_tp_capture_pending_restore_and_fork_preserve_sources_and_spending() {
    check::<false, 2, 1>(
        "managed_plain::parallel::capture::projected::saved::native_projected_tp_capture_pending_restore_and_fork_preserve_sources_and_spending",
        "EREDU_PUBLIC_PROJECTED_TP_PENDING_SAVED_MODE", "projected TP pending saved capture");
}
#[test]
#[ignore = "requires Metal and two local Ring processes"]
fn native_projected_tp_capture_committed_restore_and_fork_preserve_sources_and_spending() {
    check::<true, 2, 1>(
        "managed_plain::parallel::capture::projected::saved::native_projected_tp_capture_committed_restore_and_fork_preserve_sources_and_spending",
        "EREDU_PUBLIC_PROJECTED_TP_COMMITTED_SAVED_MODE", "projected TP committed saved capture");
}
#[test]
#[ignore = "requires Metal and two local Ring processes; run after projected pipeline decode closes"]
fn native_projected_pipeline_capture_pending_restore_and_fork_preserve_sources_and_spending() {
    check::<false, 1, 2>(
        "managed_plain::parallel::capture::projected::saved::native_projected_pipeline_capture_pending_restore_and_fork_preserve_sources_and_spending",
        "EREDU_PUBLIC_PROJECTED_PP_PENDING_SAVED_MODE", "projected PP pending saved capture");
}
#[test]
#[ignore = "requires Metal and two local Ring processes; run after projected pipeline decode closes"]
fn native_projected_pipeline_capture_committed_restore_and_fork_preserve_sources_and_spending() {
    check::<true, 1, 2>(
        "managed_plain::parallel::capture::projected::saved::native_projected_pipeline_capture_committed_restore_and_fork_preserve_sources_and_spending",
        "EREDU_PUBLIC_PROJECTED_PP_COMMITTED_SAVED_MODE", "projected PP committed saved capture");
}
#[test]
#[ignore = "requires Metal and four local Ring processes; run after projected combined decode closes"]
fn native_projected_combined_capture_pending_restore_and_fork_preserve_sources_and_spending() {
    check::<false, 2, 2>(
        "managed_plain::parallel::capture::projected::saved::native_projected_combined_capture_pending_restore_and_fork_preserve_sources_and_spending",
        "EREDU_PUBLIC_PROJECTED_COMBINED_PENDING_SAVED_MODE", "projected combined pending saved capture");
}
#[test]
#[ignore = "requires Metal and four local Ring processes; run after projected combined decode closes"]
fn native_projected_combined_capture_committed_restore_and_fork_preserve_sources_and_spending() {
    check::<true, 2, 2>(
        "managed_plain::parallel::capture::projected::saved::native_projected_combined_capture_committed_restore_and_fork_preserve_sources_and_spending",
        "EREDU_PUBLIC_PROJECTED_COMBINED_COMMITTED_SAVED_MODE", "projected combined committed saved capture");
}
