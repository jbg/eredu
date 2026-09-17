use super::*;
use crate::host_authority::Guard;
use eredu::api::{
    ControlledGenerationError, ControlledGenerationSession, ControlledGenerationSnapshot,
    GenerationBranchOptions,
};
use eredu_core::{
    Admission, ExecutionWorkspaceEstimate, InferenceGeometry, LayerSchedule, OutputDemand,
    WorkspaceBound,
};
use eredu_runtime::{
    execution_control::TextSnapshotError,
    working_memory::{InferenceExecutionIdentity, WorkingMemoryError, WorkingMemoryPool},
};

fn ignore(_: ControlledGenerationRecord) -> ControlFlow<()> {
    ControlFlow::Continue(())
}

fn unexpected_record(_: ControlledGenerationRecord) -> ControlFlow<()> {
    panic!("rejected host preparation must not publish a control record")
}

fn branch_options() -> GenerationBranchOptions {
    GenerationBranchOptions {
        trace_limits: limits(),
        capture_limits: None,
        sampling: None,
        intervention: None,
    }
}

fn copy_limits() -> SnapshotLimits {
    SnapshotLimits {
        max_snapshots: 3,
        max_branches: 2,
        retained_bytes: 64_000_000,
        cumulative_copy_bytes: 256_000_000,
    }
}

#[derive(Clone, Copy, Debug)]
enum CopyOperation {
    Snapshot,
    Restore,
    Fork,
}

impl CopyOperation {
    fn reject(
        self,
        run: &mut ControlledGenerationSession<'_, MockBackend>,
        saved: &ControlledGenerationSnapshot<MockBackend>,
    ) -> ControlledGenerationError {
        match self {
            Self::Snapshot => run.snapshot(unexpected_record).map(|_| ()),
            Self::Restore => run.restore(saved, unexpected_record),
            Self::Fork => run
                .fork(saved, branch_options(), unexpected_record)
                .map(|_| ()),
        }
        .unwrap_err()
    }
}

fn cause<'a, T: std::error::Error + 'static>(
    mut error: &'a (dyn std::error::Error + 'static),
) -> Option<&'a T> {
    loop {
        if let Some(cause) = error.downcast_ref::<T>() {
            return Some(cause);
        }
        error = error.source()?;
    }
}

// A real zero-byte finite reservation is only an exclusion fixture. This mock
// supplies no finite native/parser copy proof, and these tests claim none.
fn exclusion_admission() -> Admission {
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 1,
        max_output_tokens: 0,
        prefill_chunk_positions: 1,
        output: OutputDemand::StateOnly,
    };
    let layout = StateMemoryLayout::new(
        LayerSchedule::new(1, vec![eredu_core::cache::LayerCachePolicy::NoState]).unwrap(),
        vec![0],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let zero = || WorkspaceBound::bounded(0, "snapshot host exclusion fixture");
    let state = eredu_core::estimate_runtime_state(
        &layout,
        InputTokenCount::text(1),
        0,
        1,
        NonZeroU8::new(4).unwrap(),
    )
    .unwrap()
    .with_execution_workspace(ExecutionWorkspaceEstimate {
        geometry,
        activations: zero(),
        attention: zero(),
        vocabulary: zero(),
        state_update: zero(),
        materialization: zero(),
        retained: zero(),
    })
    .unwrap();
    Admission {
        requested_positions: 1,
        state,
        incremental_required_bytes: 0,
        available_memory_bytes: None,
    }
}

#[test]
fn zero_finite_reservation_rejects_snapshot_restore_and_fork_without_work_or_budget() {
    for operation in [
        CopyOperation::Snapshot,
        CopyOperation::Restore,
        CopyOperation::Fork,
    ] {
        let (mut model, chat, settings, first) = super::super::setup();
        let prepared = model
            .prepare_observed_chat(&chat, settings, CapturePlan::none(), limits())
            .unwrap();
        let mut run = model
            .start_controlled_text(prepared, &[], Default::default(), ignore)
            .unwrap();
        run.enable_snapshots(copy_limits()).unwrap();
        run.step(ignore).unwrap();
        let saved = run.snapshot(ignore).unwrap();
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let reservation = pool
            .reserve_with_capacity(
                &InferenceExecutionIdentity::default(),
                &exclusion_admission(),
                0,
            )
            .unwrap();
        let probe = Guard::new(&pool);
        let before = (run.status(), run.next_prediction(), run.snapshot_usage());
        let accounting = (pool.used_bytes().unwrap(), pool.peak_bytes().unwrap());
        let error = operation.reject(&mut run, &saved);
        assert!(matches!(
            &error,
            ControlledGenerationError::Snapshot(TextSnapshotError::HostPreparation(_))
        ));
        assert_eq!(
            cause::<WorkingMemoryError>(&error),
            Some(&WorkingMemoryError::ReservedWorkActive)
        );
        assert_eq!(probe.update(|p| p.attempts), 1, "{operation:?}");
        assert!(probe.update(|p| p.copies.is_empty()));
        assert_eq!(
            before,
            (run.status(), run.next_prediction(), run.snapshot_usage())
        );
        assert_eq!(run.token_ids(), [first]);
        assert_eq!(saved.token_ids(), [first]);
        assert_eq!(
            accounting,
            (pool.used_bytes().unwrap(), pool.peak_bytes().unwrap())
        );
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
        drop(reservation);
        run.step(ignore).unwrap();
        assert_eq!(run.token_ids(), [first, first + 1]);
        let retry = run.snapshot(ignore).unwrap();
        assert!(pool.unquoted_owner_count().unwrap() > 0);
        drop(retry);
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    }
}

#[test]
fn each_host_admission_rejects_before_copy_and_preserves_partial_semantic_state() {
    for operation in [
        CopyOperation::Snapshot,
        CopyOperation::Restore,
        CopyOperation::Fork,
    ] {
        for reject_at in [1, 2] {
            let (mut model, chat, settings, first) = snapshot_setup();
            let prepared = model
                .prepare_observed_chat(&chat, settings, CapturePlan::none(), limits())
                .unwrap();
            let mut run = model
                .start_controlled_chat(prepared, &[], Default::default(), ignore)
                .unwrap();
            run.enable_snapshots(copy_limits()).unwrap();
            run.step(ignore).unwrap();
            let saved = run.snapshot(ignore).unwrap();
            let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
            let probe = Guard::new(&pool);
            probe.update(|p| p.reject_at = Some(reject_at));
            let before = (run.status(), run.next_prediction(), run.snapshot_usage());
            let error = operation.reject(&mut run, &saved);
            assert!(matches!(
                &error,
                ControlledGenerationError::Snapshot(TextSnapshotError::HostPreparation(_))
            ));
            assert!(
                matches!(cause::<MockError>(&error), Some(MockError::Capture(message))
                if message == "host snapshot preparation rejected")
            );
            assert_eq!(probe.update(|p| p.attempts), reject_at, "{operation:?}");
            assert!(probe.update(|p| p.copies.is_empty()));
            assert_eq!(
                before,
                (run.status(), run.next_prediction(), run.snapshot_usage())
            );
            assert_eq!(run.token_ids(), [first]);
            assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
            assert_eq!(
                (pool.used_bytes().unwrap(), pool.peak_bytes().unwrap()),
                (0, 0)
            );
            probe.update(|p| p.reject_at = None);
            let mut records = Vec::new();
            run.run(collect(&mut records)).unwrap();
            let text: String = semantic(&records)
                .into_iter()
                .filter_map(|event| match event {
                    SemanticEvent::TextDelta(text) => Some(text.as_str().to_owned()),
                    _ => None,
                })
                .collect();
            assert_eq!(text, "é");
            assert_eq!(run.token_ids(), [first, first + 1, first + 2]);
            assert_eq!(saved.token_ids(), [first]);
        }
    }
}

#[test]
fn snapshot_and_branch_authority_follows_payload_through_exchange_and_retirement() {
    let (mut model, chat, settings, first) = snapshot_setup();
    let prepared = model
        .prepare_observed_chat(&chat, settings, CapturePlan::none(), limits())
        .unwrap();
    let mut run = model
        .start_controlled_chat(prepared, &[], Default::default(), ignore)
        .unwrap();
    run.enable_snapshots(copy_limits()).unwrap();
    run.step(ignore).unwrap();
    let snapshot_pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let branch_pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let probe = Guard::new(&snapshot_pool);
    let saved = run.snapshot(ignore).unwrap();
    assert!(snapshot_pool.unquoted_owner_count().unwrap() > 0);
    probe.update(|p| p.pool = Some(branch_pool.clone()));
    let mut branch = run.fork(&saved, branch_options(), ignore).unwrap();
    assert!(branch_pool.unquoted_owner_count().unwrap() > 0);
    run.step(ignore).unwrap();
    assert_eq!(run.token_ids(), [first, first + 1]);
    assert_eq!(branch.token_ids(), [first]);
    let before = (
        run.snapshot_usage(),
        probe.update(|p| (p.attempts, p.copies.len())),
    );
    run.exchange(&mut branch, ignore).unwrap();
    assert_eq!(
        before,
        (
            run.snapshot_usage(),
            probe.update(|p| (p.attempts, p.copies.len()))
        )
    );
    assert_eq!(run.token_ids(), [first]);
    assert_eq!(branch.token_ids(), [first, first + 1]);
    run.step(ignore).unwrap();
    assert_eq!(run.token_ids(), [first, first + 1]);
    drop(run);
    drop(model);
    drop(chat);
    // The branch now contains the original unmanaged state. Its former unquoted
    // host custody moved into the active run and retired with that run.
    assert_eq!(branch_pool.unquoted_owner_count().unwrap(), 0);
    assert!(snapshot_pool.unquoted_owner_count().unwrap() > 0);
    assert_eq!(saved.token_ids(), [first]);
    drop(saved);
    assert_eq!(snapshot_pool.unquoted_owner_count().unwrap(), 0);
    assert_eq!(branch.token_ids(), [first, first + 1]);
    drop(branch);
}

#[test]
fn restore_installs_fresh_authority_and_retains_source_after_saved_handle_drops() {
    let (mut model, chat, settings, first) = super::super::setup();
    let prepared = model
        .prepare_observed_chat(&chat, settings, CapturePlan::none(), limits())
        .unwrap();
    let mut run = model
        .start_controlled_text(prepared, &[], Default::default(), ignore)
        .unwrap();
    run.enable_snapshots(copy_limits()).unwrap();
    run.step(ignore).unwrap();
    let source_pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let destination_pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let probe = Guard::new(&source_pool);
    let saved = run.snapshot(ignore).unwrap();
    run.step(ignore).unwrap();
    probe.update(|p| p.pool = Some(destination_pool.clone()));
    run.restore(&saved, ignore).unwrap();
    drop(saved);
    assert!(source_pool.unquoted_owner_count().unwrap() > 0);
    assert!(destination_pool.unquoted_owner_count().unwrap() > 0);
    assert_eq!(run.token_ids(), [first]);
    run.step(ignore).unwrap();
    assert_eq!(run.token_ids(), [first, first + 1]);
    drop(run);
    drop(model);
    drop(chat);
    assert_eq!(source_pool.unquoted_owner_count().unwrap(), 0);
    assert_eq!(destination_pool.unquoted_owner_count().unwrap(), 0);
}

#[test]
fn post_admission_copy_failure_consumes_logical_allowance_and_releases_host_authority() {
    let (mut model, chat, settings, first) = super::super::setup();
    let prepared = model
        .prepare_observed_chat(&chat, settings, CapturePlan::none(), limits())
        .unwrap();
    let mut run = model
        .start_controlled_text(prepared, &[], Default::default(), ignore)
        .unwrap();
    run.enable_snapshots(copy_limits()).unwrap();
    run.step(ignore).unwrap();
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let probe = Guard::new(&pool);
    probe.update(|p| p.fail_copy = Some("sampling copy"));
    let before = run.snapshot_usage().unwrap();
    let status = run.status();
    let error = run.snapshot(unexpected_record).err().unwrap();
    assert!(matches!(
        &error,
        ControlledGenerationError::Snapshot(TextSnapshotError::Backend(_))
    ));
    assert!(
        matches!(cause::<MockError>(&error), Some(MockError::Capture(message))
        if message == "host snapshot copy failed")
    );
    assert_eq!(
        probe.update(|p| p.copies.clone()),
        ["input copy", "sampling copy"]
    );
    let after = run.snapshot_usage().unwrap();
    assert!(after.cumulative_copy_bytes > before.cumulative_copy_bytes);
    assert_eq!(after.retained_bytes, before.retained_bytes);
    assert_eq!(after.snapshots, before.snapshots);
    assert_eq!(run.status(), status);
    assert_eq!(run.token_ids(), [first]);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    probe.update(|p| p.fail_copy = None);
    run.step(ignore).unwrap();
    assert_eq!(run.token_ids(), [first, first + 1]);
}

#[test]
fn direct_driver_host_rejection_precedes_controller_and_composition_copy_callbacks() {
    use eredu_core::{TextGenerationDriver, TokenFilterController};
    use eredu_runtime::execution_control::{
        ManagedTextContinuation, SnapshotBudget, SnapshotTokenController, TextBranchRequest,
        TextContinuationSnapshot,
    };
    use std::{cell::Cell, convert::Infallible, rc::Rc};

    struct Controller {
        history: Vec<u32>,
        forks: Rc<Cell<usize>>,
        decisions: Rc<Cell<usize>>,
    }
    impl TokenFilterController for Controller {
        type Error = Infallible;

        fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
            self.decisions.set(self.decisions.get() + 1);
            Ok(TokenFilter::All)
        }

        fn commit_token(&mut self, token: u32) -> Result<(), Self::Error> {
            self.history.push(token);
            Ok(())
        }

        fn is_complete(&mut self) -> Result<bool, Self::Error> {
            Ok(false)
        }
    }
    impl SnapshotTokenController for Controller {
        fn snapshot_storage_bytes(&self) -> Option<u64> {
            (std::mem::size_of::<Self>() as u64)
                .checked_add((self.history.capacity() as u64).checked_mul(4)?)
        }

        fn fork_snapshot(&self) -> Result<Self, String> {
            self.forks.set(self.forks.get() + 1);
            Ok(Self {
                history: self.history.clone(),
                forks: self.forks.clone(),
                decisions: self.decisions.clone(),
            })
        }
    }

    for operation in [
        CopyOperation::Snapshot,
        CopyOperation::Restore,
        CopyOperation::Fork,
    ] {
        let mut runtime = ModelRuntime::prepare(MockBackend, ()).unwrap();
        let mut driver = TextGenerationDriver::new(&mut runtime);
        let forks = Rc::new(Cell::new(0));
        let decisions = Rc::new(Cell::new(0));
        let host_callbacks = Cell::new(0);
        let mut history = Vec::with_capacity(8);
        history.extend([19, 23]);
        let controller = Controller {
            history,
            forks: forks.clone(),
            decisions: decisions.clone(),
        };
        let config = TextGenerationConfig::new(
            eredu_core::resolve_generation_config(
                None,
                GenerationConfigOverrides {
                    max_new_tokens: Some(3),
                    ..Default::default()
                },
            )
            .unwrap(),
        );
        let mut state = ManagedTextContinuation::root(
            driver.start(vec![11, 7, 3], config, controller).unwrap(),
        );
        let budget = SnapshotBudget::new(copy_limits());
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let probe = Guard::new(&pool);
        let saved = if matches!(operation, CopyOperation::Snapshot) {
            None
        } else {
            Some(
                TextContinuationSnapshot::capture(
                    &mut state.boundary(&mut driver).unwrap(),
                    &budget,
                    Some(64),
                )
                .unwrap(),
            )
        };
        let before_forks = forks.get();
        assert_eq!(before_forks, usize::from(saved.is_some()));
        let before_copies = probe.update(|p| p.copies.len());
        let before_usage = budget.usage();
        let before_owners = pool.unquoted_owner_count().unwrap();
        let before_decisions = decisions.get();
        probe.update(|p| p.reject_at = Some(p.attempts + 1));
        let result = {
            let mut boundary = state.boundary(&mut driver).unwrap();
            match operation {
                CopyOperation::Snapshot => {
                    TextContinuationSnapshot::capture(&mut boundary, &budget, Some(64)).map(|_| ())
                }
                CopyOperation::Restore => {
                    saved
                        .as_ref()
                        .unwrap()
                        .restore_with(&mut boundary, &budget, || {
                            host_callbacks.set(host_callbacks.get() + 1);
                            Ok(())
                        })
                }
                CopyOperation::Fork => saved
                    .as_ref()
                    .unwrap()
                    .fork_with(
                        &mut boundary,
                        &budget,
                        TextBranchRequest {
                            session_id: "direct-host-rejection-child",
                            max_predictions: 3,
                            capture_limits: None,
                            intervention: None,
                            host_bytes: Some(64),
                            continuation_growth_bytes: Some(64),
                        },
                        |_, _| {
                            host_callbacks.set(host_callbacks.get() + 1);
                            Ok(())
                        },
                    )
                    .map(|_| ()),
            }
        };
        let error = result.unwrap_err();
        assert!(matches!(&error, TextSnapshotError::HostPreparation(_)));
        assert!(
            matches!(cause::<MockError>(&error), Some(MockError::Capture(message))
                if message == "host snapshot preparation rejected")
        );
        assert_eq!(forks.get(), before_forks, "{operation:?}");
        assert_eq!(host_callbacks.get(), 0, "{operation:?}");
        assert_eq!(probe.update(|p| p.copies.len()), before_copies);
        assert_eq!(budget.usage(), before_usage);
        assert_eq!(decisions.get(), before_decisions);
        assert_eq!(state.controller().history, [19, 23]);
        assert_eq!(pool.unquoted_owner_count().unwrap(), before_owners);
        assert_eq!(
            (pool.used_bytes().unwrap(), pool.peak_bytes().unwrap()),
            (0, 0)
        );
        if let Some(saved) = &saved {
            assert_eq!(saved.controller().history, [19, 23]);
        }
        probe.update(|p| p.reject_at = None);
        assert!(state.advance(&mut driver).unwrap().is_some());
        state.take_completed_step(&mut driver).unwrap();
        assert_eq!(&state.controller().history[..2], &[19, 23]);
        assert_eq!(state.controller().history.len(), 3);
        drop(saved);
        drop(state);
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    }
}
