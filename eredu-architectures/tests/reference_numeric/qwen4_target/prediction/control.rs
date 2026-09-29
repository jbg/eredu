//! Actual controlled driver over retained numeric target and prediction owners.
use super::super::sampling::{Constraint, Publisher, Sampling};
use super::super::snapshots::{COPIES, FAIL_PREDICTION};
use super::*;
use eredu_core::execution_control::{ExecutionControlError, SnapshotLimits};
use eredu_core::generation::{
    GenerationCancellationToken, GenerationSequence, SpeculativeConfig,
    SpeculativeRequestStatus as Status, SpeculativeSchedulerOptions,
};
use eredu_core::speculative::SpeculativeControlError;
use eredu_core::{
    PreparedSpeculativeLane, SpeculativeExecutionTopology, SpeculativeGenerationVisitor,
    SpeculativeOutputRuntime, SpeculativeRandomness,
};
use eredu_runtime::speculative::{
    ControlledSpeculativeOptions, ControlledSpeculativeSession, DriveControlledSpeculation,
};

type Selected = eredu_architectures::qwen4_exp::prepared::SelectedTargetExecution;
fn options() -> ControlledSpeculativeOptions {
    ControlledSpeculativeOptions {
        snapshots: Some(SnapshotLimits {
            max_snapshots: 3,
            max_branches: 2,
            retained_bytes: 16 << 20,
            cumulative_copy_bytes: 128 << 20,
        }),
        ..Default::default()
    }
}
struct Outcome {
    tokens: Vec<u32>,
    target: State,
    prediction: State,
    row_completions: usize,
}
fn run(
    selected: &Selected,
    prepared: &PreparedTarget,
    ctx: &NumericContext,
    options: ControlledSpeculativeOptions,
    drive: impl FnOnce(&mut dyn ControlledSpeculativeSession) -> Result<(), SpeculativeControlError>,
) -> Outcome {
    run_result(selected, prepared, ctx, options, drive).unwrap()
}
fn run_result(
    selected: &Selected,
    prepared: &PreparedTarget,
    ctx: &NumericContext,
    options: ControlledSpeculativeOptions,
    drive: impl FnOnce(&mut dyn ControlledSpeculativeSession) -> Result<(), SpeculativeControlError>,
) -> Result<Outcome, SpeculativeControlError> {
    let speculative = selection(selected);
    let (mut session, mut extension) = paired(selected, prepared, ctx);
    let discovery = captures::discovery(selected, &extension, &speculative);
    let mut strategy = ReplicatedMaterializedPredictionStrategy::<
        Target,
        NumericBackend,
        State,
        NumericReplicatedMechanisms,
        _,
        _,
        Inputs,
        Materializer,
        Mechanisms,
    >::new(&mut session, &mut extension, &speculative, Inputs, ctx);
    let mut cache = strategy.new_cache().unwrap();
    let mut executor = EmbeddedPredictionExecutor::<_, Mechanisms>::new(&mut strategy);
    let runtime = SpeculativeOutputRuntime::new(
        Sampling::default(),
        GenerationSequence::new(12, []),
        Constraint,
        Publisher(std::rc::Rc::new(RefCell::new(vec![]))),
        GenerationCancellationToken::new(),
    );
    let lane = PreparedSpeculativeLane::new(
        &mut cache,
        Input::new(&[3, 4, 7]),
        SpeculativeConfig {
            max_tokens: 12,
            max_draft_tokens: 2,
            temperature: 0.,
            eos_token_ids: vec![],
        },
        runtime,
        SpeculativeRandomness::new(None, None),
    );
    let mut failure = None;
    let output = DriveControlledSpeculation::new(
        SpeculativeSchedulerOptions::default().with_lookahead(false),
        options,
        drive,
        &mut failure,
    )
    .with_vocabulary(32)
    .with_activation_discovery(Some(discovery))
    .run(
        &mut executor,
        vec![lane],
        SpeculativeExecutionTopology::Single,
        false,
        false,
        ctx,
    );
    let output = output.map_err(|_| failure.take().expect("driver retains typed failure"))?;
    assert!(failure.is_none());
    drop(executor);
    drop(strategy);
    let row_completions = session
        .execution_strategy()
        .provider()
        .rows
        .as_ref()
        .unwrap()
        .providers()
        .values()
        .map(|p| p.bank().completed())
        .sum();
    Ok(Outcome {
        tokens: output.requests()[0].token_ids().to_vec(),
        target: cache.target().unwrap().clone(),
        prediction: cache.prediction().0.clone(),
        row_completions,
    })
}
fn finish(
    session: &mut dyn ControlledSpeculativeSession,
    sequence: &mut Vec<u64>,
) -> Result<(), SpeculativeControlError> {
    while let Some(step) = session.step()? {
        sequence.push(step.sequence);
        assert_eq!(step.run_id, session.run_id());
        assert_eq!(step.epoch, session.epoch());
        if step.drafted.is_some() {
            assert!(step.committed_token_ids.is_empty());
        }
    }
    Ok(())
}
fn exact(actual: &State, expected: &State, label: &str) {
    assert_eq!(actual.layout(), expected.layout());
    assert_state_exact(actual, expected, actual.layout().len(), label);
    for (actual, expected) in actual.as_ref().iter().zip(expected.as_ref()) {
        for (a, e) in RuntimeLayerState::retained_values(actual)
            .zip(RuntimeLayerState::retained_values(expected))
        {
            assert_eq!(a.exact_i32, e.exact_i32, "{label} exact integer buffers");
            assert_eq!(a.dtype, e.dtype, "{label} scalar representation");
        }
        assert_eq!(actual.streams.len(), expected.streams.len());
        for ((aslot, alane, a), (eslot, elane, e)) in actual.streams.iter().zip(&expected.streams) {
            use eredu_runtime::AppendOnlyStream;
            assert_eq!((aslot, alane), (eslot, elane));
            assert_eq!(a.specification(), e.specification());
            assert_eq!(a.len(), e.len());
        }
    }
}
pub(super) fn check(selected: &Selected, prepared: &PreparedTarget, ctx: &NumericContext) {
    let (expected, frontiers) = oracle(selected, prepared, ctx, None);
    let plain = run(selected, prepared, ctx, options(), |session| {
        finish(session, &mut vec![])
    });
    assert_eq!(plain.tokens, expected);
    let frontier = plain.target.clone().layer(0).unwrap().position() as usize - 3;
    exact(
        &plain.target,
        &frontiers[frontier].0,
        "controlled target oracle",
    );
    exact(
        &plain.prediction,
        &frontiers[frontier].1,
        "controlled prediction oracle",
    );

    let restored = run(selected, prepared, ctx, options(), |session| {
        assert!(!session.can_snapshot());
        let first = session.step()?.unwrap();
        assert!(session.can_snapshot(), "{:?}", session.snapshot_support());
        let reads = prepared
            .artifact()
            .source_diagnostics()
            .unwrap()
            .physical_reads;
        let saved = session.snapshot()?;
        assert_eq!(
            prepared
                .artifact()
                .source_diagnostics()
                .unwrap()
                .physical_reads,
            reads
        );
        let saved_prefix = session.token_ids().to_vec();
        let mut sequence = vec![first.sequence];
        for epoch in 0..3 {
            finish(session, &mut sequence)?;
            assert_eq!(session.token_ids(), expected);
            if epoch < 2 {
                let before = session.snapshot_usage();
                let reads = prepared
                    .artifact()
                    .source_diagnostics()
                    .unwrap()
                    .physical_reads;
                session.restore(&saved)?;
                assert_eq!(
                    prepared
                        .artifact()
                        .source_diagnostics()
                        .unwrap()
                        .physical_reads,
                    reads
                );
                assert_eq!(session.epoch(), epoch + 1);
                assert_eq!(session.token_ids(), saved_prefix);
                assert_eq!(
                    session.snapshot_usage().retained_bytes,
                    before.retained_bytes
                );
                assert!(
                    session.snapshot_usage().cumulative_copy_bytes > before.cumulative_copy_bytes
                );
            }
        }
        assert!(sequence.windows(2).all(|s| s[0] < s[1]));
        let copying = session.snapshot_usage().cumulative_copy_bytes;
        session.release_snapshot(&saved)?;
        assert_eq!(session.snapshot_usage().retained_bytes, 0);
        assert_eq!(session.snapshot_usage().cumulative_copy_bytes, copying);
        assert!(matches!(
            session.restore(&saved),
            Err(SpeculativeControlError::IncompatibleSnapshot)
        ));
        Ok(())
    });
    assert!(
        restored.row_completions > plain.row_completions,
        "provider accounting must survive restore"
    );
    exact(&restored.target, &plain.target, "restored target");
    exact(
        &restored.prediction,
        &plain.prediction,
        "restored prediction",
    );

    let forced = (expected[1] + 1) % 32;
    let (branch_expected, _) = oracle(selected, prepared, ctx, Some((1, forced)));
    assert_ne!(branch_expected, expected);
    let forked = run(selected, prepared, ctx, options(), |session| {
        session.step()?;
        let saved = session.snapshot()?;
        let prefix = session.token_ids().to_vec();
        let branch = session.fork(&saved)?;
        assert_eq!(session.branch_info(&branch)?.token_ids, prefix);
        session.release_snapshot(&saved)?;
        assert!(session.snapshot_usage().retained_bytes > 0);
        finish(session, &mut vec![])?;
        let copying = session.snapshot_usage().cumulative_copy_bytes;
        session.exchange(&branch)?;
        assert_eq!(session.run_id(), 1);
        assert_eq!(session.token_ids(), prefix);
        assert_eq!(session.branch_info(&branch)?.token_ids, expected);
        assert!(session.snapshot_usage().cumulative_copy_bytes > copying);
        session.force_next_token(forced)?;
        finish(session, &mut vec![])?;
        assert_eq!(session.token_ids(), branch_expected);
        session.exchange(&branch)?;
        assert_eq!(session.run_id(), 0);
        assert_eq!(session.token_ids(), expected);
        session.release_branch(&branch)?;
        assert_eq!(session.snapshot_usage().retained_bytes, 0);
        Ok(())
    });
    assert!(
        forked.row_completions > plain.row_completions,
        "branch exchange must not rewind providers"
    );
    exact(&forked.target, &plain.target, "fork target");
    exact(&forked.prediction, &plain.prediction, "fork prediction");

    for phase in [
        Status::Prefill,
        Status::ReadyToDraft,
        Status::ReadyToSubmitVerification,
        Status::TargetVerificationInFlight,
    ] {
        let cancelled = run(selected, prepared, ctx, options(), |session| {
            while session.status() != phase {
                session.step()?.unwrap();
            }
            if matches!(
                phase,
                Status::ReadyToSubmitVerification | Status::TargetVerificationInFlight
            ) {
                assert!(!session.can_snapshot());
                assert!(matches!(
                    session.snapshot(),
                    Err(SpeculativeControlError::NotQuiescent)
                ));
            }
            session.cancel()?;
            assert_eq!(session.status(), Status::Cancelled);
            assert!(!session.can_snapshot());
            Ok(())
        });
        if phase == Status::Prefill {
            assert!(cancelled.tokens.is_empty());
        } else {
            assert_eq!(cancelled.tokens, expected[..1]);
            // Pending cancellation settles and replays the already-published
            // anchor. No proposed token becomes committed output.
            let frontier = usize::from(phase == Status::TargetVerificationInFlight);
            exact(
                &cancelled.target,
                &frontiers[frontier].0,
                "cancelled target",
            );
            exact(
                &cancelled.prediction,
                &frontiers[frontier].1,
                "cancelled prediction",
            );
        }
    }
    budget_checks(selected, prepared, ctx, &plain);
    captures::check(selected, prepared, ctx, &plain);
}
fn budget_checks(
    selected: &Selected,
    prepared: &PreparedTarget,
    ctx: &NumericContext,
    plain: &Outcome,
) {
    for (retained, copy) in [(1, 128 << 20), (16 << 20, 1)] {
        let mut options = options();
        let limits = options.snapshots.as_mut().unwrap();
        limits.retained_bytes = retained;
        limits.cumulative_copy_bytes = copy;
        let result = run(selected, prepared, ctx, options, |session| {
            session.step()?;
            let before = COPIES.with(Cell::get);
            assert!(matches!(
                session.snapshot(),
                Err(SpeculativeControlError::Control(
                    ExecutionControlError::Limit(_)
                ))
            ));
            assert_eq!(
                COPIES.with(Cell::get),
                before,
                "failed admission must precede copies"
            );
            finish(session, &mut vec![])
        });
        exact(&result.target, &plain.target, "budget target");
        exact(&result.prediction, &plain.prediction, "budget prediction");
    }
    let result = run(selected, prepared, ctx, options(), |session| {
        session.step()?;
        let before = session.snapshot_usage();
        FAIL_PREDICTION.with(|v| v.set(true));
        let failed = session.snapshot();
        FAIL_PREDICTION.with(|v| v.set(false));
        assert!(failed.is_err());
        assert_eq!(
            session.snapshot_usage().retained_bytes,
            before.retained_bytes
        );
        assert!(session.snapshot_usage().cumulative_copy_bytes > before.cumulative_copy_bytes);
        let saved = session.snapshot()?;
        finish(session, &mut vec![])?;
        session.restore(&saved)?;
        finish(session, &mut vec![])
    });
    exact(&result.target, &plain.target, "failed snapshot target");
    exact(
        &result.prediction,
        &plain.prediction,
        "failed snapshot prediction",
    );
    let cost = Cell::new(0);
    run(selected, prepared, ctx, options(), |session| {
        session.step()?;
        session.snapshot()?;
        cost.set(session.snapshot_usage().cumulative_copy_bytes);
        Ok(())
    });
    let mut bounded = options();
    bounded.snapshots.as_mut().unwrap().cumulative_copy_bytes = 2 * cost.get();
    let result = run(selected, prepared, ctx, bounded, |session| {
        session.step()?;
        let saved = session.snapshot()?;
        finish(session, &mut vec![])?;
        session.restore(&saved)?;
        let copies = COPIES.with(Cell::get);
        assert!(matches!(
            session.restore(&saved),
            Err(SpeculativeControlError::Control(
                ExecutionControlError::Limit(_)
            ))
        ));
        assert_eq!(COPIES.with(Cell::get), copies);
        assert_eq!(
            session.snapshot_usage().cumulative_copy_bytes,
            2 * cost.get()
        );
        session.release_snapshot(&saved)?;
        assert_eq!(
            session.snapshot_usage().cumulative_copy_bytes,
            2 * cost.get()
        );
        finish(session, &mut vec![])
    });
    exact(&result.target, &plain.target, "cumulative budget target");
    exact(
        &result.prediction,
        &plain.prediction,
        "cumulative budget prediction",
    );
}

#[path = "captures.rs"]
pub(super) mod captures;
