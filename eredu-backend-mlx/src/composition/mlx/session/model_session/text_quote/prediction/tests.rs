use super::super::sequence::fixture::{Probe, tests as fixture};
use super::*;
use crate::backend::submission_recovery::prediction::PredictionRole as OriginalPredictionScopeRole;
use crate::backend::submission_recovery::prediction::test_counts::Calls;
use crate::backend::submission_recovery::{self, Retention, Status};
use eredu_core::{
    ControlledTextGeneration, GenerationCancellationToken, GenerationSequenceRequest,
    PendingTextInput, TextGeneration, TextGenerationInput, TextPreparationOptions, TokenFilter,
    TokenFilterController, TokenOutput,
};
use eredu_runtime::execution_control::TextSnapshotBackend;
use eredu_runtime::working_memory::OriginalPredictionRecoveryCustody;
use std::{
    error::Error as _,
    sync::mpsc,
    time::{Duration, Instant},
};

fn options(
    source: Option<&eredu_core::capture::SharedCapturePlan>,
) -> Option<TextPreparationOptions> {
    source.map(|source| TextPreparationOptions {
        interventions: None, capture: Some(source.clone()),
    })
}
fn with_foreign_runtime(f: impl FnOnce()) {
    let (ready_tx, ready_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(released) = safemlx::try_with_submission_retirement(|| {
                let _ = ready_tx.send(());
                release_rx.recv_timeout(Duration::from_secs(5)).is_ok()
            }) {
                return released;
            }
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        }
    });
    struct Release(
        Option<mpsc::Sender<()>>,
        Option<std::thread::JoinHandle<bool>>,
    );
    impl Drop for Release {
        fn drop(&mut self) {
            if let Some(tx) = self.0.take() {
                let _ = tx.send(());
            }
            if let Some(worker) = self.1.take() {
                let _ = worker.join();
            }
        }
    }
    let mut release = Release(Some(release_tx), Some(worker));
    ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    f();
    release.0.take().unwrap().send(()).unwrap();
    assert!(release.1.take().unwrap().join().unwrap());
}

#[derive(Clone, Default)]
struct Controller {
    cancel_on_decision: Option<GenerationCancellationToken>,
}
impl TokenFilterController for Controller {
    type Error = std::convert::Infallible;
    fn inference_workspace_is_run_owned(&self) -> bool {
        true
    }
    fn inference_workspace(&self, _: u64) -> Option<eredu_core::TextControllerWorkspace<'_>> {
        Some(eredu_core::TextControllerWorkspace {
            filter: (&TokenFilter::All).into(),
            additional_host_bytes: 0,
        })
    }
    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        if let Some(cancel) = &self.cancel_on_decision {
            cancel.cancel();
        }
        Ok(TokenFilter::All)
    }
    fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
        Ok(())
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        Ok(false)
    }
}

#[test]
fn five_original_roles_follow_shared_ordinary_controlled_and_resident_host_disk_execution() {
    let stream = fixture::stream();
    for residency in 0..3 {
        let mut reference = None;
        // Baseline uses the existing legacy quoted route. The three joined
        // modes exercise sequence-only, combined, and original opening rows.
        for mode in 0..4 {
            // Keep the existing resident-only opening-row fixture boundary;
            // this role bank does not supply a new host/disk opening inventory.
            if mode == 3 && residency != 0 {
                continue;
            }
            for controlled in [false, true] {
                let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
                let (mut runtime, _artifact) = fixture::load(&stream, &pool, residency);
                let source = (mode >= 2).then(|| fixture::source(&runtime));
                let probe = (mode != 0).then(|| Probe::new(&runtime, source.as_ref(), mode == 3));
                let calls = Calls::new();
                let outputs = if mode == 0 {
                    if controlled {
                        ControlledTextGeneration::from_input(
                            &mut runtime,
                            TextGenerationInput::TokenIds(vec![2, 5, 7]),
                            fixture::config(4, u64::MAX),
                            Controller::default(),
                        )
                        .unwrap()
                        .map(|token| token.unwrap().into_output())
                        .collect::<Vec<_>>()
                    } else {
                        TextGeneration::new(
                            &mut runtime,
                            vec![2, 5, 7],
                            fixture::config(4, u64::MAX),
                        )
                        .unwrap()
                        .map(Result::unwrap)
                        .collect::<Vec<_>>()
                    }
                } else if controlled {
                    let mut run = ControlledTextGeneration::from_input_with_sequence(
                        &mut runtime,
                        TextGenerationInput::TokenIds(vec![2, 5, 7]),
                        fixture::chunked_original(fixture::config(4, u64::MAX)),
                        Controller::default(),
                        options(source.as_ref()),
                        GenerationSequenceRequest::new(4, &[]),
                    )
                    .unwrap();
                    let sequence = run
                        .take_prepared_sequence()
                        .unwrap()
                        .prepare_storage()
                        .unwrap();
                    let mut outputs = Vec::new();
                    while let Some(token) = run.next() {
                        outputs.push(token.unwrap().into_output());
                        let delivery = run.take_captured_delivery().unwrap();
                        assert_eq!(delivery.is_some(), mode >= 2);
                        drop(delivery);
                    }
                    drop((run, sequence));
                    outputs
                } else {
                    let mut run = TextGeneration::from_input_with_sequence(
                        &mut runtime,
                        TextGenerationInput::TokenIds(vec![2, 5, 7]),
                        fixture::chunked_original(fixture::config(4, u64::MAX)),
                        TokenFilter::All,
                        options(source.as_ref()),
                        GenerationSequenceRequest::new(4, &[]),
                    )
                    .unwrap();
                    let sequence = run
                        .take_prepared_sequence()
                        .unwrap()
                        .prepare_storage()
                        .unwrap();
                    let mut outputs = Vec::new();
                    while let Some(token) = run.next() {
                        outputs.push(token.unwrap());
                        let delivery = run.take_captured_delivery().unwrap();
                        assert_eq!(delivery.is_some(), mode >= 2);
                        drop(delivery);
                    }
                    drop((run, sequence));
                    outputs
                };
                assert_eq!(outputs.len(), 4);
                if mode != 0 {
                    assert_eq!(
                        calls.original(),
                        [4, 4, 4, 4, if controlled { 4 } else { 0 }]
                    );
                } else {
                    assert_eq!(calls.original(), [0; 5]);
                }
                let values = outputs
                    .iter()
                    .map(|token| token.token_id().unwrap())
                    .collect::<Vec<_>>();
                let state = runtime.session().payload.model.erased().state_snapshot();
                assert!(state.iter().all(|(position, _)| *position == 6));
                if let Some(expected) = &reference {
                    assert_eq!(&(values, state), expected);
                } else {
                    reference = Some((values, state));
                }
                if mode != 0 {
                    assert_eq!(calls.original(), [4; 5]);
                }
                for output in &outputs {
                    assert!(
                        matches!(
                            output.owner.take_sampling_scope(),
                            Err(Error::PredictionScopeUnavailable)
                        ) || mode == 0
                    );
                    assert!(
                        matches!(
                            output.owner.take_sampling_event_scope(),
                            Err(Error::PredictionScopeUnavailable)
                        ) || mode == 0
                    );
                    assert!(
                        matches!(
                            output.owner.take_model_validation_scope(),
                            Err(Error::PredictionScopeUnavailable)
                        ) || mode == 0
                    );
                    assert!(
                        matches!(
                            output.owner.take_token_scalar_scope(),
                            Err(Error::PredictionScopeUnavailable)
                        ) || mode == 0
                    );
                }
                drop((outputs, probe, source, calls));
                fixture::finish(runtime, &stream);
                fixture::settle(&pool, 0);
            }
        }
    }
}

#[test]
fn original_scalar_is_once_after_run_drop_copy_rejects_and_ordinary_copy_observes_fresh() {
    let stream = fixture::stream();
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = fixture::load(&stream, &pool, 0);
    let probe = Probe::new(&runtime, None, false);
    let calls = Calls::new();
    let mut run = TextGeneration::from_input_with_sequence(
        &mut runtime,
        TextGenerationInput::TokenIds(vec![2, 5, 7]),
        fixture::chunked_original(fixture::config(4, u64::MAX)),
        TokenFilter::All,
        None,
        GenerationSequenceRequest::new(4, &[]),
    )
    .unwrap();
    let sequence = run
        .take_prepared_sequence()
        .unwrap()
        .prepare_storage()
        .unwrap();
    let token = run.next().unwrap().unwrap();
    drop((run, sequence));
    let preparation = probe.take();
    drop((preparation, probe));
    assert_eq!(calls.original(), [1, 1, 1, 1, 0]);
    let alias = token.clone();
    let id = token.token_id().unwrap();
    for _ in 0..3 {
        assert_eq!(alias.token_id().unwrap(), id);
    }
    assert_eq!(calls.original(), [1; 5]);
    // The escaped funded token still owns its original work. The public copy
    // path must preserve that exclusion; dropping the run is not a copy grant.
    let copy_error = match MlxBackend::copy_pending_input(
        &mut runtime,
        Some(PendingTextInput::Decode(&token)),
    ) {
        Err(error) => error,
        Ok(_) => panic!("live original token must exclude unquoted copy"),
    };
    let Error::Other(cause) = &copy_error else {
        panic!("exact original copy exclusion")
    };
    assert_eq!(
        cause.downcast_ref::<WorkingMemoryError>(),
        Some(&WorkingMemoryError::ReservedWorkActive)
    );
    assert_eq!(calls.legacy()[4], 0);
    assert_eq!(calls.original(), [1; 5]);
    drop(copy_error);
    assert!(matches!(
        token.owner.take_token_scalar_scope(),
        Err(Error::PredictionScopeUnavailable)
    ));
    // The same actual owner-health transition invalidates cached success;
    // cached error replay cannot reopen the already consumed scalar role.
    token.owner.reject_unresolved();
    let first = alias.token_id().unwrap_err();
    let again = alias.token_id().unwrap_err();
    let Error::OutputObservation(first) = first else {
        panic!("retained observation failure")
    };
    let Error::OutputObservation(again) = again else {
        panic!("same retained failure")
    };
    assert!(std::ptr::eq(
        first.source().unwrap().downcast_ref::<Error>().unwrap(),
        again.source().unwrap().downcast_ref::<Error>().unwrap()
    ));
    assert_eq!(calls.original(), [1; 5]);
    drop((token, alias, calls));
    // Deliberate owner poison follows existing conservative quarantine policy.
    drop(runtime);
    stream.synchronize().unwrap();
    submission_recovery::reap();

    // A distinct real ordinary producer carries no original quote. Its existing
    // unquoted copy path is eligible without weakening the rejection above.
    let ordinary_pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut ordinary, _ordinary_artifact) = fixture::load(&stream, &ordinary_pool, 0);
    let calls = Calls::new();
    let config = TextGenerationConfig::new(fixture::config(4, u64::MAX).sampling()).with_seed(19);
    assert_eq!(
        config.inference_policy().managed_memory_capacity_bytes,
        None
    );
    let mut run = TextGeneration::new(&mut ordinary, vec![2, 5, 7], config).unwrap();
    let token = run.next().unwrap().unwrap();
    drop(run);
    let id = token.token_id().unwrap();
    let alias = token.clone();
    for _ in 0..3 {
        assert_eq!(alias.token_id().unwrap(), id);
    }
    assert_eq!(calls.legacy()[4], 1);
    assert_eq!(calls.original(), [0; 5]);
    let copied =
        MlxBackend::copy_pending_input(&mut ordinary, Some(PendingTextInput::Decode(&token)))
            .unwrap()
            .unwrap();
    let PendingTextInput::Decode(copied) = copied else {
        panic!("actual copied ordinary token")
    };
    assert_eq!(calls.legacy()[4], 1);
    assert_eq!(copied.token_id().unwrap(), id);
    assert_eq!(calls.legacy()[4], 2);
    let copied_alias = copied.clone();
    for _ in 0..3 {
        assert_eq!(copied_alias.token_id().unwrap(), id);
    }
    assert_eq!(calls.legacy()[4], 2);
    assert_eq!(calls.original(), [0; 5]);
    drop((token, alias, copied, copied_alias, calls));
    fixture::finish(ordinary, &stream);
    fixture::settle(&ordinary_pool, 0);
}

fn isolated_event_role() -> (OriginalPredictionScopeRole, WorkingMemoryPool, u64) {
    isolated_event_role_with_controls(false)
}

fn isolated_event_role_with_controls(
    quoted: bool,
) -> (OriginalPredictionScopeRole, WorkingMemoryPool, u64) {
    isolated_roles_with_controls(quoted, |roles| {
        let role = roles.take_sampling_event().unwrap();
        assert!(roles.take_sampling_event().is_err());
        role
    })
}

fn isolated_roles_with_controls<T>(
    quoted: bool,
    take: impl FnOnce(&mut crate::backend::submission_recovery::prediction::PredictionSet) -> T,
) -> (T, WorkingMemoryPool, u64) {
    let stream = fixture::stream();
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = fixture::load(&stream, &pool, 0);
    let probe = Probe::new(&runtime, None, false);
    let run = TextGeneration::from_input_with_sequence(
        &mut runtime,
        TextGenerationInput::TokenIds(vec![2, 5, 7]),
        if quoted {
            fixture::chunked_original(fixture::config(4, u64::MAX))
        } else {
            fixture::config(4, u64::MAX)
        },
        TokenFilter::All,
        None,
        GenerationSequenceRequest::new(4, &[]),
    )
    .unwrap();
    let preparation = probe.take();
    let held = probe.facts().held;
    let request = preparation.request.as_ref().unwrap();
    let step = request
        .claim_step(&probe.context(), PendingTextInput::Prefill(()))
        .unwrap();
    let quote = preparation.quote.as_ref().unwrap();
    let scopes = quote.prediction_scopes.as_ref().unwrap();
    let loan = scopes.bank.borrow_mut();
    assert!(matches!(
        quote.claim_prediction_scopes(&step),
        Err(Error::PredictionScopeReentrant)
    ));
    drop(loan);
    let mut roles = quote.claim_prediction_scopes(&step).unwrap().unwrap();
    assert!(quote.claim_prediction_scopes(&step).is_err());
    let role = take(&mut roles);
    drop((roles, step, preparation, run, probe));
    fixture::finish(runtime, &stream);
    fixture::settle(&pool, held);
    (role, pool, held)
}

#[test]
fn prediction_native_and_recovery_custody_retire_independently_after_original_run_closes() {
    let (role, pool, held) = isolated_event_role();
    let (native, recovery, quota, graph) = role.into_custody();
    assert!(quota.is_none(), "existing fixture has no tracking ceiling");
    assert!(graph.is_none(), "existing fixture has no graph ceiling");
    let mut pending = safemlx::PreparedSubmissionScopeOwner::try_new(native).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut scope = loop {
        match safemlx::SubmissionScope::try_begin_retaining(pending) {
            Ok(scope) => break scope,
            Err(e) => {
                assert_eq!(e.cause(), safemlx::SubmissionScopeOwnerCause::RuntimeBusy);
                pending = e.into_parts().1;
                assert!(Instant::now() < deadline);
                std::thread::yield_now();
            }
        }
    };
    scope.seal();
    assert!(scope.status().is_settled());
    drop(recovery);
    assert_eq!(pool.used_bytes().unwrap(), held);
    drop(scope);
    assert_eq!(
        pool.used_bytes().unwrap(),
        held,
        "unlocked queue still owns exact original charge"
    );
    safemlx::reclaim_allocation_owners();
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

struct CustodyOnly {
    observed: Rc<Cell<usize>>,
    _custody: Option<OriginalPredictionRecoveryCustody>,
}

fn setup_cause(
    error: &Error,
) -> &crate::backend::submission_recovery::prediction::PredictionSetupCause {
    let mut source: &(dyn std::error::Error + 'static) = error;
    loop {
        if let Some(owned) = source.downcast_ref::<
            crate::backend::submission_recovery::prediction::PredictionSetupFailure
        >() {
            return owned.cause();
        }
        source = source.source().expect("original prediction setup source");
    }
}
impl crate::backend::submission_recovery::prediction::PredictionRetention for CustodyOnly {
    fn install_prediction_custody(&mut self, c: OriginalPredictionRecoveryCustody) {
        self._custody = Some(c);
    }
}
impl Retention for CustodyOnly {
    fn observe(&self, _: Status) {
        self.observed.set(self.observed.get() + 1);
    }
}
#[test]
fn busy_prediction_begin_keeps_never_started_recovery_custody_without_false_completion() {
    let (role, pool, held) = isolated_event_role_with_controls(true);
    let observed = Rc::new(Cell::new(0));
    with_foreign_runtime(|| {
        let result = crate::backend::submission_recovery::prediction::begin(
            Some(role),
            CustodyOnly {
                observed: observed.clone(),
                _custody: None,
            },
        );
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("foreign runtime accepted prediction"),
        };
        assert!(matches!(
            setup_cause(&error),
            crate::backend::submission_recovery::prediction::PredictionSetupCause::Backend(
                Error::PredictionScope(safemlx::SubmissionScopeOwnerCause::RuntimeBusy)
            )
        ));
        drop(error);
        safemlx::reclaim_allocation_owners();
        assert_eq!(
            pool.used_bytes().unwrap(),
            held,
            "Rust recovery queued without a native probe owns its own custody"
        );
        assert_eq!(observed.get(), 0);
    });
    assert_eq!(pool.used_bytes().unwrap(), held);
    submission_recovery::reap();
    assert_eq!(observed.get(), 0);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn original_prediction_missing_quota_refuses_before_scope_and_error_keeps_real_controls() {
    let (role, pool, held) = isolated_event_role();
    // Require strict native mode explicitly; a genuine host-only role does not
    // request either arena and uses its separate legacy execution mode.
    let role = role.requiring_native_controls();
    let observed = Rc::new(Cell::new(0));
    let error = match crate::backend::submission_recovery::prediction::begin(
        Some(role),
        CustodyOnly {
            observed: observed.clone(),
            _custody: None,
        },
    ) {
        Err(error) => error,
        Ok(_) => panic!("missing original arenas accepted"),
    };
    assert!(matches!(
        setup_cause(&error),
        crate::backend::submission_recovery::prediction::PredictionSetupCause::Backend(
            Error::OriginalNativeControl(safemlx::OriginalNativeControlError::MissingRecord)
        )
    ));
    submission_recovery::reap();
    safemlx::reclaim_allocation_owners();
    assert_eq!(observed.get(), 0);
    assert_eq!(pool.used_bytes().unwrap(), held);
    drop(error);
    fixture::settle(&pool, 0);
}

#[test]
fn original_prediction_busy_setup_error_outlives_never_started_native_and_recovery_nodes() {
    let (role, pool, held) = isolated_event_role_with_controls(true);
    let observed = Rc::new(Cell::new(0));
    let mut escaped = None;
    with_foreign_runtime(|| {
        let error = match crate::backend::submission_recovery::prediction::begin(
            Some(role),
            CustodyOnly {
                observed: observed.clone(),
                _custody: None,
            },
        ) {
            Err(error) => error,
            Ok(_) => panic!("foreign runtime accepted prediction"),
        };
        assert!(matches!(
            setup_cause(&error),
            crate::backend::submission_recovery::prediction::PredictionSetupCause::Backend(
                Error::PredictionScope(safemlx::SubmissionScopeOwnerCause::RuntimeBusy)
            )
        ));
        escaped = Some(error);
    });
    submission_recovery::reap();
    safemlx::reclaim_allocation_owners();
    assert_eq!(observed.get(), 0);
    assert_eq!(pool.used_bytes().unwrap(), held);
    drop(escaped);
    fixture::settle(&pool, 0);
}

#[test]
fn original_prediction_scoped_busy_cause_retains_carrier_after_scope_and_run_retire() {
    let (role, pool, held) = isolated_event_role_with_controls(true);
    let observed = Rc::new(Cell::new(0));
    let mut active = crate::backend::submission_recovery::prediction::begin(
        Some(role),
        CustodyOnly {
            observed: observed.clone(),
            _custody: None,
        },
    )
    .unwrap();
    // This is impossible until scoped observation, original-required, the real
    // carrier and original native controls have all been installed.
    let observer = safemlx::OriginalScopeObserver::require_current().unwrap();
    let mut escaped = None;
    with_foreign_runtime(|| {
        let (outcome, _) = observer.progress().unwrap();
        assert_eq!(outcome, safemlx::ScopedSubmissionProgress::Busy);
        let error = observer
            .observation_error(outcome)
            .expect("real Busy refusal");
        assert_eq!(
            error.scoped_evaluation_cause(),
            Some(safemlx::error::ScopedEvaluationCause::RuntimeBusy)
        );
        escaped = Some(error);
    });
    active.seal();
    let status = active.finish();
    assert!(status.settled && !status.failed && !status.blocked);
    assert!(observed.get() > 0);
    drop(observer);
    submission_recovery::reap();
    safemlx::reclaim_allocation_owners();
    assert_eq!(pool.used_bytes().unwrap(), held);
    assert_eq!(
        escaped.as_ref().unwrap().scoped_evaluation_cause(),
        Some(safemlx::error::ScopedEvaluationCause::RuntimeBusy)
    );
    drop(escaped);
    fixture::settle(&pool, 0);
}

#[test]
fn original_sampling_event_completes_nonzero_token_and_optional_rng_after_role_seal() {
    use crate::backend::MlxCompletion;
    use eredu_core::Completion;
    thread_local! {
        static HOOK_CALLS: Cell<usize> = const { Cell::new(0) };
    }
    fn hook() {
        HOOK_CALLS.with(|calls| calls.set(calls.get() + 1));
    }
    struct Hook;
    impl Drop for Hook {
        fn drop(&mut self) {
            safemlx::unregister_thread_runtime_housekeeping(hook);
        }
    }
    for with_random in [false, true] {
        let stream = fixture::stream();
        let runtime = safemlx::PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
        let scalar = Array::from_slice(&[13_u32], &[1]);
        let token = scalar.add(&scalar, &stream).unwrap();
        let seed = Array::from_slice(&[17_u32, 29], &[2]);
        let random = with_random.then(|| seed.add(&seed, &stream).unwrap());
        let sibling = random.clone();
        let (role, pool, held) = isolated_event_role_with_controls(true);
        let calls = Calls::new();
        safemlx::register_thread_runtime_housekeeping(hook);
        let hook_owner = Hook;
        HOOK_CALLS.with(|calls| calls.set(0));
        let submitted =
            MlxCompletion::submission_retaining_with_scope(token, random, &stream, Some(role))
                .unwrap();
        assert_eq!(calls.original(), [0, 0, 1, 0, 0]);
        assert_eq!(calls.legacy(), [0; 5]);
        assert_eq!(
            submitted.completion.retained_resources(),
            1 + usize::from(with_random)
        );
        assert!(
            safemlx::OriginalScopeObserver::try_current()
                .unwrap()
                .is_none()
        );
        submitted.completion.wait().unwrap();
        assert!(submitted.completion.resources_releasable());
        // The original submission, observation and descriptor-validation path
        // ends here. allocation_info is an ordinary guarded metadata query.
        assert_eq!(HOOK_CALLS.with(Cell::get), 0);
        assert!(submitted.output.allocation_info().unwrap().is_some());
        if let Some(random) = &sibling {
            assert!(random.allocation_info().unwrap().is_some());
        }
        // Measure the existing ordinary-read positive control independently of
        // the preceding ordinary allocation queries.
        HOOK_CALLS.with(|calls| calls.set(0));
        assert_eq!(
            submitted
                .output
                .evaluated()
                .unwrap()
                .try_as_slice::<u32>()
                .unwrap(),
            &[26]
        );
        if let Some(random) = &sibling {
            assert_eq!(
                random.evaluated().unwrap().try_as_slice::<u32>().unwrap(),
                &[34, 58]
            );
        }
        assert!(
            HOOK_CALLS.with(Cell::get) > 0,
            "ordinary read is the live hook control"
        );
        assert_eq!(pool.used_bytes().unwrap(), held);
        drop((submitted, sibling, calls, runtime, hook_owner));
        fixture::settle(&pool, 0);
    }
}

#[test]
fn original_sampling_busy_poll_and_guarded_drop_keep_cause_until_exact_nodes_retire() {
    use crate::backend::MlxCompletion;
    use eredu_core::Completion;
    let stream = fixture::stream();
    let runtime = safemlx::PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let token = Array::from_slice(&[23_u32], &[1]);
    let random = Array::from_slice(&[41_u32, 53], &[2]);
    let (role, pool, held) = isolated_event_role_with_controls(true);
    let nodes = submission_recovery::test_node_unbox_count();
    let submitted =
        MlxCompletion::submission_retaining_with_scope(token, Some(random), &stream, Some(role))
            .unwrap();
    submitted.completion.wait().unwrap();
    let mut escaped = None;
    with_foreign_runtime(|| {
        let error = submitted.completion.is_complete().unwrap_err();
        assert_eq!(
            MlxCompletion::sampling_event_native_cause(&error),
            Some(safemlx::error::ScopedEvaluationCause::RuntimeBusy)
        );
        assert!(!submitted.completion.resources_releasable());
        drop(submitted);
        assert_eq!(submission_recovery::test_node_unbox_count(), nodes);
        assert_eq!(pool.used_bytes().unwrap(), held);
        escaped = Some(error);
    });
    // First retire the inner Scope/resources; the SAME observed cleanup node
    // survives for the final exact-owner wrapper/record drain on a later pass.
    submission_recovery::wait_for_retirement(|| {
        submission_recovery::test_node_unbox_count() >= nodes + 2
    });
    safemlx::reclaim_allocation_owners();
    assert_eq!(pool.used_bytes().unwrap(), held);
    assert_eq!(
        MlxCompletion::sampling_event_native_cause(escaped.as_ref().unwrap()),
        Some(safemlx::error::ScopedEvaluationCause::RuntimeBusy)
    );
    drop((escaped, runtime));
    fixture::settle(&pool, 0);
}

#[test]
fn original_sampling_missing_arena_keeps_typed_setup_refusal_and_never_submits() {
    use crate::backend::MlxCompletion;
    let stream = fixture::stream();
    let token = Array::from_slice(&[31_u32], &[1]);
    let (role, pool, held) = isolated_event_role();
    let error =
        match MlxCompletion::submission_retaining_with_scope(token, None, &stream, Some(role)) {
            Ok(_) => panic!("missing original arena accepted sampling"),
            Err(error) => error,
        };
    assert!(matches!(
        setup_cause(&error),
        crate::backend::submission_recovery::prediction::PredictionSetupCause::Backend(
            Error::OriginalNativeControl(safemlx::OriginalNativeControlError::MissingRecord)
        )
    ));
    assert!(
        safemlx::OriginalScopeObserver::try_current()
            .unwrap()
            .is_none()
    );
    submission_recovery::reap();
    safemlx::reclaim_allocation_owners();
    assert_eq!(pool.used_bytes().unwrap(), held);
    drop(error);
    fixture::settle(&pool, 0);
}

#[test]
fn original_sampling_prepared_foreign_stream_completes_nonzero_token() {
    use crate::backend::MlxCompletion;
    use eredu_core::Completion;
    let (stream_tx, stream_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        // Ordinary creation publishes this exact stream's shared encoder. The
        // original no-init lookup can use that owner from the receiving thread.
        stream_tx.send(fixture::stream()).unwrap();
        release_rx.recv_timeout(Duration::from_secs(10)).is_ok()
    });
    struct ReleaseForeignStream(
        Option<mpsc::Sender<()>>,
        Option<std::thread::JoinHandle<bool>>,
    );
    impl Drop for ReleaseForeignStream {
        fn drop(&mut self) {
            if let Some(tx) = self.0.take() {
                let _ = tx.send(());
            }
            if let Some(worker) = self.1.take() {
                let _ = worker.join();
            }
        }
    }
    let mut release = ReleaseForeignStream(Some(release_tx), Some(worker));
    let stream = stream_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    let token = Array::from_slice(&[47_u32], &[1]);
    let (role, pool, held) = isolated_event_role_with_controls(true);
    let submitted =
        MlxCompletion::submission_retaining_with_scope(token, None, &stream, Some(role)).unwrap();
    assert!(
        safemlx::OriginalScopeObserver::try_current()
            .unwrap()
            .is_none()
    );
    assert_eq!(submitted.completion.retained_resources(), 1);
    submitted.completion.wait().unwrap();
    assert!(submitted.completion.resources_releasable());
    assert_eq!(
        submitted
            .output
            .evaluated()
            .unwrap()
            .try_as_slice::<u32>()
            .unwrap(),
        &[47]
    );
    assert_eq!(pool.used_bytes().unwrap(), held);
    drop(submitted);
    release.0.take().unwrap().send(()).unwrap();
    assert!(release.1.take().unwrap().join().unwrap());
    drop((stream, release));
    fixture::settle(&pool, 0);
}

#[test]
fn original_sampling_unprepared_cpu_stream_refusal_keeps_exact_native_cause_and_custody() {
    use crate::backend::MlxCompletion;
    let token = Array::from_slice(&[47_u32], &[1]);
    let (role, pool, held) = isolated_event_role_with_controls(true);
    // A fresh CPU Stream creates its encoder but no scheduler worker. Create it
    // after setup, and do not prepare, evaluate or synchronize this unique stream
    // before submission: original preflight must refuse its missing worker.
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let nodes = submission_recovery::test_node_unbox_count();
    let error =
        match MlxCompletion::submission_retaining_with_scope(token, None, &stream, Some(role)) {
            Ok(_) => panic!("unprepared CPU worker accepted original sampling"),
            Err(error) => error,
        };
    assert_eq!(
        MlxCompletion::sampling_event_native_cause(&error),
        Some(safemlx::error::ScopedEvaluationCause::Domain)
    );
    assert!(
        safemlx::OriginalScopeObserver::try_current()
            .unwrap()
            .is_none()
    );
    submission_recovery::wait_for_retirement(|| {
        submission_recovery::test_node_unbox_count() >= nodes + 2
    });
    safemlx::reclaim_allocation_owners();
    assert_eq!(pool.used_bytes().unwrap(), held);
    drop(error);
    fixture::settle(&pool, 0);
    drop(stream);
}

#[test]
fn cancellation_after_real_permit_uses_only_model_and_validation_roles() {
    let stream = fixture::stream();
    for after_permit in [false, true] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (mut runtime, _artifact) = fixture::load(&stream, &pool, 0);
        let probe = Probe::new(&runtime, None, false);
        let calls = Calls::new();
        let cancel = GenerationCancellationToken::new();
        let controller = Controller {
            cancel_on_decision: after_permit.then(|| cancel.clone()),
        };
        let mut run = ControlledTextGeneration::from_input_with_sequence(
            &mut runtime,
            TextGenerationInput::TokenIds(vec![2, 5, 7]),
            fixture::chunked_original(fixture::config(4, u64::MAX)),
            controller,
            None,
            GenerationSequenceRequest::new(4, &[]),
        )
        .unwrap();
        if !after_permit {
            cancel.cancel();
        }
        assert!(run.next_cancellable(&cancel).is_none());
        assert!(run.next().is_none());
        drop(run);
        assert_eq!(
            calls.original(),
            if after_permit {
                [1, 0, 0, 1, 0]
            } else {
                [0; 5]
            }
        );
        assert!(
            runtime
                .session()
                .payload
                .model
                .erased()
                .state_snapshot()
                .iter()
                .all(|(p, _)| *p == 0)
        );
        drop((probe, calls));
        fixture::finish(runtime, &stream);
        fixture::settle(&pool, 0);
    }
}

#[test]
fn original_prediction_custody_survives_scope_unwind_until_guarded_reclamation() {
    let (role, pool, held) = isolated_event_role_with_controls(true);
    let marker = std::sync::Arc::new(());
    let expected = marker.clone();
    let observed = Rc::new(Cell::new(0));
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _scope = crate::backend::submission_recovery::prediction::begin(
            Some(role),
            CustodyOnly {
                observed: observed.clone(),
                _custody: None,
            },
        )
        .unwrap();
        assert_eq!(pool.used_bytes().unwrap(), held);
        std::panic::panic_any(marker);
    }))
    .unwrap_err();
    assert!(std::sync::Arc::ptr_eq(
        panic.downcast_ref::<std::sync::Arc<()>>().unwrap(),
        &expected
    ));
    // Native/recovery Drop keeps the existing retirement guard exception; it
    // does not turn an unwind into successful model submission or certification.
    submission_recovery::reap();
    safemlx::reclaim_allocation_owners();
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

// Real roles come from the genuine request/step/quote above. This small neutral
// session ticket only hosts token observations; it issues no original role/Q.
fn scalar_submission(
    value: Array,
    stream: &Stream,
) -> (
    MlxTextToken,
    crate::backend::MlxCompletion,
    SessionAuthority,
    WorkingMemoryPool,
    u64,
) {
    let ((event, scalar), pool, held) = isolated_roles_with_controls(true, |roles| {
        let event = roles.take_sampling_event().unwrap();
        let scalar = roles.take_token_scalar().unwrap();
        assert!(roles.take_sampling_event().is_err());
        assert!(roles.take_token_scalar().is_err());
        (event, scalar)
    });
    let mut authority = SessionAuthority::new();
    let owner = SubmissionResources::new(
        authority.begin_submission().unwrap(),
        Rc::new(Cell::new(false)),
    );
    let submitted = crate::backend::MlxCompletion::submission_retaining_with_scope(
        value,
        None,
        stream,
        Some(event),
    )
    .unwrap();
    let token = MlxTextToken::new_with_sampling_source(
        submitted.output,
        stream.clone(),
        owner,
        Some(scalar),
        None,
        submitted.completion.original_sampling_source(),
    );
    (token, submitted.completion, authority, pool, held)
}

fn scalar_cause(error: &Error) -> &super::super::super::super::output_completion::scalar::Cause {
    use super::super::super::super::output_completion::scalar::Failure;
    let mut source: &(dyn std::error::Error + 'static) = error;
    loop {
        if let Some(failure) = source.downcast_ref::<Failure>() {
            return &failure.cause;
        }
        source = source.source().expect("original scalar failure source");
    }
}
fn same_observation_source(left: &Error, right: &Error) -> bool {
    let (Error::OutputObservation(left), Error::OutputObservation(right)) = (left, right) else {
        panic!("closed observation error");
    };
    std::ptr::eq(left.source().unwrap(), right.source().unwrap())
}

#[test]
fn original_token_scalar_early_read_matches_explicit_wait_and_clones_replay_without_native_work() {
    use eredu_core::Completion;
    for explicit_wait in [false, true] {
        let stream = fixture::stream();
        let runtime = safemlx::PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
        let seed = Array::from_slice(&[37_u32], &[1]);
        let value = seed.add(&seed, &stream).unwrap();
        let (token, completion, authority, pool, held) = scalar_submission(value, &stream);
        let calls = Calls::new();
        let alias = token.clone();
        if explicit_wait {
            completion.wait().unwrap();
        }
        assert_eq!(token.token_id().unwrap(), 74);
        assert!(token.value.allocation_info().unwrap().is_some());
        assert_eq!(calls.original(), [0, 0, 0, 0, 1]);
        assert_eq!(calls.legacy(), [0; 5]);
        completion.wait().unwrap();
        drop((completion, token));
        assert_eq!(pool.used_bytes().unwrap(), held);
        with_foreign_runtime(|| {
            for _ in 0..3 {
                assert_eq!(alias.token_id().unwrap(), 74);
            }
        });
        assert_eq!(calls.original(), [0, 0, 0, 0, 1]);
        alias.owner.request_release();
        drop((alias, runtime, calls));
        fixture::settle(&pool, 0);
        assert_eq!(authority.require_idle(), Ok(()));
    }
}

#[test]
fn original_token_scalar_keeps_producer_after_completion_drop_and_health_error_after_all_tokens() {
    let stream = fixture::stream();
    let runtime = safemlx::PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let seed = Array::from_slice(&[43_u32], &[1]);
    let (token, completion, authority, pool, held) =
        scalar_submission(seed.add(&seed, &stream).unwrap(), &stream);
    // The token source must progress the exact producer even before an event
    // wait and after its public completion owner has entered retirement.
    drop(completion);
    submission_recovery::reap();
    let alias = token.clone();
    assert_eq!(token.token_id().unwrap(), 86);
    token.owner.reject_unresolved();
    let first = token.token_id().unwrap_err();
    let second = alias.token_id().unwrap_err();
    assert!(same_observation_source(&first, &second));
    assert!(super::super::super::super::output_completion::scalar::is_fixed_poison(&first));
    drop((token, alias, runtime, authority));
    submission_recovery::reap();
    safemlx::reclaim_allocation_owners();
    assert_eq!(pool.used_bytes().unwrap(), held);
    drop(first);
    assert_eq!(pool.used_bytes().unwrap(), held);
    drop(second);
    fixture::settle(&pool, 0);
}

#[test]
fn original_token_scalar_busy_setup_is_once_and_escaped_error_retains_source_custody() {
    use super::super::super::super::output_completion::scalar::Cause;
    use eredu_core::Completion;
    let stream = fixture::stream();
    let runtime = safemlx::PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let (token, completion, authority, pool, held) =
        scalar_submission(Array::from_slice(&[59_u32], &[1]), &stream);
    completion.wait().unwrap();
    let alias = token.clone();
    let calls = Calls::new();
    let mut escaped = None;
    with_foreign_runtime(|| {
        let error = token.token_id().unwrap_err();
        let Cause::Backend(setup) = scalar_cause(&error) else {
            panic!("actual setup refusal")
        };
        assert!(matches!(
            setup_cause(setup),
            crate::backend::submission_recovery::prediction::PredictionSetupCause::Backend(
                Error::PredictionScope(safemlx::SubmissionScopeOwnerCause::RuntimeBusy)
            )
        ));
        let repeated = alias.token_id().unwrap_err();
        assert!(same_observation_source(&error, &repeated));
        assert_eq!(calls.original(), [0, 0, 0, 0, 1]);
        drop(completion);
        assert_eq!(pool.used_bytes().unwrap(), held);
        escaped = Some((error, repeated));
    });
    // Each token owns an ordinary Stream whose Drop acquires the runtime lock.
    // Keep exact error polling and completion quarantine under contention, then
    // destroy token stream wrappers after the foreign owner has released it.
    drop((token, alias));
    submission_recovery::reap();
    safemlx::reclaim_allocation_owners();
    assert_eq!(pool.used_bytes().unwrap(), held);
    drop((runtime, authority, calls, escaped));
    fixture::settle(&pool, 0);
}

#[test]
fn original_token_scalar_rejects_wrong_shape_and_dtype_without_ordinary_conversion() {
    use super::super::super::super::output_completion::scalar::Cause;
    use eredu_core::Completion;
    for wrong_dtype in [false, true] {
        let stream = fixture::stream();
        let runtime = safemlx::PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
        let value = if wrong_dtype {
            Array::from_slice(&[61_i32], &[1])
        } else {
            Array::from_slice(&[61_u32, 67], &[2])
        };
        let (token, completion, authority, pool, held) = scalar_submission(value, &stream);
        completion.wait().unwrap();
        let error = token.token_id().unwrap_err();
        if wrong_dtype {
            assert!(matches!(
                scalar_cause(&error),
                Cause::Read(safemlx::error::AsSliceError::DtypeMismatch { .. })
            ));
        } else {
            assert!(matches!(scalar_cause(&error), Cause::Elements(2)));
        }
        let repeated = token.token_id().unwrap_err();
        assert!(same_observation_source(&error, &repeated));
        drop((token, completion, runtime, authority));
        submission_recovery::reap();
        safemlx::reclaim_allocation_owners();
        assert_eq!(pool.used_bytes().unwrap(), held);
        drop((error, repeated));
        fixture::settle(&pool, 0);
    }
}

#[test]
fn original_sampling_repeated_public_busy_errors_share_cause_and_retire_after_last_alias() {
    use eredu_core::{BackendProvider, Completion};
    let stream = fixture::stream();
    let runtime = safemlx::PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let (role, pool, held) = isolated_event_role_with_controls(true);
    let submitted = crate::backend::MlxCompletion::submission_retaining_with_scope(
        Array::from_slice(&[71_u32], &[1]),
        None,
        &stream,
        Some(role),
    )
    .unwrap();
    submitted.completion.wait().unwrap();
    let mut escaped = None;
    with_foreign_runtime(|| {
        let errors = std::array::from_fn::<_, 128, _>(|_| {
            MlxBackend::into_backend_failure(submitted.completion.is_complete().unwrap_err())
        });
        let source = errors[0].source().unwrap();
        for error in &errors {
            assert!(std::ptr::eq(source, error.source().unwrap()));
        }
        drop(submitted);
        escaped = Some(errors);
    });
    drop(runtime);
    submission_recovery::reap();
    safemlx::reclaim_allocation_owners();
    assert_eq!(pool.used_bytes().unwrap(), held);
    drop(escaped);
    fixture::settle(&pool, 0);
}

#[test]
fn original_sampling_publication_keeps_old_busy_snapshot_and_new_native_source() {
    let stream = fixture::stream();
    let runtime = safemlx::PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let root = Array::from_slice(&[73_u32], &[1]);
    let (role, pool, held) = isolated_event_role_with_controls(true);
    let graph = fixture::chunked_original(fixture::config(4, u64::MAX))
        .inference_policy()
        .graph_metadata_capacity_bytes
        .unwrap()
        .get();
    let count = usize::try_from(graph).unwrap().checked_add(1).unwrap();
    let errors = crate::backend::MlxCompletion::exercise_sampling_snapshot_publication(
        role,
        &root,
        &stream,
        count,
        |work| with_foreign_runtime(work),
    );
    drop((root, runtime));
    submission_recovery::reap();
    safemlx::reclaim_allocation_owners();
    assert_eq!(pool.used_bytes().unwrap(), held);
    drop(errors);
    fixture::settle(&pool, 0);
}

#[test]
fn original_sampling_contract_failure_is_sticky_for_completion_and_escaped_source() {
    use eredu_core::{BackendProvider, Completion};
    let stream = fixture::stream();
    let runtime = safemlx::PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let (role, pool, held) = isolated_event_role_with_controls(true);
    let submitted = crate::backend::MlxCompletion::submission_retaining_with_scope(
        Array::from_slice(&[79_u32], &[1]),
        None,
        &stream,
        Some(role),
    )
    .unwrap();
    submitted.completion.wait().unwrap();
    let first = submitted.completion.exercise_sampling_terminal_contract();
    let subsequent =
        MlxBackend::into_backend_failure(submitted.completion.is_complete().unwrap_err());
    assert!(std::ptr::eq(
        first.source().unwrap(),
        subsequent.source().unwrap()
    ));
    drop((submitted, runtime));
    submission_recovery::reap();
    safemlx::reclaim_allocation_owners();
    assert_eq!(pool.used_bytes().unwrap(), held);
    drop((first, subsequent));
    fixture::settle(&pool, 0);
}

#[test]
fn original_scalar_reentrant_and_interrupted_aliases_keep_fixed_sources_and_both_roles() {
    use super::super::super::super::output_completion::scalar;
    use eredu_core::Completion;
    for interrupted in [false, true] {
        let stream = fixture::stream();
        let runtime = safemlx::PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
        let (token, completion, authority, pool, held) =
            scalar_submission(Array::from_slice(&[83_u32], &[1]), &stream);
        completion.wait().unwrap();
        let calls = Calls::new();
        let errors = scalar::exercise_fixed_observation(&token, interrupted);
        assert_eq!(
            calls.original(),
            [0; 5],
            "fixed refusal cannot consume scalar role"
        );
        if !interrupted {
            assert_eq!(token.token_id().unwrap(), 83);
            assert_eq!(calls.original(), [0, 0, 0, 0, 1]);
        }
        drop((token, completion, runtime, authority, calls));
        submission_recovery::reap();
        safemlx::reclaim_allocation_owners();
        assert_eq!(pool.used_bytes().unwrap(), held);
        drop(errors);
        fixture::settle(&pool, 0);
    }
}

#[test]
fn original_sampling_conflicting_carrier_is_terminal_and_retains_first_offending_source() {
    use eredu_core::Completion;
    let stream = fixture::stream();
    let runtime = safemlx::PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let (role, pool, held) = isolated_event_role_with_controls(true);
    let submitted = crate::backend::MlxCompletion::submission_retaining_with_scope(
        Array::from_slice(&[87_u32], &[1]),
        None,
        &stream,
        Some(role),
    )
    .unwrap();
    submitted.completion.wait().unwrap();
    let (foreign_role, foreign_pool, foreign_held) = isolated_event_role_with_controls(true);
    let mut foreign = crate::backend::submission_recovery::prediction::begin(
        Some(foreign_role),
        CustodyOnly {
            observed: Rc::new(Cell::new(0)),
            _custody: None,
        },
    )
    .unwrap();
    let observer = safemlx::OriginalScopeObserver::require_current().unwrap();
    let errors = submitted
        .completion
        .exercise_sampling_conflicting_snapshot(&observer);
    foreign.seal();
    drop((foreign, observer, submitted, runtime));
    submission_recovery::reap();
    safemlx::reclaim_allocation_owners();
    assert_eq!(pool.used_bytes().unwrap(), held);
    assert_eq!(foreign_pool.used_bytes().unwrap(), foreign_held);
    drop(errors);
    fixture::settle(&pool, 0);
    fixture::settle(&foreign_pool, 0);
}

#[test]
fn original_scalar_cached_value_rejects_shared_sampling_contract_failure() {
    use eredu_core::{BackendProvider, Completion};
    let stream = fixture::stream();
    let runtime = safemlx::PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let (token, completion, authority, pool, held) =
        scalar_submission(Array::from_slice(&[89_u32], &[1]), &stream);
    completion.wait().unwrap();
    assert_eq!(token.token_id().unwrap(), 89);
    let failure = completion.exercise_sampling_terminal_contract();
    let first = MlxBackend::into_backend_failure(token.token_id().unwrap_err());
    let repeated = MlxBackend::into_backend_failure(token.token_id().unwrap_err());
    assert!(std::ptr::eq(
        first.source().unwrap(),
        repeated.source().unwrap()
    ));
    assert!(!token.owner.is_healthy());
    let mut source = first.source().unwrap();
    while !std::ptr::eq(source, failure.source().unwrap()) {
        source = source.source().expect("same retained contract cause");
    }
    drop((token, completion, runtime, authority));
    submission_recovery::reap();
    safemlx::reclaim_allocation_owners();
    assert_eq!(pool.used_bytes().unwrap(), held);
    drop((failure, first, repeated));
    fixture::settle(&pool, 0);
}
