use super::*;
#[cfg(test)]
use crate::memory_fixture::LedgerFixture as _;

fn deferred(
    runtime: &mut ModelRuntime<MlxBackend<'static>>,
    capacity: u64,
) -> (Probe, MlxTextPreparation) {
    let probe = Probe::new(runtime, None, false);
    probe.mode(Mode::Defer);
    let result = TextGeneration::from_input_with_sequence(
        runtime,
        TextGenerationInput::TokenIds(vec![2, 5, 7]),
        fixture::config(4, capacity),
        TokenFilter::All,
        None,
        GenerationSequenceRequest::new(4, &[]),
    );
    assert!(
        result.is_err(),
        "defer the actual sequence extraction after core binding"
    );
    drop(result);
    let preparation = probe.take();
    (probe, preparation)
}

#[test]
fn unused_busy_work_returns_original_budget_for_repeated_exact_capacity_admissions() {
    let stream = fixture::stream();
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = fixture::load(&stream, &pool, 0);
    let baseline = pool.fixture_host_charge().unwrap();
    let mut original_required = None;
    for pass in 0..4 {
        let capacity = original_required.map_or(u64::MAX, |required| baseline + required);
        let (probe, preparation) = deferred(&mut runtime, capacity);
        let required = preparation
            .quote
            .as_ref()
            .unwrap()
            .request()
            .memory_reservation()
            .requirements()
            .get(crate::memory_fixture::topology().host_domain())
            .unwrap()
            .total()
            .unwrap();
        if let Some(original) = original_required {
            assert_eq!(required, original);
        } else {
            original_required = Some(required);
        }
        let quote = preparation.quote.as_ref().unwrap();
        let scopes = quote.preparation_scopes.as_ref().unwrap();
        let held = probe.facts().held;
        let mut retained = None;
        with_foreign_runtime(|| {
            busy(
                MlxBackend::prepare_text_prompt_admitted(
                    runtime.backend(),
                    vec![2, 5, 7],
                    &preparation,
                )
                .unwrap_err(),
            );
            let same = prompt_state(scopes);
            let used = pool.fixture_host_charge().unwrap();
            for _ in 0..3 {
                busy(
                    MlxBackend::prepare_text_prompt_admitted(
                        runtime.backend(),
                        vec![2, 5, 7],
                        &preparation,
                    )
                    .unwrap_err(),
                );
                assert_eq!(prompt_state(scopes), same);
                assert_eq!(pool.fixture_host_charge().unwrap(), used);
            }
            let loan = scopes.prompt.borrow();
            let Slot::Pending(pending) = &*loan else {
                panic!("original pending Work")
            };
            retained = Some(pending.work.retention());
        });
        let retained = retained.unwrap();
        assert_eq!(
            retained.phase(),
            (false, false),
            "no active funding or publication before begin"
        );
        drop((preparation, probe));
        submission_recovery::reap();
        assert_eq!(
            retained.phase(),
            (false, false),
            "no fabricated settled observation or publication"
        );
        fixture::settle(&pool, baseline + held);
        drop(retained);
        fixture::settle(&pool, baseline);
        assert_eq!(
            pool.fixture_host_charge().unwrap(),
            baseline,
            "pass {pass} returns its own unexposed scope"
        );
    }
    // The same original candidate at one byte less still rejects before any
    // Prompt creation. Reusable credit did not enlarge its fixed original Q.
    let probe = Probe::new(&runtime, None, false);
    probe.mode(Mode::Defer);
    let result = TextGeneration::from_input_with_sequence(
        &mut runtime,
        TextGenerationInput::TokenIds(vec![2, 5, 7]),
        fixture::config(4, baseline + original_required.unwrap() - 1),
        TokenFilter::All,
        None,
        GenerationSequenceRequest::new(4, &[]),
    );
    let failure = result.err().expect("original exact candidate minus one");
    let mut cause: &(dyn std::error::Error + 'static) = &failure;
    loop {
        if let Some(error) = cause.downcast_ref::<WorkingMemoryError>() {
            assert!(matches!(
                error,
                WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { .. })
            ));
            break;
        }
        cause = cause.source().expect("typed original quota failure");
    }
    drop((failure, probe));
    fixture::settle(&pool, baseline);
    fixture::finish(runtime, &stream);
    fixture::settle(&pool, 0);
}

fn original_pending(
    preparation: &MlxTextPreparation,
) -> (
    InferencePreparationStage,
    PreparedFundedWork,
    NativePreparation,
) {
    let quote = preparation.quote.as_ref().unwrap();
    let original = preparation.request.as_ref().unwrap();
    let (stage, custody) = quote
        .preparation_scopes
        .as_ref()
        .unwrap()
        .bank
        .borrow_mut()
        .claim_prompt(original)
        .unwrap();
    let work = quote.prepared_preparation_work().unwrap();
    let ready = NativePreparation::Unallocated(
        PreparedTextPreparationRetention {
            _request: original.request().clone(),
            funding: work.retention(),
        },
        custody,
    );
    (stage, work, ready)
}
fn begin(
    mut ready: NativePreparation,
) -> submission_recovery::Recovery<PreparedTextPreparationRetention> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match ready.begin() {
            Ok(active) => return active,
            Err((cause, pending)) => {
                assert_eq!(cause, SubmissionScopeOwnerCause::RuntimeBusy);
                ready = pending;
                assert!(Instant::now() < deadline);
                std::thread::yield_now();
            }
        }
    }
}

#[test]
fn panic_after_native_begin_before_funding_exposure_cancels_only_unused_scope() {
    let stream = fixture::stream();
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = fixture::load(&stream, &pool, 0);
    let (probe, preparation) = deferred(&mut runtime, u64::MAX);
    let held = probe.facts().held;
    let (stage, work, ready) = original_pending(&preparation);
    let retained = work.retention();
    let recovery = begin(ready); // A real native Scope, no worker has run.
    let marker = Arc::new(());
    let expected = marker.clone();
    let failure = catch_unwind(AssertUnwindSafe(move || {
        let _recovery = recovery;
        let _work = work;
        std::panic::panic_any(marker);
    }))
    .unwrap_err();
    assert!(Arc::ptr_eq(
        failure.downcast_ref::<Arc<()>>().unwrap(),
        &expected
    ));
    assert_eq!(retained.phase(), (false, false));
    assert!(
        preparation
            .quote
            .as_ref()
            .unwrap()
            .preparation_scopes
            .as_ref()
            .unwrap()
            .bank
            .borrow_mut()
            .claim_prompt(preparation.request.as_ref().unwrap())
            .is_err(),
        "unwind never refills the original role"
    );
    drop((stage, preparation, probe));
    fixture::finish(runtime, &stream);
    fixture::settle(&pool, held);
    assert_eq!(retained.phase(), (false, false));
    drop(retained);
    fixture::settle(&pool, 0);
}

#[test]
fn activated_original_work_keeps_actual_native_validation_error_and_ordinary_cleanup() {
    let stream = fixture::stream();
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = fixture::load(&stream, &pool, 0);
    let (probe, preparation) = deferred(&mut runtime, u64::MAX);
    let (stage, work, ready) = original_pending(&preparation);
    let identity = work.identity();
    let recovery = begin(ready);
    let work = work.activate();
    assert_eq!(std::ptr::from_ref(work.as_ref()) as usize, identity);
    assert!(!work.test_scope_is_retired());
    let mut original_source = None;
    let result: Result<(), Error> = submission_recovery::detached_with_recovery(
        recovery,
        || {
            let input = Array::from_slice(&[2_f32, -3., 5.], &[3]);
            let output = input.square(&stream).unwrap();
            safemlx::transforms::eval([&output]).unwrap();
            assert_eq!(
                output.evaluated().unwrap().as_slice::<f32>(),
                &[4., 9., 25.]
            );
            work.retain(&output);
            let actual = output.reshape(&[4], &stream).unwrap_err();
            original_source = Some((actual.what().as_ptr() as usize, actual.location()));
            Err(actual.into())
        },
        |cause| cause,
    );
    let failure = result.unwrap_err();
    let Error::Exception(actual) = &failure else {
        panic!("unchanged native exception")
    };
    let (pointer, location) = original_source.unwrap();
    assert_eq!(
        actual.what().as_ptr() as usize,
        pointer,
        "original native source String moved, not reformatted"
    );
    assert!(std::ptr::eq(actual.location(), location));
    // Eager native validation failure can be safely settled. Do not label it
    // a failed asynchronous Record or force a quarantine expectation.
    submission_recovery::wait_for_retirement(|| work.test_scope_is_retired());
    drop((failure, work, stage, preparation, probe));
    fixture::finish(runtime, &stream);
    fixture::settle(&pool, 0);
}

#[test]
fn dropping_exposed_unresolved_work_keeps_canonical_quarantine_after_child_retirement() {
    let stream = fixture::stream();
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = fixture::load(&stream, &pool, 0);
    let (probe, preparation) = deferred(&mut runtime, u64::MAX);
    let quote = preparation.quote.as_ref().unwrap();
    let required = quote
        .request()
        .memory_reservation()
        .requirements()
        .get(crate::memory_fixture::topology().host_domain())
        .unwrap()
        .total()
        .unwrap();
    let (stage, custody) = quote
        .preparation_scopes
        .as_ref()
        .unwrap()
        .bank
        .borrow_mut()
        .claim_prompt(preparation.request.as_ref().unwrap())
        .unwrap();
    let work = quote.prepared_preparation_work().unwrap();
    let mut pending = safemlx::PreparedSubmissionScopeOwner::try_new(custody).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut native = loop {
        match safemlx::SubmissionScope::try_begin_retaining(pending) {
            Ok(scope) => break scope,
            Err(error) => {
                assert_eq!(error.cause(), SubmissionScopeOwnerCause::RuntimeBusy);
                pending = error.into_parts().1;
                assert!(Instant::now() < deadline);
                std::thread::yield_now();
            }
        }
    };
    let work = work.activate();
    let mut child = safemlx::SubmissionScope::begin().unwrap();
    native.seal();
    assert!(
        !native.status().is_settled(),
        "actual outstanding native child"
    );
    drop(work); // No completion callback/certification: active Drop must fence.
    assert_eq!(
        quote
            .original_controls()
            .unwrap()
            .validate_reservation(quote.request().memory_reservation()),
        Err(WorkingMemoryError::ExecutionFenced)
    );
    child.seal();
    drop((child, native, stage, preparation, probe));
    fixture::finish(runtime, &stream);
    submission_recovery::reap();
    safemlx::reclaim_allocation_owners();
    assert!(
        pool.fixture_host_charge().unwrap() >= required,
        "native lifetime retirement does not clear the exposed scope's original quarantine"
    );
    assert!(pool.acquire_unquoted().is_err());
}
