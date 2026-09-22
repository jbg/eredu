use super::super::resume::history_extent;
use super::*;

const CAPACITY: u64 = ROOM;
const PREFIX: [u32; 5] = [3, 11, 7, 19, 5];

fn held(pool: &MemoryLedger) -> u64 {
    pool.0
        .usage
        .lock()
        .unwrap()
        .funding
        .values()
        .map(|state| state.host_held.checked_sub(state.control_floor).unwrap())
        .sum()
}

fn seeded(
    adaptive: bool,
) -> (
    MemoryLedger,
    RunOwnedTextSampler,
    InferenceTextPreparation,
    WorkingMemoryFundingRun,
) {
    let pool = crate::working_memory::memory_fixture::host_ledger(CAPACITY, 0).unwrap();
    let (mut source, preparation, run) = source(&pool, CAPACITY, 64, adaptive);
    grow(&mut source, &PREFIX);
    assert_eq!(
        (
            source.as_sampler().history_len(),
            source.as_sampler().history_capacity()
        ),
        (5, 8)
    );
    (pool, source, preparation, run)
}

fn clear_history(source: &mut RunOwnedTextSampler) {
    // Private fixture access preserves the authenticated owner's custody while
    // exercising a real cleared-but-allocated history. It is not a public escape.
    match &mut source.sampler {
        ConfiguredTextSampler::Standard(sampler) => sampler.clear_generated_tokens(),
        ConfiguredTextSampler::MirostatV2(sampler) => sampler.reset(),
    }
}

#[test]
fn resumed_nonzero_history_requires_exact_preserved_and_future_envelope() {
    let inline = std::mem::size_of::<ConfiguredTextSampler>() as u64;
    for (adaptive, source_capacity) in [(false, 8), (true, 8), (false, 5)] {
        let extents = if source_capacity == 5 {
            [(0, 5, 5), (11, 30, 20), (27, 60, 40)]
        } else {
            [(0, 8, 8), (11, 24, 16), (27, 48, 32)]
        };
        for (maximum, peak_slots, final_capacity) in extents {
            for short in [true, false] {
                let (pool, mut source, source_preparation, source_run) = seeded(adaptive);
                if source_capacity == 5 {
                    // This private scalar fixture retains its original generous
                    // host hold while installing real, exactly packed history.
                    source.sampler = ConfiguredTextSampler::Standard(
                        crate::GenerationSampler::from_resolved(config(64, false).sampling())
                            .with_generated_tokens(PREFIX),
                    );
                    assert_eq!(source.as_sampler().history_capacity(), 5);
                }
                let config = config(maximum, adaptive);
                let plan = source
                    .borrow_funded()
                    .prepare_resume(config.clone())
                    .unwrap();
                let expected = inline + 4 * peak_slots;
                assert_eq!(
                    (
                        plan.history_len(),
                        plan.history_capacity(),
                        plan.max_samples()
                    ),
                    (5, source_capacity, maximum as u64)
                );
                assert_eq!(plan.max_history_capacity(), final_capacity);
                assert_eq!(plan.required_host_bytes(), expected);
                let available = expected - u64::from(short);
                let (preparation, run, fresh) =
                    prepared_request_with_bytes(&pool, CAPACITY, maximum, adaptive, available);
                let before = (usage(&pool), held(&pool), attempts());
                let original = format!("{:?}", source.as_sampler());
                let result = preparation
                    .claim_sampling(fresh.clone())
                    .unwrap()
                    .construct_resumed_sampler(plan, run.sampler_scope().unwrap());
                if short {
                    assert!(
                        matches!(result, Err(WorkingMemoryError::DomainAllowanceExceeded { required_bytes, available_bytes, .. })
                        if required_bytes == expected && available_bytes == available)
                    );
                    assert_eq!((usage(&pool), held(&pool), attempts()), before);
                    assert!(matches!(
                        preparation.claim_sampling(fresh.clone()),
                        Err(WorkingMemoryError::PreparationAlreadyStarted)
                    ));
                    assert_eq!(format!("{:?}", source.as_sampler()), original);
                    drop((preparation, run, source, source_preparation, source_run));
                    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
                    continue;
                }
                let (mut resumed, completion) = result.unwrap();
                completion.finish().unwrap();
                assert!(matches!(
                    preparation.claim_sampling(fresh.clone()),
                    Err(WorkingMemoryError::PreparationAlreadyStarted)
                ));
                assert_eq!(attempts(), before.2 + 1);
                assert_eq!(held(&pool), before.1 + expected);
                assert_eq!(format!("{:?}", resumed.as_sampler()), original);
                assert_eq!((source.issued_samples, resumed.issued_samples), (5, 0));
                assert_ne!(
                    history(source.as_sampler()).as_ptr(),
                    history(resumed.as_sampler()).as_ptr()
                );
                let future: Vec<u32> = (0..maximum).map(|n| 23 + n as u32).collect();
                grow(&mut source, &future);
                grow(&mut resumed, &future);
                assert_eq!(
                    format!("{:?}", resumed.as_sampler()),
                    format!("{:?}", source.as_sampler())
                );
                assert_eq!(resumed.as_sampler().history_capacity(), final_capacity);
                assert_eq!(source.issued_samples, 5 + maximum as u64);
                assert!(
                    matches!(resumed.prepare_sample(), Err(WorkingMemoryError::TextOutputAllowanceExceeded { issued, limit })
                    if issued == maximum as u64 && limit == maximum as u64)
                );
                drop((source, source_preparation, source_run, preparation, run));
                assert_eq!(pool.payload_used_bytes().unwrap(), expected);
                drop(resumed);
                assert_eq!(pool.payload_used_bytes().unwrap(), 0);
            }
        }
    }
}

#[test]
fn resumed_attempts_are_new_but_dropped_permits_and_clearing_never_refund_them() {
    let (pool, source, source_preparation, source_run) = seeded(false);
    let plan = source
        .borrow_funded()
        .prepare_resume(config(2, false))
        .unwrap();
    let (preparation, run, config) =
        prepared_request_with_bytes(&pool, CAPACITY, 2, false, plan.required_host_bytes());
    let (mut resumed, completion) = preparation
        .claim_sampling(config.clone())
        .unwrap()
        .construct_resumed_sampler(plan, run.sampler_scope().unwrap())
        .unwrap();
    completion.finish().unwrap();
    drop(resumed.prepare_sample().unwrap());
    let context = Context::default();
    assert_eq!(
        resumed
            .prepare_sample()
            .unwrap()
            .sample::<Scalar>(&41, 0.7, None, &context)
            .unwrap(),
        41
    );
    assert_eq!(context.callbacks.get(), 1);
    assert_eq!(history(resumed.as_sampler()), &[3, 11, 7, 19, 5, 41]);
    assert_eq!(history(source.as_sampler()), PREFIX);
    assert_eq!((source.issued_samples, resumed.issued_samples), (5, 2));
    clear_history(&mut resumed);
    assert!(matches!(
        resumed.prepare_sample(),
        Err(WorkingMemoryError::TextOutputAllowanceExceeded {
            issued: 2,
            limit: 2
        })
    ));
    assert_eq!(context.callbacks.get(), 1);
    drop((
        resumed,
        preparation,
        run,
        source,
        source_preparation,
        source_run,
    ));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn cleared_source_copies_spare_slots_and_current_adaptive_state() {
    for adaptive in [false, true] {
        for maximum in [0, 9] {
            let (pool, mut source, source_preparation, source_run) = seeded(adaptive);
            clear_history(&mut source);
            assert_eq!(
                (
                    source.as_sampler().history_len(),
                    source.as_sampler().history_capacity(),
                    source.issued_samples
                ),
                (0, 8, 5)
            );
            let original = format!("{:?}", source.as_sampler());
            let plan = source
                .borrow_funded()
                .prepare_resume(config(maximum, adaptive))
                .unwrap();
            let expected = std::mem::size_of::<ConfiguredTextSampler>() as u64
                + 4 * if maximum == 0 { 8 } else { 24 };
            assert_eq!(plan.required_host_bytes(), expected);
            let (preparation, run, config) =
                prepared_request_with_bytes(&pool, CAPACITY, maximum, adaptive, expected);
            let (mut resumed, completion) = preparation
                .claim_sampling(config.clone())
                .unwrap()
                .construct_resumed_sampler(plan, run.sampler_scope().unwrap())
                .unwrap();
            completion.finish().unwrap();
            assert_eq!(format!("{:?}", resumed.as_sampler()), original);
            assert_eq!(
                (
                    resumed.as_sampler().history_len(),
                    resumed.as_sampler().history_capacity(),
                    resumed.issued_samples
                ),
                (0, 8, 0)
            );
            assert_ne!(
                history(source.as_sampler()).as_ptr(),
                history(resumed.as_sampler()).as_ptr()
            );
            let future: Vec<u32> = (0..maximum).map(|n| n as u32 + 31).collect();
            grow(&mut resumed, &future);
            assert_eq!(history(resumed.as_sampler()), future);
            assert_eq!(source.as_sampler().history_len(), 0);
            assert_eq!(source.issued_samples, 5);
            drop((
                resumed,
                preparation,
                run,
                source,
                source_preparation,
                source_run,
            ));
            assert_eq!(pool.payload_used_bytes().unwrap(), 0);
        }
    }
}

#[test]
fn resume_rejects_foreign_source_wrong_scope_prompt_and_repeat_claim() {
    let (pool, source, source_preparation, source_run) = seeded(false);
    let foreign = crate::working_memory::memory_fixture::host_ledger(CAPACITY, 0).unwrap();
    let (preparation, run, config) = prepared_request(&foreign, CAPACITY, 3, false);
    let before = (
        usage(&pool),
        usage(&foreign),
        held(&pool),
        held(&foreign),
        attempts(),
    );
    assert!(matches!(
        preparation
            .claim_sampling(config.clone())
            .unwrap()
            .construct_resumed_sampler(
                source
                    .borrow_funded()
                    .prepare_resume(config.clone())
                    .unwrap(),
                run.sampler_scope().unwrap()
            ),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(
        (
            usage(&pool),
            usage(&foreign),
            held(&pool),
            held(&foreign),
            attempts()
        ),
        before
    );
    drop((preparation, run));
    assert_eq!(foreign.payload_used_bytes().unwrap(), 0);
    let (preparation, run, config) = prepared_request(&pool, CAPACITY, 3, false);
    let (other, other_run, _) = prepared_request(&pool, CAPACITY, 3, false);
    let before = (usage(&pool), held(&pool), attempts());
    assert!(matches!(
        preparation
            .claim_prompt()
            .unwrap()
            .construct_resumed_sampler(
                source
                    .borrow_funded()
                    .prepare_resume(config.clone())
                    .unwrap(),
                run.sampler_scope().unwrap()
            ),
        Err(WorkingMemoryError::InvocationPhaseMismatch)
    ));
    assert_eq!((usage(&pool), held(&pool), attempts()), before);
    assert!(matches!(
        preparation
            .claim_sampling(config.clone())
            .unwrap()
            .construct_resumed_sampler(
                source
                    .borrow_funded()
                    .prepare_resume(config.clone())
                    .unwrap(),
                other_run.sampler_scope().unwrap()
            ),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!((usage(&pool), held(&pool), attempts()), before);
    assert!(matches!(
        preparation.claim_sampling(config.clone()),
        Err(WorkingMemoryError::PreparationAlreadyStarted)
    ));
    run.scope().unwrap().certify().unwrap();
    other_run.scope().unwrap().certify().unwrap();
    drop((
        preparation,
        run,
        other,
        other_run,
        source,
        source_preparation,
        source_run,
    ));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn policy_quota_and_exact_preparation_configuration_are_checked_before_copy() {
    for adaptive in [false, true] {
        let (pool, source, source_preparation, source_run) = seeded(adaptive);
        let normal = config(3, adaptive);
        let changed = if adaptive {
            normal.clone().with_mirostat_v2(4.1, 0.23).unwrap()
        } else {
            TextGenerationConfig::new(ResolvedGenerationConfig {
                top_k: 18,
                ..normal.sampling()
            })
        };
        let before = (usage(&pool), held(&pool), attempts());
        assert!(matches!(
            source.borrow_funded().prepare_resume(changed),
            Err(WorkingMemoryError::PreparationConfigurationMismatch)
        ));
        let unlimited = TextGenerationConfig::new(ResolvedGenerationConfig {
            max_new_tokens: None,
            ..normal.sampling()
        });
        let unlimited = if adaptive {
            unlimited.with_mirostat_v2(3.7, 0.23).unwrap()
        } else {
            unlimited
        };
        assert!(matches!(
            source.borrow_funded().prepare_resume(unlimited),
            Err(WorkingMemoryError::UnknownBound)
        ));
        assert_eq!((usage(&pool), held(&pool), attempts()), before);
        // Static source policy matches both plans. Their quota or full config
        // still cannot be swapped at the fresh stage boundary.
        for stale in [config(2, adaptive), normal.with_seed(41)] {
            let (preparation, run, config) = prepared_request(&pool, CAPACITY, 3, adaptive);
            let before = (usage(&pool), held(&pool), attempts());
            assert!(matches!(
                preparation
                    .claim_sampling(config.clone())
                    .unwrap()
                    .construct_resumed_sampler(
                        source.borrow_funded().prepare_resume(stale).unwrap(),
                        run.sampler_scope().unwrap()
                    ),
                Err(WorkingMemoryError::PreparationConfigurationMismatch)
            ));
            assert_eq!((usage(&pool), held(&pool), attempts()), before);
            assert!(matches!(
                preparation.claim_sampling(config.clone()),
                Err(WorkingMemoryError::PreparationAlreadyStarted)
            ));
            drop((preparation, run));
        }
        drop((source, source_preparation, source_run));
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
}

#[test]
fn source_or_destination_quarantine_after_planning_rejects_at_hold_boundary() {
    for quarantine_source in [true, false] {
        let (pool, source, source_preparation, source_run) = seeded(false);
        let (preparation, run, config) = prepared_request(&pool, CAPACITY, 3, false);
        let borrowed = source.borrow_funded();
        let plan = source
            .borrow_funded()
            .prepare_resume(config.clone())
            .unwrap();
        let mut host_scope = run.sampler_scope().unwrap();
        if quarantine_source {
            drop(source_run.scope().unwrap());
        } else {
            drop(run.scope().unwrap());
        }
        let before = (usage(&pool), held(&pool), attempts());
        // Exercise the final same-lock check independently of the earlier
        // public stage validation, which may also reject a fenced destination.
        assert_eq!(
            host_scope.hold_resumed_sampler_payload(
                plan.required_host_bytes(),
                borrowed.source(),
                borrowed.execution(),
            ),
            Err(WorkingMemoryError::ExecutionFenced)
        );
        assert_eq!((usage(&pool), held(&pool), attempts()), before);
        assert!(matches!(
            preparation
                .claim_sampling(config.clone())
                .unwrap()
                .construct_resumed_sampler(plan, host_scope),
            Err(WorkingMemoryError::ExecutionFenced)
        ));
        assert_eq!((usage(&pool), held(&pool), attempts()), before);
        if quarantine_source {
            run.scope().unwrap().certify().unwrap();
        } else {
            source_run.scope().unwrap().certify().unwrap();
        }
        drop((preparation, run, source, source_preparation, source_run));
        assert_eq!(pool.payload_used_bytes().unwrap(), SOURCE_BYTES);
        assert!(matches!(
            pool.acquire_unquoted(),
            Err(WorkingMemoryError::ReservedWorkActive)
        ));
    }
}

#[test]
fn resumed_host_hold_is_unavailable_to_native_adoption_until_payload_retirement() {
    let (pool, source, source_preparation, source_run) = seeded(true);
    let plan = source
        .borrow_funded()
        .prepare_resume(config(11, true))
        .unwrap();
    let host = plan.required_host_bytes();
    let (preparation, run, config) =
        prepared_request_with_bytes(&pool, CAPACITY, 11, true, host + 17);
    let (resumed, completion) = preparation
        .claim_sampling(config.clone())
        .unwrap()
        .construct_resumed_sampler(plan, run.sampler_scope().unwrap())
        .unwrap();
    completion.finish().unwrap();
    let native = run.scope().unwrap();
    let first = adopt(&native, publication(&pool), [(101_u32, 17)]).unwrap();
    assert!(matches!(
        adopt(&native, publication(&pool), [(102_u32, 1)]),
        Err(WorkingMemoryError::DomainAllowanceExceeded {
            available_bytes: 0,
            ..
        })
    ));
    drop((preparation, run));
    assert_eq!(history(resumed.as_sampler()), PREFIX);
    assert!(matches!(
        adopt(&native, publication(&pool), [(102_u32, host)]),
        Err(WorkingMemoryError::DomainAllowanceExceeded {
            available_bytes: 0,
            ..
        })
    ));
    drop(resumed);
    let second = adopt(&native, publication(&pool), [(102_u32, host)]).unwrap();
    native.certify().unwrap();
    drop((source, source_preparation, source_run));
    assert_eq!(pool.payload_used_bytes().unwrap(), host + 17);
    drop((first, second));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn frozen_copy_is_an_authenticated_resume_source_after_original_owners_retire() {
    let (pool, source, source_preparation, source_run) = seeded(true);
    let frozen = pool
        .copy_sampler(
            source.borrow_funded(),
            SamplerCopyLimits::new(crate::working_memory::memory_fixture::host_limits(CAPACITY)),
        )
        .unwrap();
    let original = format!("{:?}", frozen.as_sampler());
    drop((source, source_preparation, source_run));
    assert_eq!(pool.payload_used_bytes().unwrap(), frozen.bytes());
    let plan = frozen
        .borrow_funded()
        .prepare_resume(config(11, true))
        .unwrap();
    let host = plan.required_host_bytes();
    let (preparation, run, config) = prepared_request_with_bytes(&pool, CAPACITY, 11, true, host);
    let (mut resumed, completion) = preparation
        .claim_sampling(config.clone())
        .unwrap()
        .construct_resumed_sampler(plan, run.sampler_scope().unwrap())
        .unwrap();
    completion.finish().unwrap();
    assert_eq!(format!("{:?}", resumed.as_sampler()), original);
    drop((frozen, preparation, run));
    assert_eq!(pool.payload_used_bytes().unwrap(), host);
    grow(&mut resumed, &[23, 29]);
    assert_eq!(history(resumed.as_sampler()), &[3, 11, 7, 19, 5, 23, 29]);
    assert_eq!(resumed.issued_samples, 2);
    drop(resumed);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn zero_quota_keeps_inline_and_history_payload_and_overflow_never_copies() {
    let pool = crate::working_memory::memory_fixture::host_ledger(CAPACITY, 0).unwrap();
    let (source, source_preparation, source_run) = source(&pool, CAPACITY, 0, false);
    let plan = source
        .borrow_funded()
        .prepare_resume(config(0, false))
        .unwrap();
    let inline = std::mem::size_of::<ConfiguredTextSampler>() as u64;
    assert_eq!(
        (
            plan.history_len(),
            plan.history_capacity(),
            plan.max_history_capacity(),
            plan.required_host_bytes()
        ),
        (0, 0, 0, inline)
    );
    let (preparation, run, config) = prepared_request_with_bytes(&pool, CAPACITY, 0, false, inline);
    let (mut resumed, completion) = preparation
        .claim_sampling(config.clone())
        .unwrap()
        .construct_resumed_sampler(plan, run.sampler_scope().unwrap())
        .unwrap();
    completion.finish().unwrap();
    assert!(matches!(
        resumed.prepare_sample(),
        Err(WorkingMemoryError::TextOutputAllowanceExceeded {
            issued: 0,
            limit: 0
        })
    ));
    drop((
        resumed,
        preparation,
        run,
        source,
        source_preparation,
        source_run,
    ));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);

    let (pool, source, source_preparation, source_run) = seeded(false);
    let before = (usage(&pool), held(&pool), attempts());
    assert!(matches!(
        source
            .borrow_funded()
            .prepare_resume(super::config(usize::MAX, false)),
        Err(WorkingMemoryError::Overflow)
    ));
    assert_eq!(
        history_extent(2, 1, 0),
        Err(WorkingMemoryError::IdentityMismatch)
    );
    assert_eq!(
        history_extent(usize::MAX, usize::MAX, 1),
        Err(WorkingMemoryError::Overflow)
    );
    assert_eq!(
        history_extent(0, isize::MAX as usize / 4 + 1, 0),
        Err(WorkingMemoryError::Overflow)
    );
    assert_eq!((usage(&pool), held(&pool), attempts()), before);
    drop((source, source_preparation, source_run));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn panic_after_hold_releases_only_host_custody_and_never_certifies_native_work() {
    for retain_native in [false, true] {
        let (pool, source, source_preparation, source_run) = seeded(false);
        let plan = source
            .borrow_funded()
            .prepare_resume(config(11, false))
            .unwrap();
        let host = plan.required_host_bytes();
        let (preparation, run, config) =
            prepared_request_with_bytes(&pool, CAPACITY, 11, false, host);
        let native = retain_native.then(|| run.scope().unwrap());
        let before = (usage(&pool), held(&pool), attempts());
        let original = format!("{:?}", source.as_sampler());
        FAIL_COPY.with(|flag| assert!(!flag.replace(true)));
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = preparation
                .claim_sampling(config.clone())
                .unwrap()
                .construct_resumed_sampler(plan, run.sampler_scope().unwrap());
        }));
        assert!(failed.is_err());
        assert_eq!(attempts(), before.2 + 1);
        assert_eq!((usage(&pool), held(&pool)), (before.0, before.1));
        assert_eq!(format!("{:?}", source.as_sampler()), original);
        assert!(matches!(
            preparation.claim_sampling(config.clone()),
            Err(WorkingMemoryError::PreparationAlreadyStarted)
        ));
        drop((preparation, run));
        if let Some(native) = native {
            assert_eq!(pool.payload_used_bytes().unwrap(), SOURCE_BYTES + host);
            // The host-only cleanup cannot certify this independent scope.
            drop(native);
            drop((source, source_preparation, source_run));
            assert_eq!(pool.payload_used_bytes().unwrap(), host);
            assert!(matches!(
                pool.acquire_unquoted(),
                Err(WorkingMemoryError::ReservedWorkActive)
            ));
        } else {
            assert_eq!(pool.payload_used_bytes().unwrap(), SOURCE_BYTES);
            drop((source, source_preparation, source_run));
            assert_eq!(pool.payload_used_bytes().unwrap(), 0);
        }
    }
}
