use super::*;
use eredu_text::bpe_storage::{BpeModelPlan, BpeModelSourceErrorKind};
use std::error::Error as _;

const INPUT: &str = r#"{"vocab":{"h":2,"i":5,"hi":90,"😃":4294967295},"merges":[["h","i"]]}"#;
const CONFIGURED: &str = r###"{"vocab":{"a":0,"##b</w>":1,"ab</w>":2,"[UNK]":3},"merges":[["a","##b</w>"]],"unk_token":"[UNK]","continuing_subword_prefix":"##","end_of_word_suffix":"</w>"}"###;
fn plan(input: &str) -> BpeModelPlan<'_> {
    BpeModelPlan::prepare_model_json(input.as_bytes()).unwrap()
}
fn model_values(model: &OriginalBpeModel) {
    assert_eq!(model.token_count(), 4);
    assert_eq!(model.token_id("hi"), Some(90));
    assert_eq!(model.spelling(u32::MAX), Some("😃"));
    assert_eq!(model.spelling(3), None);
    let mut ids: Vec<_> = model.ids().collect();
    ids.sort_unstable();
    assert_eq!(ids, [2, 5, 90, u32::MAX]);
}

#[test]
fn exact_original_model_charge_precedes_compile_and_remains_held_when_idle() {
    let input = INPUT.to_owned();
    let bytes = WorkingMemoryPool::bpe_model_required_bytes(&plan(&input)).unwrap();
    let short = WorkingMemoryPool::new(bytes - 1, 0).unwrap();
    let error = short
        .compile_bpe_model_with(plan(&input), || panic!("short admission compiled"))
        .unwrap_err();
    assert!(
        matches!(error.accounting_failure(),Some(WorkingMemoryError::BudgetExceeded {required_bytes,available_bytes}) if *required_bytes==bytes && *available_bytes==bytes-1)
    );
    assert_eq!(error.retained_bytes(), 0);
    assert_eq!(short.used_bytes().unwrap(), 0);
    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let model = pool
        .compile_bpe_model_with(plan(&input), || {
            assert_eq!(pool.used_bytes().unwrap(), bytes);
            assert!(matches!(
                pool.acquire_unquoted(),
                Err(WorkingMemoryError::ReservedWorkActive)
            ));
        })
        .unwrap();
    drop(input);
    model_values(&model);
    assert_eq!(model.original_bytes(), bytes);
    drop(pool.acquire_unquoted().unwrap());
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    let rejected = pool.compile_bpe_model(plan(INPUT)).unwrap_err();
    assert!(matches!(
        rejected.accounting_failure(),
        Some(WorkingMemoryError::BudgetExceeded {
            available_bytes: 0,
            ..
        })
    ));
    drop(model);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(pool.peak_bytes().unwrap(), bytes);
    // Rejected replay-like attempts own no model/charge after the actual source retires.
    assert_eq!(rejected.retained_bytes(), 0);
}

#[test]
fn seven_actual_partial_failures_retain_their_original_charge_through_core_erasure() {
    let mut previous = 0;
    for stage in 0..7 {
        let input = CONFIGURED.to_owned();
        let plan = plan(&input).fail_reservation(stage);
        let bytes = WorkingMemoryPool::bpe_model_required_bytes(&plan).unwrap();
        let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
        let error = pool.compile_bpe_model(plan).unwrap_err();
        drop(input);
        assert_eq!(error.retained_bytes(), bytes);
        let failure = error.compiler_failure().unwrap();
        assert!(failure.allocation_error().is_some());
        assert!(failure.source_error().is_none());
        let retained = failure.allocated_bytes().unwrap();
        assert_eq!(retained == 0, stage == 0);
        if stage > 0 {
            assert!(retained > previous);
        }
        previous = retained;
        let capacities = failure.buffer_capacities();
        assert!(capacities[..stage].iter().all(|&n| n > 0));
        assert!(capacities[stage..].iter().all(|&n| n == 0));
        drop(pool.acquire_unquoted().unwrap());
        let erased = BackendFailure::new(eredu_core::BackendFailureKind::ResourceExhausted, error);
        let concrete = erased
            .source()
            .unwrap()
            .downcast_ref::<OriginalBpeModelError>()
            .unwrap();
        assert_eq!(
            concrete.compiler_failure().unwrap().allocated_bytes(),
            Some(retained)
        );
        assert_eq!(concrete.retained_bytes(), bytes);
        assert_eq!(pool.used_bytes().unwrap(), bytes);
        drop(erased);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn actual_late_semantic_error_keeps_buffers_after_input_and_public_error_moves() {
    let input = String::from(r#"{"vocab":{"a":7,"b":7},"merges":[]}"#);
    let plan = plan(&input);
    let payload = plan.requirements().buffer_bytes();
    let bytes = WorkingMemoryPool::bpe_model_required_bytes(&plan).unwrap();
    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let error = pool.compile_bpe_model(plan).unwrap_err();
    drop(input);
    let failure = error.compiler_failure().unwrap();
    assert_eq!(
        failure.source_error().unwrap().kind(),
        BpeModelSourceErrorKind::AmbiguousId
    );
    assert_eq!(failure.allocated_bytes(), Some(payload));
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    let error = BackendFailure::new(eredu_core::BackendFailureKind::InvalidInput, error)
        .with_operation("model component");
    let moved = error;
    assert_eq!(moved.operation(), "model component");
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    drop(moved);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn closed_aliases_escape_inputs_and_callers_but_preserve_exact_source_and_domain() {
    let bytes = WorkingMemoryPool::bpe_model_required_bytes(&plan(INPUT)).unwrap();
    let pool = WorkingMemoryPool::new(bytes * 2, 0).unwrap();
    let witness = pool.clone();
    let first = pool.compile_bpe_model(plan(INPUT)).unwrap();
    let second = pool.compile_bpe_model(plan(INPUT)).unwrap();
    let alias = first.clone();
    assert!(first.same_source(&alias));
    assert!(!first.same_source(&second));
    let foreign = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let rejections: Vec<_> = (0..64)
        .map(|_| first.validate_pool(&foreign).unwrap_err())
        .collect();
    assert!(rejections
        .iter()
        .all(|e| matches!(e, WorkingMemoryError::IdentityMismatch)));
    assert_eq!(foreign.used_bytes().unwrap(), 0);
    first.validate_pool(&pool).unwrap();
    drop(first);
    drop(pool);
    model_values(&alias);
    assert_eq!(witness.used_bytes().unwrap(), bytes * 2);
    let peer = alias.clone();
    std::thread::scope(|s| {
        s.spawn(move || drop(alias));
        s.spawn(move || drop(peer));
    });
    assert_eq!(witness.used_bytes().unwrap(), bytes);
    model_values(&second);
    drop(second);
    assert_eq!(witness.used_bytes().unwrap(), 0);
    assert_eq!(rejections.len(), 64);
}

#[test]
fn independent_active_compilers_share_one_atomic_capacity_comparison() {
    let bytes = WorkingMemoryPool::bpe_model_required_bytes(&plan(INPUT)).unwrap();
    let pool = WorkingMemoryPool::new(bytes * 2, 0).unwrap();
    let (first, second, observations, arrivals) = std::thread::scope(|s| {
        let (notify, notifications) = std::sync::mpsc::channel();
        let (release_first, resume_first) = std::sync::mpsc::channel::<()>();
        let (release_second, resume_second) = std::sync::mpsc::channel::<()>();
        let make = |notify: std::sync::mpsc::Sender<bool>,
                    resume: std::sync::mpsc::Receiver<()>| {
            let mut admitted = false;
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                pool.compile_bpe_model_with(plan(INPUT), || {
                    admitted = true;
                    let _ = notify.send(true);
                    // Disconnection releases the worker even if the parent unwinds.
                    let _ = resume.recv();
                })
            }));
            if !admitted {
                // Admission rejection and a pre-hook panic must both wake the parent.
                let _ = notify.send(false);
            }
            result
        };
        let peer_notify = notify.clone();
        let first = s.spawn(move || make(peer_notify, resume_first));
        let second = s.spawn(move || make(notify, resume_second));
        let arrivals = [notifications.recv(), notifications.recv()];
        let observations = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let used = pool.used_bytes();
            let excluded = matches!(
                pool.acquire_unquoted(),
                Err(WorkingMemoryError::ReservedWorkActive)
            );
            let reservations = pool.0.usage.lock().unwrap().reservations;
            (used, excluded, reservations)
        }));
        // Release and join both workers before any assertion, including observations.
        drop(release_first);
        drop(release_second);
        (first.join(), second.join(), observations, arrivals)
    });
    assert!(arrivals.iter().all(|result| matches!(result, Ok(true))));
    let (used, excluded, reservations) = observations.unwrap();
    assert_eq!(used.unwrap(), bytes * 2);
    assert!(excluded);
    assert_eq!(reservations, 2);
    let first = first.unwrap().unwrap().unwrap();
    let second = second.unwrap().unwrap().unwrap();
    assert!(!first.same_source(&second));
    model_values(&first);
    model_values(&second);
    {
        let usage = pool.0.usage.lock().unwrap();
        assert_eq!(usage.reservations, 0);
    }
    drop(pool.acquire_unquoted().unwrap());
    drop(first);
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    drop(second);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn unknown_host_ownership_and_unwind_do_not_reach_an_unguarded_constructor() {
    let bytes = WorkingMemoryPool::bpe_model_required_bytes(&plan(INPUT)).unwrap();
    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let host = pool.acquire_unquoted().unwrap();
    let error = pool
        .compile_bpe_model_with(plan(INPUT), || panic!("unknown host compiled"))
        .unwrap_err();
    assert!(matches!(
        error.accounting_failure(),
        Some(WorkingMemoryError::UnknownBound)
    ));
    assert_eq!(error.retained_bytes(), 0);
    drop(host);
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pool.compile_bpe_model_with(plan(INPUT), || {
            assert_eq!(pool.used_bytes().unwrap(), bytes);
            panic!("after actual original admission");
        })
    }));
    assert!(panic.is_err());
    assert_eq!(pool.used_bytes().unwrap(), 0);
    drop(pool.acquire_unquoted().unwrap());
    let model = pool.compile_bpe_model(plan(INPUT)).unwrap();
    model_values(&model);
    drop(model);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn poisoned_settlement_keeps_actual_completed_or_partial_model_and_unsettled_charge() {
    for partial in [false, true] {
        let plan = if partial {
            plan(INPUT).fail_reservation(2)
        } else {
            plan(INPUT)
        };
        let bytes = WorkingMemoryPool::bpe_model_required_bytes(&plan).unwrap();
        let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
        let error = pool
            .compile_bpe_model_with(plan, || {
                assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _usage = pool.0.usage.lock().unwrap();
                    panic!("poison exact source account");
                }))
                .is_err());
            })
            .unwrap_err();
        assert!(matches!(
            error.accounting_failure(),
            Some(WorkingMemoryError::Poisoned)
        ));
        assert_eq!(error.retained_bytes(), bytes);
        if partial {
            assert!(error.compiler_failure().unwrap().allocated_bytes().unwrap() > 0);
            assert!(error._completed.is_none());
        } else {
            assert_eq!(error._completed.as_ref().unwrap().spelling(90), Some("hi"));
        }
        {
            let usage = pool.0.usage.lock().unwrap_err().into_inner();
            assert_eq!(usage.reservations, 1);
            assert_eq!(usage.reserved, bytes);
        }
        drop(error);
        let usage = pool.0.usage.lock().unwrap_err().into_inner();
        assert_eq!(usage.reservations, 1);
        assert_eq!(usage.reserved, bytes);
    }
}
