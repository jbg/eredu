use super::*;
use crate::working_memory::{HostDestinationCause, HostDestinationFacts};

fn request(
    pool: &WorkingMemoryPool,
    bytes: u64,
    attempts: usize,
) -> Option<IncrementalInferenceQuote> {
    let host_facts = match HostDestinationFacts::new(bytes, attempts) {
        Ok(facts) => facts,
        Err(WorkingMemoryError::UnknownBound) => return None,
        Err(cause) => panic!("host destination facts: {cause}"),
    };
    let quote = replacement_quote(pool, geometry(), 0).into_incremental();
    let controls = PreparedTextControlWorkspace::prepare_controls(
        geometry(),
        quote.span_workspace().plan(),
        facts(),
    )
    .unwrap()
    .with_host_destinations(host_facts)
    .unwrap();
    assert!(matches!(
        controls.clone().with_host_destinations(host_facts),
        Err(WorkingMemoryError::AlreadyStarted)
    ));
    Some(
        quote
            .with_span_workspace_and_text_controls(controls)
            .unwrap(),
    )
}

#[test]
fn admitted_fresh_vec_keeps_exact_hold_through_prefix_error_and_final_storage() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let Some(quote) = request(&pool, 12, 3) else {
        assert!(matches!(
            HostDestinationFacts::new(12, 3),
            Err(WorkingMemoryError::UnknownBound)
        ));
        return;
    };
    let exact = 64 + quote.incremental_bytes();
    assert!(matches!(
        sealed_plan(&pool, &quote, exact - 1),
        Err(PrefillPlanningError::Reservation(
            WorkingMemoryError::BudgetExceeded { .. }
        ))
    ));
    let (reservation, accepted) = sealed_plan(&pool, &quote, exact).unwrap();
    let (reservation, run) = reservation.into_funding().unwrap();
    let (mut span, _) = accepted
        .into_funded_text_span_workspace(&run, &reservation)
        .unwrap();
    let protected = span.protected_host_bytes();
    let mut bank = span.take_host_destinations().unwrap().unwrap();
    assert!(matches!(
        span.take_host_destinations(),
        Err(WorkingMemoryError::AlreadyStarted)
    ));
    let mut values = bank.try_vec::<u32>(2).unwrap();
    assert_eq!(
        (
            values.capacity(),
            bank.remaining_bytes(),
            bank.remaining_attempts()
        ),
        (2, 4, 2)
    );
    let pointer = values.as_slice().as_ptr();
    assert!(matches!(
        values.try_fill(2, [11, 23, 37]),
        Err(HostDestinationCause::Extent {
            expected: 2,
            actual: 3
        })
    ));
    assert_eq!(values.as_slice(), &[11, 23]);
    assert_eq!(values.as_slice().as_ptr(), pointer);
    assert!(matches!(
        values.try_fill(2, [0, 0]),
        Err(HostDestinationCause::Consumed)
    ));
    let error = bank.try_vec::<u64>(1).unwrap_err();
    assert!(matches!(
        error.cause(),
        HostDestinationCause::Capacity {
            required: 8,
            remaining: 4
        }
    ));
    assert_eq!(error.destination().unwrap().capacity(), 0);
    assert_eq!((bank.remaining_bytes(), bank.remaining_attempts()), (4, 1));
    let mut tail = bank.try_vec::<u32>(1).unwrap();
    tail.try_fill(1, [41]).unwrap();
    assert!(matches!(
        bank.try_vec::<u8>(0).unwrap_err().cause(),
        HostDestinationCause::Attempts
    ));
    // Native storage cannot spend the disjoint host hold.
    let native = run.scope().unwrap();
    let free = reservation.bytes() - protected;
    assert!(
        matches!(native.adopt_storage_individually([(90u32, free + 1)]), Err(WorkingMemoryError::BudgetExceeded { available_bytes, .. }) if available_bytes == free)
    );
    native.certify().unwrap();
    drop((quote, span, bank, reservation, run, root, tail, error));
    assert_eq!(pool.used_bytes().unwrap(), protected);
    drop(values);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn admitted_vec_checks_real_iterator_completion_and_does_not_refund_or_change_request_owner() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let Some(quote) = request(&pool, 4, 2) else {
        return;
    };
    let (r, run, accepted) = accept(&pool, quote);
    let (mut span, _) = accepted.into_funded_text_span_workspace(&run, &r).unwrap();
    let mut bank = span.take_host_destinations().unwrap().unwrap();
    assert!(bank.belongs_to(&span.control_guard()));
    let foreign_pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let foreign_root = foreign_pool.register_storage([(1u32, 64)]).unwrap();
    let (fr, frun, fq) = accept(&foreign_pool, request(&foreign_pool, 4, 2).unwrap());
    let (foreign, _) = fq.into_funded_text_span_workspace(&frun, &fr).unwrap();
    assert!(!bank.belongs_to(&foreign.control_guard()));
    drop((foreign, frun, fr, foreign_root));
    assert_eq!(foreign_pool.used_bytes().unwrap(), 0);
    let mut storage = bank.try_vec::<u8>(4).unwrap();
    struct Short;
    impl Iterator for Short {
        type Item = u8;
        fn next(&mut self) -> Option<u8> {
            None
        }
        fn size_hint(&self) -> (usize, Option<usize>) {
            (4, Some(4))
        }
    }
    impl ExactSizeIterator for Short {}
    assert!(matches!(
        storage.try_fill(4, Short),
        Err(HostDestinationCause::Extent {
            expected: 4,
            actual: 0
        })
    ));
    drop(storage);
    assert_eq!(bank.remaining_bytes(), 0);
    let error = bank.try_vec::<u8>(1).unwrap_err();
    assert!(matches!(
        error.cause(),
        HostDestinationCause::Capacity {
            required: 1,
            remaining: 0
        }
    ));
    let protected = span.protected_host_bytes();
    drop((span, bank, r, run, root));
    assert_eq!(pool.used_bytes().unwrap(), protected);
    drop(error);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn admitted_empty_vec_has_no_payload_allocation_and_zst_is_outside_qualified_contract() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let Some(quote) = request(&pool, 0, 2) else {
        return;
    };
    let (r, run, quote) = accept(&pool, quote);
    let (mut span, _) = quote.into_funded_text_span_workspace(&run, &r).unwrap();
    let mut bank = span.take_host_destinations().unwrap().unwrap();
    let mut empty = bank.try_vec::<u64>(0).unwrap();
    empty.try_fill(0, []).unwrap();
    assert_eq!((empty.len(), empty.capacity()), (0, 0));
    assert!(matches!(
        bank.try_vec::<()>(1).unwrap_err().cause(),
        HostDestinationCause::Layout
    ));
    assert_eq!((bank.remaining_bytes(), bank.remaining_attempts()), (0, 0));
    drop((empty, bank, span, r, run, root));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn exhausted_calls_cannot_retain_new_custody_and_closed_health_spends_its_only_attempt() {
    for attempts in [0, 1] {
        let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
        let root = pool.register_storage([(1u32, 64)]).unwrap();
        let Some(quote) = request(&pool, 0, attempts) else {
            return;
        };
        let (r, run, quote) = accept(&pool, quote);
        let (mut span, _) = quote.into_funded_text_span_workspace(&run, &r).unwrap();
        let protected = span.protected_host_bytes();
        let mut bank = span.take_host_destinations().unwrap().unwrap();
        drop(run); // Genuine account closure; health validation must refuse.
        let owned = if attempts == 1 {
            let error = bank.try_vec::<u8>(0).unwrap_err();
            assert!(matches!(
                error.cause(),
                HostDestinationCause::Memory(WorkingMemoryError::ExecutionFenced)
            ));
            assert_eq!(error.destination().unwrap().capacity(), 0);
            assert_eq!(bank.remaining_attempts(), 0);
            Some(error)
        } else {
            None
        };
        let refusals: Vec<_> = (0..32)
            .map(|_| bank.try_vec::<u8>(0).unwrap_err())
            .collect();
        assert!(refusals.iter().all(|error| matches!(
            error.cause(),
            HostDestinationCause::Attempts
        ) && error.destination().is_none()));
        drop((span, bank, r, root));
        assert_eq!(
            pool.used_bytes().unwrap(),
            if attempts == 1 { protected } else { 0 }
        );
        drop(owned);
        // All 32 fixed refusal values still exist. None owns/cloned the hold.
        assert_eq!(pool.used_bytes().unwrap(), 0);
        assert_eq!(refusals.len(), 32);
        drop(refusals);
    }
}

mod source;

mod paired;
