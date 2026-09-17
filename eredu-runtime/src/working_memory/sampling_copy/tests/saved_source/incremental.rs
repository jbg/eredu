use super::*;
use crate::working_memory::{quote_inference_workspace, CopyPreparationInferenceQuote};

fn quote_with_copy(
    pool: &WorkingMemoryPool,
    key: u32,
    bytes: u64,
    copies: usize,
    outside_bytes: u64,
) -> Result<CopyPreparationInferenceQuote<u32>, crate::working_memory::ResidualQuoteError> {
    let basis = full(0);
    let geometry = basis.state.execution_workspace.as_ref().unwrap().geometry;
    let context = WorkspaceContext::new(Facts::default());
    let equations = quote_inference_workspace(geometry, |_| {
        context.begin_state_span([])?;
        context.report(&[])
    })
    .unwrap();
    let mut outside = basis.state.execution_workspace.clone().unwrap();
    outside.retained = WorkspaceBound::bounded(
        outside_bytes,
        "independent host/future fixture contribution",
    );
    copy_plan(pool, key, bytes, copies).compose_inference(&equations, basis.state, outside)
}

fn admission(quote: &CopyPreparationInferenceQuote<u32>) -> Admission {
    Admission {
        requested_positions: quote.state().assumptions.requested_positions,
        state: quote.state().clone(),
        incremental_required_bytes: quote.incremental_bytes(),
        available_memory_bytes: None,
    }
}

fn full_bytes(quote: &CopyPreparationInferenceQuote<u32>) -> u64 {
    quote.state().requested_state_bytes
        + quote
            .state()
            .execution_workspace
            .as_ref()
            .unwrap()
            .peak_bytes()
            .unwrap()
            .unwrap()
}

#[test]
fn exact_increment_excludes_old_root_once_but_keeps_independent_copies_and_history() {
    for adaptive in [false, true] {
        let pool = WorkingMemoryPool::new(8192, 0).unwrap();
        let (sampler, custody, native, registered) = saved(&pool, adaptive);
        native.certify().unwrap();
        let witness = custody
            .bind_saved_sampling_source(&sampler, registered.clone())
            .unwrap();
        let quote = quote_with_copy(&pool, 2, 16, 2, 137).unwrap();
        assert_eq!(quote.incremental_bytes(), COPY_BYTES * 2 + 137);
        assert_eq!(full_bytes(&quote), quote.incremental_bytes() + 16);
        let before = usage(&pool);
        let copies = attempts();
        let history_pointer = history(sampler.as_sampler()).as_ptr();
        let execution = InferenceExecutionIdentity::default();
        let admission = admission(&quote);
        let capacity = before.0 + quote.incremental_bytes();
        assert!(matches!(
            quote.reserve_saved_source_with_capacity_handoff(
                &pool,
                &execution,
                &admission,
                capacity - 1,
                &[],
                &witness,
            ),
            Err(WorkingMemoryError::BudgetExceeded { .. })
        ));
        assert_eq!(usage(&pool), before);
        // The same full diagnostic cannot fit by the old full-only route.
        let mut full_admission = admission.clone();
        full_admission.incremental_required_bytes = full_bytes(&quote);
        assert!(matches!(
            pool.reserve_saved_source_with_capacity_handoff(
                &execution,
                &full_admission,
                capacity,
                &[],
                &witness,
            ),
            Err(WorkingMemoryError::BudgetExceeded { .. })
        ));
        assert_eq!(usage(&pool), before);
        let reservation = quote
            .reserve_saved_source_with_capacity_handoff(
                &pool,
                &execution,
                &admission,
                capacity,
                &[],
                &witness,
            )
            .unwrap();
        assert_eq!(pool.used_bytes().unwrap(), capacity);
        assert_eq!(reservation.admission().state, *quote.state());
        assert_eq!(attempts(), copies);
        assert_eq!(history(sampler.as_sampler()), &[3, 11, 7, 19, 5]);
        assert_eq!(history(sampler.as_sampler()).as_ptr(), history_pointer);
        drop((reservation, quote, witness));
        assert_eq!(pool.used_bytes().unwrap(), before.0);
        drop((registered, sampler, custody));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn saved_and_credited_origins_are_both_rechecked_after_cold_composition() {
    for fail_saved in [false, true] {
        let pool = WorkingMemoryPool::new(8192, 0).unwrap();
        let (sampler, custody, native, registered) = saved(&pool, false);
        let witness = custody
            .bind_saved_sampling_source(&sampler, registered.clone())
            .unwrap();
        let (preparation, other_run, _) = prepared_request_with_bytes(&pool, 8192, 3, false, 80);
        let other = other_run.scope().unwrap();
        let other_root = other.adopt_storage_individually([(77u32, 32)]).unwrap();
        let quote = quote_with_copy(
            &pool,
            if fail_saved { 2 } else { 77 },
            if fail_saved { 16 } else { 32 },
            1,
            13,
        )
        .unwrap();
        if fail_saved {
            drop(native);
            other.certify().unwrap();
        } else {
            native.certify().unwrap();
            drop(other);
        }
        let before = usage(&pool);
        let copies = attempts();
        assert!(matches!(
            quote.reserve_saved_source_with_capacity_handoff(
                &pool,
                &InferenceExecutionIdentity::default(),
                &admission(&quote),
                8192,
                &[],
                &witness,
            ),
            Err(WorkingMemoryError::ExecutionFenced)
        ));
        assert_eq!(usage(&pool), before);
        assert_eq!(attempts(), copies);
        drop((other_root, preparation, other_run));
    }
}

#[test]
fn uncredited_complete_source_including_zero_key_cannot_hide_later_quarantine() {
    for bytes in [0, 32] {
        let pool = WorkingMemoryPool::new(8192, 0).unwrap();
        let (sampler, custody, native, registered) = saved(&pool, false);
        native.certify().unwrap();
        let (preparation, other_run, _) = prepared_request_with_bytes(&pool, 8192, 3, false, 80);
        let other = other_run.scope().unwrap();
        let other_root = other.adopt_storage_individually([(77u32, bytes)]).unwrap();
        let complete = pool
            .pin_registered_storage([(2u32, 16), (77u32, bytes)])
            .unwrap();
        let witness = custody
            .bind_saved_sampling_source(&sampler, complete)
            .unwrap();
        let quote = quote_with_copy(&pool, 2, 16, 1, 13).unwrap();
        drop(other);
        let before = usage(&pool);
        assert!(matches!(
            quote.reserve_saved_source_with_capacity_handoff(
                &pool,
                &InferenceExecutionIdentity::default(),
                &admission(&quote),
                8192,
                &[],
                &witness,
            ),
            Err(WorkingMemoryError::ExecutionFenced)
        ));
        assert_eq!(usage(&pool), before);
        drop((registered, other_root, preparation, other_run));
    }
}

#[test]
fn reduced_reservation_retains_old_physical_charge_through_completion_or_quarantine() {
    for certify in [true, false] {
        let pool = WorkingMemoryPool::new(8192, 0).unwrap();
        let (sampler, custody, native, registered) = saved(&pool, false);
        native.certify().unwrap();
        let witness = custody
            .bind_saved_sampling_source(&sampler, registered.clone())
            .unwrap();
        let quote = quote_with_copy(&pool, 2, 16, 1, 13).unwrap();
        let bytes = quote.incremental_bytes();
        let reservation = quote
            .reserve_saved_source_with_capacity_handoff(
                &pool,
                &InferenceExecutionIdentity::default(),
                &admission(&quote),
                8192,
                &[],
                &witness,
            )
            .unwrap();
        let (reservation, run) = reservation.into_funding().unwrap();
        let scope = run.scope().unwrap();
        drop((witness, quote));
        drop((registered, sampler, custody, reservation, run));
        assert_eq!(pool.used_bytes().unwrap(), bytes + 16);
        if certify {
            scope.certify().unwrap();
            assert_eq!(pool.used_bytes().unwrap(), 0);
        } else {
            drop(scope);
            assert_eq!(pool.used_bytes().unwrap(), bytes + 16);
        }
    }
}

#[test]
fn altered_full_diagnostics_understatement_foreign_domain_and_overflow_reject() {
    let pool = WorkingMemoryPool::new(8192, 0).unwrap();
    let foreign = WorkingMemoryPool::new(8192, 0).unwrap();
    let (sampler, custody, native, registered) = saved(&pool, false);
    native.certify().unwrap();
    let witness = custody
        .bind_saved_sampling_source(&sampler, registered)
        .unwrap();
    let quote = quote_with_copy(&pool, 2, 16, 1, 13).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let before = usage(&pool);
    for kind in 0..3 {
        let mut admission = admission(&quote);
        if kind == 0 {
            admission.incremental_required_bytes -= 1;
        }
        if kind == 1 {
            admission
                .state
                .execution_workspace
                .as_mut()
                .unwrap()
                .retained = WorkspaceBound::bounded(12, "altered diagnostic");
        }
        let selected_pool = if kind == 2 { &foreign } else { &pool };
        assert!(matches!(
            quote.reserve_saved_source_with_capacity_handoff(
                selected_pool,
                &execution,
                &admission,
                8192,
                &[],
                &witness,
            ),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
    }
    assert!(matches!(
        quote_with_copy(&pool, 2, 16, 1, u64::MAX),
        Err(crate::working_memory::ResidualQuoteError::Storage(
            WorkingMemoryError::Overflow
        )) | Err(crate::working_memory::ResidualQuoteError::Estimate(
            eredu_core::CapabilityError::ArithmeticOverflow { .. }
        ))
    ));
    assert_eq!(usage(&pool), before);
    assert_eq!(foreign.used_bytes().unwrap(), 0);
}
