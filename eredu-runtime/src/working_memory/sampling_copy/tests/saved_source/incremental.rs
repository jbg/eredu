use super::*;
use crate::working_memory::{CopyPreparationInferenceQuote, quote_inference_workspace};

fn quote_with_copy(
    pool: &MemoryLedger,
    key: u32,
    bytes: u64,
    copies: usize,
    outside_bytes: u64,
) -> Result<CopyPreparationInferenceQuote<u32>, crate::working_memory::ResidualQuoteError> {
    let basis = full(pool, 0);
    let geometry = basis.state.execution_workspace.as_ref().unwrap().geometry;
    let context = WorkspaceContext::new(Facts::default().with_pool(pool));
    let equations = quote_inference_workspace(geometry, |_| {
        context.begin_state_span([])?;
        context.report(&[])
    })
    .unwrap();
    let mut outside = basis.state.execution_workspace.clone().unwrap();
    outside.physical_domains.as_mut().unwrap().retained =
        crate::working_memory::memory_fixture::host_requirements(pool, outside_bytes);
    outside.retained = WorkspaceBound::bounded(
        outside_bytes,
        "independent host/future fixture contribution",
    );
    copy_plan(pool, key, bytes, copies).compose_inference(&equations, basis.state, outside)
}

fn admission(quote: &CopyPreparationInferenceQuote<u32>) -> Admission {
    Admission {
        additional_headroom: eredu_core::MemoryHeadroomDeclarations::none(),
        memory_limits: eredu_core::MemoryLimitDeclarations::unlimited(),
        requested_positions: quote.state().assumptions.requested_positions,
        state: quote.state().clone(),
        incremental_required_bytes: quote.incremental_bytes(),
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

fn terminal_copy_quote(pool: &MemoryLedger) -> CopyPreparationInferenceQuote<u32> {
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 7,
        input_positions: 0,
        max_output_tokens: 0,
        prefill_chunk_positions: 0,
        output: OutputDemand::StateOnly,
    };
    let equations = quote_inference_workspace(
        geometry,
        |_| -> Result<eredu_nn::workspace::WorkspaceTraceReport, eredu_nn::Error> {
            panic!("terminal copy preparation must not inspect a model equation");
        },
    )
    .unwrap();
    let layout = StateMemoryLayout::new(
        LayerSchedule::empty(),
        vec![],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let mut state = eredu_core::estimate_runtime_state(
        &layout,
        InputTokenCount::text(geometry.cached_positions),
        0,
        1,
        std::num::NonZeroU8::new(4).unwrap(),
    )
    .unwrap()
    .with_selected_state_backing(
        geometry,
        WorkspaceBound::bounded(32, "independently priced copied state"),
    )
    .unwrap();
    let mut domains = crate::working_memory::memory_fixture::empty_state(pool, geometry);
    domains.decoder_state = crate::working_memory::memory_fixture::host_requirements(pool, 32);
    state.physical_domains = Some(domains);
    let mut outside_domains =
        crate::working_memory::memory_fixture::host_workspace(pool, geometry, 0);
    outside_domains.retained = crate::working_memory::memory_fixture::host_requirements(pool, 13);
    let zero = || WorkspaceBound::bounded(0, "no additional terminal mechanism");
    let outside = ExecutionWorkspaceEstimate {
        physical_domains: Some(outside_domains),
        geometry,
        activations: zero(),
        attention: zero(),
        vocabulary: zero(),
        state_update: zero(),
        materialization: zero(),
        retained: WorkspaceBound::bounded(13, "retained terminal host payload"),
    };
    copy_plan(pool, 2, 16, 2)
        .compose_inference(&equations, state, outside)
        .unwrap()
}

#[test]
fn terminal_copy_seals_empty_schedule_with_exact_cost_and_source_custody() {
    let pool = crate::working_memory::memory_fixture::host_ledger(16 << 20, 0).unwrap();
    // This source ceiling must also fit the explicitly sealed execution
    // controls, unlike the smaller unsealed-copy fixtures below.
    let (sampler, custody, native, registered) = saved_with_capacity(&pool, false, 16 << 20);
    native.certify().unwrap();
    let witness = custody
        .bind_saved_sampling_source(&sampler, registered.clone())
        .unwrap();
    let copies = attempts();
    let quote = terminal_copy_quote(&pool);
    assert_eq!(quote.state().requested_state_bytes, 32);
    assert_eq!(quote.incremental_bytes().unwrap(), 32 + COPY_BYTES * 2 + 13);
    assert_eq!(full_bytes(&quote), quote.incremental_bytes().unwrap() + 16);
    let unsealed = quote.incremental_bytes().unwrap();
    let quote = quote.with_span_workspace().unwrap();
    assert!(quote.span_workspace().plan().records().is_empty());
    assert_eq!(
        quote.span_workspace().plan().generation_forward_count(),
        Some(0)
    );
    assert!(quote.incremental_bytes().unwrap() > unsealed);
    assert_eq!(full_bytes(&quote), quote.incremental_bytes().unwrap() + 16);
    let before = usage(&pool);
    let capacity = current(&pool)
        + pool
            .saved_source_reservation_requirements(
                &admission(&quote),
                quote.incremental_requirements(),
                &witness,
            )
            .unwrap()
            .get(pool.topology().host_domain())
            .unwrap()
            .total()
            .unwrap();
    let execution = InferenceExecutionIdentity::default();
    assert!(matches!(
        quote.reserve_saved_source_with_capacity_handoff(
            &pool,
            &execution,
            &admission(&quote),
            crate::working_memory::memory_fixture::resolved_host_limits(&pool, capacity - 1),
            &[],
            &witness,
        ),
        Err(WorkingMemoryError::Domain(
            eredu_core::MemoryDomainError::BudgetExceeded { .. }
        ))
    ));
    assert_eq!(usage(&pool), before);
    let reservation = quote
        .reserve_saved_source_with_capacity_handoff(
            &pool,
            &execution,
            &admission(&quote),
            crate::working_memory::memory_fixture::resolved_host_limits(&pool, capacity),
            &[],
            &witness,
        )
        .unwrap();
    assert!(
        quote
            .reserved_span_workspace(&reservation)
            .unwrap()
            .workspace()
            .plan()
            .records()
            .is_empty()
    );
    let foreign = terminal_copy_quote(&pool).with_span_workspace().unwrap();
    assert!(matches!(
        foreign.reserved_span_workspace(&reservation),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    drop(foreign);
    assert_eq!(attempts(), copies);
    let bytes = quote.incremental_bytes().unwrap();
    let (reservation, run) = reservation.into_funding().unwrap();
    let scope = run.scope().unwrap();
    drop((quote, witness));
    drop((registered, sampler, custody, reservation, run));
    assert_eq!(pool.payload_used_bytes().unwrap(), bytes + 16);
    scope.certify().unwrap();
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn terminal_copy_empty_schedule_does_not_bypass_saved_source_health() {
    let pool = crate::working_memory::memory_fixture::host_ledger(16 << 20, 0).unwrap();
    let (sampler, custody, native, registered) = saved_with_capacity(&pool, false, 16 << 20);
    let witness = custody
        .bind_saved_sampling_source(&sampler, registered)
        .unwrap();
    let quote = terminal_copy_quote(&pool).with_span_workspace().unwrap();
    drop(native);
    let before = usage(&pool);
    assert!(matches!(
        quote.reserve_saved_source_with_capacity_handoff(
            &pool,
            &InferenceExecutionIdentity::default(),
            &admission(&quote),
            crate::working_memory::memory_fixture::resolved_host_limits(&pool, 16 << 20),
            &[],
            &witness,
        ),
        Err(WorkingMemoryError::ExecutionFenced)
    ));
    assert_eq!(usage(&pool), before);
}

#[test]
fn exact_increment_excludes_old_root_once_but_keeps_independent_copies_and_history() {
    for adaptive in [false, true] {
        let pool = crate::working_memory::memory_fixture::host_ledger((1 << 20), 0).unwrap();
        let (sampler, custody, native, registered) = saved(&pool, adaptive);
        native.certify().unwrap();
        let witness = custody
            .bind_saved_sampling_source(&sampler, registered.clone())
            .unwrap();
        let quote = quote_with_copy(&pool, 2, 16, 2, 137).unwrap();
        assert_eq!(quote.incremental_bytes().unwrap(), COPY_BYTES * 2 + 137);
        assert_eq!(full_bytes(&quote), quote.incremental_bytes().unwrap() + 16);
        let before = usage(&pool);
        let copies = attempts();
        let history_pointer = history(sampler.as_sampler()).as_ptr();
        let execution = InferenceExecutionIdentity::default();
        let admission = admission(&quote);
        let capacity = current(&pool)
            + pool
                .saved_source_reservation_requirements(
                    &admission,
                    quote.incremental_requirements(),
                    &witness,
                )
                .unwrap()
                .get(pool.topology().host_domain())
                .unwrap()
                .total()
                .unwrap();
        assert!(matches!(
            quote.reserve_saved_source_with_capacity_handoff(
                &pool,
                &execution,
                &admission,
                crate::working_memory::memory_fixture::resolved_host_limits(&pool, capacity - 1),
                &[],
                &witness,
            ),
            Err(WorkingMemoryError::Domain(
                eredu_core::MemoryDomainError::BudgetExceeded { .. }
            ))
        ));
        assert_eq!(usage(&pool), before);
        // The full route includes the source backing, while each route also
        // pays its own concrete pin controls. Check its exact physical bound.
        let mut full_admission = admission.clone();
        full_admission.incremental_required_bytes = Some(full_bytes(&quote));
        let full_capacity = current(&pool)
            + pool
                .saved_source_reservation_requirements(&full_admission, None, &witness)
                .unwrap()
                .get(pool.topology().host_domain())
                .unwrap()
                .total()
                .unwrap();
        assert!(matches!(
            pool.reserve_saved_source_with_capacity_handoff(
                &execution,
                &full_admission,
                crate::working_memory::memory_fixture::resolved_host_limits(
                    &pool,
                    full_capacity - 1
                ),
                &[],
                &witness,
            ),
            Err(WorkingMemoryError::Domain(
                eredu_core::MemoryDomainError::BudgetExceeded { .. }
            ))
        ));
        assert_eq!(usage(&pool), before);
        let reservation = quote
            .reserve_saved_source_with_capacity_handoff(
                &pool,
                &execution,
                &admission,
                crate::working_memory::memory_fixture::resolved_host_limits(&pool, capacity),
                &[],
                &witness,
            )
            .unwrap();
        assert_eq!(
            pool.payload_used_bytes().unwrap(),
            before.0 + quote.incremental_bytes().unwrap()
        );
        assert_eq!(reservation.admission().state, *quote.state());
        assert_eq!(attempts(), copies);
        assert_eq!(history(sampler.as_sampler()), &[3, 11, 7, 19, 5]);
        assert_eq!(history(sampler.as_sampler()).as_ptr(), history_pointer);
        drop((reservation, quote, witness));
        assert_eq!(pool.payload_used_bytes().unwrap(), before.0);
        drop((registered, sampler, custody));
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
}

#[test]
fn saved_and_credited_origins_are_both_rechecked_after_cold_composition() {
    for fail_saved in [false, true] {
        let pool = crate::working_memory::memory_fixture::host_ledger((1 << 20), 0).unwrap();
        let (sampler, custody, native, registered) = saved(&pool, false);
        let witness = custody
            .bind_saved_sampling_source(&sampler, registered.clone())
            .unwrap();
        let (preparation, other_run, _) =
            prepared_request_with_bytes(&pool, (1 << 20), 3, false, 80);
        let other = other_run.scope().unwrap();
        let other_root = other.publish_host_storage_fixture([(77u32, 32)]).unwrap();
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
                crate::working_memory::memory_fixture::resolved_host_limits(&pool, (1 << 20)),
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
        let pool = crate::working_memory::memory_fixture::host_ledger((1 << 20), 0).unwrap();
        let (sampler, custody, native, registered) = saved(&pool, false);
        native.certify().unwrap();
        let (preparation, other_run, _) =
            prepared_request_with_bytes(&pool, (1 << 20), 3, false, 80);
        let other = other_run.scope().unwrap();
        let other_root = other
            .publish_host_storage_fixture([(77u32, bytes)])
            .unwrap();
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
                crate::working_memory::memory_fixture::resolved_host_limits(&pool, (1 << 20)),
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
        let pool = crate::working_memory::memory_fixture::host_ledger((1 << 20), 0).unwrap();
        let (sampler, custody, native, registered) = saved(&pool, false);
        native.certify().unwrap();
        let witness = custody
            .bind_saved_sampling_source(&sampler, registered.clone())
            .unwrap();
        let quote = quote_with_copy(&pool, 2, 16, 1, 13).unwrap();
        let bytes = quote.incremental_bytes().unwrap();
        let reservation = quote
            .reserve_saved_source_with_capacity_handoff(
                &pool,
                &InferenceExecutionIdentity::default(),
                &admission(&quote),
                crate::working_memory::memory_fixture::resolved_host_limits(&pool, (1 << 20)),
                &[],
                &witness,
            )
            .unwrap();
        let (reservation, run) = reservation.into_funding().unwrap();
        let scope = run.scope().unwrap();
        drop((witness, quote));
        drop((registered, sampler, custody, reservation, run));
        assert_eq!(pool.payload_used_bytes().unwrap(), bytes + 16);
        if certify {
            scope.certify().unwrap();
            assert_eq!(pool.payload_used_bytes().unwrap(), 0);
        } else {
            drop(scope);
            assert_eq!(pool.payload_used_bytes().unwrap(), bytes + 16);
        }
    }
}

#[test]
fn altered_full_diagnostics_understatement_foreign_domain_and_overflow_reject() {
    let pool = crate::working_memory::memory_fixture::host_ledger((1 << 20), 0).unwrap();
    let foreign = crate::working_memory::memory_fixture::host_ledger((1 << 20), 0).unwrap();
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
            admission.incremental_required_bytes =
                admission.incremental_required_bytes.map(|bytes| bytes - 1);
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
                crate::working_memory::memory_fixture::resolved_host_limits(&pool, (1 << 20)),
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
                | WorkingMemoryError::Domain(eredu_core::MemoryDomainError::Overflow)
        )) | Err(crate::working_memory::ResidualQuoteError::Estimate(
            eredu_core::CapabilityError::ArithmeticOverflow { .. }
                | eredu_core::CapabilityError::MemoryDomain(
                    eredu_core::MemoryDomainError::Overflow
                )
        ))
    ));
    assert_eq!(usage(&pool), before);
    assert_eq!(foreign.payload_used_bytes().unwrap(), 0);
}
