//! Actual qualified bank construction, admission and retained publication.
use super::*;

fn qualified_plan(
    pool: &WorkingMemoryPool,
    mechanism: &Mechanism,
    attempts: usize,
    rows: usize,
) -> Result<IncrementalInferenceQuote, ResidualQuoteError> {
    let q = replacement_quote(pool, geometry(), 0).into_incremental();
    // Concrete neutral provider: one Rc<Budget>, inline sidecar payloads, fixed
    // borrowed observations and static-string failure. No native proof is used.
    let provider = crate::working_memory::qualified_storage::shared_bytes::<Budget>().ok();
    let plan = PreparedNativeStoragePlan::<Mechanism>::prepare_qualified(
        q.span_workspace(),
        mechanism,
        Some(32),
        Some((attempts, rows)),
        (0..q.span_workspace().plan().records().len())
            .map(|i| q.span_workspace().span_bytes(i).map(|n| n.min(32))),
        provider,
    )
    .map_err(ResidualQuoteError::Storage)?;
    let controls = PreparedTextControlWorkspace::prepare_controls(
        geometry(),
        q.span_workspace().plan(),
        facts(),
    )
    .unwrap()
    .with_native_storage(plan)
    .unwrap();
    q.with_span_workspace_and_text_controls(controls)
}

fn ready(
    pool: &WorkingMemoryPool,
    attempts: usize,
    rows: usize,
) -> Option<(
    WorkingMemoryReservation,
    WorkingMemoryFundingRun,
    OwnedTextSpanWorkspace,
    OriginalNativeStorageBank<Mechanism>,
    Mechanism,
)> {
    let mut mechanism = Mechanism::new(pool);
    mechanism.nested_key_bytes = Some(0); // u32 clone has no nested storage.
    let quote = match qualified_plan(pool, &mechanism, attempts, rows) {
        Ok(quote) => quote,
        Err(ResidualQuoteError::Storage(WorkingMemoryError::UnknownBound)) => {
            assert!(!crate::working_memory::qualified_storage::qualified());
            assert_eq!(mechanism.calls.get(), 0);
            return None;
        }
        Err(cause) => panic!("qualified component quote: {cause}"),
    };
    let (r, run, accepted) = accept(pool, quote);
    let (mut span, _) = accepted.into_funded_text_span_workspace(&run, &r).unwrap();
    let mut bank = span
        .take_native_storage_bank::<Mechanism>(&run, &mechanism.selection)
        .unwrap()
        .unwrap();
    bank.install(mechanism.clone()).unwrap();
    Some((r, run, span, bank, mechanism))
}

#[test]
fn qualified_controls_require_key_fact_and_exact_admission_before_provider_work() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let namespace = pool.register_storage([(1u32, 64)]).unwrap();
    let mut mechanism = Mechanism::new(&pool);
    assert!(matches!(
        qualified_plan(&pool, &mechanism, 2, 3),
        Err(ResidualQuoteError::Storage(
            WorkingMemoryError::UnknownBound
        ))
    ));
    assert_eq!(mechanism.calls.get(), 0);
    assert_eq!(pool.used_bytes().unwrap(), 64);
    mechanism.nested_key_bytes = Some(0);
    if !crate::working_memory::qualified_storage::qualified() {
        assert!(qualified_plan(&pool, &mechanism, 2, 3).is_err());
        return;
    }
    let quote = qualified_plan(&pool, &mechanism, 2, 3).unwrap();
    let exact = 64 + quote.incremental_bytes();
    assert!(matches!(
        sealed_plan(&pool, &quote, exact - 1),
        Err(PrefillPlanningError::Reservation(
            WorkingMemoryError::BudgetExceeded { .. }
        ))
    ));
    assert_eq!(mechanism.calls.get(), 0);
    assert_eq!(pool.used_bytes().unwrap(), 64);
    let (_, r, accepted) = sealed_plan(&pool, &quote, exact).unwrap();
    let (r, run) = r.into_funding().unwrap();
    let (mut span, _) = accepted.into_funded_text_span_workspace(&run, &r).unwrap();
    let mut bank = span
        .take_native_storage_bank::<Mechanism>(&run, &mechanism.selection)
        .unwrap()
        .unwrap();
    bank.install(mechanism.clone()).unwrap();
    assert_eq!(mechanism.calls.get(), 1);
    let mut scope = run.scope().unwrap();
    let attempt = bank.claim_publication(&mut scope).unwrap();
    assert_eq!(attempt.control_capacities(), [3; 6]);
    scope.certify().unwrap();
    drop((attempt, bank, span, r, run, namespace));
    drop(quote);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn qualified_controls_actual_roots_ignore_size_hints_and_do_not_grow_or_retry() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let namespace = pool.register_storage([(1u32, 64)]).unwrap();
    let Some((r, run, span, mut bank, mechanism)) = ready(&pool, 2, 1) else {
        return;
    };
    let mut scope = run.scope().unwrap();
    let root = Root::native(7, 16, bank.budget_for_scope(&scope).unwrap());
    struct Liar<'a>(&'a Root, usize);
    impl<'a> Iterator for Liar<'a> {
        type Item = &'a Root;
        fn next(&mut self) -> Option<Self::Item> {
            if self.1 == 0 {
                None
            } else {
                self.1 -= 1;
                Some(self.0)
            }
        }
        fn size_hint(&self) -> (usize, Option<usize>) {
            (usize::MAX, Some(usize::MAX))
        }
    }
    let mut refused = bank.claim_publication(&mut scope).unwrap();
    let calls = mechanism.calls.get();
    assert!(matches!(
        refused.publish(&scope, Liar(&root, 2), &[]),
        Err(NativeStorageError::Memory(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    assert_eq!(mechanism.calls.get(), calls);
    assert_eq!(refused.control_capacities(), [1; 6]);
    let mut exact = bank.claim_publication(&mut scope).unwrap();
    exact.publish(&scope, Liar(&root, 1), &[]).unwrap();
    assert_eq!(root.attachments(), 1);
    assert_eq!(root.values, [1.25, -3.5]);
    assert_eq!(exact.control_capacities(), [1; 6]);
    for _ in 0..16 {
        assert!(matches!(
            bank.claim_publication(&mut scope),
            Err(WorkingMemoryError::PreparationAlreadyStarted)
        ));
    }
    scope.certify().unwrap();
    drop((refused, exact, root, bank, span, r, run, namespace));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn qualified_controls_distinct_scope_shells_remain_exact_and_moving_scope_preserves_association() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let namespace = pool.register_storage([(1u32, 64)]).unwrap();
    let Some((r, run, span, mut bank, mechanism)) = ready(&pool, 3, 2) else {
        return;
    };
    let mut first = run.scope().unwrap();
    let mut other = run.scope().unwrap();
    let root = Root::native(7, 16, bank.budget_for_scope(&first).unwrap());
    let mut refused = bank.claim_publication(&mut first).unwrap();
    let calls = mechanism.calls.get();
    assert!(matches!(
        refused.publish(&other, [&root], &[]),
        Err(NativeStorageError::Memory(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    assert_eq!(mechanism.calls.get(), calls);
    let mut moved = first;
    let mut same = bank.claim_publication(&mut moved).unwrap();
    same.publish(&moved, [&root], &[]).unwrap();
    let mut distinct = bank.claim_publication(&mut other).unwrap();
    distinct.publish(&other, [&root], &[]).unwrap();
    assert_eq!(root.attachments(), 2);
    moved.certify().unwrap();
    other.certify().unwrap();
    drop((same, distinct, refused, root, bank, span, r, run, namespace));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn qualified_controls_keep_native_prefix_and_failed_preparation_until_actual_retirement() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let namespace = pool.register_storage([(1u32, 64)]).unwrap();
    let Some((r, run, span, mut bank, _)) = ready(&pool, 1, 2) else {
        return;
    };
    let held = span.protected_host_bytes();
    let mut scope = run.scope().unwrap();
    let first = Root::native(7, 16, bank.budget_for_scope(&scope).unwrap());
    let refused = Root::native(99, 16, bank.budget_for_scope(&scope).unwrap());
    let before = balances(&pool);
    let mut attempt = bank.claim_publication(&mut scope).unwrap();
    assert!(matches!(
        attempt.publish(&scope, [&first, &refused], &[]),
        Err(NativeStorageError::Native(Failure(
            "second physical attachment refused"
        )))
    ));
    assert_eq!((first.attachments(), refused.attachments()), (1, 0));
    assert_eq!(balances(&pool), before);
    assert_eq!(attempt.control_capacities(), [2; 6]);
    drop((first, refused));
    scope.certify().unwrap();
    drop((bank, span, r, run, namespace));
    assert_eq!(pool.used_bytes().unwrap(), held + 32);
    drop(attempt);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn qualified_controls_existing_alias_keeps_donor_and_same_vec_registry_without_double_charge() {
    for birth_first in [false, true] {
        let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
        let namespace = pool.register_storage([(1u32, 64)]).unwrap();
        let Some((ar, a_run, a_span, mut a, _)) = ready(&pool, 1, 2) else {
            return;
        };
        let (br, b_run, b_span, mut b, _) = ready(&pool, 1, 2).unwrap();
        let protected = a_span.protected_host_bytes();
        let b_protected = b_span.protected_host_bytes();
        let mut ascope = a_run.scope().unwrap();
        let mut bscope = b_run.scope().unwrap();
        let root = Root::native(7, 16, a.budget_for_scope(&ascope).unwrap());
        let before = balances(&pool);
        let mut ap = a.claim_publication(&mut ascope).unwrap();
        ap.publish(&ascope, [&root, &root], &[]).unwrap();
        let mut bp = b.claim_publication(&mut bscope).unwrap();
        bp.publish(&bscope, [&root], &[]).unwrap();
        assert_eq!(balances(&pool), before);
        assert_eq!(root.attachments(), 2);
        ascope.certify().unwrap();
        bscope.certify().unwrap();
        drop((ap, bp, a, b, a_span, b_span, ar, br, a_run, b_run));
        assert_eq!(
            pool.used_bytes().unwrap(),
            64 + protected + b_protected + 32
        );
        if birth_first {
            drop(root.budget.borrow_mut().take());
        } else {
            root.owners
                .borrow_mut()
                .iter_mut()
                .for_each(|owner| drop(owner.take()));
        }
        assert_eq!(
            pool.used_bytes().unwrap(),
            64 + protected + 32 + if birth_first { b_protected } else { 0 }
        );
        drop((root, namespace));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn qualified_controls_overflow_is_not_hidden_by_missing_native_facts() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let namespace = pool.register_storage([(1u32, 64)]).unwrap();
    let mut mechanism = Mechanism::new(&pool);
    mechanism.nested_key_bytes = Some(u64::MAX);
    let q = replacement_quote(&pool, geometry(), 0).into_incremental();
    if !crate::working_memory::qualified_storage::qualified() {
        return;
    }
    let n = q.span_workspace().plan().records().len();
    assert!(matches!(
        PreparedNativeStoragePlan::<Mechanism>::prepare_qualified(
            q.span_workspace(),
            &mechanism,
            None,
            Some((1, 1)),
            vec![None; n],
            None
        ),
        Err(WorkingMemoryError::Overflow)
    ));
    assert_eq!(mechanism.calls.get(), 0);
    assert_eq!(pool.used_bytes().unwrap(), 64);
    drop(q);
    drop(namespace);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn qualified_zero_attempt_bank_refuses_repeatedly_without_scope_identity_or_provider_calls() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let namespace = pool.register_storage([(1u32, 64)]).unwrap();
    let Some((r, run, span, mut bank, mechanism)) = ready(&pool, 0, 0) else {
        return;
    };
    let mut scope = run.scope().unwrap();
    let calls = mechanism.calls.get();
    let before = balances(&pool);
    let mut refused = Vec::new();
    for _ in 0..32 {
        refused.push(
            bank.claim_publication(&mut scope)
                .err()
                .expect("zero attempts refuse"),
        );
    }
    assert!(refused
        .iter()
        .all(|e| *e == WorkingMemoryError::PreparationAlreadyStarted));
    assert!(scope.native_publication_identity.is_none());
    assert_eq!(mechanism.calls.get(), calls);
    assert_eq!(balances(&pool), before);
    scope.certify().unwrap();
    drop((bank, span, r, run, namespace));
    assert_eq!(
        pool.used_bytes().unwrap(),
        0,
        "nonowning refusals cannot pin original credit"
    );
    drop(refused);
}

mod namespace;

#[test]
fn qualified_source_attachment_prefix_keeps_original_rows_and_moves_only_unattached_sources() {
    for refuse_attachment in [true, false] {
        let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
        let namespace = pool.register_storage([(1u32, 64)]).unwrap();
        let Some((r, run, span, mut bank, _)) = ready(&pool, 1, 3) else {
            return;
        };
        let mut scope = run.scope().unwrap();
        let mut attempt = bank.claim_publication(&mut scope).unwrap();
        assert!(attempt.take_remaining_sources().is_none());
        attempt
            .publish(
                &scope,
                std::iter::empty::<&Root>(),
                &[(71, 4), (72, 8), (72, 8)],
            )
            .unwrap();
        assert!(
            attempt.source(2).is_none(),
            "duplicate has no second registration"
        );
        let attachment = attempt.clone_source_for_attachment(0).unwrap();
        assert_eq!(attachment.bytes(), 4);
        assert!(attempt.clone_source_for_attachment(0).is_none());
        assert_eq!(
            attempt.source(0).unwrap().bytes(),
            4,
            "the actual attempt retains the original through the attachment call"
        );
        scope.certify().unwrap();
        if refuse_attachment {
            drop(attachment);
            assert_eq!(attempt.source(0).unwrap().bytes(), 4);
            assert_eq!(attempt.source(1).unwrap().bytes(), 8);
            drop((bank, span, r, run, namespace));
            assert!(
                pool.used_bytes().unwrap() > 0,
                "failed prefix remains owned by its attempt"
            );
            drop(attempt);
        } else {
            let remaining = attempt.take_remaining_sources().unwrap();
            assert_eq!(remaining.len(), 1);
            assert_eq!(remaining[0].bytes(), 8);
            assert_eq!(
                remaining.capacity(),
                3,
                "moves the prepaid backing, without collecting a new vector"
            );
            assert!(attempt.take_remaining_sources().is_none());
            assert_eq!(
                attempt.source(0).unwrap().bytes(),
                4,
                "attached originals stay with the attempt, not the published source vector"
            );
            drop((attempt, bank, span, r, run, namespace));
            assert!(pool.used_bytes().unwrap() > 0);
            drop(remaining);
            assert!(
                pool.used_bytes().unwrap() > 0,
                "the real attached owner still holds its row"
            );
            drop(attachment);
        }
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn retained_equation_generations_replace_the_peak_and_fund_one_exact_partition() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let namespace = pool.register_storage([(1u32, 64)]).unwrap();
    let mut mechanism = Mechanism::new(&pool);
    mechanism.nested_key_bytes = Some(0);
    let quote = replacement_quote(&pool, geometry(), 0).into_incremental();
    let generations: Vec<u64> = quote
        .span_workspace()
        .plan()
        .records()
        .iter()
        .map(|record| record.new_tensor_allocation_bytes().unwrap() + 32)
        .collect();
    let capacity: u64 = generations.iter().sum();
    let old_peak = quote.equation_incremental_bytes.unwrap();
    let host_peak = quote
        .span_workspace()
        .plan()
        .records()
        .iter()
        .map(|record| record.host_workspace_bytes().unwrap())
        .max()
        .unwrap();
    assert!(capacity + host_peak > old_peak);
    let make_plan = || {
        PreparedNativeStoragePlan::prepare_qualified(
            quote.span_workspace(),
            &mechanism,
            Some(capacity),
            Some((0, 0)),
            quote
                .span_workspace()
                .plan()
                .records()
                .iter()
                .map(|record| record.new_tensor_allocation_bytes()),
            crate::working_memory::qualified_storage::shared_bytes::<Budget>().ok(),
        )
        .unwrap()
    };
    let seal = |plan| {
        let controls = PreparedTextControlWorkspace::prepare_controls(
            geometry(),
            quote.span_workspace().plan(),
            facts(),
        )
        .unwrap()
        .with_native_storage(plan)
        .unwrap();
        quote
            .clone()
            .with_span_workspace_and_text_controls(controls)
    };
    if !crate::working_memory::qualified_storage::qualified() {
        assert!(matches!(
            seal(make_plan()),
            Err(ResidualQuoteError::Storage(
                WorkingMemoryError::UnknownBound
            ))
        ));
        return;
    }
    let old = seal(make_plan()).unwrap();
    let retained = seal(
        make_plan()
            .with_retained_equation_generations(generations.iter().copied().map(Some))
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        retained.incremental_bytes() - old.incremental_bytes(),
        capacity + host_peak - old_peak
    );
    let exact = 64 + retained.incremental_bytes();
    assert!(matches!(
        sealed_plan(&pool, &retained, exact - 1),
        Err(PrefillPlanningError::Reservation(
            WorkingMemoryError::BudgetExceeded { .. }
        ))
    ));
    assert_eq!(mechanism.calls.get(), 0);
    let (_, reservation, accepted) = sealed_plan(&pool, &retained, exact).unwrap();
    let (reservation, run) = reservation.into_funding().unwrap();
    let (mut span, _) = accepted
        .into_funded_text_span_workspace(&run, &reservation)
        .unwrap();
    let mut bank = span
        .take_native_storage_bank::<Mechanism>(&run, &mechanism.selection)
        .unwrap()
        .unwrap();
    bank.install(mechanism.clone()).unwrap();
    let scope = run.scope().unwrap();
    assert_eq!(
        bank.budget_for_scope(&scope).unwrap().0.capacity_bytes(),
        capacity
    );
    assert_eq!(bank.remaining_publications(), 0);
    scope.certify().unwrap();
    drop((
        bank,
        span,
        reservation,
        run,
        old,
        retained,
        quote,
        namespace,
    ));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
