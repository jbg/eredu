//! Real source-free component admission: no synthetic row seeds the namespace.
use super::*;

fn fresh(
    pool: &WorkingMemoryPool,
    rows: usize,
) -> Option<(
    WorkingMemoryReservation,
    WorkingMemoryFundingRun,
    OwnedTextSpanWorkspace,
    OriginalNativeStorageBank<Mechanism>,
    Mechanism,
)> {
    let mut mechanism = Mechanism::new(pool);
    mechanism.nested_key_bytes = Some(0);
    let context = WorkspaceContext::new(Facts::default());
    let storage = RegisteredWorkspaceStorage::<u32>::bind(pool, &context, []).unwrap();
    let report = quote_inference_workspace(geometry(), |_| {
        context.begin_state_span([])?;
        context.report(&[])
    })
    .unwrap();
    // The fixture creates one explicit 32-byte native budget. It is funded
    // independently of Q; empty span reports claim no covered native work.
    let q = ResidualInferenceQuote::compose(
        &report,
        state(geometry()),
        outside(geometry(), 32),
        &storage,
    )
    .unwrap()
    .into_incremental();
    let plan = PreparedNativeStoragePlan::<Mechanism>::prepare_qualified(
        q.span_workspace(),
        &mechanism,
        Some(32),
        Some((1, rows)),
        q.span_workspace().plan().records().iter().map(|_| Some(0)),
        crate::working_memory::qualified_storage::shared_bytes::<Budget>().ok(),
    )
    .unwrap();
    let controls = PreparedTextControlWorkspace::prepare_controls(
        geometry(),
        q.span_workspace().plan(),
        facts(),
    )
    .unwrap()
    .with_native_storage(plan)
    .unwrap();
    let q = match q.with_span_workspace_and_text_controls(controls) {
        Ok(q) => q,
        Err(ResidualQuoteError::Storage(WorkingMemoryError::UnknownBound)) => {
            assert!(!crate::working_memory::qualified_storage::qualified());
            assert_eq!(mechanism.calls.get(), 0);
            return None;
        }
        Err(error) => panic!("fresh namespace component: {error}"),
    };
    // Same quote, exact shared ceiling and one-short: the candidate layouts
    // enter admission before a provider call or namespace allocation occurs.
    let exact = pool.used_bytes().unwrap() + q.incremental_bytes();
    assert!(matches!(
        sealed_plan(pool, &q, exact - 1),
        Err(PrefillPlanningError::Reservation(
            WorkingMemoryError::BudgetExceeded { .. }
        ))
    ));
    assert_eq!(mechanism.calls.get(), 0);
    let (probe, accepted_probe) = sealed_plan(pool, &q, exact).unwrap();
    drop((probe, accepted_probe));
    // Subsequent A and B share the domain's full ceiling, as separate requests.
    let (r, run, accepted) = accept(pool, q);
    let (mut span, _) = accepted.into_funded_text_span_workspace(&run, &r).unwrap();
    let mut bank = span
        .take_native_storage_bank::<Mechanism>(&run, &mechanism.selection)
        .unwrap()
        .unwrap();
    bank.install(mechanism.clone()).unwrap();
    Some((r, run, span, bank, mechanism))
}

#[test]
fn fresh_namespace_keeps_only_original_metadata_after_birth_and_foreign_alias_retire() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let Some((ar, a_run, a_span, mut a, am)) = fresh(&pool, 1) else {
        return;
    };
    let (br, b_run, b_span, mut b, bm) = fresh(&pool, 1).unwrap();
    assert!(pool.0.usage.lock().unwrap().storage.is_empty());
    let a_held = a_span.protected_host_bytes();
    let mut ascope = a_run.scope().unwrap();
    let mut bscope = b_run.scope().unwrap();
    let root = Root::native(7, 16, a.budget_for_scope(&ascope).unwrap());
    let before = balances(&pool);
    let mut ap = a.claim_publication(&mut ascope).unwrap();
    ap.publish(&ascope, [&root], &[]).unwrap();
    let mut bp = b.claim_publication(&mut bscope).unwrap();
    bp.publish(&bscope, [&root], &[]).unwrap();
    assert_eq!(root.attachments(), 2);
    assert_eq!(root.values, [1.25, -3.5]);
    assert_eq!(
        balances(&pool),
        before,
        "native alias never charges P again"
    );
    let ordinary = pool.register_storage([(8u32, 7)]).unwrap();
    ascope.certify().unwrap();
    bscope.certify().unwrap();
    drop((
        ap, bp, a, b, a_span, b_span, ar, br, a_run, b_run, am, bm, root,
    ));
    assert_eq!(pool.used_bytes().unwrap(), a_held + 7);
    {
        let usage = pool.0.usage.lock().unwrap();
        assert_eq!(usage.storage.len(), 1);
        assert_eq!(
            usage.funding.len(),
            1,
            "unused B candidate does not stay in A namespace"
        );
        let account = usage.funding.values().next().unwrap();
        assert_eq!(
            account.protected_remaining().unwrap(),
            a_held,
            "namespace must retain only H after the nonzero P partition retires"
        );
        assert_eq!(account.host_held, a_held);
    }
    drop(ordinary);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    let usage = pool.0.usage.lock().unwrap();
    assert!(usage.storage.is_empty());
    assert!(usage.funding.is_empty());
}

#[test]
fn fresh_namespace_attachment_failure_retains_successful_prefix_and_exact_failed_owners() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let Some((r, run, span, mut bank, mechanism)) = fresh(&pool, 2) else {
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
    assert_eq!(pool.0.usage.lock().unwrap().storage.len(), 1);
    assert_eq!(balances(&pool), before);
    scope.certify().unwrap();
    drop((first, refused, bank, span, r, run, mechanism));
    assert_eq!(pool.used_bytes().unwrap(), held + 32);
    drop(attempt);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert!(pool.0.usage.lock().unwrap().storage.is_empty());
}
