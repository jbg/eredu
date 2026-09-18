use super::*;
use crate::working_memory::PreparedPrefillStoragePublicationPlan;
#[test]
fn finite_publication_layout_requires_exact_plan_and_same_key_original_c_before_seal() {
    let pool = WorkingMemoryPool::new(2_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let source = capture_source();
    let raw = replacement_quote(&pool, geometry(), 0).into_incremental();
    let other = replacement_quote(&pool, geometry(), 0).into_incremental();
    let rows = || {
        PreparedPrefillStoragePublicationPlan::<Key>::prepare(raw.span_workspace().plan(), |_| {
            Some(3)
        })
        .unwrap()
    };
    assert!(matches!(
        prepared(&source, &raw).with_prefill_storage_publications(rows()),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let publication = || {
        PreparedCapturePlanPublication::prepare(
            &pool,
            raw.span_workspace().plan(),
            &source,
            key(&source),
            None,
        )
        .unwrap()
    };
    let controls = prepared(&source, &raw)
        .with_capture_plan_publication(publication())
        .unwrap();
    let wrong = PreparedPrefillStoragePublicationPlan::<OtherKey>::prepare(
        raw.span_workspace().plan(),
        |_| Some(3),
    )
    .unwrap();
    assert!(matches!(
        controls.clone().with_prefill_storage_publications(wrong),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let foreign = PreparedPrefillStoragePublicationPlan::<Key>::prepare(
        other.span_workspace().plan(),
        |_| Some(3),
    )
    .unwrap();
    assert!(matches!(
        controls.clone().with_prefill_storage_publications(foreign),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert!(matches!(
        PreparedPrefillStoragePublicationPlan::<Key>::prepare(raw.span_workspace().plan(), |_| {
            None
        }),
        Err(WorkingMemoryError::UnknownBound)
    ));
    assert!(matches!(
        PreparedPrefillStoragePublicationPlan::<Key>::prepare(raw.span_workspace().plan(), |_| {
            Some(usize::MAX)
        }),
        Err(WorkingMemoryError::Overflow)
    ));
    let plain = raw
        .clone()
        .with_span_workspace_and_text_controls(controls.clone())
        .unwrap();
    let rows = rows();
    let s = rows.control_peak_bytes();
    let controls = controls.with_prefill_storage_publications(rows).unwrap();
    assert!(!controls.same_binding(plain.span_workspace().text_controls().unwrap()));
    assert!(controls.same_binding(&controls.clone()));
    assert!(matches!(
        controls.clone().with_prefill_storage_publications(
            PreparedPrefillStoragePublicationPlan::<Key>::prepare(
                raw.span_workspace().plan(),
                |_| Some(3)
            )
            .unwrap()
        ),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let quote = raw
        .with_span_workspace_and_text_controls(controls.clone())
        .unwrap();
    assert_eq!(quote.incremental_bytes(), plain.incremental_bytes() + s);
    assert!(matches!(
        quote
            .clone()
            .with_span_workspace_and_text_controls(controls),
        Err(ResidualQuoteError::Storage(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    assert_eq!(pool.used_bytes().unwrap(), 64);
    drop((root, source, plain, quote, other));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn original_single_hold_prices_finite_publication_rows_once_and_c_remains_separate() {
    let pool = WorkingMemoryPool::new(2_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let source = capture_source();
    let c = source.capacity_bytes().unwrap();
    let raw = replacement_quote(&pool, geometry(), 0).into_incremental();
    let publication = PreparedCapturePlanPublication::prepare(
        &pool,
        raw.span_workspace().plan(),
        &source,
        key(&source),
        None,
    )
    .unwrap();
    let rows =
        PreparedPrefillStoragePublicationPlan::<Key>::prepare(raw.span_workspace().plan(), |_| {
            Some(2)
        })
        .unwrap();
    let controls = prepared(&source, &raw)
        .with_capture_plan_publication(publication)
        .unwrap()
        .with_prefill_storage_publications(rows)
        .unwrap();
    let quote = raw.with_span_workspace_and_text_controls(controls).unwrap();
    let exact = 64 + quote.incremental_bytes();
    assert!(matches!(
        sealed_plan(&pool, &quote, exact - 1),
        Err(PrefillPlanningError::Reservation(
            WorkingMemoryError::BudgetExceeded { .. }
        ))
    ));
    let (r, accepted) = sealed_plan(&pool, &quote, exact).unwrap();
    let (r, run) = r.into_funding().unwrap();
    let native = run.scope().unwrap();
    let pending = accepted
        .begin_capture_plan_publication::<Key>(&run, &r, &source)
        .unwrap();
    let before = account(&pool, &r);
    let (mut owner, witness) = pending.publish_and_finish(&native).unwrap();
    let protected = owner.protected_host_bytes();
    assert_eq!(before.1, protected);
    assert_eq!(account(&pool, &r), (before.0 - c, protected, before.2));
    let bank = owner.take_prefill_storage_publications::<Key>().unwrap();
    assert_eq!(bank.spent_rows(), 0);
    assert!(owner.take_prefill_storage_publications::<Key>().is_err());
    assert_eq!(account(&pool, &r).1, protected);
    native.certify().unwrap();
    drop((owner, witness, quote, root, r, run, source));
    assert_eq!(
        pool.used_bytes().unwrap(),
        protected + c,
        "unissued bank retains original full witness"
    );
    drop(bank);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
