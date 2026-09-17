use super::*;
use crate::working_memory::{
    CaptureRunHostPlan, PreparedTextControlWorkspace, ReservedTextSpanWorkspace,
    TextHostControlFacts, WorkingMemoryFundingRun,
};
use eredu_core::{
    capture::*, DescriptionCompleteness, ObservationCatalog, ObservationDtype, ObservationPoint,
    ObservationPosition, ObservationRequirement, ObservationSupport, ObservationSupportReport,
    ObservationSupportStatus, ObservationValueType, SymbolicDimension, TensorAxis,
};
use std::panic::{catch_unwind, AssertUnwindSafe};

fn capture_source() -> SharedCapturePlan {
    capture_source_for_geometry(geometry())
}
fn capture_source_for_geometry(g: InferenceGeometry) -> SharedCapturePlan {
    let point = ObservationPoint {
        path: "block.output".into(),
        node_id: "block".into(),
        meaning: "fixture values".into(),
        value_type: ObservationValueType::Tensor,
        dtype: ObservationDtype::Floating,
        axes: Some(vec![
            TensorAxis {
                name: "sequence".into(),
                dimension: SymbolicDimension::Sequence,
            },
            TensorAxis {
                name: "width".into(),
                dimension: SymbolicDimension::Known(2),
            },
        ]),
        prefill: true,
        decode: true,
        requirements: vec![ObservationRequirement::ActivationHooks],
        position: ObservationPosition::BeforeIntervention,
        retained_bytes: None,
        host_bytes: None,
    };
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: Default::default(),
        points: vec![ObservationSupport {
            path: point.path.clone(),
            prefill: ObservationSupportStatus::Supported,
            decode: ObservationSupportStatus::Supported,
            floating_to_f32: true,
        }],
    };
    let catalog = ObservationCatalog {
        schema_version: 1,
        points: vec![point],
        completeness: DescriptionCompleteness::Complete,
    };
    let caps = CaptureCapabilities {
        transformations: vec![CaptureTransformKind::FullTensor],
        max_histogram_bins: 0,
        physical_native_limit: false,
        conditions: vec![],
    };
    let mut raw = CapturePlan::none();
    raw.selections.push(CaptureSelection {
        id: "row".into(),
        path: "block.output".into(),
        schedule: Default::default(),
        slices: vec![],
        transform: CaptureTransform::FullTensor,
    });
    let unlimited = CaptureUsage {
        captures: u64::MAX,
        retained_bytes: u64::MAX,
        host_bytes: u64::MAX,
        encoded_bytes: u64::MAX,
    };
    raw.limits.per_step = unlimited;
    raw.limits.cumulative = unlimited;
    SharedCapturePlan::new(
        raw.admit_with_text_origin(
            &catalog,
            &support,
            &caps,
            CaptureRequestShape {
                batch: g.batch_size,
                prompt_tokens: g.input_positions,
                max_predictions: g.max_output_tokens,
            },
            CaptureTextOrigin {
                cached_positions: g.cached_positions,
            },
        )
        .unwrap(),
    )
}
fn facts() -> TextHostControlFacts {
    TextHostControlFacts::new(Some(11), Some(17), Some(23))
}
fn prepared(
    source: &SharedCapturePlan,
    q: &IncrementalInferenceQuote,
) -> PreparedTextControlWorkspace {
    PreparedTextControlWorkspace::prepare(source, q.geometry(), q.span_workspace().plan(), facts())
        .unwrap()
}
fn quote(pool: &WorkingMemoryPool, source: &SharedCapturePlan) -> IncrementalInferenceQuote {
    let q = replacement_quote(pool, geometry(), 0).into_incremental();
    let c = prepared(source, &q);
    q.with_span_workspace_and_text_controls(c).unwrap()
}
fn accept(
    pool: &WorkingMemoryPool,
    q: IncrementalInferenceQuote,
) -> (
    WorkingMemoryReservation,
    WorkingMemoryFundingRun,
    IncrementalInferenceQuote,
) {
    let (_, r, q) = sealed_plan(pool, &q, 1_000_000).unwrap();
    let (r, run) = r.into_funding().unwrap();
    (r, run, q)
}
fn account(pool: &WorkingMemoryPool, r: &WorkingMemoryReservation) -> (u64, u64, usize) {
    let usage = pool.0.usage.lock().unwrap();
    let a = &usage.funding[&r.0.funding.unwrap()];
    (a.remaining, a.host_held, a.scopes)
}
#[test]
fn text_seal_adds_named_q_and_actual_p_once_with_exact_original_capacity() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let source = capture_source();
    let original = replacement_quote(&pool, geometry(), 0).into_incremental();
    let c = prepared(&source, &original);
    let earlier = c.clone();
    assert_eq!(c.facts(), facts());
    assert_eq!(c.source_identity(), Some(source.storage_identity()));
    let before = original.incremental_bytes();
    let q = original.with_span_workspace_and_text_controls(c).unwrap();
    let p = q.span_workspace().retention_peak_bytes().unwrap();
    assert_eq!(q.incremental_bytes(), before + p + 51);
    assert_eq!(pool.used_bytes().unwrap(), 64);
    assert!(matches!(
        q.clone().with_span_workspace(),
        Err(ResidualQuoteError::Storage(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    assert!(matches!(
        q.clone()
            .with_span_workspace_and_text_controls(earlier.clone()),
        Err(ResidualQuoteError::Storage(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    let exact = 64 + q.incremental_bytes();
    assert!(matches!(
        sealed_plan(&pool, &q, exact - 1),
        Err(PrefillPlanningError::Reservation(
            WorkingMemoryError::BudgetExceeded { .. }
        ))
    ));
    let (_, r, accepted) = sealed_plan(&pool, &q, exact).unwrap();
    let (r, run) = r.into_funding().unwrap();
    let (owner, _) = accepted.into_funded_text_span_workspace(&run, &r).unwrap();
    assert_eq!(owner.protected_host_bytes(), p + 51);
    assert_eq!(account(&pool, &r).1, p + 51);
    assert!(owner.workspace().plan().same_plan(earlier.plan()));
    drop((owner, r, run, q, earlier, source, root));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn text_facts_bind_original_plan_geometry_source_and_preserve_unknown_overflow() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let source = capture_source();
    let q = replacement_quote(&pool, geometry(), 0).into_incremental();
    let other = replacement_quote(&pool, geometry(), 0).into_incremental();
    let c = prepared(&source, &other);
    assert!(matches!(
        q.clone().with_span_workspace_and_text_controls(c),
        Err(ResidualQuoteError::Storage(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    let changed = InferenceGeometry {
        cached_positions: 5,
        ..geometry()
    };
    assert!(matches!(
        PreparedTextControlWorkspace::prepare(&source, changed, q.span_workspace().plan(), facts()),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let equal = SharedCapturePlan::new(source.admission().clone());
    let c = prepared(&equal, &q);
    assert_ne!(c.source_identity(), Some(source.storage_identity()));
    for f in [
        TextHostControlFacts::new(None, Some(7), Some(9)),
        TextHostControlFacts::new(Some(0), None, Some(0)),
    ] {
        let c = PreparedTextControlWorkspace::prepare(
            &source,
            geometry(),
            q.span_workspace().plan(),
            f,
        )
        .unwrap();
        assert_eq!(c.facts().total_bytes().unwrap(), None);
        assert!(matches!(
            q.clone().with_span_workspace_and_text_controls(c),
            Err(ResidualQuoteError::Storage(
                WorkingMemoryError::UnknownBound
            ))
        ));
    }
    assert!(matches!(
        PreparedTextControlWorkspace::prepare(
            &source,
            geometry(),
            q.span_workspace().plan(),
            TextHostControlFacts::new(None, Some(u64::MAX), Some(1))
        ),
        Err(WorkingMemoryError::Overflow)
    ));
    let c = PreparedTextControlWorkspace::prepare(
        &source,
        geometry(),
        q.span_workspace().plan(),
        TextHostControlFacts::new(Some(u64::MAX), Some(0), Some(0)),
    )
    .unwrap();
    assert!(matches!(
        q.clone().with_span_workspace_and_text_controls(c),
        Err(ResidualQuoteError::Storage(WorkingMemoryError::Overflow))
    ));
    assert_eq!(pool.used_bytes().unwrap(), 64);
    drop((q, other, root));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn aggregate_hold_excludes_every_q_byte_and_coexists_with_original_h() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let source = capture_source();
    let h = CaptureRunHostPlan::prepare(&source)
        .unwrap()
        .initialization_peak_bytes();
    let context = WorkspaceContext::new(Facts::default());
    let root_view = WorkspaceExistingStorage::new(Some(64), &context);
    let storage =
        RegisteredWorkspaceStorage::bind(&pool, &context, [(1u32, root_view.clone())]).unwrap();
    let report = replacement_report(&context, &root_view, geometry());
    let mut enclosing = outside(geometry(), 0);
    enclosing.retained = WorkspaceBound::bounded(h, "actual closed cumulative capture host plan");
    let original = ResidualInferenceQuote::compose(&report, state(geometry()), enclosing, &storage)
        .unwrap()
        .into_incremental();
    let c = prepared(&source, &original);
    let q = original.with_span_workspace_and_text_controls(c).unwrap();
    let (r, run, q) = accept(&pool, q);
    let native = run.scope().unwrap();
    let (owner, _) = q.into_funded_text_span_workspace(&run, &r).unwrap();
    let pq = owner.protected_host_bytes();
    let bank = run
        .prepare_capture_run(&r, CaptureRunHostPlan::prepare(&source).unwrap())
        .unwrap();
    assert_eq!(account(&pool, &r).1, pq + h);
    let n = r.bytes() - pq - h;
    assert!(
        matches!(native.adopt_storage_individually([(20u32,n+1)]),Err(WorkingMemoryError::BudgetExceeded{required_bytes,available_bytes}) if required_bytes==n+1 && available_bytes==n)
    );
    let payload = native.adopt_storage_individually([(20u32, n)]).unwrap();
    assert_eq!(account(&pool, &r).0 - account(&pool, &r).1, 0);
    owner
        .control_guard()
        .validate_native_scope(&native)
        .unwrap();
    drop(payload);
    native.certify().unwrap();
    drop((bank, owner, r, run, storage, root));
    // The original report is an earlier alias of the same retained span plan.
    assert!(pool.used_bytes().unwrap() >= pq);
    drop(report);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn failed_pq_promotion_preserves_quote_and_alias_then_exact_retry_succeeds() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let source = capture_source();
    let q = quote(&pool, &source);
    let (r, run, q) = accept(&pool, q);
    let alias = q.span_workspace().plan().clone();
    let pq = q.span_workspace().retention_peak_bytes().unwrap() + 51;
    let native = run.scope().unwrap();
    let pressure = native
        .adopt_storage_individually([(21u32, r.bytes() - pq + 1)])
        .unwrap();
    let before = account(&pool, &r);
    let error = q.into_funded_text_span_workspace(&run, &r).unwrap_err();
    assert!(
        matches!(error.cause(),WorkingMemoryError::BudgetExceeded{required_bytes,available_bytes} if *required_bytes==pq && *available_bytes+1==pq)
    );
    assert_eq!(account(&pool, &r), before);
    let (q, _) = error.into_parts();
    assert!(q.span_workspace().plan().same_plan(&alias));
    drop(pressure);
    let (owner, _) = q.into_funded_text_span_workspace(&run, &r).unwrap();
    assert_eq!(account(&pool, &r).1, pq);
    native.certify().unwrap();
    drop((owner, alias, r, run, root));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn compact_control_guard_is_last_custody_owner_without_retaining_plan_records() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let source = capture_source();
    let q = quote(&pool, &source);
    let (r, run, q) = accept(&pool, q);
    let (owner, _) = q.into_funded_text_span_workspace(&run, &r).unwrap();
    let pq = owner.protected_host_bytes();
    let guard = owner.control_guard();
    let native = run.scope().unwrap();
    let rest = native
        .adopt_storage_individually([(22u32, r.bytes() - pq)])
        .unwrap();
    native.certify().unwrap();
    assert_eq!(
        owner.workspace().plan().strong_owner_count(),
        2,
        "workspace and retained diagnostic field share plan; guard has no plan"
    );
    let used = pool.used_bytes().unwrap();
    drop((owner, run, r, root));
    assert_eq!(pool.used_bytes().unwrap(), used - 64);
    drop(guard);
    assert_eq!(pool.used_bytes().unwrap(), used - 64 - pq);
    drop(rest);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn early_diagnostic_alias_keeps_pq_through_owner_unwind_without_native_grant() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let source = capture_source();
    let original = replacement_quote(&pool, geometry(), 0).into_incremental();
    let controls = prepared(&source, &original);
    let alias = controls.clone();
    let q = original
        .with_span_workspace_and_text_controls(controls)
        .unwrap();
    let (r, run, q) = accept(&pool, q);
    let pq = q.span_workspace().retention_peak_bytes().unwrap() + 51;
    let result = catch_unwind(AssertUnwindSafe(|| {
        let (_owner, _) = q.into_funded_text_span_workspace(&run, &r).unwrap();
        panic!("after successful P+Q attachment");
    }));
    assert!(result.is_err());
    assert_eq!(account(&pool, &r).1, pq);
    assert_eq!(alias.plan().records().len(), 3);
    drop((r, run, root));
    assert!(pool.used_bytes().unwrap() >= pq);
    drop(alias);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn foreign_accounts_and_concurrent_duplicate_promotions_cannot_replace_custody() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let source = capture_source();
    let q = quote(&pool, &source);
    let (r, run, q) = accept(&pool, q);
    let oq = quote(&pool, &source);
    let (other, other_run, oq) = accept(&pool, oq);
    let before = account(&pool, &r);
    let e = q
        .into_funded_text_span_workspace(&other_run, &r)
        .unwrap_err();
    let (q, cause) = e.into_parts();
    assert!(matches!(cause, WorkingMemoryError::IdentityMismatch));
    let e = q.into_funded_text_span_workspace(&run, &other).unwrap_err();
    let (q, _) = e.into_parts();
    assert_eq!(account(&pool, &r), before);
    let duplicate = q.clone();
    let (a, b) = std::thread::scope(|scope| {
        let a = scope.spawn(|| q.into_funded_text_span_workspace(&run, &r));
        let b = scope.spawn(|| duplicate.into_funded_text_span_workspace(&run, &r));
        (a.join().unwrap(), b.join().unwrap())
    });
    let (owner, error) = match (a, b) {
        (Ok((owner, _)), Err(e)) | (Err(e), Ok((owner, _))) => (owner, e),
        _ => panic!("exactly one attachment"),
    };
    assert!(matches!(
        error.cause(),
        WorkingMemoryError::IdentityMismatch
    ));
    let guard = owner.control_guard();
    assert!(matches!(
        guard.validate_reservation(&other),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let other_scope = other_run.scope().unwrap();
    assert!(matches!(
        guard.validate_native_scope(&other_scope),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    other_scope.certify().unwrap();
    drop((owner, error, guard, oq, other, other_run, r, run, root));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn p_only_and_wrong_control_receipts_reject_under_the_existing_usage_lock() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let source = capture_source();
    let q = quote(&pool, &source);
    let (r, run, q) = accept(&pool, q);
    let (owner, _) = q.into_funded_text_span_workspace(&run, &r).unwrap();
    let guard = owner.control_guard();
    let base = replacement_quote(&pool, geometry(), 0).into_incremental();
    let extra = r.bytes()
        - base.incremental_bytes()
        - base.span_workspace().retention_peak_bytes().unwrap();
    drop(base);
    let plain = replacement_quote(&pool, geometry(), extra)
        .into_incremental()
        .with_span_workspace()
        .unwrap();
    let (pr, prun, plain) = accept(&pool, plain);
    assert_eq!(
        pr.bytes(),
        r.bytes(),
        "equal bytes cannot substitute P-only authority"
    );
    let e = plain
        .into_funded_text_span_workspace(&prun, &pr)
        .unwrap_err();
    let (plain, _) = e.into_parts();
    let (po, _) = plain.into_funded_span_workspace(&prun, &pr).unwrap();
    let forged = ReservedTextSpanWorkspace {
        span: po.as_reserved_span_workspace(),
        controls: &guard,
    };
    let oq = quote(&pool, &source);
    let (orr, orun, oq) = accept(&pool, oq);
    let (other, _) = oq.into_funded_text_span_workspace(&orun, &orr).unwrap();
    let other_guard = other.control_guard();
    let swapped = ReservedTextSpanWorkspace {
        span: owner.as_reserved_text_span_workspace().span,
        controls: &other_guard,
    };
    let usage = pool.0.usage.lock().unwrap();
    owner
        .as_reserved_text_span_workspace()
        .validate_locked(&pool, &usage)
        .unwrap();
    assert!(matches!(
        forged.validate_locked(&pool, &usage),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert!(matches!(
        swapped.validate_locked(&pool, &usage),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    drop(usage);
    drop((forged, swapped));
    drop((
        owner,
        guard,
        po,
        pr,
        prun,
        other,
        other_guard,
        orr,
        orun,
        r,
        run,
        root,
    ));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn every_explicit_zero_and_nonzero_source_is_checked_before_and_after_attachment() {
    for bytes in [0, 24] {
        for phase in 0..3 {
            let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
            let root = pool.register_storage([(1u32, 64)]).unwrap();
            let source = capture_source();
            let origin = replacement_quote(&pool, geometry(), 0).into_incremental();
            let (sr, srun, _) = accept(&pool, origin);
            let source_scope = srun.scope().unwrap();
            let registration = source_scope
                .adopt_storage_individually([(30u32, bytes)])
                .unwrap()
                .into_values()
                .next()
                .unwrap();
            let q = quote(&pool, &source)
                .with_registered_sources(registration.clone())
                .unwrap();
            let (r, run, q) = accept(&pool, q);
            let alias = q.span_workspace().plan().clone();
            let pq = q.span_workspace().retention_peak_bytes().unwrap() + 51;
            match phase {
                0 => {
                    drop(source_scope);
                    let before = account(&pool, &r);
                    let e = q.into_funded_text_span_workspace(&run, &r).unwrap_err();
                    assert!(matches!(e.cause(), WorkingMemoryError::ExecutionFenced));
                    assert_eq!(account(&pool, &r), before);
                    drop(e);
                }
                1 => {
                    crate::working_memory::funding::quarantine_source_after_next_attachment(
                        source_scope,
                    );
                    let e = q.into_funded_text_span_workspace(&run, &r).unwrap_err();
                    assert!(matches!(e.cause(), WorkingMemoryError::ExecutionFenced));
                    assert_eq!(account(&pool, &r).1, pq);
                    let (q, _) = e.into_parts();
                    assert!(q.span_workspace().plan().same_plan(&alias));
                    drop(q);
                    assert_eq!(
                        account(&pool, &r).1,
                        pq,
                        "failed promotion leaves aggregate on actual earlier owner"
                    );
                }
                _ => {
                    let (owner, witness) = q.into_funded_text_span_workspace(&run, &r).unwrap();
                    witness.unwrap().validate(&pool).unwrap();
                    let guard = owner.control_guard();
                    guard.validate_reservation(&r).unwrap();
                    drop(source_scope);
                    assert!(matches!(
                        guard.validate_reservation(&r),
                        Err(WorkingMemoryError::ExecutionFenced)
                    ));
                    let native = run.scope().unwrap();
                    assert!(matches!(
                        guard.validate_native_scope(&native),
                        Err(WorkingMemoryError::ExecutionFenced)
                    ));
                    native.certify().unwrap();
                    drop((owner, guard));
                }
            }
            drop((alias, r, run, sr, srun, registration, root));
            assert!(
                pool.used_bytes().unwrap() > 0,
                "quarantined source keeps its original envelope"
            );
        }
    }
}
#[test]
fn foreign_pool_closed_run_and_poisoned_usage_cannot_validate_or_release_live_controls() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let source = capture_source();
    let q = quote(&pool, &source);
    let (r, run, q) = accept(&pool, q);
    let (owner, _) = q.into_funded_text_span_workspace(&run, &r).unwrap();
    let guard = owner.control_guard();
    let foreign = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let froot = foreign.register_storage([(1u32, 64)]).unwrap();
    let fq = quote(&foreign, &source);
    let (fr, frun, fq) = accept(&foreign, fq);
    let fs = frun.scope().unwrap();
    assert!(matches!(
        guard.validate_native_scope(&fs),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    fs.certify().unwrap();
    drop((fq, fr, frun, froot));
    assert_eq!(foreign.used_bytes().unwrap(), 0);
    drop(run);
    assert!(matches!(
        guard.validate_reservation(&r),
        Err(WorkingMemoryError::ExecutionFenced)
    ));
    let before = account(&pool, &r);
    let result = catch_unwind(AssertUnwindSafe(|| {
        let _usage = pool.0.usage.lock().unwrap();
        panic!("poison while aggregate lives");
    }));
    assert!(result.is_err());
    assert!(matches!(
        guard.validate_reservation(&r),
        Err(WorkingMemoryError::Poisoned)
    ));
    {
        let usage = pool.0.usage.lock().unwrap_err().into_inner();
        let s = &usage.funding[&r.0.funding.unwrap()];
        assert_eq!((s.remaining, s.host_held, s.scopes), before);
    }
    let account_id = r.0.funding.unwrap();
    drop((owner, guard, r, root));
    let usage = pool.0.usage.lock().unwrap_err().into_inner();
    // Existing host-scope cleanup cannot certify through a poisoned Usage
    // lock. Its underlying scope conservatively quarantines the remaining
    // original envelope even after the actual P+Q payload owners have retired.
    let account = &usage.funding[&account_id];
    assert!(matches!(
        account.validate_registered_copy_origin(),
        Err(WorkingMemoryError::ExecutionFenced)
    ));
    assert_eq!((account.host_held, account.scopes), (0, 0));
    assert_eq!(account.remaining, before.0);
    assert_eq!(usage.reserved, before.0);
    assert_eq!(usage.registered, 0);
}

mod bounded_pins;

mod capture_publication;

mod sequence;

mod preparation;

mod prediction;

mod graph_metadata;
mod tracking;

mod prefill;

mod host_destinations;

mod native_storage;
