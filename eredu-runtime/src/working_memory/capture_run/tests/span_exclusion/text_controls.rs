//! Actual accepted quote, original bank and existing stamped-channel fixture.
//! This is not a native completion or SessionPrefill integration claim.
use super::*;

#[test]
fn original_capture_source_pin_uses_paid_controls_and_rejects_foreign_custody() {
    fn accepted(
        pool: &MemoryLedger,
        source: &SharedCapturePlan,
    ) -> (
        InferenceRequest,
        WorkingMemoryFundingRun,
        OwnedTextSpanWorkspace,
    ) {
        let q = quote(pool, plan(source).initialization_peak_bytes());
        let controls = PreparedTextControlWorkspace::prepare(
            source,
            geometry(),
            q.span_workspace().plan(),
            TextHostControlFacts::new(
                Some(MemoryLedger::capture_source_pin_control_bytes::<u32>().unwrap()),
                Some(17),
                Some(23),
            ),
        )
        .unwrap();
        let (r, run, q) = reserve_sealed(
            pool,
            q.with_span_workspace_and_text_controls(controls).unwrap(),
        );
        let (owner, _) = q
            .into_funded_text_span_workspace(&run, r.memory_reservation())
            .unwrap();
        (r, run, owner)
    }
    for foreign in [false, true] {
        let pool = capture_test_ledger(4_000_000, 0).unwrap();
        let other = capture_test_ledger(4_000_000, 0).unwrap();
        let source = source();
        let roots = pool.register_host_storage([(73u32, 8), (74, 12)]).unwrap();
        let (r, run, owner) = accepted(&pool, &source);
        let (other_r, other_run, other_owner) = accepted(&other, &source);
        let custody = if foreign {
            other_owner.control_guard()
        } else {
            owner.control_guard()
        }
        .metadata_custody();
        let mut bank = bank(&run, &r, &source);
        let mut native = run.scope().unwrap();
        let mut frame = bank
            .begin_step(CapturePhase::Prefill, 0)
            .unwrap()
            .prepare()
            .unwrap();
        let claim = frame.take_tensor(0).unwrap();
        let elements = claim.geometry().elements();
        let before = pool.snapshot().unwrap();
        let result = claim.prepare_with_original_source(
            &mut native,
            &custody,
            [Some((73u32, 8)), Some((74, 12))],
        );
        if foreign {
            assert!(matches!(
                result,
                Err(CaptureRunHostError::Memory(
                    WorkingMemoryError::IdentityMismatch
                ))
            ));
        } else {
            let mut transfer = result.unwrap();
            for index in 0..elements {
                transfer.push_f32(index as f32 + 0.5).unwrap();
            }
            let captured = transfer.finish().unwrap();
            assert_eq!(
                captured.observation().data(),
                &TensorObservationData::F32(
                    (0..elements).map(|index| index as f32 + 0.5).collect()
                )
            );
            drop(captured);
        }
        assert_eq!(
            pool.snapshot().unwrap(),
            before,
            "prepaid source controls do not reserve another host allowance"
        );
        drop(frame);
        native.certify().unwrap();
        drop((
            bank,
            custody,
            owner,
            r,
            run,
            roots,
            other_owner,
            other_r,
            other_run,
        ));
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
        assert_eq!(other.payload_used_bytes().unwrap(), 0);
    }
}

#[test]
fn text_activation_uses_matching_pq_custody_and_exact_original_headroom() {
    for short in [false, true] {
        let pool = capture_test_ledger(4_000_000, 0).unwrap();
        let source = source();
        let opening_root = pool.register_host_storage([(73u32, 8)]).unwrap();
        let original = quote(&pool, plan(&source).initialization_peak_bytes());
        let pins =
            PreparedPrefillStoragePinPlan::<u32>::prepare(original.span_workspace().plan(), |_| {
                Some(1)
            })
            .unwrap();
        let controls = PreparedTextControlWorkspace::prepare(
            &source,
            geometry(),
            original.span_workspace().plan(),
            TextHostControlFacts::new(Some(11), Some(17), Some(23)),
        )
        .unwrap()
        .with_prefill_storage_pins(pins)
        .unwrap();
        let (r, run, q) = reserve_sealed(
            &pool,
            original
                .with_span_workspace_and_text_controls(controls)
                .unwrap(),
        );
        let (mut owner, _) = q
            .into_funded_text_span_workspace(&run, r.memory_reservation())
            .unwrap();
        let mut pins = owner.take_prefill_storage_pins::<u32>().unwrap();
        let bank = bank(&run, &r, &source);
        let mut native = run.scope().unwrap();
        let c = chunk();
        let cx = context(&r, &c, 83);
        let (mut segment, reg) = bank
            .prefill_source_bootstrap()
            .unwrap()
            .begin_segment(&mut native, &cx)
            .unwrap();
        let receipt = owner.as_reserved_text_span_workspace();
        // The old private P-only dispatch cannot discard a text control guard.
        let before = ledger(&pool);
        assert!(matches!(
            segment.activate_reserved_span(&mut native, &receipt.span, &cx),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
        assert_eq!(ledger(&pool), before);
        // An accepted text receipt cannot activate without its exact group.
        assert!(matches!(
            segment.activate_reserved_text_span(&mut native, &receipt, &cx),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
        assert_eq!(ledger(&pool), before);
        let mut attempt = pins.begin(&cx, &native, &segment).unwrap();
        attempt.push_owned(73u32, 8).unwrap();
        let mut opening = Some(attempt.pin_registered(&native, &segment).unwrap());
        assert_eq!(opening.as_ref().unwrap().bytes(), Some(8));
        segment
            .install_opening_group(&mut native, &cx, &mut opening)
            .unwrap();
        assert!(opening.is_none());
        assert!(matches!(
            segment.retire_after_settled_boundary(&mut native),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
        assert_eq!(ledger(&pool), before);
        let n = receipt
            .span_bytes(&InferenceWorkspaceSpan::Prefill(c.clone()))
            .unwrap();
        assert_eq!(n, 16);
        let expected_host =
            owner.protected_host_bytes() + plan(&source).initialization_peak_bytes();
        {
            let usage = pool.0.usage.lock().unwrap();
            let account = &usage.funding[&r.memory_reservation().0.funding.unwrap()];
            assert_eq!(account.host_held - account.control_floor, expected_host);
        }
        let pressure = headroom(&pool, &r) - n + u64::from(short);
        let mut storage = native
            .adopt_capture_host_storage([(71u32, pressure)])
            .unwrap();
        let root = storage.remove(&71).unwrap();
        let alias = root.clone();
        let before = ledger(&pool);
        let result = segment.activate_reserved_text_span(&mut native, &receipt, &cx);
        if short {
            assert!(matches!(
                result,
                Err(WorkingMemoryError::DomainAllowanceExceeded {
                    required_bytes: 16,
                    available_bytes: 15,
                    ..
                })
            ));
            assert_eq!(ledger(&pool), before);
            drop(root);
            assert_eq!(headroom(&pool, &r), 15);
            assert!(
                segment
                    .activate_reserved_text_span(&mut native, &receipt, &cx)
                    .is_err()
            );
            drop(alias);
            segment
                .activate_reserved_text_span(&mut native, &receipt, &cx)
                .unwrap();
        } else {
            result.unwrap();
            assert_eq!(ledger(&pool), before);
            drop((root, alias));
        }
        owner
            .control_guard()
            .validate_native_scope(&native)
            .unwrap();
        fenced(run.scope());
        let payload = native.adopt_capture_host_storage([(72u32, n)]).unwrap();
        let settled = ticket(reg, &cx);
        let parcel = segment.take_settled_sources(&mut native, &settled).unwrap();
        run.scope().unwrap().certify().unwrap();
        // Terminal numerical payload precedes its matching accounting parcel.
        drop(payload);
        drop((parcel, settled, segment, receipt, storage));
        native.certify().unwrap();
        drop((bank, pins, owner, r, run, source, opening_root));
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
}

#[test]
fn joined_text_without_original_pin_layout_cannot_bypass_required_opening_group() {
    let pool = capture_test_ledger(4_000_000, 0).unwrap();
    let source = source();
    let original = quote(&pool, plan(&source).initialization_peak_bytes());
    let controls = PreparedTextControlWorkspace::prepare(
        &source,
        geometry(),
        original.span_workspace().plan(),
        TextHostControlFacts::new(Some(11), Some(17), Some(23)),
    )
    .unwrap();
    let (r, run, q) = reserve_sealed(
        &pool,
        original
            .with_span_workspace_and_text_controls(controls)
            .unwrap(),
    );
    let (owner, _) = q
        .into_funded_text_span_workspace(&run, r.memory_reservation())
        .unwrap();
    let bank = bank(&run, &r, &source);
    let mut native = run.scope().unwrap();
    let c = chunk();
    let cx = context(&r, &c, 83);
    let (segment, reg) = bank
        .prefill_source_bootstrap()
        .unwrap()
        .begin_segment(&mut native, &cx)
        .unwrap();
    let receipt = owner.as_reserved_text_span_workspace();
    let before = ledger(&pool);
    for result in [
        segment.activate_reserved_span(&mut native, &receipt.span, &cx),
        segment.activate_reserved_text_span(&mut native, &receipt, &cx),
    ] {
        assert!(matches!(result, Err(WorkingMemoryError::IdentityMismatch)));
    }
    assert_eq!(ledger(&pool), before);
    let t = ticket(reg, &cx);
    drop(segment.take_settled_sources(&mut native, &t).unwrap());
    native.certify().unwrap();
    drop(receipt);
    drop((t, segment, bank, owner, r, run, source));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn opening_physical_origins_are_rechecked_in_same_text_activation_transaction() {
    for bytes in [0, 8] {
        let pool = capture_test_ledger(4_000_000, 0).unwrap();
        let source = source();
        let (origin_r, origin_run) = fresh(&pool, 8);
        let origin = origin_run.scope().unwrap();
        let roots = origin.adopt_capture_host_storage([(71u32, bytes)]).unwrap();
        let original = quote(&pool, plan(&source).initialization_peak_bytes());
        let pins =
            PreparedPrefillStoragePinPlan::<u32>::prepare(original.span_workspace().plan(), |_| {
                Some(1)
            })
            .unwrap();
        let controls = PreparedTextControlWorkspace::prepare(
            &source,
            geometry(),
            original.span_workspace().plan(),
            TextHostControlFacts::new(Some(11), Some(17), Some(23)),
        )
        .unwrap()
        .with_prefill_storage_pins(pins)
        .unwrap();
        let (r, run, q) = reserve_sealed(
            &pool,
            original
                .with_span_workspace_and_text_controls(controls)
                .unwrap(),
        );
        let (mut owner, _) = q
            .into_funded_text_span_workspace(&run, r.memory_reservation())
            .unwrap();
        let mut pins = owner.take_prefill_storage_pins::<u32>().unwrap();
        let bank = bank(&run, &r, &source);
        let mut native = run.scope().unwrap();
        let c = chunk();
        let cx = context(&r, &c, 83);
        let (segment, reg) = bank
            .prefill_source_bootstrap()
            .unwrap()
            .begin_segment(&mut native, &cx)
            .unwrap();
        let mut attempt = pins.begin(&cx, &native, &segment).unwrap();
        attempt.push_owned(71u32, bytes).unwrap();
        let mut opening = Some(attempt.pin_registered(&native, &segment).unwrap());
        segment
            .install_opening_group(&mut native, &cx, &mut opening)
            .unwrap();
        let receipt = owner.as_reserved_text_span_workspace();
        drop(origin); // A real registered origin becomes quarantined after pin/install.
        let before = ledger(&pool);
        assert!(matches!(
            segment.activate_reserved_text_span(&mut native, &receipt, &cx),
            Err(WorkingMemoryError::ExecutionFenced)
        ));
        assert_eq!(ledger(&pool), before);
        // Failed activation did not install a marker or debit the original span.
        run.scope().unwrap().certify().unwrap();
        // The failed source inventory stays in this original scope's quarantine.
        drop(native);
        drop(receipt);
        drop((
            reg, segment, bank, pins, owner, r, run, roots, origin_r, origin_run, source,
        ));
        assert!(pool.payload_used_bytes().unwrap() > 0);
    }
}

mod native_coverage;

#[test]
fn original_text_metadata_keeps_fixed_slot_and_raw_custody_until_final_alias() {
    use crate::input::{
        PreparedInputCacheIdentity, SharedPreparedInputCacheIdentity, TextInputIdentityPlan,
    };
    use crate::working_memory::qualified_storage;
    use std::{
        mem::size_of,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
    };
    if !qualified_storage::qualified() {
        assert_ne!(
            std::env::var_os("EREDU_REQUIRE_QUALIFIED_PROMPT_INPUT"),
            Some("1".into())
        );
        return;
    }
    struct Probe {
        pool: MemoryLedger,
        retired: Arc<AtomicBool>,
    }
    impl Drop for Probe {
        fn drop(&mut self) {
            assert!(
                self.pool.0.usage.try_lock().is_ok(),
                "boxed provider retires outside Usage"
            );
            assert!(
                self.pool.payload_used_bytes().unwrap() > 0,
                "raw custody outlives Box payload and storage"
            );
            self.retired.store(true, Ordering::SeqCst);
        }
    }
    let plan_identity = TextInputIdentityPlan::new(1, 3).unwrap();
    let (inner, key, control) = SharedPreparedInputCacheIdentity::text_control_request().unwrap();
    let bytes = plan_identity.peak_bytes() + qualified_storage::shared_layout_bytes(inner).unwrap()
        - size_of::<PreparedInputCacheIdentity>() as u64
        + qualified_storage::shared_layout_bytes(key).unwrap()
        + crate::working_memory::fixed_baseline::pal_mutex_bytes().unwrap()
        + control as u64
        + size_of::<Probe>() as u64;
    let pool = capture_test_ledger(4_000_000, 0).unwrap();
    let source = source();
    let original = quote(&pool, plan(&source).initialization_peak_bytes());
    let controls = PreparedTextControlWorkspace::prepare(
        &source,
        geometry(),
        original.span_workspace().plan(),
        TextHostControlFacts::new(Some(bytes), Some(0), Some(0)),
    )
    .unwrap();
    let (r, run, q) = reserve_sealed(
        &pool,
        original
            .with_span_workspace_and_text_controls(controls)
            .unwrap(),
    );
    let (owner, _) = q
        .into_funded_text_span_workspace(&run, r.memory_reservation())
        .unwrap();
    let identity = plan_identity
        .bind(&[7, 13, 29])
        .unwrap()
        .construct_original(owner.control_guard().metadata_custody())
        .unwrap();
    let alias = identity.clone();
    let domain = pool.shared_storage_accounting_id();
    let retired = Arc::new(AtomicBool::new(false));
    assert!(
        identity
            .try_attach(domain, || Ok::<Box<dyn Send + Sync>, WorkingMemoryError>(
                Box::new(Probe {
                    pool: pool.clone(),
                    retired: retired.clone()
                })
            ))
            .unwrap()
    );
    assert!(
        !alias
            .try_attach::<WorkingMemoryError>(domain, || panic!(
                "same-ledger alias must reuse exact slot"
            ))
            .unwrap()
    );
    assert!(
        alias
            .try_attach::<WorkingMemoryError>(
                &eredu_core::SharedStorageAccountingId::default(),
                || panic!("foreign ledger must refuse before allocation/provider")
            )
            .is_err()
    );
    let before = pool.payload_used_bytes().unwrap();
    drop((identity, owner, r, run, source));
    assert!(!retired.load(Ordering::SeqCst));
    assert!(pool.payload_used_bytes().unwrap() > 0);
    assert!(pool.payload_used_bytes().unwrap() <= before);
    drop(alias);
    assert!(retired.load(Ordering::SeqCst));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
