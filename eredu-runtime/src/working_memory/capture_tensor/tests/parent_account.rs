//! Parent-account behavior uses the same actual admitted tensor fixtures.
use super::*;
use crate::working_memory::capture_run::tests::{
    CaptureFundingFixture, capture_test_ledger, fresh, ledger,
};

use crate::working_memory::{
    InferenceExecutionIdentity, WorkingMemoryFundingRun, WorkingMemoryReservation,
};
use eredu_core::{cache::LayerCachePolicy, *};
use std::num::NonZeroU8;

#[test]
fn parent_exact_and_one_short_use_only_the_original_reservation() {
    let source = ordinary();
    let required = plan(&source).initialization_peak_bytes();
    for bytes in [required - 1, required] {
        let pool = capture_test_ledger(2 * required, 0).unwrap();
        let (reservation, run) = fresh(&pool, bytes);
        let before = ledger(&pool);
        let allocations = ALLOCATIONS.get();
        let result = run.prepare_capture_tensor(&reservation, plan(&source));
        if bytes < required {
            assert!(matches!(
                result,
                Err(CaptureTensorConstructionError::Memory(
                    WorkingMemoryError::DomainAllowanceExceeded { required_bytes, available_bytes, .. }
                )) if required_bytes == required && available_bytes == bytes
            ));
            assert_eq!(ledger(&pool), before);
            assert_eq!(ALLOCATIONS.get(), allocations);
        } else {
            let value = fill(result.unwrap());
            assert_eq!(value.shape(), &[3, 4]);
            assert_eq!(ALLOCATIONS.get(), allocations + 1);
            let after = ledger(&pool);
            assert_eq!((after.0, after.3, after.4), (before.0, 1, 1));
            assert_eq!((after.1, after.2), (required, 1));
            drop(value);
            assert_eq!(ledger(&pool), before);
        }
        drop((run, reservation));
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
}

#[test]
fn simultaneous_host_destinations_and_native_storage_share_exact_parent_pressure() {
    let source = ordinary();
    let required = plan(&source).initialization_peak_bytes();
    let native_bytes = 73;
    let total = 2 * required + native_bytes;
    let pool = capture_test_ledger(total, 0).unwrap();
    let (reservation, run) = fresh(&pool, total);
    let native = run.scope().unwrap();
    let first = fill(
        run.prepare_capture_tensor(&reservation, plan(&source))
            .unwrap(),
    );
    let second = fill(
        run.prepare_capture_tensor(&reservation, plan(&source))
            .unwrap(),
    );
    assert!(!first.same_storage(&second));
    assert_eq!(ledger(&pool), (total, 2 * required, 3, 1, 1));
    let before = ledger(&pool);
    assert!(matches!(
        native.adopt_capture_host_storage([(41u32, native_bytes + 1)]),
        Err(WorkingMemoryError::DomainAllowanceExceeded { .. })
    ));
    assert_eq!(ledger(&pool), before);
    let roots = native
        .adopt_capture_host_storage([(41u32, native_bytes)])
        .unwrap();
    assert_eq!(ledger(&pool), before);
    let alias = first.clone();
    drop((first, second, run, reservation));
    native.certify().unwrap();
    assert_eq!(pool.payload_used_bytes().unwrap(), required + native_bytes);
    // Only the surviving destination hold and independently published native
    // root remain; the retired sibling's unused host headroom is released.
    assert_eq!(ledger(&pool).1, required);
    assert_eq!(alias.shape(), &[3, 4]);
    drop(alias);
    assert_eq!(pool.payload_used_bytes().unwrap(), native_bytes);
    drop(roots);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn completed_alias_keeps_values_and_charge_after_all_parent_metadata_retires() {
    let source = ordinary();
    let required = plan(&source).initialization_peak_bytes();
    let pool = capture_test_ledger(required, 0).unwrap();
    let (reservation, run) = fresh(&pool, required);
    let value = fill(
        run.prepare_capture_tensor(&reservation, plan(&source))
            .unwrap(),
    );
    let alias = value.clone();
    let pointer = match alias.data() {
        TensorObservationData::F32(values) => values.as_ptr(),
        _ => unreachable!(),
    };
    drop((source, run, reservation, value));
    assert_eq!(pool.payload_used_bytes().unwrap(), required);
    assert_eq!(alias.shape(), &[3, 4]);
    let TensorObservationData::F32(values) = alias.data() else {
        unreachable!()
    };
    assert_eq!(values.as_ptr(), pointer);
    assert_eq!(
        values,
        &(0..12).map(|n| n as f32 + 0.25).collect::<Vec<_>>()
    );
    drop(alias);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    assert!(pool.acquire_unquoted().is_ok());
}

#[test]
fn wrong_reservation_domain_and_quarantined_parent_reject_before_allocation() {
    let source = ordinary();
    let required = plan(&source).initialization_peak_bytes();
    let pool = capture_test_ledger(4 * required, 0).unwrap();
    let foreign = capture_test_ledger(4 * required, 0).unwrap();
    let (reservation, run) = fresh(&pool, required);
    let (other, other_run) = fresh(&pool, required);
    let (foreign_reservation, foreign_run) = fresh(&foreign, required);
    for wrong in [&other, &foreign_reservation] {
        let before = ledger(&pool);
        let allocations = ALLOCATIONS.get();
        assert!(matches!(
            run.prepare_capture_tensor(wrong, plan(&source)),
            Err(CaptureTensorConstructionError::Memory(
                WorkingMemoryError::IdentityMismatch
            ))
        ));
        assert_eq!(ledger(&pool), before);
        assert_eq!(ALLOCATIONS.get(), allocations);
    }
    drop(run.scope().unwrap());
    let before = ledger(&pool);
    let allocations = ALLOCATIONS.get();
    assert!(matches!(
        run.prepare_capture_tensor(&reservation, plan(&source)),
        Err(CaptureTensorConstructionError::Memory(
            WorkingMemoryError::ExecutionFenced
        ))
    ));
    assert_eq!(ledger(&pool), before);
    assert_eq!(ALLOCATIONS.get(), allocations);
    drop((other, other_run, foreign_reservation, foreign_run));
    assert_eq!(foreign.payload_used_bytes().unwrap(), 0);
}

#[test]
fn partial_fill_and_finish_reject_closed_or_quarantined_parent_without_payload_escape() {
    for failure in 0..3 {
        let source = ordinary();
        let required = plan(&source).initialization_peak_bytes();
        let pool = capture_test_ledger(required, 0).unwrap();
        let (reservation, run) = fresh(&pool, required);
        let mut builder = run
            .prepare_capture_tensor(&reservation, plan(&source))
            .unwrap();
        builder.push_f32(7.5).unwrap();
        let mut run = Some(run);
        let mut reservation = Some(reservation);
        match failure {
            0 => drop(run.take()),
            1 => drop(reservation.take()),
            _ => drop(run.as_ref().unwrap().scope().unwrap()),
        }
        assert!(matches!(
            builder.push_f32(99.0),
            Err(WorkingMemoryError::ExecutionFenced)
        ));
        assert_eq!(builder.initialized_count(), 1);
        let error = builder.finish().unwrap_err();
        assert!(
            std::error::Error::source(&error)
                .unwrap()
                .downcast_ref::<WorkingMemoryError>()
                .is_some_and(|error| matches!(error, WorkingMemoryError::ExecutionFenced))
        );
        let error = error.into_builder().unwrap_err();
        assert_eq!(ledger(&pool).1, required);
        drop(error);
        assert_eq!(ledger(&pool).1, 0);
        drop((run, reservation));
        assert_eq!(
            pool.payload_used_bytes().unwrap(),
            if failure == 2 { required } else { 0 }
        );
    }
}

#[test]
fn health_is_rechecked_between_hold_commit_and_actual_buffer_allocation() {
    for close in [true, false] {
        let source = ordinary();
        let plan = plan(&source);
        let required = plan.initialization_peak_bytes();
        let pool = capture_test_ledger(required, 0).unwrap();
        let (reservation, run) = fresh(&pool, required);
        let custody = run.hold_capture_tensor(&reservation, &plan).unwrap();
        assert_eq!(ledger(&pool).1, required);
        if !close {
            drop(run.scope().unwrap());
        }
        drop(run);
        let allocations = ALLOCATIONS.get();
        assert!(matches!(
            super::super::allocate(plan, custody),
            Err(CaptureTensorConstructionError::Memory(
                WorkingMemoryError::ExecutionFenced
            ))
        ));
        assert_eq!(ALLOCATIONS.get(), allocations);
        assert_eq!(ledger(&pool).1, 0);
        drop(reservation);
        assert_eq!(
            pool.payload_used_bytes().unwrap(),
            if close { 0 } else { required }
        );
    }
}

#[test]
fn partial_recovery_and_competing_destinations_preserve_fixed_holds() {
    let source = ordinary();
    let required = plan(&source).initialization_peak_bytes();
    let total = 2 * required - 1;
    let pool = capture_test_ledger(total, 0).unwrap();
    let (reservation, run) = fresh(&pool, total);
    let mut builder = run
        .prepare_capture_tensor(&reservation, plan(&source))
        .unwrap();
    builder.push_f32(3.5).unwrap();
    let mut builder = builder.finish().unwrap_err().into_builder().unwrap();
    let before = ledger(&pool);
    let allocations = ALLOCATIONS.get();
    assert!(matches!(
        run.prepare_capture_tensor(&reservation, plan(&source)),
        Err(CaptureTensorConstructionError::Memory(
            WorkingMemoryError::DomainAllowanceExceeded { .. }
        ))
    ));
    assert_eq!(ledger(&pool), before);
    assert_eq!(ALLOCATIONS.get(), allocations);
    while builder.initialized_count() < builder.len() {
        builder.push_f32(-2.0).unwrap();
    }
    assert!(matches!(
        builder.push_f32(99.0),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let value = builder.finish().unwrap();
    let TensorObservationData::F32(values) = value.data() else {
        unreachable!()
    };
    assert_eq!(values[0], 3.5);
    assert!(values[1..].iter().all(|value| *value == -2.0));
    drop(value);
    let retry = run
        .prepare_capture_tensor(&reservation, plan(&source))
        .unwrap();
    drop((retry, run, reservation));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn host_unwind_never_certifies_or_discards_the_independent_native_scope() {
    for partial in [false, true] {
        let source = ordinary();
        let required = plan(&source).initialization_peak_bytes();
        let total = required + 53;
        let pool = capture_test_ledger(total, 0).unwrap();
        let (reservation, run) = fresh(&pool, total);
        let native = run.scope().unwrap();
        let result = catch_unwind(AssertUnwindSafe(|| {
            if partial {
                let mut builder = run
                    .prepare_capture_tensor(&reservation, plan(&source))
                    .unwrap();
                builder.push_f32(2.25).unwrap();
                panic!("injected partial capture failure");
            } else {
                PANIC_BEFORE_BUFFERS.set(true);
                let _ = run.prepare_capture_tensor(&reservation, plan(&source));
            }
        }));
        assert!(result.is_err());
        assert_eq!(ledger(&pool), (total, 0, 1, 1, 1));
        // Cleanup did not fence the parent or retire the independent native scope.
        run.scope().unwrap().certify().unwrap();
        drop((run, reservation));
        assert_eq!(pool.payload_used_bytes().unwrap(), total);
        drop(native);
        assert_eq!(pool.payload_used_bytes().unwrap(), total);
        assert!(matches!(
            pool.acquire_unquoted(),
            Err(WorkingMemoryError::ReservedWorkActive)
        ));
    }
}

mod transfer;

mod shared_retirement;
