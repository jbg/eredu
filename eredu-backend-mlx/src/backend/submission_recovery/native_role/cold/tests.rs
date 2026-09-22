use super::*;
use safemlx::{Device, DeviceType, Dtype, OperationEvent, PrefillRootsRuntime, Stream};
use std::{cell::Cell, rc::Rc};

struct DropProbe(Rc<Cell<bool>>);
impl Drop for DropProbe {
    fn drop(&mut self) {
        self.0.set(true);
    }
}

fn wait_for_source_retirement(pool: &MemoryLedger) {
    crate::backend::submission_recovery::wait_for_retirement(|| {
        // Recovery can release native owners under its no-hooks runtime guard.
        // Their account-only Rust destructors run on the ordinary host afterward.
        safemlx::reclaim_allocation_owners();
        pool.fixture_host_charge() == Ok(0)
    });
}

fn capacity(runtime: &PreparedInputRuntime) -> NativeRoleCapacity {
    let layout = OperationEvent::cpu_affine_quantize_submission_layout(
        Dtype::Float32,
        Dtype::Float16,
        2,
        2,
        64,
        32,
        4,
    )
    .expect("qualified CPU affine layout");
    NativeRoleCapacity {
        graph: layout.graph_capacity(),
        records: layout.record_capacity(),
        backing: layout.physical_capacity(runtime).unwrap(),
    }
}

#[test]
fn admission_rejects_before_invocation_and_preserves_the_plan() {
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let dropped = Rc::new(Cell::new(false));
    let called = Cell::new(false);
    let plan = Plan::new(
        &runtime,
        capacity(&runtime),
        None,
        DropProbe(dropped.clone()),
        |_, _| {
            called.set(true);
            Ok(Ok::<_, ()>(()))
        },
    );
    let bytes = plan.required_bytes().unwrap();
    let pool = crate::memory_fixture::ledger(bytes - 1, 0).unwrap();
    let error = plan.submit(&pool).unwrap_err();
    assert!(matches!(error.accounting_failure(),
        Some(WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { requested_bytes: required_bytes, limit_bytes, existing_bytes, .. }))
        if *required_bytes == bytes && limit_bytes.checked_sub(*existing_bytes).unwrap() == bytes - 1));
    assert!(error.rejected_plan().is_some());
    assert!(error.constructor_failure().is_none());
    assert!(!called.get());
    assert!(!dropped.get());
    assert_eq!(pool.fixture_host_charge().unwrap(), 0);
    drop(error);
    assert!(dropped.get());
}

#[test]
fn exact_admission_runs_once_and_retires_invocation() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let _roots = PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let dropped = Rc::new(Cell::new(false));
    let called = Cell::new(0);
    let plan = Plan::new(
        &runtime,
        capacity(&runtime),
        None,
        DropProbe(dropped.clone()),
        |_, context| {
            called.set(called.get() + 1);
            assert!(context
                .observer()
                .same_scope(&OriginalScopeObserver::require_current().unwrap()));
            assert!(!dropped.get());
            Ok(Ok::<_, ()>(42))
        },
    );
    let pool = crate::memory_fixture::ledger(plan.required_bytes().unwrap(), 0).unwrap();
    assert_eq!(plan.submit(&pool).unwrap().finish().unwrap().unwrap(), 42);
    assert_eq!(called.get(), 1);
    wait_for_source_retirement(&pool);
    assert!(dropped.get());
    assert_eq!(pool.fixture_host_charge().unwrap(), 0);
}

#[test]
fn callback_failure_retains_source_account_until_error_and_recovery_retire() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let _roots = PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let dropped = Rc::new(Cell::new(false));
    let plan = Plan::new(
        &runtime,
        capacity(&runtime),
        None,
        DropProbe(dropped.clone()),
        |_, _| Err::<Result<(), ()>, _>(Error::PrefillScopeReentrant),
    );
    let bytes = plan.required_bytes().unwrap();
    let pool = crate::memory_fixture::ledger(bytes, 0).unwrap();
    let error = plan.submit(&pool).unwrap_err();
    assert!(error.rejected_plan().is_none());
    assert!(error.accounting_failure().is_none());
    assert!(error.constructor_failure().is_some());
    crate::backend::submission_recovery::wait_for_retirement(|| dropped.get());
    assert_eq!(pool.fixture_host_charge().unwrap(), bytes);
    drop(error);
    wait_for_source_retirement(&pool);
    assert_eq!(pool.fixture_host_charge().unwrap(), 0);
}

#[test]
fn cold_root_refuses_an_active_original_parent_before_callback() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let _roots = PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let called = Cell::new(false);
    let inner = Plan::new(&runtime, capacity(&runtime), None, (), |_, _| {
        called.set(true);
        Ok(Ok::<_, ()>(()))
    });
    let inner_bytes = inner.required_bytes().unwrap();
    let pool_cell = std::cell::OnceCell::<MemoryLedger>::new();
    let outer_bytes = Cell::new(0);
    let outer = Plan::new(&runtime, capacity(&runtime), None, (), |_, context| {
        let pool = pool_cell.get().unwrap();
        let error = inner.submit(pool).unwrap_err();
        assert!(!called.get());
        assert!(error.accounting_failure().is_none());
        assert!(error.constructor_failure().is_some());
        assert_eq!(
            pool.fixture_host_charge().unwrap(),
            outer_bytes.get() + inner_bytes
        );
        assert!(context
            .observer()
            .same_scope(&OriginalScopeObserver::require_current().unwrap()));
        drop(error);
        assert_eq!(pool.fixture_host_charge().unwrap(), outer_bytes.get());
        Ok(Ok::<_, ()>(()))
    });
    outer_bytes.set(outer.required_bytes().unwrap());
    pool_cell
        .set(crate::memory_fixture::ledger(outer_bytes.get() + inner_bytes, 0).unwrap())
        .unwrap();
    let pool = pool_cell.get().unwrap();
    outer.submit(pool).unwrap().finish().unwrap().unwrap();
    wait_for_source_retirement(pool);
    assert_eq!(pool.fixture_host_charge().unwrap(), 0);
}

#[test]
fn submitted_invocations_keep_independent_scopes_and_retirement() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let _roots = PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let first_dropped = Rc::new(Cell::new(false));
    let second_dropped = Rc::new(Cell::new(false));
    let make = |dropped| {
        Plan::new(
            &runtime,
            capacity(&runtime),
            None,
            DropProbe(dropped),
            |_, context| Ok(Ok::<_, ()>(context.observer().clone())),
        )
    };
    let first = make(first_dropped.clone());
    let second = make(second_dropped.clone());
    let first_bytes = first.required_bytes().unwrap();
    let second_bytes = second.required_bytes().unwrap();
    let pool = crate::memory_fixture::ledger(first_bytes + second_bytes, 0).unwrap();
    let first = first.submit(&pool).unwrap();
    assert!(OriginalScopeObserver::try_current().unwrap().is_none());
    let second = second.submit(&pool).unwrap();
    assert!(OriginalScopeObserver::try_current().unwrap().is_none());
    assert!(!first
        .result()
        .as_ref()
        .unwrap()
        .same_scope(second.result().as_ref().unwrap()));
    assert!(!first_dropped.get() && !second_dropped.get());
    assert_eq!(
        pool.fixture_host_charge().unwrap(),
        first_bytes + second_bytes
    );
    drop(first.finish().unwrap().unwrap());
    crate::backend::submission_recovery::wait_for_retirement(|| {
        safemlx::reclaim_allocation_owners();
        pool.fixture_host_charge() == Ok(second_bytes)
    });
    assert!(first_dropped.get());
    assert!(!second_dropped.get());
    // Abandoning a submitted invocation uses deferred recovery, even when the
    // caller never asks it for a result or explicit completion.
    drop(second);
    wait_for_source_retirement(&pool);
    assert!(second_dropped.get());
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;
