use super::*;
use crate::{
    ops::indexing::TryIndexOp, utils::allocation_test::measure, Device, DeviceType, Stream,
};
use std::{
    cell::Cell,
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc, Arc,
    },
    time::Duration,
};
thread_local! { static HOOKS: Cell<usize> = const { Cell::new(0) }; }
fn hook() {
    HOOKS.with(|n| n.set(n.get() + 1));
}
struct Hook;
impl Hook {
    fn new() -> Self {
        runtime_lock::register_housekeeping_hook(hook);
        HOOKS.with(|n| n.set(0));
        Self
    }
}
impl Drop for Hook {
    fn drop(&mut self) {
        runtime_lock::unregister_housekeeping_hook(hook);
    }
}
struct Retired(Arc<AtomicUsize>);
impl Drop for Retired {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
fn stream() -> Stream {
    Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
}

#[test]
fn borrowed_descriptor_loan_and_exact_fill_allocate_nothing_and_share_ordinary_backing_semantics() {
    let stream = stream();
    let source = Array::from_slice(&[2_i32, -3, 5, 7, 11, -13], &[2, 3]);
    let expected = source.try_allocation_info().unwrap().unwrap();
    let tail = source.try_index_device((.., 2..), &stream).unwrap();
    tail.evaluated().unwrap();
    let lazy = source.square(&stream).unwrap();
    let mut runtime = crate::RuntimeCallDeadline::new(Duration::from_secs(10))
        .unwrap()
        .enter()
        .unwrap();
    let _hook = Hook::new();
    let mut target = [-1; 2];
    let ((facts, borrowed, fill, short, unknown), allocations) = measure(|| {
        let loan = runtime.descriptor(&tail).unwrap();
        let facts = loan.facts();
        assert_eq!(loan.row_contiguous(), Some(false));
        let borrowed = [loan.shape()[0], loan.shape()[1]];
        let fill = loan.fill_shape(&mut target);
        let short = loan.fill_shape(&mut []);
        let lazy = runtime.descriptor(&lazy).unwrap();
        assert_eq!(lazy.row_contiguous(), None);
        let unknown = lazy.facts().allocation();
        drop(lazy);
        assert_eq!(
            runtime.descriptor(&source).unwrap().row_contiguous(),
            Some(true)
        );
        (facts, borrowed, fill, short, unknown)
    });
    assert_eq!(allocations, 0);
    assert_eq!(facts.dtype(), Dtype::Int32);
    assert_eq!(facts.rank(), 2);
    assert_eq!(facts.logical_bytes(), 8);
    assert_eq!(facts.allocation(), Some(expected));
    assert!(expected.bytes() > facts.logical_bytes());
    assert_eq!(borrowed, [2, 1]);
    assert_eq!(target, [2, 1]);
    assert_eq!(fill, Ok(()));
    assert_eq!(short, Err(ArrayDescriptorError::DestinationLength));
    assert_eq!(unknown, None);
    assert_eq!(HOOKS.with(Cell::get), 0);
    // Positive instrumentation control: the preserved ordinary snapshot owns Vec.
    let (snapshot, allocations) = measure(|| tail.try_metadata_snapshot().unwrap());
    assert!(allocations > 0);
    assert_eq!(snapshot.allocation(), facts.allocation());
}

#[test]
fn fixed_descriptor_busy_refusal_has_no_allocation_or_housekeeping_and_retains_no_new_owner() {
    let source = Array::from_slice(&[17_i32, -19], &[2]);
    let (ready_tx, ready_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        runtime_lock::try_retire(|| {
            ready_tx.send(()).unwrap();
            release_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        })
        .unwrap();
    });
    ready_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    let _hook = Hook::new();
    let (error, allocations) = measure(|| source.try_descriptor().unwrap_err());
    release_tx.send(()).unwrap();
    worker.join().unwrap();
    assert_eq!(error, ArrayDescriptorError::RuntimeBusy);
    assert_eq!(allocations, 0);
    assert_eq!(HOOKS.with(Cell::get), 0);
    let loan = source.try_descriptor().unwrap();
    assert_eq!(loan.shape(), [2]);
}

#[test]
fn borrowed_descriptor_loan_preserves_source_and_does_not_reclaim_unrelated_queued_custody() {
    let retired = Arc::new(AtomicUsize::new(0));
    let source = Array::from_slice(&[23_i32, 29], &[2]);
    source
        .retain_allocation_owner(Retired(retired.clone()))
        .unwrap();
    let mut runtime = crate::RuntimeCallDeadline::new(Duration::from_secs(10))
        .unwrap()
        .enter()
        .unwrap();
    let queued = Array::from_slice(&[31_i32, 37], &[2]);
    queued
        .retain_allocation_owner(Retired(retired.clone()))
        .unwrap();
    runtime_lock::try_retire(|| drop(queued)).unwrap();
    let _hook = Hook::new();
    let loan = runtime.descriptor(&source).unwrap();
    let mut shape = [0];
    loan.fill_shape(&mut shape).unwrap();
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    assert_eq!(HOOKS.with(Cell::get), 0);
    assert_eq!(shape, [2]);
    drop(loan);
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    drop(_hook);
    drop(runtime);
    crate::memory::clear_cache().unwrap();
    crate::reclaim_allocation_owners();
    assert_eq!(retired.load(Ordering::SeqCst), 1);
    drop(source);
    crate::memory::clear_cache().unwrap();
    crate::reclaim_allocation_owners();
    assert_eq!(retired.load(Ordering::SeqCst), 2);
}

#[test]
fn fixed_descriptor_scalar_empty_and_wrong_destination_keep_exact_shape_without_rank_caps() {
    let scalar = Array::from_int(41);
    let empty = Array::from_slice::<i32>(&[], &[2, 0, 3]);
    let rank = 37;
    let dimensions = vec![1; rank];
    let wide = Array::from_slice(&[43_i32], &dimensions);
    let mut destination = vec![-1; rank];
    let (result, allocations) = measure(|| {
        let scalar = scalar.try_descriptor().unwrap();
        assert_eq!(scalar.facts().rank(), 0);
        scalar.fill_shape(&mut []).unwrap();
        let empty = empty.try_descriptor().unwrap();
        assert_eq!(empty.shape(), [2, 0, 3]);
        assert_eq!(empty.facts().logical_bytes(), 0);
        let wide = wide.try_descriptor().unwrap();
        let result = wide.fill_shape(&mut destination);
        let mut wrong = [97, 101];
        assert_eq!(
            wide.fill_shape(&mut wrong),
            Err(ArrayDescriptorError::DestinationLength)
        );
        assert_eq!(wrong, [97, 101]);
        result
    });
    assert_eq!(allocations, 0);
    result.unwrap();
    assert_eq!(destination, dimensions);
    assert!(
        Array::descriptor_control_bytes().unwrap()
            >= std::mem::size_of::<ArrayDescriptorLoan<'static>>()
    );
}

#[test]
fn fixed_descriptor_fill_rechecks_changed_lazy_state_before_touching_destination() {
    let stream = stream();
    let source = Array::from_slice(&[47_i32, -53], &[2]);
    let lazy = source.square(&stream).unwrap();
    let loan = lazy.try_descriptor().unwrap();
    assert_eq!(loan.facts().allocation(), None);
    // Same-thread reentrant work is an explicit different API. A metadata loan
    // does not pretend to freeze native completion or turn unknown into known.
    lazy.evaluated().unwrap();
    let mut destination = [107];
    let (result, allocations) = measure(|| loan.fill_shape(&mut destination));
    assert_eq!(result, Err(ArrayDescriptorError::SourceChanged));
    assert_eq!(destination, [107]);
    assert_eq!(allocations, 0);
    assert_eq!(loan.shape(), [2]);
    drop(loan);
    assert!(lazy
        .try_descriptor()
        .unwrap()
        .facts()
        .allocation()
        .is_some());
}

#[test]
fn fixed_descriptor_runtime_loan_stays_held_between_count_and_fill() {
    let source = Array::from_slice(&[59_i32, 61], &[2]);
    let alias = source.try_clone_for_inspection().unwrap();
    let loan = source.try_descriptor().unwrap();
    let rank = loan.facts().rank();
    let worker = std::thread::spawn(move || {
        let busy = matches!(
            alias.try_descriptor(),
            Err(ArrayDescriptorError::RuntimeBusy)
        );
        busy
    });
    assert!(worker.join().unwrap());
    let mut destination = vec![0; rank];
    loan.fill_shape(&mut destination).unwrap();
    assert_eq!(destination, [2]);
}

#[test]
fn borrowed_descriptor_drop_keeps_the_actual_outer_runtime_guard_until_its_own_retirement() {
    let source = Array::from_slice(&[67_i32, 71], &[2]);
    let mut runtime = crate::RuntimeCallDeadline::new(Duration::from_secs(10))
        .unwrap()
        .enter()
        .unwrap();
    let _hook = Hook::new();
    let mut shape = [0];
    let ((), allocations) = measure(|| {
        let loan = runtime.descriptor(&source).unwrap();
        loan.fill_shape(&mut shape).unwrap();
        drop(loan);
    });
    assert_eq!(allocations, 0);
    assert_eq!(shape, [2]);
    assert_eq!(HOOKS.with(Cell::get), 0);
    let blocked = std::thread::spawn(|| runtime_lock::try_enter_for_recovery().is_none())
        .join()
        .unwrap();
    assert!(blocked);
    drop(_hook);
    // The outer owner is deliberately outside the measured fixed borrowed body.
    drop(runtime);
    let available = std::thread::spawn(|| runtime_lock::try_enter_for_recovery().is_some())
        .join()
        .unwrap();
    assert!(available);
}

#[test]
fn owning_descriptor_read_fill_and_final_release_allocate_nothing_without_hooks() {
    let source = Array::from_slice(&[109_i32, -113, 127, 131], &[2, 2]);
    let ordinary = source.try_metadata_snapshot().unwrap();
    let _hook = Hook::new();
    let mut shape = [0; 2];
    let ((facts, wrong), allocations) = measure(|| {
        let loan = source.try_descriptor().unwrap();
        let facts = loan.facts();
        loan.fill_shape(&mut shape).unwrap();
        let wrong = loan.fill_shape(&mut []);
        drop(loan);
        (facts, wrong)
    });
    assert_eq!(allocations, 0);
    assert_eq!(HOOKS.with(Cell::get), 0);
    assert_eq!(shape, [2, 2]);
    assert_eq!(facts.dtype(), ordinary.dtype());
    assert_eq!(facts.logical_bytes(), ordinary.nbytes());
    assert_eq!(facts.allocation(), ordinary.allocation());
    assert_eq!(wrong, Err(ArrayDescriptorError::DestinationLength));
    // Successful foreign entry proves the measured owner actually unlocked.
    assert!(
        std::thread::spawn(|| runtime_lock::try_enter_for_recovery().is_some())
            .join()
            .unwrap()
    );
}
