use super::*;
use eredu_core::SharedStorageDomain;
use std::{
    cell::Cell,
    convert::Infallible,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Allocation {
    len: usize,
    capacity: usize,
    vector: usize,
    boxed: Option<usize>,
}
thread_local! {
    static ALLOCATION: Cell<Option<Allocation>> = const { Cell::new(None) };
    static ZST_DROPS: Cell<usize> = const { Cell::new(0) };
}

pub(super) fn record_initialization(len: usize, capacity: usize, vector: usize) {
    ALLOCATION.set(Some(Allocation {
        len,
        capacity,
        vector,
        boxed: None,
    }));
}
pub(super) fn record_boxed(boxed: usize) {
    let mut record = ALLOCATION.get().expect("dense initialization was recorded");
    record.boxed = Some(boxed);
    ALLOCATION.set(Some(record));
}
fn witness() -> Option<Allocation> {
    ALLOCATION.get()
}
fn reset_witness() {
    ALLOCATION.set(None);
}

// No Clone or Default; nested drop counters are test observations only and are
// not included in the primitive's inline-payload bound.
struct Value {
    number: u32,
    state: Cell<u8>,
    drops: Arc<AtomicUsize>,
}
impl Drop for Value {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}
fn value(number: u32, drops: &Arc<AtomicUsize>) -> Value {
    Value {
        number,
        state: Cell::new(number as u8),
        drops: drops.clone(),
    }
}

#[test]
fn actual_source_and_dense_destination_use_one_buffer_without_clone_or_default() {
    let source_drops = Arc::new(AtomicUsize::new(0));
    let source = HostSlotTable::new(Box::new([
        value(3, &source_drops),
        value(7, &source_drops),
        value(11, &source_drops),
    ]));
    reset_witness();
    let optional = source.prepare_copy_slots().unwrap();
    let dense: DenseHostSlotInitialization<'_, Value, Value> =
        optional.for_dense_destination().unwrap();
    assert_eq!(
        witness(),
        None,
        "preparation does not allocate a destination"
    );
    assert_eq!(dense.len(), 3);
    assert!(!dense.is_empty());
    assert!(dense.source_metadata().same_storage(source.metadata()));
    assert!(std::ptr::eq(
        dense.source_at(1).unwrap(),
        &source.slots()[1]
    ));
    assert!(dense.source_at(3).is_none());
    source.slots()[1].state.set(23);
    assert_eq!(dense.source_at(1).unwrap().state.get(), 23);
    let slot = size_of::<Value>() as u64;
    assert_eq!(
        (dense.retained_bytes(), dense.initialization_peak_bytes()),
        (3 * slot, 5 * slot)
    );
    let numbers = [
        dense.source_at(0).unwrap().number,
        dense.source_at(1).unwrap().number,
        dense.source_at(2).unwrap().number,
    ];
    let destination_drops = Arc::new(AtomicUsize::new(0));
    let mut builder = dense.initialize();
    let allocation = witness().unwrap();
    assert_eq!(
        (allocation.len, allocation.capacity, allocation.boxed),
        (3, 3, None)
    );
    assert_eq!((builder.len(), builder.initialized_count()), (3, 0));
    for (index, number) in numbers.into_iter().enumerate() {
        builder.push(value(number, &destination_drops)).unwrap();
        assert_eq!(builder.initialized_count(), index + 1);
        assert_eq!(builder.slots.capacity(), 3);
        assert_eq!(builder.slots.as_ptr() as usize, allocation.vector);
    }
    let complete = builder.finish().unwrap();
    assert_eq!(witness().unwrap().boxed, Some(allocation.vector));
    assert_eq!(complete.slots.slots().as_ptr() as usize, allocation.vector);
    assert_eq!(complete.metadata().capacity_bytes(), Some(3 * slot));
    assert!(!complete.metadata().same_storage(source.metadata()));
    assert_eq!(
        complete.iter().map(|item| item.number).collect::<Vec<_>>(),
        numbers
    );
    complete.get(1).unwrap().state.set(29);
    assert_eq!(source.slots()[1].state.get(), 23);
    assert_eq!(destination_drops.load(Ordering::SeqCst), 0);
    drop(source);
    assert_eq!(source_drops.load(Ordering::SeqCst), 3);
    assert_eq!(complete.get(2).unwrap().number, 11);
    drop(complete);
    assert_eq!(destination_drops.load(Ordering::SeqCst), 3);
}

#[test]
fn partial_finish_retains_original_capacity_and_resumes_without_reallocation() {
    let source = HostSlotTable::new(Box::new([3_u8, 5, 7]));
    let drops = Arc::new(AtomicUsize::new(0));
    let mut builder = source
        .prepare_copy_slots()
        .unwrap()
        .for_dense_destination::<Value>()
        .unwrap()
        .initialize();
    let address = witness().unwrap().vector;
    builder.push(value(17, &drops)).unwrap();
    let incomplete = builder.finish().unwrap_err();
    assert_eq!((incomplete.expected(), incomplete.initialized()), (3, 1));
    assert_eq!(incomplete.builder.slots.capacity(), 3);
    assert_eq!(incomplete.builder.slots.as_ptr() as usize, address);
    assert_eq!(witness().unwrap().boxed, None);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    let mut builder = incomplete.into_builder();
    builder.push(value(19, &drops)).unwrap();
    builder.push(value(23, &drops)).unwrap();
    let complete = builder.finish().unwrap();
    assert_eq!(complete.get(2).unwrap().number, 23);
    assert_eq!(witness().unwrap().boxed, Some(address));
    assert_eq!(complete.slots.slots().as_ptr() as usize, address);
    drop(complete);
    assert_eq!(drops.load(Ordering::SeqCst), 3);
}

#[test]
fn full_push_rejection_returns_original_value_and_preserves_complete_buffer() {
    let source = HostSlotTable::new(Box::new([1_u8]));
    let drops = Arc::new(AtomicUsize::new(0));
    let mut builder = source
        .prepare_copy_slots()
        .unwrap()
        .for_dense_destination::<Value>()
        .unwrap()
        .initialize();
    builder.push(value(31, &drops)).unwrap();
    let rejected = builder.push(value(37, &drops)).unwrap_err();
    assert_eq!(rejected.capacity(), 1);
    assert_eq!(builder.initialized_count(), 1);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    let rejected = rejected.into_value();
    assert_eq!(rejected.number, 37);
    drop(rejected);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    let complete = builder.finish().unwrap();
    assert_eq!(complete.get(0).unwrap().number, 31);
    drop(complete);
    assert_eq!(drops.load(Ordering::SeqCst), 2);
}

#[test]
fn partial_error_and_unwind_drop_only_initialized_values_without_boxing() {
    for unwind in [false, true] {
        let source = HostSlotTable::new(Box::new([2_u8, 3, 5, 7]));
        let source_metadata = source.metadata().clone();
        let drops = Arc::new(AtomicUsize::new(0));
        let mut builder = source
            .prepare_copy_slots()
            .unwrap()
            .for_dense_destination::<Value>()
            .unwrap()
            .initialize();
        builder.push(value(41, &drops)).unwrap();
        builder.push(value(43, &drops)).unwrap();
        // No source borrow or payload owner is hidden in the partial result.
        drop(source);
        assert!(matches!(
            source_metadata.try_attach::<Infallible>(&SharedStorageDomain::default(), || {
                panic!("the destination must not retain its source")
            }),
            Err(crate::HostSlotAttachmentError::Retired)
        ));
        if unwind {
            let result = catch_unwind(AssertUnwindSafe(|| {
                let _partial = builder;
                panic!("failure after two produced values");
            }));
            assert!(result.is_err());
        } else {
            let incomplete = builder.finish().unwrap_err();
            assert_eq!(incomplete.initialized(), 2);
            assert_eq!(incomplete.builder.slots.capacity(), 4);
            assert_eq!(drops.load(Ordering::SeqCst), 0);
            drop(incomplete);
        }
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        assert_eq!(witness().unwrap().boxed, None);
    }
}

#[test]
fn completed_optional_and_dense_sources_preserve_exact_identity_and_value_type() {
    let original = HostSlotTable::new(Box::new([3_u8, 7]));
    let mut optional_builder = original
        .prepare_copy_slots()
        .unwrap()
        .for_destination::<Option<u32>>()
        .unwrap()
        .initialize();
    optional_builder.push(Some(13)).unwrap();
    optional_builder.push(None).unwrap();
    let optional = optional_builder.finish().unwrap();
    let dense_plan = optional
        .prepare_copy_slots()
        .unwrap()
        .for_dense_destination::<Option<u32>>()
        .unwrap();
    assert!(dense_plan
        .source_metadata()
        .same_storage(optional.metadata()));
    assert!(std::ptr::eq(
        dense_plan.source_at(0).unwrap(),
        optional.get(0).unwrap()
    ));
    assert_eq!(*dense_plan.source_at(1).unwrap(), None);
    let retained = 2 * size_of::<Option<u32>>() as u64;
    assert_eq!(dense_plan.retained_bytes(), retained);
    let values = [
        *dense_plan.source_at(0).unwrap(),
        *dense_plan.source_at(1).unwrap(),
    ];
    let mut builder = dense_plan.initialize();
    for item in values {
        builder.push(item).unwrap();
    }
    let dense = builder.finish().unwrap();
    assert_eq!(dense.metadata().capacity_bytes(), Some(retained));
    assert_eq!(dense.iter().copied().collect::<Vec<_>>(), [Some(13), None]);
    drop((original, optional));

    reset_witness();
    let repeat = dense
        .prepare_copy_slots()
        .unwrap()
        .for_dense_destination::<u64>()
        .unwrap();
    assert_eq!(witness(), None);
    assert!(repeat.source_metadata().same_storage(dense.metadata()));
    assert!(std::ptr::eq(
        repeat.source_at(1).unwrap(),
        dense.get(1).unwrap()
    ));
    assert_eq!(repeat.retained_bytes(), 2 * size_of::<u64>() as u64);
    let values = [
        repeat.source_at(0).unwrap().unwrap_or(0) as u64,
        repeat.source_at(1).unwrap().unwrap_or(0) as u64,
    ];
    let mut builder = repeat.initialize();
    for item in values {
        builder.push(item).unwrap();
    }
    let wide = builder.finish().unwrap();
    drop(dense);
    assert_eq!(wide.iter().copied().collect::<Vec<_>>(), [13, 0]);
    assert_eq!(wide.metadata().slot_size(), size_of::<u64>());
    assert_eq!(
        wide.metadata().capacity_bytes(),
        Some(2 * size_of::<u64>() as u64)
    );
}

#[test]
fn completed_table_transfer_keeps_payload_pointer_identity_and_retirement_order() {
    struct Charge {
        drops: Arc<AtomicUsize>,
        retired: Arc<AtomicUsize>,
    }
    impl Drop for Charge {
        fn drop(&mut self) {
            assert_eq!(self.drops.load(Ordering::SeqCst), 2);
            self.retired.fetch_add(1, Ordering::SeqCst);
        }
    }
    let source = HostSlotTable::new(Box::new([2_u8, 3]));
    let drops = Arc::new(AtomicUsize::new(0));
    let retired = Arc::new(AtomicUsize::new(0));
    let mut builder = source
        .prepare_copy_slots()
        .unwrap()
        .for_dense_destination::<Value>()
        .unwrap()
        .initialize();
    builder.push(value(47, &drops)).unwrap();
    builder.push(value(53, &drops)).unwrap();
    let dense = builder.finish().unwrap();
    let token = dense.metadata().clone();
    let address = dense.slots.slots().as_ptr();
    token
        .try_attach(&SharedStorageDomain::default(), || {
            Ok::<Box<dyn Send + Sync>, Infallible>(Box::new(Charge {
                drops: drops.clone(),
                retired: retired.clone(),
            }))
        })
        .unwrap();
    // This crate-private transfer grants no funding. A future admitted handoff
    // must carry its host owner with it; this test covers only table continuity.
    let mut table = dense.into_table();
    assert!(table.metadata().same_storage(&token));
    assert_eq!(table.slots().as_ptr(), address);
    table.slots_mut()[1].number = 59;
    assert_eq!(table.slots()[1].number, 59);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(table);
    assert_eq!(drops.load(Ordering::SeqCst), 2);
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    assert!(matches!(
        token.try_attach::<Infallible>(&SharedStorageDomain::default(), || {
            panic!("escaped accounting token cannot reacquire after table retirement")
        }),
        Err(crate::HostSlotAttachmentError::Retired)
    ));
    drop(token);
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}

#[test]
fn empty_and_zero_sized_dense_destinations_keep_count_without_payload_allocation() {
    struct Zst;
    impl Drop for Zst {
        fn drop(&mut self) {
            ZST_DROPS.set(ZST_DROPS.get() + 1);
        }
    }
    enum Never {}
    let empty = HostSlotTable::<u8>::new(Box::new([]));
    let plan = empty
        .prepare_copy_slots()
        .unwrap()
        .for_dense_destination::<[u8; 64]>()
        .unwrap();
    assert_eq!(
        (plan.retained_bytes(), plan.initialization_peak_bytes()),
        (0, 0)
    );
    let builder = plan.initialize();
    assert_eq!(witness().unwrap().capacity, 0);
    let complete = builder.finish().unwrap();
    assert!(complete.is_empty());
    assert_eq!(complete.metadata().capacity_bytes(), Some(0));
    let plan = empty
        .prepare_copy_slots()
        .unwrap()
        .for_dense_destination::<Never>()
        .unwrap();
    assert!(plan.is_empty());
    assert_eq!(
        (plan.retained_bytes(), plan.initialization_peak_bytes()),
        (0, 0)
    );
    let builder = plan.initialize();
    assert!(builder.is_empty());
    let empty_complete = builder.finish().unwrap();
    assert!(empty_complete.is_empty());
    assert_eq!(empty_complete.metadata().capacity_bytes(), Some(0));

    let source = HostSlotTable::new(Box::new([(), (), ()]));
    assert_eq!(source.metadata().capacity_bytes(), Some(0));
    let plan = source
        .prepare_copy_slots()
        .unwrap()
        .for_dense_destination::<Zst>()
        .unwrap();
    assert_eq!(
        (
            plan.len(),
            plan.retained_bytes(),
            plan.initialization_peak_bytes()
        ),
        (3, 0, 0)
    );
    ZST_DROPS.set(0);
    let mut builder = plan.initialize();
    assert_eq!(witness().unwrap().capacity, usize::MAX);
    for _ in 0..3 {
        builder.push(Zst).unwrap();
    }
    let rejected = builder.push(Zst).unwrap_err();
    assert_eq!(
        rejected.capacity(),
        3,
        "logical count must not use ZST Vec capacity"
    );
    drop(rejected);
    assert_eq!(ZST_DROPS.get(), 1);
    let complete = builder.finish().unwrap();
    assert_eq!(complete.len(), 3);
    assert_eq!(complete.iter().len(), 3);
    assert_eq!(complete.metadata().capacity_bytes(), Some(0));
    assert_eq!(complete.metadata().slot_size(), 0);
    drop(complete);
    assert_eq!(ZST_DROPS.get(), 4);
}

#[test]
fn dense_overflow_is_rejected_from_actual_source_before_allocating_worker() {
    // The individual type is well below Rust's object layout limit. The actual
    // 257-slot source makes its combined requested payload exceed isize::MAX.
    type TooWide = [u8; isize::MAX as usize / 256];
    let source = HostSlotTable::new(Box::new([3_u8; 257]));
    reset_witness();
    let error = source
        .prepare_copy_slots()
        .unwrap()
        .for_dense_destination::<TooWide>()
        .unwrap_err();
    assert_eq!(
        error,
        HostSlotInitializationError::Overflow {
            component: "dense destination payload",
        }
    );
    assert_eq!(witness(), None);
    assert_eq!(source.slots(), &[3_u8; 257]);
    let slot = size_of::<u64>();
    let maximum = isize::MAX as usize / slot;
    let exact = DenseExtent::for_type::<u64>(maximum).unwrap();
    assert_eq!(exact.retained, (maximum * slot) as u64);
    assert_eq!(exact.peak, exact.retained + 2 * slot as u64);
    assert_eq!(
        DenseExtent::for_type::<u64>(maximum + 1).unwrap_err(),
        error
    );
    assert_eq!(DenseExtent::for_type::<u64>(usize::MAX).unwrap_err(), error);
    assert_eq!(witness(), None);
}
