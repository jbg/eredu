use super::*;
use eredu_core::SharedStorageAccountingId;
use std::{
    cell::Cell,
    convert::Infallible,
    num::NonZeroU32,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct InitializationWitness {
    len: usize,
    capacity: usize,
    vector: usize,
    boxed: usize,
}
thread_local! {
    static INITIALIZATION: Cell<Option<InitializationWitness>> = const { Cell::new(None) };
}
pub(super) fn record_initialization(len: usize, capacity: usize, vector: usize, boxed: usize) {
    INITIALIZATION.set(Some(InitializationWitness {
        len,
        capacity,
        vector,
        boxed,
    }));
}
fn witness() -> Option<InitializationWitness> {
    INITIALIZATION.take()
}

#[test]
fn fixed_worker_never_requires_element_clone_default_or_an_initializer() {
    struct NoCloneOrDefault {
        id: u32,
        nested: Box<[u8]>,
    }
    let source = HostSlotTable::new(Box::new([
        NoCloneOrDefault {
            id: 11,
            nested: Box::new([1, 3, 5]),
        },
        NoCloneOrDefault {
            id: 17,
            nested: Box::new([2, 4]),
        },
        NoCloneOrDefault {
            id: 23,
            nested: Box::new([7]),
        },
    ]));
    witness();
    let plan = source.prepare_copy_slots().unwrap();
    assert_eq!(
        witness(),
        None,
        "preparing does not enter the allocating worker"
    );
    assert!(plan.source_metadata().same_storage(source.metadata()));
    assert!(std::ptr::eq(plan.source_at(1).unwrap(), &source.slots()[1]));
    assert!(plan.source_at(3).is_none());
    let slot = size_of::<Option<NoCloneOrDefault>>() as u64;
    assert_eq!(plan.retained_bytes(), 3 * slot);
    assert_eq!(plan.initialization_peak_bytes(), 5 * slot);
    let mut builder = plan.initialize();
    let allocation = witness().unwrap();
    assert_eq!(allocation.len, 3);
    assert_eq!(allocation.capacity, 3);
    assert_eq!(
        allocation.vector, allocation.boxed,
        "boxing reuses the initialized buffer"
    );
    assert_eq!(builder.slots.slots().as_ptr() as usize, allocation.boxed);
    assert!(builder.slots.slots().iter().all(Option::is_none));
    assert_eq!(builder.metadata().capacity_bytes(), Some(3 * slot));
    let identity = builder.metadata().clone();
    builder
        .push(NoCloneOrDefault {
            id: 31,
            nested: Box::new([8]),
        })
        .unwrap();
    builder
        .push(NoCloneOrDefault {
            id: 37,
            nested: Box::new([9, 10]),
        })
        .unwrap();
    builder
        .push(NoCloneOrDefault {
            id: 41,
            nested: Box::new([]),
        })
        .unwrap();
    let completed = builder.finish().unwrap();
    assert_eq!(completed.slots.slots().as_ptr() as usize, allocation.boxed);
    assert!(identity.same_storage(completed.metadata()));
    assert_eq!(
        completed.iter().map(|value| value.id).collect::<Vec<_>>(),
        [31, 37, 41]
    );
    assert_eq!(&*completed.get(1).unwrap().nested, &[9, 10]);
    assert_eq!(source.slots()[1].id, 17);
    assert_eq!(&*source.slots()[1].nested, &[2, 4]);
    assert_eq!(completed.iter().len(), 3);
    assert_eq!(completed.iter().next_back().unwrap().id, 41);
}

#[test]
fn incomplete_finish_preserves_builder_and_inner_optional_values_are_real_slots() {
    let source = HostSlotTable::new(Box::new([Some(7_u32), None, Some(19)]));
    let mut builder = source.prepare_copy_slots().unwrap().initialize();
    let identity = builder.metadata().clone();
    let pointer = builder.slots.slots().as_ptr();
    builder.push(None).unwrap();
    let error = builder.finish().unwrap_err();
    assert_eq!(error.expected(), 3);
    assert_eq!(error.initialized(), 1);
    assert!(error.to_string().contains("expected 3"));
    let mut builder = error.into_builder();
    assert_eq!(builder.slots.slots().as_ptr(), pointer);
    assert!(identity.same_storage(builder.metadata()));
    assert_eq!(builder.initialized_count(), 1);
    builder.push(Some(23)).unwrap();
    builder.push(None).unwrap();
    let completed = builder.finish().unwrap();
    assert_eq!(
        completed.iter().copied().collect::<Vec<_>>(),
        [None, Some(23), None]
    );
    assert_eq!(completed.get(0), Some(&None));
    assert_eq!(completed.get(3), None);
    assert_eq!(completed.slots.slots().as_ptr(), pointer);
}

struct Dropped {
    id: usize,
    drops: Arc<AtomicUsize>,
}
impl Drop for Dropped {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn full_push_returns_original_value_without_disturbing_completed_cells() {
    let source_drops = Arc::new(AtomicUsize::new(0));
    let copied_drops = Arc::new(AtomicUsize::new(0));
    let rejected_drops = Arc::new(AtomicUsize::new(0));
    let source = HostSlotTable::new(Box::new([Dropped {
        id: 1,
        drops: source_drops.clone(),
    }]));
    let mut builder = source.prepare_copy_slots().unwrap().initialize();
    builder
        .push(Dropped {
            id: 5,
            drops: copied_drops.clone(),
        })
        .unwrap();
    let error = builder
        .push(Dropped {
            id: 9,
            drops: rejected_drops.clone(),
        })
        .unwrap_err();
    assert_eq!(error.capacity(), 1);
    assert_eq!(rejected_drops.load(Ordering::SeqCst), 0);
    assert_eq!(copied_drops.load(Ordering::SeqCst), 0);
    let rejected = error.into_value();
    assert_eq!(rejected.id, 9);
    let completed = builder.finish().unwrap();
    assert_eq!(completed.get(0).unwrap().id, 5);
    drop(rejected);
    assert_eq!(rejected_drops.load(Ordering::SeqCst), 1);
    assert_eq!(copied_drops.load(Ordering::SeqCst), 0);
    drop(completed);
    assert_eq!(copied_drops.load(Ordering::SeqCst), 1);
    assert_eq!(source_drops.load(Ordering::SeqCst), 0);
    drop(source);
    assert_eq!(source_drops.load(Ordering::SeqCst), 1);
}

#[test]
fn dropping_incomplete_error_retires_only_inserted_payload() {
    let source = HostSlotTable::new(Box::new([0_u32, 1, 2]));
    // Preserve coverage of the original same-type preparation path. Distinct
    // source/destination representations have separate tests below.
    let source_drops = Arc::new(AtomicUsize::new(0));
    let dropped_source = HostSlotTable::new(
        source
            .slots()
            .iter()
            .map(|id| Dropped {
                id: *id as usize,
                drops: source_drops.clone(),
            })
            .collect(),
    );
    let inserted_drops = Arc::new(AtomicUsize::new(0));
    let mut builder = dropped_source.prepare_copy_slots().unwrap().initialize();
    builder
        .push(Dropped {
            id: 13,
            drops: inserted_drops.clone(),
        })
        .unwrap();
    let metadata = builder.metadata().clone();
    let error = builder.finish().unwrap_err();
    assert_eq!(error.initialized(), 1);
    assert_eq!(inserted_drops.load(Ordering::SeqCst), 0);
    drop(error);
    assert_eq!(inserted_drops.load(Ordering::SeqCst), 1);
    assert_eq!(source_drops.load(Ordering::SeqCst), 0);
    assert!(matches!(
        metadata.try_attach::<Infallible>(&SharedStorageAccountingId::default(), || {
            panic!("retired partial slots cannot acquire custody")
        }),
        Err(crate::HostSlotAttachmentError::Retired)
    ));
}

#[test]
fn repeated_preparation_borrows_saved_owner_without_growing_optional_representation() {
    let source = HostSlotTable::new(Box::new([
        NonZeroU32::new(3).unwrap(),
        NonZeroU32::new(7).unwrap(),
    ]));
    let plan = source.prepare_copy_slots().unwrap();
    let bytes = plan.retained_bytes();
    let peak = plan.initialization_peak_bytes();
    let values = [*plan.source_at(0).unwrap(), *plan.source_at(1).unwrap()];
    let mut builder = plan.initialize();
    for value in values {
        builder.push(value).unwrap();
    }
    let saved = builder.finish().unwrap();
    drop(source);
    let plan = saved.prepare_copy_slots().unwrap();
    assert!(plan.source_metadata().same_storage(saved.metadata()));
    assert_eq!(plan.source_metadata().capacity_bytes(), Some(bytes));
    assert!(std::ptr::eq(
        plan.source_at(1).unwrap(),
        saved.get(1).unwrap()
    ));
    assert_eq!(plan.retained_bytes(), bytes);
    assert_eq!(plan.initialization_peak_bytes(), peak);
    assert_eq!(bytes, 2 * size_of::<Option<NonZeroU32>>() as u64);
    assert!(size_of::<Option<Option<NonZeroU32>>>() > size_of::<Option<NonZeroU32>>());
    let values = [*plan.source_at(0).unwrap(), *plan.source_at(1).unwrap()];
    let mut next = plan.initialize();
    for value in values {
        next.push(value).unwrap();
    }
    let next = next.finish().unwrap();
    assert!(!saved.metadata().same_storage(next.metadata()));
    assert_eq!(next.metadata().capacity_bytes(), Some(bytes));
    assert_eq!(saved.iter().copied().collect::<Vec<_>>(), values);
    drop(saved);
    assert_eq!(next.iter().copied().collect::<Vec<_>>(), values);
}

#[test]
fn empty_and_zero_sized_source_extents_use_actual_destination_cell_size() {
    let empty = HostSlotTable::<u32>::new(Box::new([]));
    let plan = empty.prepare_copy_slots().unwrap();
    assert!(plan.is_empty());
    assert_eq!(plan.retained_bytes(), 0);
    assert_eq!(plan.initialization_peak_bytes(), 0);
    let empty = plan.initialize().finish().unwrap();
    assert!(empty.is_empty());
    assert_eq!(empty.metadata().capacity_bytes(), Some(0));
    let source = HostSlotTable::new(Box::new([(), (), ()]));
    assert_eq!(source.metadata().capacity_bytes(), Some(0));
    let plan = source.prepare_copy_slots().unwrap();
    let slot = size_of::<Option<()>>() as u64;
    assert!(slot > 0);
    assert_eq!(plan.retained_bytes(), 3 * slot);
    assert_eq!(plan.initialization_peak_bytes(), 5 * slot);
    let mut builder = plan.initialize();
    for _ in 0..3 {
        builder.push(()).unwrap();
    }
    let completed = builder.finish().unwrap();
    assert_eq!(completed.iter().len(), 3);
    assert_eq!(completed.metadata().capacity_bytes(), Some(3 * slot));
}

#[test]
fn checked_extent_rejects_host_allocation_overflow_before_initialization() {
    witness();
    let slot = size_of::<Option<u64>>();
    let maximum = isize::MAX as usize / slot;
    let exact = InitializationExtent::for_type::<u64>(maximum).unwrap();
    assert_eq!(exact.retained, maximum as u64 * slot as u64);
    assert_eq!(exact.peak, exact.retained + 2 * slot as u64);
    for count in [maximum + 1, usize::MAX] {
        assert_eq!(
            InitializationExtent::for_type::<u64>(count).unwrap_err(),
            HostSlotInitializationError::Overflow {
                component: "destination payload"
            }
        );
    }
    assert_eq!(witness(), None);
}

#[test]
fn borrowed_values_can_have_interior_mutability_without_changing_fixed_extent() {
    let source = HostSlotTable::new(Box::new([Cell::new(5_u32)]));
    let plan = source.prepare_copy_slots().unwrap();
    source.slots()[0].set(11);
    assert_eq!(plan.source_at(0).unwrap().get(), 11);
    let bytes = plan.retained_bytes();
    let mut builder = plan.initialize();
    builder.push(Cell::new(17)).unwrap();
    let completed = builder.finish().unwrap();
    completed.get(0).unwrap().set(23);
    assert_eq!(completed.get(0).unwrap().get(), 23);
    assert_eq!(completed.metadata().capacity_bytes(), Some(bytes));
    assert_eq!(source.slots()[0].get(), 11);
}

#[test]
fn partial_unwind_drops_values_before_last_metadata_custody() {
    struct Charge {
        values: Arc<AtomicUsize>,
        retired: Arc<AtomicUsize>,
    }
    impl Drop for Charge {
        fn drop(&mut self) {
            assert_eq!(self.values.load(Ordering::SeqCst), 2);
            self.retired.fetch_add(1, Ordering::SeqCst);
        }
    }
    let source_drops = Arc::new(AtomicUsize::new(0));
    let source = HostSlotTable::new(
        (0..3)
            .map(|id| Dropped {
                id,
                drops: source_drops.clone(),
            })
            .collect(),
    );
    let values = Arc::new(AtomicUsize::new(0));
    let retired = Arc::new(AtomicUsize::new(0));
    let mut builder = source.prepare_copy_slots().unwrap().initialize();
    let token = builder.metadata().clone();
    token
        .try_attach(&SharedStorageAccountingId::default(), || {
            Ok::<Box<dyn Send + Sync>, Infallible>(Box::new(Charge {
                values: values.clone(),
                retired: retired.clone(),
            }))
        })
        .unwrap();
    let result = catch_unwind(AssertUnwindSafe(|| {
        builder
            .push(Dropped {
                id: 31,
                drops: values.clone(),
            })
            .unwrap();
        builder
            .push(Dropped {
                id: 37,
                drops: values.clone(),
            })
            .unwrap();
        let _partial = builder;
        panic!("closed worker failure after partial fill");
    }));
    assert!(result.is_err());
    assert_eq!(values.load(Ordering::SeqCst), 2);
    assert_eq!(source_drops.load(Ordering::SeqCst), 0);
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    assert!(matches!(
        token.try_attach::<Infallible>(&SharedStorageAccountingId::default(), || {
            panic!("retired partial slots must not acquire")
        }),
        Err(crate::HostSlotAttachmentError::Retired)
    ));
    drop(token);
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}

#[test]
fn destination_type_changes_extent_without_copying_or_replacing_actual_source() {
    struct WideSource {
        state: Cell<u8>,
        bytes: [u8; 64],
    }
    struct SmallDestination(u8);
    struct WideDestination {
        state: u8,
        bytes: [u8; 96],
    }
    let source = HostSlotTable::new(Box::new([
        WideSource {
            state: Cell::new(3),
            bytes: [7; 64],
        },
        WideSource {
            state: Cell::new(11),
            bytes: [13; 64],
        },
    ]));
    witness();
    let same: HostSlotInitialization<'_, WideSource> = source.prepare_copy_slots().unwrap();
    let same_bytes = same.retained_bytes();
    let plan: HostSlotInitialization<'_, WideSource, SmallDestination> =
        same.for_destination().unwrap();
    assert_eq!(witness(), None);
    assert_eq!(plan.len(), source.len());
    assert!(plan.source_metadata().same_storage(source.metadata()));
    assert_eq!(
        plan.source_metadata().capacity_bytes(),
        source.metadata().capacity_bytes()
    );
    assert!(std::ptr::eq(plan.source_at(1).unwrap(), &source.slots()[1]));
    source.slots()[1].state.set(17);
    assert_eq!(plan.source_at(1).unwrap().state.get(), 17);
    assert_eq!(plan.source_at(0).unwrap().bytes, [7; 64]);
    let slot = size_of::<Option<SmallDestination>>() as u64;
    assert_eq!(plan.retained_bytes(), 2 * slot);
    assert_eq!(plan.initialization_peak_bytes(), 4 * slot);
    assert!(same_bytes > plan.retained_bytes());
    let values = [
        plan.source_at(0).unwrap().state.get(),
        plan.source_at(1).unwrap().state.get(),
    ];
    let mut builder = plan.initialize();
    let allocation = witness().unwrap();
    assert_eq!((allocation.len, allocation.capacity), (2, 2));
    assert_eq!(allocation.vector, allocation.boxed);
    assert_eq!(builder.metadata().capacity_bytes(), Some(2 * slot));
    for value in values {
        builder.push(SmallDestination(value)).unwrap();
    }
    let small = builder.finish().unwrap();
    assert_eq!(small.get(1).unwrap().0, 17);
    assert!(!small.metadata().same_storage(source.metadata()));
    assert_eq!(source.slots()[1].bytes, [13; 64]);

    let plan = small
        .prepare_copy_slots()
        .unwrap()
        .for_destination::<WideDestination>()
        .unwrap();
    assert!(plan.source_metadata().same_storage(small.metadata()));
    assert!(std::ptr::eq(
        plan.source_at(0).unwrap(),
        small.get(0).unwrap()
    ));
    let slot = size_of::<Option<WideDestination>>() as u64;
    assert_eq!(plan.retained_bytes(), 2 * slot);
    assert_eq!(plan.initialization_peak_bytes(), 4 * slot);
    assert!(plan.retained_bytes() > small.metadata().capacity_bytes().unwrap());
    let values = [plan.source_at(0).unwrap().0, plan.source_at(1).unwrap().0];
    let mut builder = plan.initialize();
    for value in values {
        builder
            .push(WideDestination {
                state: value,
                bytes: [value; 96],
            })
            .unwrap();
    }
    let wide = builder.finish().unwrap();
    drop((small, source));
    assert_eq!(wide.get(0).unwrap().state, 3);
    assert_eq!(wide.get(1).unwrap().bytes, [17; 96]);
    assert_eq!(wide.metadata().capacity_bytes(), Some(2 * slot));
}

#[test]
fn distinct_destination_partial_error_retires_values_before_custody() {
    struct Charge {
        values: Arc<AtomicUsize>,
        retired: Arc<AtomicUsize>,
    }
    impl Drop for Charge {
        fn drop(&mut self) {
            assert_eq!(self.values.load(Ordering::SeqCst), 1);
            self.retired.fetch_add(1, Ordering::SeqCst);
        }
    }
    let source = HostSlotTable::new(Box::new([3_u16, 5]));
    let source_metadata = source.metadata().clone();
    let values = Arc::new(AtomicUsize::new(0));
    let retired = Arc::new(AtomicUsize::new(0));
    let mut builder = source
        .prepare_copy_slots()
        .unwrap()
        .for_destination::<Dropped>()
        .unwrap()
        .initialize();
    let destination_metadata = builder.metadata().clone();
    assert_eq!(
        builder.metadata().capacity_bytes(),
        Some(2 * size_of::<Option<Dropped>>() as u64)
    );
    destination_metadata
        .try_attach(&SharedStorageAccountingId::default(), || {
            Ok::<Box<dyn Send + Sync>, Infallible>(Box::new(Charge {
                values: values.clone(),
                retired: retired.clone(),
            }))
        })
        .unwrap();
    builder
        .push(Dropped {
            id: 17,
            drops: values.clone(),
        })
        .unwrap();
    let error = builder.finish().unwrap_err();
    assert_eq!((error.expected(), error.initialized()), (2, 1));
    drop(source);
    assert!(matches!(
        source_metadata.try_attach::<Infallible>(&SharedStorageAccountingId::default(), || {
            panic!("the destination does not retain the actual source table")
        }),
        Err(crate::HostSlotAttachmentError::Retired)
    ));
    assert_eq!(values.load(Ordering::SeqCst), 0);
    drop(error);
    assert_eq!(values.load(Ordering::SeqCst), 1);
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    assert!(matches!(
        destination_metadata
            .try_attach::<Infallible>(&SharedStorageAccountingId::default(), || {
                panic!("a retired partial destination cannot acquire custody")
            }),
        Err(crate::HostSlotAttachmentError::Retired)
    ));
    drop(destination_metadata);
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}

#[test]
fn completed_distinct_destination_reprepares_its_values_without_optional_nesting() {
    let original = HostSlotTable::new(Box::new([3_u8, 7]));
    let plan = original
        .prepare_copy_slots()
        .unwrap()
        .for_destination::<NonZeroU32>()
        .unwrap();
    let first_bytes = plan.retained_bytes();
    let values = [
        *plan.source_at(0).unwrap() as u32,
        *plan.source_at(1).unwrap() as u32,
    ];
    let mut builder = plan.initialize();
    for value in values {
        builder.push(NonZeroU32::new(value).unwrap()).unwrap();
    }
    let first = builder.finish().unwrap();
    drop(original);
    witness();
    let same: HostSlotInitialization<'_, NonZeroU32> = first.prepare_copy_slots().unwrap();
    assert_eq!(same.retained_bytes(), first_bytes);
    assert_eq!(first_bytes, 2 * size_of::<Option<NonZeroU32>>() as u64);
    assert!(size_of::<Option<Option<NonZeroU32>>>() > size_of::<Option<NonZeroU32>>());
    // Re-selecting a destination always recomputes from the exact same source;
    // the intermediate destination choice never becomes a new source wrapper.
    let plan = same
        .for_destination::<[u8; 64]>()
        .unwrap()
        .for_destination::<u16>()
        .unwrap();
    assert_eq!(witness(), None);
    assert!(std::ptr::eq(
        plan.source_at(1).unwrap(),
        first.get(1).unwrap()
    ));
    assert!(plan.source_metadata().same_storage(first.metadata()));
    assert_eq!(plan.source_metadata().capacity_bytes(), Some(first_bytes));
    assert_eq!(plan.retained_bytes(), 2 * size_of::<Option<u16>>() as u64);
    let values = [
        plan.source_at(0).unwrap().get() as u16,
        plan.source_at(1).unwrap().get() as u16,
    ];
    let mut builder = plan.initialize();
    for value in values {
        builder.push(value).unwrap();
    }
    let second = builder.finish().unwrap();
    assert!(!first.metadata().same_storage(second.metadata()));
    drop(first);
    assert_eq!(second.iter().copied().collect::<Vec<_>>(), [3_u16, 7]);
    assert_eq!(
        second.prepare_copy_slots().unwrap().retained_bytes(),
        second.metadata().capacity_bytes().unwrap()
    );
}

#[test]
fn destination_selection_handles_empty_zero_sized_and_uninhabited_values() {
    enum Never {}
    let empty = HostSlotTable::<u32>::new(Box::new([]));
    let plan = empty
        .prepare_copy_slots()
        .unwrap()
        .for_destination::<[u8; 96]>()
        .unwrap();
    assert_eq!(
        (
            plan.len(),
            plan.retained_bytes(),
            plan.initialization_peak_bytes()
        ),
        (0, 0, 0)
    );
    let completed = plan.initialize().finish().unwrap();
    assert_eq!(completed.metadata().capacity_bytes(), Some(0));

    let zero_source = HostSlotTable::new(Box::new([(), (), ()]));
    assert_eq!(zero_source.metadata().capacity_bytes(), Some(0));
    let plan = zero_source
        .prepare_copy_slots()
        .unwrap()
        .for_destination::<u64>()
        .unwrap();
    let slot = size_of::<Option<u64>>() as u64;
    assert_eq!(
        (plan.retained_bytes(), plan.initialization_peak_bytes()),
        (3 * slot, 5 * slot)
    );
    let mut builder = plan.initialize();
    for value in [13_u64, 17, 23] {
        builder.push(value).unwrap();
    }
    let completed = builder.finish().unwrap();
    let plan = completed
        .prepare_copy_slots()
        .unwrap()
        .for_destination::<()>()
        .unwrap();
    let slot = size_of::<Option<()>>() as u64;
    assert!(
        slot > 0,
        "vacancy is still represented for inhabited ZST values"
    );
    assert_eq!(
        (plan.retained_bytes(), plan.initialization_peak_bytes()),
        (3 * slot, 5 * slot)
    );
    let mut builder = plan.initialize();
    for _ in 0..3 {
        builder.push(()).unwrap();
    }
    let saved_zst = builder.finish().unwrap();
    assert_eq!(saved_zst.len(), 3);
    assert_eq!(saved_zst.metadata().capacity_bytes(), Some(3 * slot));

    let plan = zero_source
        .prepare_copy_slots()
        .unwrap()
        .for_destination::<Never>()
        .unwrap();
    assert_eq!(size_of::<Option<Never>>(), 0);
    assert_eq!(
        (
            plan.len(),
            plan.retained_bytes(),
            plan.initialization_peak_bytes()
        ),
        (3, 0, 0)
    );
    let error = plan.initialize().finish().unwrap_err();
    assert_eq!((error.expected(), error.initialized()), (3, 0));
    drop(error);
    let empty = empty
        .prepare_copy_slots()
        .unwrap()
        .for_destination::<Never>()
        .unwrap()
        .initialize()
        .finish()
        .unwrap();
    assert!(empty.is_empty());
}

#[test]
fn oversized_destination_rejects_from_real_small_source_before_worker() {
    // The type itself fits the Rust object limit. The requested Option cells
    // do not. No value of this large type or large allocation is constructed.
    type TooWide = [u8; isize::MAX as usize / 256];
    let source = HostSlotTable::new(Box::new([3_u8; 257]));
    witness();
    let error = source
        .prepare_copy_slots()
        .unwrap()
        .for_destination::<TooWide>()
        .unwrap_err();
    assert_eq!(
        error,
        HostSlotInitializationError::Overflow {
            component: "destination payload"
        }
    );
    assert_eq!(witness(), None);
    assert_eq!(source.slots(), &[3_u8; 257]);
    let plan = source
        .prepare_copy_slots()
        .unwrap()
        .for_destination::<NonZeroU32>()
        .unwrap();
    assert_eq!(plan.len(), 257);
    assert!(plan.source_metadata().same_storage(source.metadata()));
    assert_eq!(
        plan.retained_bytes(),
        257 * size_of::<Option<NonZeroU32>>() as u64
    );
    let slot = size_of::<Option<u64>>();
    let maximum = isize::MAX as usize / slot;
    assert!(InitializationExtent::for_type::<u64>(maximum).is_ok());
    assert_eq!(
        InitializationExtent::for_type::<u64>(maximum + 1).unwrap_err(),
        error
    );
    assert_eq!(witness(), None);
}
