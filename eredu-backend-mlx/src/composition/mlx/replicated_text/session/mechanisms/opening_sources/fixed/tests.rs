use super::*;
use eredu_checkpoint::store::SourceStorageRef;
use std::panic::{catch_unwind, AssertUnwindSafe};

thread_local! { static CALLS: Cell<usize> = const { Cell::new(0) }; }
fn housekeep() {
    CALLS.set(CALLS.get() + 1);
}
struct Cold;
impl Cold {
    fn start() -> Self {
        safemlx::register_thread_runtime_housekeeping(housekeep);
        CALLS.set(0);
        Self
    }
}
impl Drop for Cold {
    fn drop(&mut self) {
        safemlx::unregister_thread_runtime_housekeeping(housekeep);
    }
}

fn fixed(n: OpeningSlots) -> FixedOpeningOwners {
    let mut result = FixedOpeningOwners::empty(n);
    result.reserve().unwrap();
    result.begin().unwrap();
    result
}
#[test]
fn aliases_and_zero_source_owners_take_slots_before_an_extra_retain() {
    let root = Arc::new(vec![3u8, 7, 11]);
    let empty = Arc::new(Vec::<u8>::new());
    let mut sink = fixed(OpeningSlots {
        sources: 3,
        ..OpeningSlots::default()
    });
    for source in [
        SourceStorageRef::new(&root, 3),
        SourceStorageRef::new(&root, 3),
        SourceStorageRef::new(&empty, 0),
    ] {
        sink.retain(RetainedStorageRef::Source(source)).unwrap();
    }
    let before = Arc::strong_count(&root);
    assert!(matches!(
        sink.retain(RetainedStorageRef::Source(SourceStorageRef::new(&root, 3))),
        Err(OpeningError::Capacity)
    ));
    assert_eq!(Arc::strong_count(&root), before);
    assert_eq!(sink.sources.values.len(), 3);
    assert_eq!(sink.sources.values.capacity(), 3);
    assert!(sink.sources.values.iter().all(|s| s.fact.is_none()));
    sink.inspect_facts().unwrap();
    let mut visits = 0;
    sink.visit(&mut |entry| -> Result<(), OpeningError> {
        let OpeningEntry::Source(source) = entry else {
            panic!("wrong owner category")
        };
        assert!(
            source.identity() == SourceStorageRef::new(&root, 3).identity()
                || source.identity() == SourceStorageRef::new(&empty, 0).identity()
        );
        visits += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(visits, 3);
    assert_eq!(
        sink.sources
            .values
            .iter()
            .map(|s| s.fact.unwrap())
            .collect::<Vec<_>>(),
        [3, 3, 0]
    );
    assert!(matches!(sink.begin(), Err(OpeningError::Used)));
    drop(sink);
    assert_eq!(Arc::strong_count(&root), 1);
    assert_eq!(Arc::strong_count(&empty), 1);
}

#[test]
fn source_prefix_survives_callback_unwind_and_drops_only_with_capsule() {
    let root = Arc::new(vec![19u8]);
    let weak = Arc::downgrade(&root);
    let mut sink = fixed(OpeningSlots {
        sources: 2,
        ..OpeningSlots::default()
    });
    let panic = Arc::new(());
    let payload = panic.clone();
    let result = catch_unwind(AssertUnwindSafe(|| {
        sink.retain(RetainedStorageRef::Source(SourceStorageRef::new(&root, 1)))
            .unwrap();
        std::panic::panic_any(payload);
    }));
    let caught = result.unwrap_err().downcast::<Arc<()>>().unwrap();
    assert!(Arc::ptr_eq(&caught, &panic));
    drop(root);
    assert!(weak.upgrade().is_some());
    assert_eq!(sink.sources.values.len(), 1);
    assert!(matches!(sink.begin(), Err(OpeningError::Used)));
    drop(sink);
    assert!(weak.upgrade().is_none());
}

#[test]
fn native_retention_is_cold_and_unknown_facts_keep_every_descriptor() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let root = Array::from_slice(&[2_f32, -3., 5., -7.], &[4]);
    let _ = root.evaluated().unwrap();
    let lazy = root.square(&stream).unwrap();
    assert!(lazy.try_allocation_info().unwrap().is_none());
    let mut sink = fixed(OpeningSlots {
        arrays: 3,
        ..OpeningSlots::default()
    });
    let reserved = OpeningSlots {
        arrays: 3,
        ..OpeningSlots::default()
    };
    assert_eq!(
        FixedOpeningOwners::storage_bytes(reserved).unwrap()
            - FixedOpeningOwners::buffer_bytes(reserved).unwrap(),
        (3 * Array::inspection_clone_handle_bytes()) as u64
    );
    let hook = Cold::start();
    sink.retain_array(&root).unwrap();
    sink.retain_array(&root).unwrap();
    sink.retain_array(&lazy).unwrap();
    assert!(sink.arrays.values.iter().all(|s| s.fact.is_none()));
    assert!(matches!(sink.inspect_facts(), Err(OpeningError::Unknown)));
    assert_eq!(CALLS.get(), 0);
    drop(hook);
    assert_eq!(sink.arrays.values.len(), 3);
    assert_eq!(sink.arrays.values.capacity(), 3);
    assert_eq!(sink.arrays.values[0].fact, sink.arrays.values[1].fact);
    assert!(sink.arrays.values[0].fact.is_some());
    assert!(sink.arrays.values[2].fact.is_none());
    let mut visited = false;
    assert!(matches!(
        sink.visit(&mut |_| -> Result<(), OpeningError> {
            visited = true;
            Ok(())
        }),
        Err(OpeningError::Incomplete)
    ));
    assert!(!visited);
    assert!(matches!(sink.begin(), Err(OpeningError::Used)));
}

#[test]
fn all_fixed_categories_are_measured_and_host_facts_follow_retention() {
    let mut host = safemlx::HostTransferBuffer::new(
        &[2],
        safemlx::Dtype::Int32,
        safemlx::HostTransferPolicy::Transfer,
    )
    .unwrap();
    host.as_bytes_mut().unwrap().copy_from_slice(
        &[3_i32, -11]
            .into_iter()
            .flat_map(i32::to_ne_bytes)
            .collect::<Vec<_>>(),
    );
    let host = Arc::new(host.freeze());
    let bytes: Arc<[u8]> = Arc::from([2u8, 3, 5]);
    let slots = OpeningSlots {
        hosts: 2,
        bytes: 1,
        ..OpeningSlots::default()
    };
    assert_eq!(
        FixedOpeningOwners::buffer_bytes(slots).unwrap(),
        (2 * size_of::<Slot<Arc<ImmutableHostTransferBuffer>, AllocationInfo>>()
            + size_of::<Slot<Arc<[u8]>, u64>>()) as u64
    );
    let mut sink = fixed(slots);
    sink.retain(RetainedStorageRef::Host(&host)).unwrap();
    sink.retain(RetainedStorageRef::Host(&host)).unwrap();
    sink.retain(RetainedStorageRef::Bytes(&bytes)).unwrap();
    assert_eq!(Arc::strong_count(&host), 3);
    assert_eq!(Arc::strong_count(&bytes), 2);
    sink.inspect_facts().unwrap();
    assert_eq!(sink.hosts.values[0].fact, sink.hosts.values[1].fact);
    assert_eq!(sink.bytes.values[0].fact, Some(3));
    assert!(matches!(
        FixedOpeningOwners::buffer_bytes(OpeningSlots {
            arrays: usize::MAX,
            ..OpeningSlots::default()
        }),
        Err(OpeningError::Overflow)
    ));
}

thread_local! {
    static LOCK_GATE: RefCell<Option<(std::sync::mpsc::SyncSender<()>, std::sync::mpsc::Receiver<()>)>> = const { RefCell::new(None) };
}
fn hold_runtime_from_housekeeping() {
    let gate = LOCK_GATE.with(|slot| slot.borrow_mut().take());
    if let Some((entered, release)) = gate {
        entered.send(()).unwrap();
        release
            .recv_timeout(std::time::Duration::from_secs(10))
            .unwrap();
    }
}
#[test]
fn actual_inspection_clone_busy_preserves_prefix_without_extra_owner() {
    let root = Array::from_slice(&[23_f32, -29.], &[2]);
    let _ = root.evaluated().unwrap();
    let other = root.clone();
    let mut sink = fixed(OpeningSlots {
        arrays: 2,
        ..OpeningSlots::default()
    });
    sink.retain_array(&root).unwrap();
    let (entered_tx, entered_rx) = std::sync::mpsc::sync_channel(0);
    let (release_tx, release_rx) = std::sync::mpsc::sync_channel(0);
    let worker = std::thread::spawn(move || {
        LOCK_GATE.with(|slot| *slot.borrow_mut() = Some((entered_tx, release_rx)));
        safemlx::register_thread_runtime_housekeeping(hold_runtime_from_housekeeping);
        // nbytes() bypasses runtime entry and would never run this hook.
        // Query the already completed backing through an ordinary lock entry.
        let result = other.allocation_info();
        safemlx::unregister_thread_runtime_housekeeping(hold_runtime_from_housekeeping);
        result
    });
    entered_rx
        .recv_timeout(std::time::Duration::from_secs(10))
        .unwrap();
    let result = sink.retain_array(&root);
    release_tx.send(()).unwrap();
    let worker_fact = worker.join().unwrap().unwrap().unwrap();
    assert_eq!(Some(worker_fact), root.try_allocation_info().unwrap());
    let error = result.unwrap_err();
    assert!(matches!(
        error,
        OpeningError::Array(ArrayMetadataError::RuntimeBusy)
    ));
    assert_eq!(sink.arrays.values.len(), 1);
    assert!(sink.arrays.values[0].fact.is_none());
    assert_eq!(
        sink.arrays.values[0].owner.try_allocation_info().unwrap(),
        root.try_allocation_info().unwrap()
    );
    assert!(matches!(sink.begin(), Err(OpeningError::Used)));
}
