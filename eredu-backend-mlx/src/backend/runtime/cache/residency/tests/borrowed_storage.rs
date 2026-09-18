use super::*;
use crate::backend::runtime::residency::storage::{
    RetainedStorageInspectionError as InspectError, RetainedStorageRef,
};
use safemlx::ImmutableHostTransferBuffer;
use std::{
    cell::Cell,
    sync::atomic::{AtomicBool, Ordering},
};

thread_local! { static HOUSEKEEPING: Cell<usize> = const { Cell::new(0) }; }
fn housekeeping() {
    HOUSEKEEPING.with(|count| count.set(count.get() + 1));
}
struct Hook;
impl Drop for Hook {
    fn drop(&mut self) {
        safemlx::unregister_thread_runtime_housekeeping(housekeeping);
    }
}
fn cold<R>(f: impl FnOnce() -> R) -> R {
    safemlx::register_thread_runtime_housekeeping(housekeeping);
    let hook = Hook;
    HOUSEKEEPING.with(|count| count.set(0));
    let result = f();
    assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
    drop(hook);
    result
}

enum Held {
    Array(Array),
    Host(Arc<ImmutableHostTransferBuffer>),
    Bytes(Arc<[u8]>),
}
fn retain(value: RetainedStorageRef<'_>) -> Held {
    match value {
        RetainedStorageRef::Array(value) => Held::Array(value.try_clone_for_inspection().unwrap()),
        RetainedStorageRef::CanonicalArray(_) => panic!("cache has no canonical parameter cells"),
        RetainedStorageRef::Host(value) => Held::Host(Arc::clone(value)),
        RetainedStorageRef::RetainedHost(_) => panic!("cache has no residency-manager host owner"),
        RetainedStorageRef::Bytes(value) => Held::Bytes(Arc::clone(value)),
        RetainedStorageRef::Source(_) => panic!("cache has no checkpoint source"),
    }
}

#[test]
fn borrowed_cache_fields_cover_nonzero_arrays_hosts_and_buffered_aliases_without_io() {
    let manager =
        CacheResidencyManager::new(PagedCacheOptions::new(1, 4096, 4096, 1).unwrap()).unwrap();
    let stream = cpu_stream();
    let root = Array::from_slice(&[0.5_f32, -1.5], &[2]);
    let arrays = CacheBlockArrays::KeyValue {
        keys: root.clone(),
        values: root.clone(),
    };
    let host = HostCacheBlock::from_device_arrays(&arrays, &stream).unwrap();
    let bytes: Arc<[u8]> = Arc::from([7_u8, 3, 1, 9]);
    let weak = Arc::downgrade(&bytes);
    let missing = tempfile::tempdir().unwrap();
    let location = DiskLocation::ordinary(DiskLocationData {
        buffered: Some(Arc::clone(&bytes)),
        ..missing_location_data(missing.path(), "missing")
    });
    let device_id = disk_test_id(0);
    let host_id = disk_test_id(1);
    {
        let mut state = manager.inner.state.lock().unwrap();
        for physical in [
            MlxCacheBlockStorage::device(device_id, arrays, Some(location.clone())),
            MlxCacheBlockStorage::host(host_id, host, Some(location)),
        ] {
            insert_test_record(
                &mut state,
                CacheBlockRecord {
                    physical,
                    bytes: 16,
                    shapes: [vec![2], vec![2]],
                    dtypes: ["Float32".into(), "Float32".into()],
                    imported: false,
                    original_discard: None,
                    _metadata_funding: None,
                },
                false,
                0,
            );
        }
    }
    let mut expected = [0_usize; 6];
    let mut length = 0;
    {
        let state = manager.inner.state.lock().unwrap();
        for record in state.blocks.values() {
            if let Some(arrays) = record.physical.device_resource() {
                for array in arrays.arrays() {
                    expected[length] = array as *const Array as usize;
                    length += 1;
                }
            }
            if let Some(host) = record.host_block() {
                let HostCacheBlock::KeyValue { keys, values } = host else {
                    unreachable!()
                };
                for buffer in [keys, values] {
                    expected[length] = buffer as *const Arc<ImmutableHostTransferBuffer> as usize;
                    length += 1;
                }
            }
            if let Some(bytes) = record
                .disk()
                .and_then(|location| location.buffered.as_ref())
            {
                expected[length] = bytes as *const Arc<[u8]> as usize;
                length += 1;
            }
        }
    }
    assert_eq!(length, 6);
    let mut actual = [0_usize; 6];
    let mut count = 0;
    let mut held = Vec::with_capacity(6);
    assert!(
        cold(
            || manager.try_visit_retained_storage(&mut |value| -> Result<(), InspectError> {
                actual[count] = match value {
                    RetainedStorageRef::Array(value) => value as *const Array as usize,
                    RetainedStorageRef::Host(value) => {
                        value as *const Arc<ImmutableHostTransferBuffer> as usize
                    }
                    RetainedStorageRef::Bytes(value) => value as *const Arc<[u8]> as usize,
                    _ => unreachable!(),
                };
                count += 1;
                held.push(retain(value));
                Ok(())
            })
        )
        .unwrap()
    );
    assert_eq!(actual, expected);
    assert_eq!(count, 6);
    drop((manager, bytes, root));
    assert!(weak.upgrade().is_some());
    for value in &held {
        match value {
            Held::Array(value) => assert_eq!(
                value.evaluated().unwrap().try_to_vec::<f32>().unwrap(),
                [0.5, -1.5]
            ),
            Held::Host(value) => assert_eq!(
                value.as_bytes().unwrap(),
                [0.5_f32, -1.5]
                    .into_iter()
                    .flat_map(f32::to_ne_bytes)
                    .collect::<Vec<_>>()
            ),
            Held::Bytes(value) => assert_eq!(value.as_ref(), [7, 3, 1, 9]),
        }
    }
    drop(held);
    assert!(weak.upgrade().is_none());
}

#[test]
fn borrowed_cache_scan_keeps_completed_but_undrained_host_ticket_incomplete() {
    let manager =
        CacheResidencyManager::new(PagedCacheOptions::new(1, 4096, 4096, 1).unwrap()).unwrap();
    let arrays = CacheBlockArrays::KeyValue {
        keys: Array::from_slice(&[2.0_f32], &[1]),
        values: Array::from_slice(&[-3.0_f32], &[1]),
    };
    let completion = Arc::new(HostDemotionCompletion::default());
    completion.finish(Ok(HostCacheBlock::from_device_arrays(
        &arrays,
        &cpu_stream(),
    )
    .unwrap()));
    let block = disk_test_id(0);
    let ticket = HostDemotionTicket {
        operation_id: 97,
        id: block.clone(),
        reserved_host_bytes: 8,
        completion: completion.clone(),
    };
    insert_test_record(
        &mut manager.inner.state.lock().unwrap(),
        CacheBlockRecord {
            physical: test_demoting(arrays, ticket),
            bytes: 8,
            shapes: [vec![1], vec![1]],
            dtypes: ["Float32".into(), "Float32".into()],
            imported: false,
            original_discard: None,
            _metadata_funding: None,
        },
        false,
        0,
    );
    let phase = manager.inner.state.lock().unwrap().blocks[&block]
        .physical
        .phase();
    let mut count = 0;
    assert!(
        !cold(
            || manager.try_visit_retained_storage(&mut |_| -> Result<(), InspectError> {
                count += 1;
                Ok(())
            })
        )
        .unwrap()
    );
    assert_eq!(count, 2);
    assert_eq!(
        manager.inner.state.lock().unwrap().blocks[&block]
            .physical
            .phase(),
        phase
    );
    assert!(completion.result.lock().unwrap().as_ref().unwrap().is_ok());
}

#[test]
fn borrowed_cache_scan_observes_real_worker_control_outside_block_map() {
    let manager =
        CacheResidencyManager::new(PagedCacheOptions::new(1, 4096, 4096, 1).unwrap()).unwrap();
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let (finished_tx, finished_rx) = mpsc::channel();
    let payload = Arc::new(());
    let weak = Arc::downgrade(&payload);
    assert!(
        manager
            .inner
            .host_demotion_worker
            .sender
            .send(super::super::HostDemotionRequest::Pause {
                started: started_tx,
                release: release_rx,
                retained: payload,
                finished: finished_tx
            })
            .is_ok()
    );
    started_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    let result = cold(|| manager.try_visit_retained_storage(&mut |_| Ok::<_, InspectError>(())));
    let still_owned = weak.upgrade().is_some();
    release_tx.send(()).unwrap();
    finished_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    assert!(!result.unwrap());
    assert!(still_owned);
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while manager
        .inner
        .host_demotion_worker
        .active_payload
        .load(Ordering::Acquire)
    {
        assert!(std::time::Instant::now() < deadline);
        std::thread::yield_now();
    }
    assert!(weak.upgrade().is_none());
    assert!(
        manager
            .try_visit_retained_storage(&mut |_| Ok::<_, InspectError>(()))
            .unwrap()
    );
}

#[test]
fn borrowed_cache_state_busy_returns_before_callback_and_positive_retry() {
    let manager =
        CacheResidencyManager::new(PagedCacheOptions::new(1, 4096, 4096, 1).unwrap()).unwrap();
    let other = manager.clone();
    let (locked_tx, locked_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let thread = std::thread::spawn(move || {
        let _state = other.inner.state.lock().unwrap();
        locked_tx.send(()).unwrap();
        release_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    });
    locked_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    let called = AtomicBool::new(false);
    let result = manager.try_visit_retained_storage(&mut |_| -> Result<(), InspectError> {
        called.store(true, Ordering::Release);
        Ok(())
    });
    release_tx.send(()).unwrap();
    thread.join().unwrap();
    assert!(matches!(result, Err(InspectError::Busy)));
    assert!(!called.load(Ordering::Acquire));
    assert!(
        manager
            .try_visit_retained_storage(&mut |_| Ok::<_, InspectError>(()))
            .unwrap()
    );
}

#[test]
fn borrowed_cache_disk_worker_prepared_and_active_records_stay_incomplete() {
    let directory = tempfile::tempdir().unwrap();
    let options = PagedCacheOptions::new(1, 4096, 4096, 1)
        .unwrap()
        .with_live_disk(directory.path(), 4096, 1)
        .unwrap();
    let manager = CacheResidencyManager::new(options).unwrap();
    let worker = manager.inner.disk_worker.as_ref().unwrap();
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let submission = worker
        .prepare(
            CacheIoOperationKey {
                generation: 0,
                id: disk_test_id(99),
                kind: CacheIoOperationKind::Read,
            },
            DiskTask::Pause {
                started: started_tx,
                release: release_rx,
            },
        )
        .unwrap();
    let ticket = submission.ticket.clone();
    // The real prepared registry owns a task, before any block-map entry exists.
    let prepared = cold(|| manager.try_visit_retained_storage(&mut |_| Ok::<_, InspectError>(())));
    submission.enqueue().unwrap();
    started_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    let active = cold(|| manager.try_visit_retained_storage(&mut |_| Ok::<_, InspectError>(())));
    release_tx.send(()).unwrap();
    ticket.wait().unwrap();
    ticket.wait_for_task_resources().unwrap();
    // The real worker retires its own completed record. Resource completion
    // precedes the final active-owner guard, so await that actual cold state.
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while !matches!(worker.inner.try_has_retained_work(), Ok(Some(false))) {
        assert!(std::time::Instant::now() < deadline);
        std::thread::yield_now();
    }
    assert!(!prepared.unwrap());
    assert!(!active.unwrap());
    assert!(
        manager
            .try_visit_retained_storage(&mut |_| Ok::<_, InspectError>(()))
            .unwrap()
    );
    assert!(
        std::fs::read_dir(directory.path())
            .unwrap()
            .next()
            .is_none()
    );
}

#[derive(Debug)]
enum CallbackError {
    Inspection(InspectError),
    Sentinel(u32),
}
impl From<InspectError> for CallbackError {
    fn from(error: InspectError) -> Self {
        Self::Inspection(error)
    }
}

#[test]
fn borrowed_cache_error_unwind_and_racing_failure_keep_actual_array_prefix() {
    for mode in 0..3 {
        let manager =
            CacheResidencyManager::new(PagedCacheOptions::new(1, 4096, 4096, 1).unwrap()).unwrap();
        let root = Array::from_slice(&[5.0_f32, -7.0], &[2]);
        let arrays = CacheBlockArrays::KeyValue {
            keys: root.clone(),
            values: root.clone(),
        };
        insert_test_record(
            &mut manager.inner.state.lock().unwrap(),
            CacheBlockRecord {
                physical: MlxCacheBlockStorage::device(disk_test_id(0), arrays, None),
                bytes: 16,
                shapes: [vec![2], vec![2]],
                dtypes: ["Float32".into(), "Float32".into()],
                imported: false,
                original_discard: None,
                _metadata_funding: None,
            },
            false,
            0,
        );
        let mut held = Vec::with_capacity(2);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            manager.try_visit_retained_storage(&mut |value| -> Result<(), CallbackError> {
                held.push(retain(value));
                match mode {
                    0 => Err(CallbackError::Sentinel(59)),
                    1 => std::panic::panic_any(61_u32),
                    _ => {
                        // Failure-only flag injection: no completion or credit is
                        // inferred. Real asynchronous ownership is covered above.
                        manager
                            .inner
                            .host_demotion_worker
                            .active_payload
                            .store(true, Ordering::Release);
                        Ok(())
                    }
                }
            })
        }));
        match mode {
            0 => assert!(matches!(result.unwrap(), Err(CallbackError::Sentinel(59)))),
            1 => assert_eq!(*result.err().unwrap().downcast::<u32>().unwrap(), 61),
            _ => {
                assert!(!result.unwrap().unwrap());
                manager
                    .inner
                    .host_demotion_worker
                    .active_payload
                    .store(false, Ordering::Release);
            }
        }
        assert_eq!(held.len(), if mode == 2 { 2 } else { 1 });
        drop((manager, root));
        for value in held {
            let Held::Array(value) = value else {
                panic!("array source")
            };
            assert_eq!(
                value.evaluated().unwrap().try_to_vec::<f32>().unwrap(),
                [5.0, -7.0]
            );
        }
    }
}
