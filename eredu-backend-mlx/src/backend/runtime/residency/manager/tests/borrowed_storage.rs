use super::*;
use crate::backend::runtime::residency::storage::{
    RetainedStorageInspectionError as InspectError, RetainedStorageRef,
};
use safemlx::Array;
use std::cell::Cell;

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

fn loaded() -> (tempfile::TempDir, ResidencyManager) {
    let (dir, store) = fixture_store();
    let manager = manager(
        store,
        OffloadConfig::new(Some(8), Some(8), 1).unwrap(),
        [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
        [single("a", "a")],
    );
    manager.initialize().unwrap();
    manager.prefetch(&id("a"), MemoryTier::Host).unwrap();
    manager.prefetch(&id("a"), MemoryTier::Device).unwrap();
    (dir, manager)
}

#[test]
fn borrowed_weight_fields_are_exact_cold_and_survive_actual_eviction() {
    let (dir, manager) = loaded();
    let (array_field, host_field, weak_host) = {
        let state = manager.inner.state.lock().unwrap();
        let entry = &state.storage[&id("a")];
        let array = &entry.device.as_ref().unwrap().arrays["weight"];
        let host = &entry.host.as_ref().unwrap().buffers["weight"];
        (
            array as *const Array as usize,
            host as *const RetainedHostBuffer as usize,
            host.observe_for_test(),
        )
    };
    std::fs::remove_file(dir.path().join("model.safetensors")).unwrap();
    let mut array = None;
    let mut host = None;
    assert!(cold(|| manager.try_visit_retained_storage(
        &mut |value| -> Result<(), InspectError> {
            match value {
                RetainedStorageRef::Array(value) => {
                    assert_eq!(value as *const Array as usize, array_field);
                    array = Some(value.try_clone_for_inspection().unwrap());
                }
                RetainedStorageRef::RetainedHost(value) => {
                    assert_eq!(value as *const RetainedHostBuffer as usize, host_field);
                    host = Some(value.clone());
                }
                _ => panic!("file source owns no payload and weight manager owns no shard bytes"),
            }
            Ok(())
        }
    ))
    .unwrap());
    assert!(manager.evict(&id("a"), MemoryTier::Host).unwrap());
    assert!(manager.evict(&id("a"), MemoryTier::Device).unwrap());
    drop(manager);
    assert_eq!(
        array
            .as_ref()
            .unwrap()
            .evaluated()
            .unwrap()
            .try_to_vec::<i32>()
            .unwrap(),
        [1, 2]
    );
    assert_eq!(
        host.as_ref().unwrap().as_bytes().unwrap(),
        [1_i32, 2]
            .into_iter()
            .flat_map(i32::to_ne_bytes)
            .collect::<Vec<_>>()
    );
    assert!(weak_host.upgrade().is_some());
    drop((array, host));
    assert!(weak_host.upgrade().is_none());
}

#[test]
fn borrowed_weight_scan_checks_pending_ledger_without_storage_and_rechecks_failure() {
    let (_dir, store) = fixture_store();
    let manager = manager(
        store,
        OffloadConfig::new(Some(8), Some(0), 1).unwrap(),
        [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
        [single("a", "a")],
    );
    manager.initialize().unwrap();
    let mut transfer = manager
        .acquire_many_with_transfer(&[(id("a"), 1)], MemoryTier::Device)
        .unwrap();
    // Keep real submitted resources owned while reproducing the ledger-before-map
    // boundary. The live transfer and moved entry both remain alive; no credit.
    let removed = manager.inner.state.lock().unwrap().storage.remove(&id("a"));
    let mut calls = 0;
    assert!(!cold(
        || manager.try_visit_retained_storage(&mut |_| -> Result<(), InspectError> {
            calls += 1;
            Ok(())
        })
    )
    .unwrap());
    assert_eq!(calls, 0);
    {
        let mut state = manager.inner.state.lock().unwrap();
        assert!(state
            .control
            .ledger()
            .copy_status(&id("a"), MemoryTier::Device)
            .unwrap()
            .unwrap()
            .in_flight()
            .is_some());
        if let Some(removed) = removed {
            assert!(state.storage.insert(id("a"), removed).is_none());
        }
    }
    transfer.synchronize().unwrap();
    drop(transfer);
    let mut visits = 0;
    assert!(!manager
        .try_visit_retained_storage(&mut |_| -> Result<(), InspectError> {
            visits += 1;
            // Failure-only injection models the recovery worker's independently
            // published flag. It grants no completion or retained-work release.
            manager
                .inner
                .failed_transfer
                .store(true, std::sync::atomic::Ordering::Release);
            Ok(())
        })
        .unwrap());
    assert!(visits > 0);
}

#[test]
fn borrowed_weight_sources_visit_primary_and_all_unit_aliases() {
    use eredu_checkpoint::store::{MemoryWeightStore, SharedCheckpointSource};
    let memory = |value: i32, capacity: usize| {
        let mut bytes = Vec::with_capacity(capacity);
        bytes.extend([value, -value].into_iter().flat_map(i32::to_ne_bytes));
        Arc::new(
            MemoryWeightStore::from_safetensors([("a".into(), Dtype::I32, vec![2], bytes)])
                .unwrap(),
        ) as SharedCheckpointSource
    };
    let primary = memory(3, 257);
    let other = memory(7, 129);
    let mut expected = primary.source_storage().unwrap().unwrap();
    expected
        .merge(other.source_storage().unwrap().unwrap())
        .unwrap();
    let manager = ResidencyManager::new_shared_sources(
        primary.clone(),
        BTreeMap::from([(id("alias"), primary), (id("other"), other)]),
        OffloadPlan::new(
            OffloadConfig::new(Some(24), None, 1).unwrap(),
            [
                spec("target", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                spec("alias", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                spec("other", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            ],
        )
        .unwrap(),
        [
            single("target", "a"),
            single("alias", "a"),
            single("other", "a"),
        ],
        cpu_stream(),
        cpu_stream(),
    )
    .unwrap();
    let mut sources = Vec::with_capacity(3);
    assert!(cold(|| manager.try_visit_retained_storage(
        &mut |value| -> Result<(), InspectError> {
            let RetainedStorageRef::Source(value) = value else {
                panic!("no materialization")
            };
            sources.push(value.retain());
            Ok(())
        }
    ))
    .unwrap());
    assert_eq!(sources.len(), 3);
    assert_eq!(sources[0].identity(), sources[1].identity());
    let actual: BTreeMap<_, _> = sources
        .iter()
        .map(|source| (source.identity(), source.bytes()))
        .collect();
    assert_eq!(actual, expected.capacities().collect());
    drop((expected, manager));
    assert!(sources.iter().all(|source| source.bytes() > 8));
}

#[derive(Debug)]
enum CallbackError {
    Inspection(InspectError),
    Sentinel(u32),
}
impl From<InspectError> for CallbackError {
    fn from(value: InspectError) -> Self {
        Self::Inspection(value)
    }
}

#[test]
fn borrowed_weight_callback_error_and_unwind_preserve_retained_prefix() {
    for panic in [false, true] {
        let (_dir, manager) = loaded();
        let mut host = None;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            manager.try_visit_retained_storage(&mut |value| -> Result<(), CallbackError> {
                let RetainedStorageRef::RetainedHost(value) = value else {
                    panic!("host is first")
                };
                host = Some(value.clone());
                if panic {
                    std::panic::panic_any(47_u32);
                }
                Err(CallbackError::Sentinel(43))
            })
        }));
        if panic {
            assert_eq!(*result.unwrap_err().downcast::<u32>().unwrap(), 47);
        } else {
            assert!(matches!(result.unwrap(), Err(CallbackError::Sentinel(43))));
        }
        drop(manager);
        assert_eq!(
            host.unwrap().as_bytes().unwrap(),
            [1_i32, 2]
                .into_iter()
                .flat_map(i32::to_ne_bytes)
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn borrowed_weight_state_contention_returns_busy_before_any_callback() {
    let (_dir, manager) = loaded();
    let other = manager.clone();
    let (locked_tx, locked_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let thread = std::thread::spawn(move || {
        let _state = other.inner.state.lock().unwrap();
        locked_tx.send(()).unwrap();
        release_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    });
    locked_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    let mut called = false;
    let result = manager.try_visit_retained_storage(&mut |_| -> Result<(), InspectError> {
        called = true;
        Ok(())
    });
    release_tx.send(()).unwrap();
    thread.join().unwrap();
    assert!(matches!(result, Err(InspectError::Busy)));
    assert!(!called);
    assert!(manager
        .try_visit_retained_storage(&mut |_| Ok::<_, InspectError>(()))
        .unwrap());
}

#[derive(Clone, Copy)]
enum SourceEnd {
    Incomplete,
    Error,
    Panic,
}
struct SourceWithTerminal {
    memory: eredu_checkpoint::store::MemoryWeightStore,
    end: SourceEnd,
}
impl CheckpointSource for SourceWithTerminal {
    fn source_keys(&self) -> Vec<String> {
        self.memory.source_keys()
    }
    fn source_metadata(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorMetadata, eredu_checkpoint::store::StoreError> {
        self.memory.source_metadata(key)
    }
    fn acquire_lease(
        &self,
        request: eredu_checkpoint::store::TensorReadRequest,
    ) -> Result<eredu_checkpoint::store::CheckpointLease, eredu_checkpoint::store::StoreError> {
        self.memory.acquire_lease(request)
    }
    fn source_diagnostics(
        &self,
    ) -> Result<eredu_checkpoint::store::WeightStoreDiagnostics, eredu_checkpoint::store::StoreError>
    {
        self.memory.source_diagnostics()
    }
    fn visit_source_storage(
        &self,
        visitor: &mut dyn FnMut(eredu_checkpoint::store::SourceStorageRef<'_>),
    ) -> Result<bool, eredu_checkpoint::store::StoreError> {
        self.memory.visit_source_storage(visitor)?;
        self.memory.visit_source_storage(visitor)?;
        match self.end {
            SourceEnd::Incomplete => Ok(false),
            SourceEnd::Error => Err(eredu_checkpoint::store::StoreError::Internal(
                "source after callback".into(),
            )),
            SourceEnd::Panic => std::panic::panic_any(53_u32),
        }
    }
}
fn terminal_source(end: SourceEnd) -> eredu_checkpoint::store::SharedCheckpointSource {
    Arc::new(SourceWithTerminal {
        memory: eredu_checkpoint::store::MemoryWeightStore::from_safetensors([(
            "a".into(),
            Dtype::I32,
            vec![2],
            [5_i32, -7].into_iter().flat_map(i32::to_ne_bytes).collect(),
        )])
        .unwrap(),
        end,
    })
}
fn source_only_manager(
    source: eredu_checkpoint::store::SharedCheckpointSource,
    later: eredu_checkpoint::store::SharedCheckpointSource,
) -> ResidencyManager {
    ResidencyManager::new_shared_sources(
        source,
        BTreeMap::from([(id("later"), later)]),
        OffloadPlan::new(
            OffloadConfig::new(Some(16), None, 1).unwrap(),
            [
                spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                spec("later", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            ],
        )
        .unwrap(),
        [single("a", "a"), single("later", "a")],
        cpu_stream(),
        cpu_stream(),
    )
    .unwrap()
}

#[test]
fn borrowed_weight_unknown_source_still_visits_later_physical_sources() {
    let unknown = terminal_source(SourceEnd::Incomplete);
    // The same physical source is intentionally repeated under a distinct unit.
    let manager = source_only_manager(unknown.clone(), unknown);
    let mut held = Vec::with_capacity(4);
    assert!(!cold(|| manager.try_visit_retained_storage(
        &mut |value| -> Result<(), InspectError> {
            let RetainedStorageRef::Source(value) = value else {
                panic!("not materialized")
            };
            held.push(value.retain());
            Ok(())
        }
    ))
    .unwrap());
    assert_eq!(held.len(), 4);
    assert!(held
        .iter()
        .all(|value| value.identity() == held[0].identity()));
    drop(manager);
    assert_eq!(held[0].bytes(), 8);
}

struct ErrorDropProbe {
    manager: ResidencyManager,
    unlocked: Arc<std::sync::atomic::AtomicBool>,
}
impl Drop for ErrorDropProbe {
    fn drop(&mut self) {
        // Poison is expected after provider unwind; it still proves the lock
        // is no longer held. Never wait or reenter a native operation here.
        let unlocked = !matches!(
            self.manager.inner.state.try_lock(),
            Err(std::sync::TryLockError::WouldBlock)
        );
        self.unlocked
            .store(unlocked, std::sync::atomic::Ordering::Release);
    }
}
enum ErrorWithDrop {
    Sentinel(ErrorDropProbe),
    Inspection(InspectError),
}
impl From<InspectError> for ErrorWithDrop {
    fn from(error: InspectError) -> Self {
        Self::Inspection(error)
    }
}

#[test]
fn borrowed_weight_source_error_and_provider_unwind_drop_callback_cause_after_unlock() {
    for end in [SourceEnd::Error, SourceEnd::Panic] {
        let (_dir, later) = fixture_store();
        let manager = source_only_manager(terminal_source(end), later);
        let unlocked = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut cause = Some(ErrorWithDrop::Sentinel(ErrorDropProbe {
            manager: manager.clone(),
            unlocked: unlocked.clone(),
        }));
        let mut held = Vec::with_capacity(1);
        let mut calls = 0;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            manager.try_visit_retained_storage(&mut |value| -> Result<(), ErrorWithDrop> {
                calls += 1;
                let RetainedStorageRef::Source(value) = value else {
                    panic!("not materialized")
                };
                held.push(value.retain());
                Err(cause
                    .take()
                    .expect("callback must stop after first failure"))
            })
        }));
        assert_eq!(calls, 1);
        match end {
            SourceEnd::Error => {
                let error = result.unwrap().err().unwrap();
                assert!(
                    matches!(error, ErrorWithDrop::Sentinel(_)),
                    "secondary source error must not replace callback cause"
                );
                assert!(!unlocked.load(std::sync::atomic::Ordering::Acquire));
                drop(error);
            }
            SourceEnd::Panic => assert_eq!(*result.err().unwrap().downcast::<u32>().unwrap(), 53),
            SourceEnd::Incomplete => unreachable!(),
        }
        assert!(unlocked.load(std::sync::atomic::Ordering::Acquire));
        assert_eq!(held.len(), 1);
        drop(manager);
        assert_eq!(held[0].bytes(), 8);
    }
}
