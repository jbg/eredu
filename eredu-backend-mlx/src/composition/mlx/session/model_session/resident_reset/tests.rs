//! Actual native publication with a genuine core claim. Ordinary loading,
//! execution, synchronization and reclamation are external test preparation;
//! this does not establish the pending public native readiness contract.
use super::super::{disk_layerwise_tests as disk, host_layerwise_tests as host};
use super::*;
use eredu_core::{SessionResetClaim, SessionResetLimits};
use eredu_runtime::working_memory::{InferenceStateRetention, ResidentResetError};

struct Slot {
    target: std::rc::Weak<Cell<bool>>,
    reject: bool,
    retired: Rc<Cell<usize>>,
}
thread_local! { static CURRENT: RefCell<Option<Slot>> = const { RefCell::new(None) }; }

struct Scope(Rc<Cell<usize>>);
impl Scope {
    fn enter(session: &MlxModelSession) -> Self {
        let retired = Rc::new(Cell::new(0));
        CURRENT.with_borrow_mut(|slot| {
            assert!(slot.is_none());
            *slot = Some(Slot {
                target: Rc::downgrade(&session.poison),
                reject: false,
                retired: Rc::clone(&retired),
            });
        });
        Self(retired)
    }
    fn reject(&self, reject: bool) {
        CURRENT.with_borrow_mut(|slot| slot.as_mut().unwrap().reject = reject);
    }
    fn retired(&self, expected: usize) {
        crate::backend::submission_recovery::wait_for_retirement(|| {
            reclaim();
            self.0.get() == expected
        });
        assert_eq!(self.0.get(), expected);
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        let old = CURRENT.with_borrow_mut(Option::take);
        drop(old);
    }
}

/// Observes destruction of this node's displaced Rust fields, not native
/// completion or an allocator event. Separate pool checks follow actual reaping.
pub(super) struct NodeWitness(Rc<Cell<usize>>);
impl Drop for NodeWitness {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}
pub(super) fn node_witness() -> Option<NodeWitness> {
    CURRENT.with_borrow(|slot| slot.as_ref().map(|s| NodeWitness(Rc::clone(&s.retired))))
}
fn selected(slot: &Slot, session: &MlxModelSession) -> bool {
    slot.target.as_ptr() == Rc::as_ptr(&session.poison)
}
pub(super) fn reject_constructed(session: &MlxModelSession) -> bool {
    CURRENT.with_borrow(|slot| {
        slot.as_ref()
            .is_some_and(|s| selected(s, session) && s.reject)
    })
}
pub(in super::super) fn admitted(
    backend: &MlxBackend<'_>,
    session: &mut MlxModelSession,
    claim: SessionResetClaim<'_>,
) -> Result<(), BackendFailure> {
    let selected = CURRENT.with_borrow(|slot| slot.as_ref().is_some_and(|s| selected(s, session)));
    if !selected {
        return Err(BackendFailure::new(
            BackendFailureKind::Unsupported,
            eredu_core::SessionResetRejection::Unsupported,
        ));
    }
    session.publish_prepared_resident_reset(backend.memory_pool(), claim)
}

fn reclaim() {
    // Healthy terminal status alone does not remove the native Record registry.
    let _ = safemlx::try_retire_completed_submissions();
    disk::reclaim();
}
fn settle(pool: &WorkingMemoryPool, bytes: u64) {
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        pool.used_bytes().unwrap() == bytes && pool.unquoted_owner_count().unwrap() == 0
    });
    assert_eq!(pool.used_bytes().unwrap(), bytes);
}
fn required(runtime: &ModelRuntime<MlxBackend<'_>>, pool: &WorkingMemoryPool) -> u64 {
    runtime
        .session()
        .resident_reset_plan(pool)
        .unwrap()
        .publication_required_bytes::<Prepared>()
        .unwrap()
}
// Complete actual arrays in the state's canonical layer/key/value order.
// The generic fixed_numeric_snapshot hook is intentionally empty for whole KV.
// Host reads and their buffers are ordinary fixture observation, outside reset.
fn numeric(runtime: &ModelRuntime<MlxBackend<'_>>) -> Vec<(Vec<i32>, Vec<f32>)> {
    let source = runtime
        .session()
        .payload
        .model
        .erased()
        .resident_reset_source()
        .unwrap();
    source
        .state()
        .retained_arrays()
        .into_iter()
        .map(|array| {
            (
                array.shape().to_vec(),
                array.evaluated().unwrap().try_to_vec::<f32>().unwrap(),
            )
        })
        .collect()
}
fn warm(runtime: &mut ModelRuntime<MlxBackend<'_>>, pool: &WorkingMemoryPool, controlled: bool) {
    let tokens = disk::outputs(
        runtime,
        disk::tokens(),
        disk::config(0.0, 2, u64::MAX),
        controlled,
    );
    assert_eq!(disk::token_ids(&tokens).len(), 4);
    drop(tokens);
    runtime.synchronize().unwrap();
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        pool.unquoted_owner_count().unwrap() == 0
    });
}
fn runtime(
    stream: &Stream,
    pool: &WorkingMemoryPool,
    route: usize,
) -> (ModelRuntime<MlxBackend<'static>>, tempfile::TempDir) {
    match route {
        0 => host::runtime(stream, pool, None),
        1 => host::runtime(stream, pool, Some(1)),
        2 => disk::load_runtime(stream, pool, true),
        _ => unreachable!(),
    }
}

#[test]
fn actual_selected_kv_reset_exact_short_and_publication_preserve_source_then_clear_state() {
    for route in 0..3 {
        for controlled in [false, true] {
            let stream =
                Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
            let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
            let (mut runtime, _artifact) = runtime(&stream, &pool, route);
            let empty = runtime.session().payload.model.erased().state_snapshot();
            warm(&mut runtime, &pool, controlled);
            let before_values = numeric(&runtime);
            assert_eq!(
                before_values.len(),
                6,
                "both arrays of all three fixture KV layers"
            );
            assert!(before_values
                .iter()
                .any(|(_, v)| v.iter().any(|v| *v != 0.0)));
            assert!(runtime
                .session()
                .payload
                .model
                .erased()
                .resident_copy_input_identity()
                .unwrap()
                .is_some());
            let source = runtime
                .session()
                .payload
                .model
                .erased()
                .resident_reset_source()
                .unwrap();
            let old_table = source
                .state()
                .resident_reset_layers()
                .metadata()
                .identity()
                .registry_key()
                .clone();
            let layout = source
                .state()
                .resident_reset_layout()
                .identity()
                .registry_key()
                .clone();
            let start = source.state().resident_reset_global_start();
            let revision = source.state().inference_retention().revision().clone();
            let scope = Scope::enter(runtime.session());
            let bytes = required(&runtime, &pool);
            let before = pool.used_bytes().unwrap();
            let error = runtime
                .reset_admitted(SessionResetLimits::new(before + bytes - 1))
                .unwrap_err();
            let typed = disk::cause::<ResidentResetError<MlxKeyValueState>>(&error).unwrap();
            assert_eq!(typed.retained_bytes(), 0);
            assert_eq!(
                scope.0.get(),
                0,
                "no retirement node before original acceptance"
            );
            assert_eq!(pool.used_bytes().unwrap(), before);
            assert_eq!(numeric(&runtime), before_values);
            let source = runtime
                .session()
                .payload
                .model
                .erased()
                .resident_reset_source()
                .unwrap();
            assert_eq!(
                source
                    .state()
                    .resident_reset_layers()
                    .metadata()
                    .identity()
                    .registry_key(),
                &old_table
            );
            assert_eq!(source.state().inference_retention().revision(), &revision);
            assert!(runtime
                .session()
                .payload
                .model
                .erased()
                .resident_copy_input_identity()
                .unwrap()
                .is_some());
            drop(error);
            runtime
                .reset_admitted(SessionResetLimits::new(before + bytes))
                .unwrap();
            assert_eq!(
                runtime.session().payload.model.erased().state_snapshot(),
                empty
            );
            assert!(numeric(&runtime).is_empty());
            assert!(runtime
                .session()
                .payload
                .model
                .erased()
                .resident_copy_input_identity()
                .unwrap()
                .is_none());
            let source = runtime
                .session()
                .payload
                .model
                .erased()
                .resident_reset_source()
                .unwrap();
            assert_ne!(
                source
                    .state()
                    .resident_reset_layers()
                    .metadata()
                    .identity()
                    .registry_key(),
                &old_table
            );
            assert_eq!(
                source
                    .state()
                    .resident_reset_layout()
                    .identity()
                    .registry_key(),
                &layout
            );
            assert_eq!(source.state().resident_reset_global_start(), start);
            assert_ne!(source.state().inference_retention().revision(), &revision);
            assert_eq!(
                pool.pin_original_reset_slots(source.state().resident_reset_layers().metadata())
                    .unwrap()
                    .original_bytes(),
                bytes
            );
            scope.retired(1);
            // These payload-free keys do not retain an old reset account.
            drop((old_table, layout, revision));
            drop(runtime);
            settle(&pool, 0);
        }
    }
}

#[test]
fn actual_original_table_second_reset_has_no_predecessor_account_chain() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = runtime(&stream, &pool, 0);
    runtime.synchronize().unwrap();
    let scope = Scope::enter(runtime.session());
    let first = required(&runtime, &pool);
    runtime
        .reset_admitted(SessionResetLimits::new(u64::MAX))
        .unwrap();
    scope.retired(1);
    let metadata = runtime
        .session()
        .payload
        .model
        .erased()
        .resident_reset_source()
        .unwrap()
        .state()
        .resident_reset_layers()
        .metadata()
        .clone();
    assert_eq!(
        pool.pin_original_reset_slots(&metadata)
            .unwrap()
            .original_bytes(),
        first
    );
    let second = required(&runtime, &pool);
    runtime
        .reset_admitted(SessionResetLimits::new(u64::MAX))
        .unwrap();
    scope.retired(2);
    let source = runtime
        .session()
        .payload
        .model
        .erased()
        .resident_reset_source()
        .unwrap();
    let current = source.state().resident_reset_layers().metadata();
    assert_ne!(
        current.identity().registry_key(),
        metadata.identity().registry_key()
    );
    assert_eq!(
        pool.pin_original_reset_slots(current)
            .unwrap()
            .original_bytes(),
        second
    );
    let before_drop = pool.used_bytes().unwrap();
    assert!(before_drop >= first + second);
    drop(metadata);
    settle(&pool, before_drop - first);
    assert_eq!(
        pool.pin_original_reset_slots(
            runtime
                .session()
                .payload
                .model
                .erased()
                .resident_reset_source()
                .unwrap()
                .state()
                .resident_reset_layers()
                .metadata()
        )
        .unwrap()
        .original_bytes(),
        second
    );
    drop(runtime);
    settle(&pool, 0);
}

#[test]
fn actual_constructed_rejection_keeps_destination_and_original_allowance() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = runtime(&stream, &pool, 0);
    warm(&mut runtime, &pool, false);
    let scope = Scope::enter(runtime.session());
    let before_values = numeric(&runtime);
    let source = runtime
        .session()
        .payload
        .model
        .erased()
        .resident_reset_source()
        .unwrap();
    let key = source
        .state()
        .resident_reset_layers()
        .metadata()
        .identity()
        .registry_key()
        .clone();
    let revision = source.state().inference_retention().revision().clone();
    let before = pool.used_bytes().unwrap();
    let bytes = required(&runtime, &pool);
    scope.reject(true);
    let error = runtime
        .reset_admitted(SessionResetLimits::new(u64::MAX))
        .unwrap_err();
    assert_eq!(
        disk::cause::<ResidentResetError<MlxKeyValueState>>(&error)
            .unwrap()
            .retained_bytes(),
        bytes
    );
    assert!(matches!(
        disk::cause::<WorkingMemoryError>(&error),
        Some(WorkingMemoryError::ExecutionFenced)
    ));
    scope.retired(1);
    assert_eq!(pool.used_bytes().unwrap(), before + bytes);
    assert_eq!(numeric(&runtime), before_values);
    let source = runtime
        .session()
        .payload
        .model
        .erased()
        .resident_reset_source()
        .unwrap();
    assert_eq!(
        source
            .state()
            .resident_reset_layers()
            .metadata()
            .identity()
            .registry_key(),
        &key
    );
    assert_eq!(source.state().inference_retention().revision(), &revision);
    assert!(runtime
        .session()
        .payload
        .model
        .erased()
        .resident_copy_input_identity()
        .unwrap()
        .is_some());
    drop(error);
    settle(&pool, before);
    scope.reject(false);
    runtime
        .reset_admitted(SessionResetLimits::new(u64::MAX))
        .unwrap();
    scope.retired(2);
    drop((key, revision));
    drop(runtime);
    settle(&pool, 0);
}

#[test]
fn payload_alias_rejects_before_construction_and_native_array_alias_survives_displacement() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = runtime(&stream, &pool, 0);
    warm(&mut runtime, &pool, false);
    let scope = Scope::enter(runtime.session());
    let alias = runtime.session().payload.clone();
    let before = pool.used_bytes().unwrap();
    let error = runtime
        .reset_admitted(SessionResetLimits::new(u64::MAX))
        .unwrap_err();
    assert!(matches!(
        disk::cause::<WorkingMemoryError>(&error),
        Some(WorkingMemoryError::ResetAdmissionBusy)
    ));
    assert_eq!(scope.0.get(), 0);
    assert_eq!(pool.used_bytes().unwrap(), before);
    drop((alias, error));
    let source = runtime
        .session()
        .payload
        .model
        .erased()
        .resident_reset_source()
        .unwrap();
    let values = source.state().retained_arrays();
    let array = Array::clone(values[0]);
    drop(values);
    let expected = array.evaluated().unwrap().try_to_vec::<f32>().unwrap();
    assert!(expected.iter().any(|v| *v != 0.0));
    runtime
        .reset_admitted(SessionResetLimits::new(u64::MAX))
        .unwrap();
    scope.retired(1);
    assert_eq!(
        array.evaluated().unwrap().try_to_vec::<f32>().unwrap(),
        expected
    );
    drop(runtime);
    reclaim();
    assert!(
        pool.used_bytes().unwrap() > 0,
        "original backing custody follows surviving actual native alias"
    );
    assert_eq!(
        array.evaluated().unwrap().try_to_vec::<f32>().unwrap(),
        expected
    );
    drop(array);
    settle(&pool, 0);
}

#[test]
fn installed_original_table_keeps_unfunded_publication_consumer_closed() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = runtime(&stream, &pool, 0);
    runtime.synchronize().unwrap();
    let before = pool.used_bytes().unwrap();
    let closed = runtime
        .reset_admitted(SessionResetLimits::new(u64::MAX))
        .unwrap_err();
    assert!(matches!(
        disk::cause::<eredu_core::SessionResetRejection>(&closed),
        Some(eredu_core::SessionResetRejection::Unsupported)
    ));
    assert_eq!(pool.used_bytes().unwrap(), before);
    drop(closed);
    let scope = Scope::enter(runtime.session());
    runtime
        .reset_admitted(SessionResetLimits::new(u64::MAX))
        .unwrap();
    scope.retired(1);
    let (storage, _) = runtime
        .session()
        .payload
        .retained_idle_storage()
        .unwrap()
        .into_parts();
    let before = pool.used_bytes().unwrap();
    let error = match storage.pin_registered(&pool) {
        Ok(_) => panic!("pool-only original-table population is not funded"),
        Err(error) => error,
    };
    assert!(matches!(
        disk::cause::<WorkingMemoryError>(&error),
        Some(WorkingMemoryError::UnknownBound)
    ));
    assert_eq!(pool.used_bytes().unwrap(), before);
    drop((error, storage));
    drop(runtime);
    settle(&pool, 0);
}

#[cfg(test)]
pub(in crate::composition::mlx::session::model_session) mod consumer;
