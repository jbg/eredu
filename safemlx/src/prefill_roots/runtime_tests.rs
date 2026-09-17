use super::*;
use crate::{Device, DeviceType, Stream, SubmissionScope};
use std::cell::Cell;

thread_local! { static HOUSEKEEPING: Cell<usize> = const { Cell::new(0) }; }
fn housekeeping() {
    HOUSEKEEPING.with(|calls| calls.set(calls.get() + 1));
}
struct Hook;
impl Drop for Hook {
    fn drop(&mut self) {
        crate::unregister_thread_runtime_housekeeping(housekeeping);
    }
}

#[test]
fn selected_runtime_initializes_exact_new_streams_without_numerical_warmup() {
    let device = Device::new(DeviceType::Cpu, 0);
    let operation = Stream::new_with_device(&device);
    let weights = Stream::new_with_device(&device);
    let runtime = PrefillRootsRuntime::prepare_for_stream(&operation, &weights).unwrap();
    let facts = runtime.baseline().unwrap();
    assert_eq!(facts.selected_streams, 2);
    assert_eq!(facts.cpu_workers, 2);
    assert_eq!(facts.native_threads, 2);
    assert!(facts.scheduler_object_bytes > 0);
    assert!(facts.worker_object_bytes > 0);
    assert!(facts.fixed_header_bytes().unwrap() > facts.worker_object_bytes);
    assert_eq!(facts.unpriced_populations & 7, 7);
    let one = PrefillRootsRuntime::prepare_for_stream(&operation, &operation)
        .unwrap()
        .baseline()
        .unwrap();
    assert_eq!(one.selected_streams, 1);
    assert_eq!(one.cpu_workers, 1);
    assert_eq!(one.native_threads, 1);
    assert_eq!(facts.worker_object_bytes, 2 * one.worker_object_bytes);
    assert_eq!(facts.scheduler_object_bytes, one.scheduler_object_bytes);
    assert_eq!(runtime.clone().baseline(), Some(facts));
    assert_eq!(PrefillRootsRuntime::prepare().baseline(), None);
}

#[test]
fn selected_runtime_original_scope_refusal_precedes_housekeeping_and_keeps_mode() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    crate::register_thread_runtime_housekeeping(housekeeping);
    let _hook = Hook;
    let mut scope = SubmissionScope::try_begin().unwrap();
    scope.enable_scoped_observation().unwrap();
    scope.require_original_native_controls().unwrap();
    HOUSEKEEPING.with(|calls| calls.set(0));
    assert!(matches!(
        PrefillRootsRuntime::prepare_for_stream(&stream, &stream),
        Err(PrefillRuntimePreparationError::ActiveOriginalScope)
    ));
    assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
    assert_eq!(
        scope.enable_original_native_controls(),
        Err(crate::OriginalNativeControlError::MissingGraph)
    );
    drop(scope);
    assert_eq!(
        PrefillRootsRuntime::prepare_for_stream(&stream, &stream)
            .unwrap()
            .baseline()
            .unwrap()
            .cpu_workers,
        1
    );
}
