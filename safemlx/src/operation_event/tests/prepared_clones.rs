use super::*;
use crate::PreparedArrayClone;

fn clone_completion(stream: &Stream) {
    let _runtime = PrefillRootsRuntime::prepare_for_stream(stream, stream).unwrap();
    let input = Array::from_slice(&[2.0_f32, -3.0, 7.0], &[3]);
    input.evaluated().unwrap();
    let mut slot = PreparedArrayClone::try_new().unwrap();
    let unused = PreparedArrayClone::try_new().unwrap();
    let original = Original::for_stream(stream, 1);
    let observer = OriginalScopeObserver::require_current().unwrap();
    let graph = original.graph.occupied_bytes();
    let records = original._records.occupied_bytes();
    let source = input.add(&input, stream).unwrap();
    crate::register_thread_runtime_housekeeping(hook);
    let hooks = Hook;
    HOOKS.with(|value| value.set(0));
    let cloned = slot.fill_in_original_scope(&source, &observer).unwrap();
    let error = slot.fill_in_original_scope(&source, &observer).unwrap_err();
    assert_eq!(
        error.scoped_evaluation_cause(),
        Some(crate::error::ScopedEvaluationCause::Domain)
    );
    drop(error);
    let event = crate::transforms::async_eval_with_original_operation_event_on_stream_exact(
        std::slice::from_ref(&cloned),
        &observer,
        stream,
    )
    .unwrap();
    event.synchronize().unwrap();
    assert_eq!(HOOKS.with(Cell::get), 0);
    drop(hooks);
    settle(&observer);
    observer.validate_completed_array(&cloned).unwrap();
    assert_eq!(
        cloned
            .completed_in_original_scope(&observer)
            .unwrap()
            .try_as_slice::<f32>()
            .unwrap(),
        &[4.0, -6.0, 14.0]
    );
    assert_eq!(
        source.allocation_info().unwrap(),
        cloned.allocation_info().unwrap()
    );
    crate::try_with_submission_retirement(|| drop((event, cloned, source, slot, unused))).unwrap();
    settle(&observer);
    assert_eq!(original.graph.occupied_bytes(), graph);
    assert_eq!(original._records.occupied_bytes(), records);
}

#[test]
fn prepared_clone_cpu_nonzero_roots_retire_through_ordinary_array_drop() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    clone_completion(&stream);
}
#[cfg(all(feature = "metal", target_vendor = "apple", not(feature = "cuda")))]
#[test]
fn prepared_clone_metal_nonzero_roots_retire_through_ordinary_array_drop() {
    with_prepared_metal_stream(
        "operation_event::tests::prepared_clones::prepared_clone_metal_nonzero_roots_retire_through_ordinary_array_drop",
        clone_completion,
    );
}

#[test]
fn prepared_clone_busy_and_foreign_owner_refuse_without_consuming_slot() {
    let input = Array::from_slice(&[7.0_f32], &[1]);
    input.evaluated().unwrap();
    let mut slot = PreparedArrayClone::try_new().unwrap();
    let mut first = Original::new();
    let first_observer = OriginalScopeObserver::require_current().unwrap();
    let (entered, ready) = mpsc::channel();
    let (release, released) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        crate::try_with_submission_retirement(|| {
            entered.send(()).unwrap();
            released.recv().unwrap();
        })
        .unwrap();
    });
    ready.recv().unwrap();
    let error = slot
        .fill_in_original_scope(&input, &first_observer)
        .unwrap_err();
    assert_eq!(
        error.scoped_evaluation_cause(),
        Some(crate::error::ScopedEvaluationCause::RuntimeBusy)
    );
    drop(error);
    release.send(()).unwrap();
    worker.join().unwrap();
    first.scope.seal();
    let second = Original::new();
    let second_observer = OriginalScopeObserver::require_current().unwrap();
    let error = slot
        .fill_in_original_scope(&input, &first_observer)
        .unwrap_err();
    assert_eq!(
        error.scoped_evaluation_cause(),
        Some(crate::error::ScopedEvaluationCause::Domain)
    );
    drop(error);
    let result = slot
        .fill_in_original_scope(&input, &second_observer)
        .unwrap();
    assert_eq!(
        result.allocation_info().unwrap(),
        input.allocation_info().unwrap()
    );
    drop((result, slot));
    settle(&second_observer);
    drop(second);
    settle(&first_observer);
}
