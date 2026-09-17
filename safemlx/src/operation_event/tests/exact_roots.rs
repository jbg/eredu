use super::*;
use crate::transforms::async_eval_with_original_operation_event_on_stream_exact as exact;
use std::error::Error as _;

#[test]
fn exact_root_layout_does_not_enter_runtime_and_separates_object_controls() {
    crate::register_thread_runtime_housekeeping(hook);
    let hooks = Hook;
    HOOKS.with(|value| value.set(0));
    let zero = OperationEvent::root_storage_layout(0).unwrap();
    let one = OperationEvent::root_storage_layout(1).unwrap();
    assert_eq!(zero.root_count(), 0);
    assert_eq!(zero.roots_bytes(), 0);
    assert_eq!(zero.roots_graph_extent(), 0);
    assert_eq!(zero.graph_blocks(), 1);
    assert_eq!(one.graph_blocks(), 2);
    assert_eq!(
        one.graph_request_extent(),
        one.object_graph_extent() + one.roots_graph_extent()
    );
    assert_eq!(
        OperationEvent::control_bytes().unwrap(),
        OperationEvent::non_object_control_bytes().unwrap() + one.object_bytes()
    );
    assert!(OperationEvent::root_storage_layout(usize::MAX).is_none());
    assert_eq!(HOOKS.with(Cell::get), 0);
    drop(hooks);
}

#[test]
fn exact_root_reserve_error_retires_partial_native_wrapper_and_keeps_actual_cause() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let _runtime = PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let original = Original::new();
    let observer = OriginalScopeObserver::require_current().unwrap();
    let baseline = original.graph.occupied_bytes();
    let records = original._records.occupied_bytes();
    let element = OperationEvent::root_storage_layout(1)
        .unwrap()
        .roots_bytes();
    // The actual fixture's configured Graph capacity, not a guessed root cap.
    let count = GRAPH_CAPACITY / element + 1;
    assert!(
        OperationEvent::root_storage_layout(count)
            .unwrap()
            .roots_bytes()
            > GRAPH_CAPACITY
    );
    let error = match ScopedOperation::for_exact_roots(observer.clone(), &stream, count) {
        Ok(_) => panic!("oversized actual root buffer accepted"),
        Err(error) => error,
    };
    assert_eq!(
        error.scoped_evaluation_cause(),
        Some(crate::error::ScopedEvaluationCause::Failed)
    );
    assert!(error.to_string().contains("scoped"));
    let native = error.source().unwrap().source().unwrap();
    assert_eq!(native.to_string(), "graph metadata capacity exhausted");
    assert_eq!(original.graph.occupied_bytes(), baseline);
    assert_eq!(original._records.occupied_bytes(), records);
    assert!(!observer.status().failed()); // no Eval Record was accepted
    drop(original);
    drop(observer);
    assert_eq!(
        error.source().unwrap().source().unwrap().to_string(),
        "graph metadata capacity exhausted"
    );
}

fn exact_frontier(kind: DeviceType) {
    let stream = Stream::new_with_device(&Device::new(kind, 0));
    let _runtime = PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let input = Array::from_slice(&[2.0_f32, -3.0, 7.0], &[3]);
    input.evaluated().unwrap();
    let mut original = Original::new();
    let observer = OriginalScopeObserver::require_current().unwrap();
    let graph = original.graph.occupied_bytes();
    let records = original._records.occupied_bytes();
    for empty in [true, false] {
        let roots = [
            input.clone(),
            input.clone(),
            input.add(&input, &stream).unwrap(),
        ];
        let selected = if empty { &roots[..0] } else { &roots[..] };
        crate::register_thread_runtime_housekeeping(hook);
        let hooks = Hook;
        HOOKS.with(|value| value.set(0));
        let event = exact(selected, &observer, &stream).unwrap();
        event.synchronize().unwrap();
        assert_eq!(HOOKS.with(Cell::get), 0);
        drop(hooks);
        settle(&observer);
        if !empty {
            for root in &roots {
                observer.validate_completed_array(root).unwrap();
            }
            assert_eq!(
                roots[2]
                    .completed_in_original_scope(&observer)
                    .unwrap()
                    .try_as_slice::<f32>()
                    .unwrap(),
                &[4.0, -6.0, 14.0]
            );
            let identity = exact(&roots, &observer, &stream).unwrap();
            identity.synchronize().unwrap();
            settle(&observer);
            drop(identity);
        }
        crate::try_with_submission_retirement(|| drop((event, roots))).unwrap();
        settle(&observer);
        assert_eq!(original.graph.occupied_bytes(), graph);
        assert_eq!(original._records.occupied_bytes(), records);
    }
    original.scope.seal();
    settle(&observer);
}

#[test]
fn exact_root_cpu_slice_keeps_duplicates_values_and_empty_frontier_without_hooks() {
    exact_frontier(DeviceType::Cpu);
}

#[cfg(all(feature = "metal", target_vendor = "apple", not(feature = "cuda")))]
#[test]
fn exact_root_metal_slice_keeps_duplicates_values_and_empty_frontier_without_hooks() {
    exact_frontier(DeviceType::Gpu);
}

#[test]
fn exact_root_busy_and_ended_owner_refuse_without_losing_later_completion() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let _runtime = PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let mut original = Original::new();
    let observer = OriginalScopeObserver::require_current().unwrap();
    let graph = original.graph.occupied_bytes();
    let (ready_tx, ready_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let thread = std::thread::spawn(move || {
        let _guard = runtime_lock::enter();
        ready_tx.send(()).unwrap();
        release_rx.recv_timeout(Duration::from_secs(5)).is_ok()
    });
    ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    let error = exact(&[], &observer, &stream).err().unwrap();
    let sent = release_tx.send(()).is_ok();
    assert!(thread.join().unwrap() && sent);
    assert_eq!(
        error.scoped_evaluation_cause(),
        Some(crate::error::ScopedEvaluationCause::RuntimeBusy)
    );
    assert_eq!(original.graph.occupied_bytes(), graph);
    let completion = exact(&[], &observer, &stream).unwrap();
    completion.synchronize().unwrap();
    settle(&observer);
    original.scope.seal();
    let occupied = original.graph.occupied_bytes();
    assert_eq!(
        exact(&[], &observer, &stream)
            .err()
            .unwrap()
            .scoped_evaluation_cause(),
        Some(crate::error::ScopedEvaluationCause::Domain)
    );
    assert_eq!(original.graph.occupied_bytes(), occupied);
    assert!(completion.is_complete().unwrap());
    drop(completion);
    settle(&observer);
    assert_eq!(original.graph.occupied_bytes(), graph);
    drop(error);
}
