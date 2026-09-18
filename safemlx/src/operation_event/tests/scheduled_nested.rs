use super::*;
use std::sync::Arc;

const VALUES: [f32; 3] = [2.0, -3.0, 7.0];

fn source() -> crate::ImmutableHostTransferBuffer {
    let mut source =
        HostTransferBuffer::new(&[3], Dtype::Float32, HostTransferPolicy::Transfer).unwrap();
    for (bytes, value) in source
        .as_bytes_mut()
        .unwrap()
        .chunks_exact_mut(4)
        .zip(VALUES)
    {
        bytes.copy_from_slice(&value.to_ne_bytes());
    }
    source.freeze()
}

fn bank(observer: &OriginalScopeObserver, lazy: bool, attempts: usize) -> PreparedResidentGraph {
    // Two actual Host leaves and CopyFromHost primitives; the refusal fixture
    // also constructs one Add whose lazy root must never reach scheduled Eval.
    let layout = OperationEvent::resident_graph_layout(2 + usize::from(lazy), 2, 1).unwrap();
    let traversal = OperationEvent::eval_traversal_layout(OperationEvalTraversalLimits {
        roots: 2,
        arrays: 3,
        tape_entries: 2,
        input_edges: 2,
        output_slots: 2,
        streams: 1,
        captures: OperationEvent::eval_record_layout(2, 1, 2)
            .unwrap()
            .capture_slots(),
    })
    .unwrap();
    let mut bank = OperationEvent::prepare_resident_graph(layout, observer).unwrap();
    bank.configure_nested_completions(&traversal, attempts)
        .unwrap();
    bank
}

fn completed_host_roots(device: DeviceType, stream: &Stream) {
    let host = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let _runtime = PrefillRootsRuntime::prepare_for_stream(stream, &host).unwrap();
    let source = source();
    let original = Original::new();
    let observer = OriginalScopeObserver::require_current().unwrap();
    let graph = original.graph.occupied_bytes();
    let records = original._records.occupied_bytes();
    for _ in 0..3 {
        // Two copy submissions followed by exactly one aggregate Synchronizer.
        let bank = bank(&observer, false, 3);
        let (left, first) = source
            .copy_to_array_in_original_scope(stream, &observer)
            .unwrap();
        let (right, second) = source
            .copy_to_array_in_original_scope(stream, &observer)
            .unwrap();
        let completion =
            OperationEvent::submit_nested_scheduled([&left, &right], 2, &observer, stream).unwrap();
        if device == DeviceType::Cpu {
            assert!(
                observer.status().is_settled(),
                "CPU aggregate returns only after its signal task's Record settles"
            );
            assert_eq!(
                original._records.occupied_bytes(),
                records,
                "CPU aggregate retires its actual Records before any caller settlement"
            );
        }
        completion.synchronize().unwrap();
        settle(&observer);
        for value in [&left, &right] {
            assert_eq!(
                value
                    .completed_in_original_scope(&observer)
                    .unwrap()
                    .try_as_slice::<f32>()
                    .unwrap(),
                &VALUES
            );
        }
        // Success consumed the final attempt; completion never refunds it.
        assert!(OperationEvent::validate_nested_completion(2).is_err());
        crate::try_with_submission_retirement(|| drop((bank, right, first, second, completion)))
            .unwrap();
        settle(&observer);
        assert!(
            original.graph.occupied_bytes() > graph,
            "the escaped actual output still retains its graph allocation"
        );
        assert_eq!(original._records.occupied_bytes(), records);
        crate::try_with_submission_retirement(|| drop(left)).unwrap();
        settle(&observer);
        assert_eq!(original.graph.occupied_bytes(), graph);
    }
}

#[test]
fn cpu_scheduled_host_roots_settle_before_bank_restore() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    completed_host_roots(DeviceType::Cpu, &stream);
}

#[cfg(all(feature = "metal", target_vendor = "apple", not(feature = "cuda")))]
#[test]
fn metal_scheduled_host_roots_preserve_async_completion() {
    // Device/scheduler singleton births cannot adopt an ordinary predecessor
    // from another library test. Authenticate their actual original sources in
    // a fresh process, before constructing the exact GPU command encoder.
    const CHILD: &str = "SAFEMLX_SCHEDULED_METAL_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "operation_event::tests::scheduled_nested::metal_scheduled_host_roots_preserve_async_completion",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CHILD, "1")
            .output().unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    let device = crate::PreparedMetalDevice::try_new(())
        .unwrap()
        .try_initialize()
        .unwrap();
    let scheduler = crate::PreparedScheduler::try_new(())
        .unwrap()
        .try_initialize()
        .unwrap();
    let _allocator = crate::PreparedInputAllocator::try_new(())
        .unwrap()
        .try_initialize()
        .unwrap();
    let target = crate::GpuStreamTarget::for_initialized(&device, &scheduler).unwrap();
    let layout = crate::PreparedGpuStream::<()>::layout().unwrap();
    let stream = crate::PreparedGpuStream::with_layout(layout, ())
        .unwrap()
        .try_initialize(target)
        .unwrap();
    stream.try_borrow().unwrap();
    completed_host_roots(DeviceType::Gpu, stream.as_stream());
    stream.try_observe_idle().unwrap();
}

#[test]
fn scheduled_host_roots_reject_lazy_foreign_and_count_mismatch_with_custody() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let _runtime = PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let source = source();
    let mut old = Original::new();
    let foreign = OriginalScopeObserver::require_current().unwrap();
    old.scope.seal();
    drop(old);
    let custody = Arc::new(());
    let weak = Arc::downgrade(&custody);
    let mut original = Original::with_failure_owner(custody);
    let observer = OriginalScopeObserver::require_current().unwrap();
    // Foreign rejection precedes a bank attempt; lazy/count refusals consume
    // their reached attempts, followed by one successful original aggregate.
    let bank = bank(&observer, true, 5);
    let (left, first) = source
        .copy_to_array_in_original_scope(&stream, &observer)
        .unwrap();
    let (right, second) = source
        .copy_to_array_in_original_scope(&stream, &observer)
        .unwrap();
    let foreign_error =
        OperationEvent::submit_nested_scheduled([&left], 1, &foreign, &stream).unwrap_err();
    assert_eq!(
        foreign_error.scoped_evaluation_cause(),
        Some(crate::error::ScopedEvaluationCause::Domain)
    );
    let lazy = left.add(&right, &stream).unwrap();
    let lazy_error =
        OperationEvent::submit_nested_scheduled([&lazy], 1, &observer, &stream).unwrap_err();
    assert_eq!(
        lazy_error.scoped_evaluation_cause(),
        Some(crate::error::ScopedEvaluationCause::Domain)
    );
    let mismatch =
        OperationEvent::submit_nested_scheduled([&left], 2, &observer, &stream).unwrap_err();
    assert_eq!(
        mismatch.scoped_evaluation_cause(),
        Some(crate::error::ScopedEvaluationCause::Domain)
    );
    assert!(!observer.status().failed());
    let completion =
        OperationEvent::submit_nested_scheduled([&left, &right], 2, &observer, &stream).unwrap();
    completion.synchronize().unwrap();
    settle(&observer);
    assert_eq!(
        left.completed_in_original_scope(&observer)
            .unwrap()
            .try_as_slice::<f32>()
            .unwrap(),
        &VALUES
    );
    assert!(OperationEvent::validate_nested_completion(1).is_err());
    crate::try_with_submission_retirement(|| {
        drop((
            lazy, left, right, first, second, completion, bank, lazy_error,
        ))
    })
    .unwrap();
    settle(&observer);
    original.scope.seal();
    drop((observer, original, foreign_error, foreign));
    crate::reclaim_allocation_owners();
    assert!(
        weak.upgrade().is_some(),
        "the exact returned failure retains source custody"
    );
    drop(mismatch);
    crate::reclaim_allocation_owners();
    assert!(weak.upgrade().is_none());
}
