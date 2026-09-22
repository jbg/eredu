use super::*;

mod affine_quantize_submission;
mod eval_traversal;
mod exact_roots;
mod graph_construction;
mod original_array;
mod owned_host_copy;
mod prepared_clones;
mod quantize_submission;
mod scheduled_nested;
use crate::{
    Device, DeviceType, Dtype, HostTransferBuffer, HostTransferPolicy, PrefillRootsRuntime,
    PreparedPrefillFailure, PreparedSubmissionGraphQuota, PreparedSubmissionRecordQuota,
    PreparedSubmissionScopeOwner, RetainedPrefillFailure, ScopedSubmissionProgress,
    SubmissionGraphQuota, SubmissionRecordQuota, SubmissionRetirement, SubmissionScope,
};
use std::{
    cell::Cell,
    sync::mpsc,
    time::{Duration, Instant},
};

struct Original {
    scope: SubmissionScope,
    graph: SubmissionGraphQuota,
    _records: SubmissionRecordQuota,
    _failure: RetainedPrefillFailure,
}
const GRAPH_CAPACITY: usize = 4 << 20;
impl Original {
    fn new() -> Self {
        Self::with_failure_owner(())
    }
    fn for_stream(stream: &Stream, pipelines: usize) -> Self {
        let pipelines = if stream.get_device().unwrap().get_type().unwrap() == DeviceType::Gpu {
            pipelines
        } else {
            0
        };
        Self::with_resources((), GRAPH_CAPACITY, 1 << 20, pipelines)
    }
    fn with_failure_owner(owner: impl Send + 'static) -> Self {
        Self::with_capacities(owner, GRAPH_CAPACITY, 1 << 20)
    }
    fn with_capacities(
        owner: impl Send + 'static,
        graph_bytes: usize,
        record_bytes: usize,
    ) -> Self {
        Self::with_resources(owner, graph_bytes, record_bytes, 0)
    }
    fn with_resources(
        owner: impl Send + 'static,
        graph_bytes: usize,
        record_bytes: usize,
        pipelines: usize,
    ) -> Self {
        let graph = PreparedSubmissionGraphQuota::try_new(graph_bytes, ())
            .unwrap()
            .try_allocate()
            .unwrap();
        if pipelines != 0 {
            crate::PreparedPipelineCachePlan::new(pipelines)
                .realize(())
                .unwrap()
                .install(&graph)
                .unwrap();
        }
        let records = PreparedSubmissionRecordQuota::try_new(record_bytes, ())
            .unwrap()
            .try_allocate()
            .unwrap();
        let failure = PreparedPrefillFailure::try_new(owner)
            .unwrap()
            .try_allocate()
            .unwrap();
        let mut scope = SubmissionScope::try_begin_retaining(
            PreparedSubmissionScopeOwner::try_new(())
                .unwrap()
                .with_graph_quota(graph.clone())
                .with_record_quota(records.clone()),
        )
        .unwrap();
        scope.enable_scoped_observation().unwrap();
        scope.require_original_native_controls().unwrap();
        failure.bind_original_scope(&scope).unwrap();
        scope.enable_original_native_controls().unwrap();
        Self {
            scope,
            graph,
            _records: records,
            _failure: failure,
        }
    }
}

thread_local! { static HOOKS: Cell<usize> = const { Cell::new(0) }; }
fn hook() {
    HOOKS.with(|value| value.set(value.get() + 1));
}
struct Hook;
impl Drop for Hook {
    fn drop(&mut self) {
        crate::unregister_thread_runtime_housekeeping(hook);
    }
}
fn settle(observer: &OriginalScopeObserver) {
    let limit = Instant::now() + Duration::from_secs(5);
    loop {
        let (outcome, status) = observer.progress().unwrap();
        assert_eq!(outcome, ScopedSubmissionProgress::Observed);
        assert!(!status.failed() && !status.blocked());
        if status.is_settled() {
            break;
        }
        assert!(Instant::now() < limit);
        std::thread::yield_now();
    }
    assert_eq!(
        observer.retire_completed_records().unwrap(),
        SubmissionRetirement::CompleteSnapshot
    );
}
fn round_trip(stream: &Stream) {
    round_trip_shape(stream, &[3]);
}
fn round_trip_shape(stream: &Stream, shape: &[i32]) {
    let source_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let _runtime = PrefillRootsRuntime::prepare_for_stream(stream, &source_stream).unwrap();
    let mut source =
        HostTransferBuffer::new(shape, Dtype::Float32, HostTransferPolicy::Transfer).unwrap();
    let elements = shape
        .iter()
        .map(|&n| usize::try_from(n).unwrap())
        .product::<usize>();
    let expected = (0..elements)
        .map(|i| [2.0f32, -3.0, 7.0][i % 3] + (i / 3) as f32)
        .collect::<Vec<_>>();
    for (destination, value) in source
        .as_bytes_mut()
        .unwrap()
        .chunks_exact_mut(4)
        .zip(expected.iter().copied())
    {
        destination.copy_from_slice(&value.to_ne_bytes());
    }
    let source = source.freeze();
    let allocator = crate::PreparedInputRuntime::prepare().unwrap();
    let physical =
        crate::OriginalBufferBudget::request_layout(&allocator, elements * size_of::<f32>())
            .unwrap()
            .capacity();
    let budget = crate::PreparedOriginalBufferBudget::try_new(&allocator, physical * 2, ())
        .unwrap()
        .try_allocate()
        .unwrap();
    let mut destinations = Vec::new();
    for from_writer in [false, true] {
        let plan =
            crate::PreparedHostTransferPlan::new(&allocator, shape, Dtype::Float32, 0).unwrap();
        let quota = PreparedSubmissionGraphQuota::try_new(plan.metadata_bytes(), ()).unwrap();
        let arena = crate::PreparedInputArena::try_allocate(quota).unwrap();
        destinations.push(if from_writer {
            let writer = plan.construct_writer(&arena).unwrap();
            // A worker cannot turn its byte-only handoff into a native owner.
            let writer = std::thread::spawn(move || {
                crate::PreparedHostCopyDestination::from_writer(writer).unwrap_err()
            })
            .join()
            .unwrap();
            crate::PreparedHostCopyDestination::from_writer(writer).unwrap()
        } else {
            plan.construct_copy_destination(&arena).unwrap()
        });
    }
    // Shared-Metal stores each select one actual vector-copy pipeline.
    let mut original = Original::for_stream(stream, 2);
    original.scope.bind_original_buffer_budget(&budget).unwrap();
    let observer = OriginalScopeObserver::try_current().unwrap().unwrap();
    let baseline = original.graph.occupied_bytes();
    for mut destination in destinations {
        let layout = OperationEvent::resident_graph_layout(3, 1, shape.len()).unwrap();
        let traversal = OperationEvent::eval_traversal_layout(OperationEvalTraversalLimits {
            roots: 1,
            arrays: 4,
            tape_entries: 3,
            input_edges: 3,
            output_slots: 3,
            streams: 1,
            captures: 8,
        })
        .unwrap();
        let mut bank = OperationEvent::prepare_resident_graph(layout, &observer).unwrap();
        bank.configure_nested_completions(&traversal, 2).unwrap();
        crate::register_thread_runtime_housekeeping(hook);
        let handler = crate::error::mlx_error_handler_state_for_test();
        let hooks = Hook;
        HOOKS.with(|value| value.set(0));
        let (value, producer) = source
            .copy_to_array_in_original_scope(stream, &observer)
            .unwrap();
        producer.wait_on(stream).unwrap();
        // The original wait submits its own receipt on the consumer stream.
        producer.synchronize().unwrap();
        observer.validate_completed_array(&value).unwrap();
        assert_eq!(HOOKS.with(Cell::get), 0);
        assert_eq!(crate::error::mlx_error_handler_state_for_test(), handler);
        drop(hooks);
        // The prepared destination owns its final backing before submission.
        destination
            .submit(&value, stream, &observer)
            .unwrap_or_else(|cause| {
                panic!(
                    "{cause:?}; native source: {:?}",
                    original
                        ._failure
                        .error()
                        .map(|source| String::from_utf8_lossy(
                            source.message_bytes().unwrap_or_default()
                        )
                        .into_owned())
                )
            });
        drop(bank);
        destination.synchronize().unwrap();
        let copied = destination.take_completed().unwrap();
        assert_eq!(value.shape(), shape);
        assert_eq!(copied.shape().unwrap(), shape);
        producer.synchronize().unwrap();
        assert_eq!(
            value
                .completed_in_original_scope(&observer)
                .unwrap()
                .try_as_slice::<f32>()
                .unwrap(),
            expected
        );
        let expected_bytes: Vec<u8> = expected
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        assert_eq!(copied.as_bytes().unwrap(), expected_bytes);
        crate::try_with_submission_retirement(|| drop((copied, destination, value, producer)))
            .unwrap();
        settle(&observer);
        assert_eq!(original.graph.occupied_bytes(), baseline);
        assert_eq!(budget.occupied_bytes(), 0);
        assert!(OriginalScopeObserver::try_current().unwrap().is_some());
    }
    original.scope.seal();
    settle(&observer);
}
#[test]
fn original_operation_cpu_round_trip_reclaims_each_active_role_window_without_hooks() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    round_trip(&stream);
}
#[test]
fn original_operation_cpu_rank_five_host_round_trip_reclaims_all_owners() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    // Multiple nonunit axes exercise the actual General-copy shape and stride
    // census in both directions while each request retains its full backing.
    round_trip_shape(&stream, &[2, 1, 3, 1, 2]);
}
#[cfg(all(feature = "metal", target_vendor = "apple", not(feature = "cuda")))]
// Prepared singleton ownership must precede ordinary fixture initialization.
// Each selected test establishes those owners in its own process.
fn with_prepared_metal_stream(test: &str, run: impl FnOnce(&Stream)) {
    with_prepared_metal_registration(test, |source, _| run(source.as_stream()));
}
#[cfg(all(feature = "metal", target_vendor = "apple", not(feature = "cuda")))]
fn with_prepared_metal_registration(
    test: &str,
    run: impl FnOnce(&crate::RegisteredGpuStream, crate::GpuStreamTarget),
) {
    const CHILD: &str = "SAFEMLX_PREPARED_OPERATION_STREAM_CHILD";
    if std::env::var(CHILD).ok().as_deref() != Some(test) {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test, "--nocapture", "--test-threads=1"])
            .env(CHILD, test)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success() && stdout.contains("1 passed; 0 failed;"),
            "{stdout}\n{}",
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
    run(&stream, target);
    stream.try_observe_idle().unwrap();
}

#[cfg(all(feature = "metal", target_vendor = "apple", not(feature = "cuda")))]
#[test]
fn retained_gpu_registration_keeps_identity_across_execution_contexts() {
    with_prepared_metal_registration(
        "operation_event::tests::retained_gpu_registration_keeps_identity_across_execution_contexts",
        |stream, target| {
            let layout = crate::PreparedGpuStream::<()>::layout().unwrap();
            let pending = crate::PreparedGpuStream::with_layout(layout, ()).unwrap();
            let mut ordinary = SubmissionScope::begin().unwrap();
            stream.try_borrow().unwrap();
            let rejected = pending.try_initialize(target).unwrap_err();
            assert_eq!(rejected.cause(), crate::GpuStreamRegistrationCause::Invalid);
            let (_, pending) = rejected.into_parts();
            ordinary.seal();
            drop(ordinary);
            stream.try_borrow().unwrap();

            let mut original = Original::for_stream(stream.as_stream(), 0);
            let graph = original.graph.occupied_bytes();
            let records = original._records.occupied_bytes();
            crate::register_thread_runtime_housekeeping(hook);
            let hooks = Hook;
            HOOKS.with(|value| value.set(0));
            stream.try_borrow().unwrap();
            stream.try_observe_idle().unwrap();
            let rejected = pending.try_initialize(target).unwrap_err();
            assert_eq!(rejected.cause(), crate::GpuStreamRegistrationCause::Invalid);
            let (_, pending) = rejected.into_parts();
            stream.try_borrow().unwrap();
            assert_eq!(original.graph.occupied_bytes(), graph);
            assert_eq!(original._records.occupied_bytes(), records);
            assert_eq!(HOOKS.with(Cell::get), 0);
            drop(hooks);
            {
                let mut child = SubmissionScope::begin().unwrap();
                stream.try_borrow().unwrap();
                child.seal();
            }
            stream.try_borrow().unwrap();
            original.scope.seal();
            stream.try_borrow().unwrap();
            drop(original);
            // Both refusals preserved the same untransferred constructor token.
            let fresh = pending.try_initialize(target).unwrap();
            fresh.try_borrow().unwrap();
            assert_ne!(
                fresh.as_stream().get_index().unwrap(),
                stream.as_stream().get_index().unwrap()
            );
            stream.try_borrow().unwrap();
        },
    );
}

#[cfg(all(feature = "metal", target_vendor = "apple", not(feature = "cuda")))]
#[test]
fn original_operation_metal_round_trip_reclaims_each_active_role_window_without_hooks() {
    with_prepared_metal_stream(
        "operation_event::tests::original_operation_metal_round_trip_reclaims_each_active_role_window_without_hooks",
        round_trip,
    );
}

#[test]
fn original_operation_rejects_explicit_child_and_observes_after_role_seal() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let _runtime = PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let host = HostTransferBuffer::new(&[1], Dtype::Float32, HostTransferPolicy::Transfer)
        .unwrap()
        .freeze();
    let mut original = Original::new();
    let observer = OriginalScopeObserver::try_current().unwrap().unwrap();
    {
        let mut child = SubmissionScope::begin().unwrap();
        assert_eq!(
            OriginalScopeObserver::try_current()
                .err()
                .unwrap()
                .scoped_evaluation_cause(),
            Some(crate::error::ScopedEvaluationCause::Domain)
        );
        assert!(!observer.status().is_settled());
        child.seal();
    }
    let event =
        crate::transforms::async_eval_with_operation_event(std::iter::empty::<&Array>()).unwrap();
    original.scope.seal();
    let occupied = original.graph.occupied_bytes();
    assert_eq!(
        crate::transforms::async_eval_with_original_operation_event(
            std::iter::empty::<&Array>(),
            &observer
        )
        .err()
        .unwrap()
        .scoped_evaluation_cause(),
        Some(crate::error::ScopedEvaluationCause::Domain)
    );
    assert_eq!(
        host.copy_to_array_in_original_scope(&stream, &observer)
            .err()
            .unwrap()
            .scoped_evaluation_cause(),
        Some(crate::error::ScopedEvaluationCause::Domain)
    );
    assert_eq!(original.graph.occupied_bytes(), occupied);
    assert!(event.is_complete().unwrap());
    assert_eq!(
        event
            .wait_on(&stream)
            .unwrap_err()
            .scoped_evaluation_cause(),
        Some(crate::error::ScopedEvaluationCause::Domain)
    );
    drop(event);
    settle(&observer);
}
#[test]
fn original_operation_busy_drop_keeps_wrapper_until_exact_owner_retirement() {
    let original = Original::new();
    let observer = OriginalScopeObserver::try_current().unwrap().unwrap();
    let before = original.graph.occupied_bytes();
    let event =
        crate::transforms::async_eval_with_operation_event(std::iter::empty::<&Array>()).unwrap();
    let occupied = original.graph.occupied_bytes();
    assert!(occupied > before);
    let (ready_tx, ready_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let thread = std::thread::spawn(move || {
        let _guard = runtime_lock::enter();
        ready_tx.send(()).unwrap();
        release_rx.recv_timeout(Duration::from_secs(5)).is_ok()
    });
    ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(
        event.is_complete().unwrap_err().scoped_evaluation_cause(),
        Some(crate::error::ScopedEvaluationCause::RuntimeBusy)
    );
    assert_eq!(
        observer.progress().unwrap().0,
        ScopedSubmissionProgress::Busy
    );
    drop(event);
    assert_eq!(original.graph.occupied_bytes(), occupied);
    let sent = release_tx.send(()).is_ok();
    assert!(thread.join().unwrap() && sent);
    settle(&observer);
    assert_eq!(original.graph.occupied_bytes(), before);
}

#[test]
fn original_observers_match_actual_scope_owner_and_refuse_equal_capacity_other_role() {
    assert!(OriginalScopeObserver::try_current().unwrap().is_none());
    assert_eq!(
        OriginalScopeObserver::require_current()
            .err()
            .unwrap()
            .scoped_evaluation_cause(),
        Some(crate::error::ScopedEvaluationCause::Domain)
    );
    let mut first = Original::new();
    let observed = OriginalScopeObserver::require_current().unwrap();
    assert!(observed.belongs_to(&first.scope));
    assert!(observed.same_scope(&observed.clone()));
    // Independent owners must not be nested. Keep the first observer and
    // arenas alive while a second, independently active owner is checked.
    first.scope.seal();
    {
        let second = Original::new();
        let other = OriginalScopeObserver::require_current().unwrap();
        assert!(other.belongs_to(&second.scope));
        assert!(!other.belongs_to(&first.scope));
        assert!(!observed.same_scope(&other));
        let occupied = first.graph.occupied_bytes();
        assert_eq!(
            crate::transforms::async_eval_with_original_operation_event(
                std::iter::empty::<&Array>(),
                &observed
            )
            .err()
            .unwrap()
            .scoped_evaluation_cause(),
            Some(crate::error::ScopedEvaluationCause::Domain)
        );
        assert_eq!(first.graph.occupied_bytes(), occupied);
    }
    assert!(observed.belongs_to(&first.scope));
    assert!(OriginalScopeObserver::try_current().unwrap().is_none());
}

#[test]
fn original_completed_array_after_seal_skips_hooks_and_preserves_busy_and_unscheduled() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let _runtime = PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let source = Array::from_slice(&[2.0f32, -3.0, 7.0], &[3]);
    let mut original = Original::new();
    let observer = OriginalScopeObserver::require_current().unwrap();
    let lazy = source.add(&source, &stream).unwrap();
    let output = source.add(&source, &stream).unwrap();
    let event = crate::transforms::async_eval_with_original_operation_event_on_stream(
        [&output],
        &observer,
        &stream,
    )
    .unwrap();
    event.synchronize().unwrap();
    original.scope.seal();

    let (ready_tx, ready_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let thread = std::thread::spawn(move || {
        let _guard = runtime_lock::enter();
        ready_tx.send(()).unwrap();
        release_rx.recv_timeout(Duration::from_secs(5)).is_ok()
    });
    ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    let busy = observer.validate_completed_array(&output).unwrap_err();
    let sent = release_tx.send(()).is_ok();
    assert!(thread.join().unwrap() && sent);
    assert_eq!(
        busy.scoped_evaluation_cause(),
        Some(crate::error::ScopedEvaluationCause::RuntimeBusy)
    );
    drop(busy);

    crate::register_thread_runtime_housekeeping(hook);
    let hooks = Hook;
    HOOKS.with(|value| value.set(0));
    let handler = crate::error::mlx_error_handler_state_for_test();
    let graph = original.graph.occupied_bytes();
    let records = original._records.occupied_bytes();
    assert_eq!(
        observer
            .validate_completed_array(&lazy)
            .unwrap_err()
            .scoped_evaluation_cause(),
        Some(crate::error::ScopedEvaluationCause::Invalid)
    );
    let evaluated = output.completed_in_original_scope(&observer).unwrap();
    assert_eq!(evaluated.try_as_slice::<f32>().unwrap(), &[4.0, -6.0, 14.0]);
    observer.validate_completed_array(&output).unwrap();
    assert_eq!(
        observer
            .validate_completed_array(&lazy)
            .unwrap_err()
            .scoped_evaluation_cause(),
        Some(crate::error::ScopedEvaluationCause::Invalid)
    );
    assert!(original.graph.occupied_bytes() <= graph);
    assert_eq!(original._records.occupied_bytes(), records);
    assert_eq!(HOOKS.with(Cell::get), 0);
    assert_eq!(crate::error::mlx_error_handler_state_for_test(), handler);
    drop(hooks);
    crate::try_with_submission_retirement(|| drop((event, output, lazy))).unwrap();
    settle(&observer);
}

#[test]
fn original_wait_record_layout_qualified_or_unknown_contract() {
    let one = OperationEvent::wait_record_layout(1);
    if std::env::var("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").as_deref() == Ok("1") {
        assert!(
            one.is_some(),
            "pinned validation requires the qualified wait producer"
        );
    }
    let Some(one) = one else {
        for count in [0, 1, 7, usize::MAX] {
            assert!(OperationEvent::wait_record_layout(count).is_none());
        }
        return; // Explicit unknown contract, not positive producer coverage.
    };
    let many = OperationEvent::wait_record_layout(7).unwrap();
    assert_eq!(one.capture_slots(), 8);
    assert_eq!(one.stream_receipts(), 1);
    assert_eq!(one.allocations_per_wait(), 3);
    assert_eq!(many.object_bytes(), one.object_bytes());
    assert_eq!(many.object_alignment(), one.object_alignment());
    assert_eq!(many.total_record_allocations(), 21);
    assert_eq!(
        many.total_record_requested_bytes(),
        7 * one.requested_bytes_per_wait()
    );
    assert_eq!(
        OperationEvent::wait_record_layout(0)
            .unwrap()
            .total_record_requested_bytes(),
        0
    );
    assert!(OperationEvent::wait_record_layout(usize::MAX).is_none());
}

#[test]
fn eval_record_layout_qualified_or_unknown_contract() {
    let one = OperationEvent::eval_record_layout(1, 1, 1);
    if std::env::var("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").as_deref() == Ok("1") {
        assert!(
            one.is_some(),
            "pinned validation requires the qualified Eval producer"
        );
    }
    let Some(one) = one else {
        for (tape, streams, outputs) in [
            (1, 1, 1),
            (3, 2, 4),
            (0, 0, 0),
            (1, 0, 1),
            (1, 2, 2),
            (2, 1, 1),
            (usize::MAX, 1, usize::MAX),
            (1, 1, usize::MAX),
        ] {
            assert!(OperationEvent::eval_record_layout(tape, streams, outputs).is_none());
        }
        return; // Explicit unknown contract, not positive producer coverage.
    };
    let three = OperationEvent::eval_record_layout(3, 2, 4).unwrap();
    assert_eq!(three.tape_entries(), 3);
    assert_eq!(three.stream_count(), 2);
    assert_eq!(three.output_slots(), 4);
    assert_eq!(three.record_allocations(), 6);
    assert_eq!(three.capture_slots(), 8);
    assert_eq!(three.object_bytes(), one.object_bytes());
    assert_eq!(three.capture_bytes(), one.capture_bytes());
    assert_eq!(three.stream_state_bytes(), 2 * one.stream_state_bytes());
    assert_eq!(three.stream_receipt_bytes(), 2 * one.stream_receipt_bytes());
    assert_eq!(
        three.primitive_owner_bytes(),
        3 * one.primitive_owner_bytes()
    );
    assert_eq!(three.output_pin_bytes(), 4 * one.output_pin_bytes());
    assert_eq!(
        three.record_requested_bytes(),
        three.object_bytes()
            + three.capture_bytes()
            + three.stream_state_bytes()
            + three.stream_receipt_bytes()
            + three.primitive_owner_bytes()
            + three.output_pin_bytes()
    );
    assert!(three.query_control_bytes().unwrap() > std::mem::size_of_val(&three));
    assert_eq!(three.host_graph_blocks(), 4);
    let constructors = three.host_graph_requests();
    let owners = three.host_graph_owner_requests();
    assert!(
        constructors
            .iter()
            .chain(&owners)
            .all(|(bytes, align)| *bytes > 0 && align.is_power_of_two())
    );
    let requested = constructors
        .iter()
        .chain(&owners)
        .map(|(bytes, _)| bytes)
        .sum::<usize>();
    assert_eq!(requested, three.host_graph_requested_bytes());
    assert!(three.host_graph_allocation_extents() > requested);
    assert!(three.host_graph_event_controls() > 0);
    assert!(three.query_control_bytes().unwrap() > three.host_graph_event_controls());
    assert_eq!(one.host_graph_requests(), three.host_graph_requests());
    assert_eq!(
        one.host_graph_owner_requests(),
        three.host_graph_owner_requests()
    );

    for shape in [
        (0, 0, 0),
        (1, 0, 1),
        (1, 2, 2),
        (2, 1, 1),
        (usize::MAX, 1, usize::MAX),
        (1, 1, usize::MAX),
    ] {
        assert!(OperationEvent::eval_record_layout(shape.0, shape.1, shape.2).is_none());
    }
    assert_eq!(OperationEvent::eval_record_layout(3, 2, 4), Some(three));
}

mod gpu_eval_prologue;

#[cfg(all(feature = "metal", not(feature = "cuda")))]
mod ordinary_metal_router;
