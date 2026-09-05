use super::*;
use eredu_core::BoundedCompletion as _;
use safemlx::{
    ops::indexing::TryIndexOp,
    transforms::{async_eval, async_eval_with_event},
    Device, DeviceType,
};

fn cpu_stream() -> Stream {
    Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
}

#[test]
fn bounded_communication_timeout_quarantines_all_retained_resources() {
    COMMUNICATION_ORPHANS.with(|orphans| {
        let mut orphans = orphans.borrow_mut();
        for work in &mut orphans.work {
            work.force_pending = false;
        }
        orphans.reap();
        assert!(orphans.work.is_empty());
    });
    let stream = cpu_stream();
    let native = safemlx::distributed::init(false, safemlx::distributed::Backend::Ring).unwrap();
    let group = Group::uncontracted(&native);
    let output = Array::ones::<f32>(&[1], &stream).unwrap();
    let mut completion = MlxCommunicationCompletion::submit(
        [&output],
        vec![output.clone()],
        vec![vec![1]],
        vec![group.clone()],
        Vec::new(),
        vec![stream],
    )
    .unwrap();
    completion.force_pending = true;
    let policy = eredu_core::BoundedCompletionWait::new(
        std::time::Duration::from_millis(1),
        eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
    )
    .unwrap();
    assert_eq!(
        completion.wait_bounded(policy).unwrap(),
        eredu_core::BoundedCompletionOutcome::DeadlineExceeded {
            cancellation: eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
        }
    );
    COMMUNICATION_ORPHANS.with(|orphans| {
        let orphans = orphans.borrow();
        assert_eq!(orphans.work.len(), 1);
        assert_eq!(orphans.work[0].retained_arrays(), 1);
        assert_eq!(orphans.work[0].retained_count_buffers(), 1);
        assert_eq!(orphans.work[0].retained_groups(), 1);
        assert_eq!(orphans.work[0].retained_streams(), 1);
    });
    assert!(ensure_group_available(&group).is_err());
    COMMUNICATION_ORPHANS.with(|orphans| {
        let mut orphans = orphans.borrow_mut();
        orphans.work[0].force_pending = false;
        orphans.work[0].event.synchronize().unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
        while !orphans.work.is_empty() {
            orphans.reap();
            assert!(
                std::time::Instant::now() < deadline,
                "completed quarantined work was not reaped"
            );
            std::thread::yield_now();
        }
    });
    assert!(ensure_group_available(&group).is_ok());
}

#[test]
fn completed_quarantine_is_reaped_on_an_unrelated_runtime_entry() {
    COMMUNICATION_ORPHANS.with(|orphans| {
        let mut orphans = orphans.borrow_mut();
        for work in &mut orphans.work {
            work.force_pending = false;
        }
        orphans.reap();
        assert!(orphans.work.is_empty());
    });
    let stream = cpu_stream();
    let native = safemlx::distributed::init(false, safemlx::distributed::Backend::Ring).unwrap();
    let group = Group::uncontracted(&native);
    let output = Array::ones::<f32>(&[1], &stream).unwrap();
    let mut completion = MlxCommunicationCompletion::submit(
        [&output],
        vec![output.clone()],
        Vec::new(),
        vec![group],
        Vec::new(),
        vec![stream.clone()],
    )
    .unwrap();
    completion.force_pending = true;
    let policy = eredu_core::BoundedCompletionWait::new(
        std::time::Duration::from_millis(1),
        eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
    )
    .unwrap();
    assert!(matches!(
        completion.wait_bounded(policy).unwrap(),
        eredu_core::BoundedCompletionOutcome::DeadlineExceeded { .. }
    ));
    COMMUNICATION_ORPHANS.with(|orphans| {
        let mut orphans = orphans.borrow_mut();
        assert_eq!(orphans.work.len(), 1);
        orphans.work[0].force_pending = false;
    });

    stream.synchronize().unwrap();
    let _unrelated = Array::ones::<f32>(&[1], &stream).unwrap();

    COMMUNICATION_ORPHANS.with(|orphans| {
        assert!(
            orphans.borrow().work.is_empty(),
            "a later same-thread MLX runtime entry did not reap completed quarantined work"
        );
    });
}

#[test]
fn owner_thread_exit_waits_then_releases_quarantined_native_resources() {
    let completion_observed = Arc::new(AtomicBool::new(false));
    let teardown_observed = Arc::new(AtomicBool::new(false));
    let worker_completion = Arc::clone(&completion_observed);
    let worker_observed = Arc::clone(&teardown_observed);
    std::thread::spawn(move || {
        let stream = cpu_stream();
        let native =
            safemlx::distributed::init(false, safemlx::distributed::Backend::Ring).unwrap();
        let group = Group::uncontracted(&native);
        let lhs = Array::ones::<f32>(&[256, 256], &stream).unwrap();
        let rhs = Array::ones::<f32>(&[256, 256], &stream).unwrap();
        let output = lhs.matmul(&rhs, &stream).unwrap();
        let mut completion = MlxCommunicationCompletion::submit(
            [&output],
            vec![lhs, rhs, output.clone()],
            vec![vec![1]],
            vec![group],
            Vec::new(),
            vec![stream],
        )
        .unwrap();
        completion.force_pending = true;
        completion.teardown_observed = Some(worker_observed);
        completion.owner_exit_completion_observed = Some(worker_completion);
        let policy = eredu_core::BoundedCompletionWait::new(
            std::time::Duration::from_millis(1),
            eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap();
        assert!(matches!(
            completion.wait_bounded(policy).unwrap(),
            eredu_core::BoundedCompletionOutcome::DeadlineExceeded { .. }
        ));
        COMMUNICATION_ORPHANS.with(|orphans| assert_eq!(orphans.borrow().work.len(), 1));
        // TLS teardown owns the outstanding completion after this return.
    })
    .join()
    .unwrap();
    assert!(
        completion_observed.load(Ordering::Acquire),
        "owner-thread exit released resources before its exact completion wait returned"
    );
    assert!(
        teardown_observed.load(Ordering::Acquire),
        "owner-thread exit leaked the quarantined completion and its retained resources"
    );
}

#[test]
fn distributed_completion_orders_multiple_cpu_consumers() {
    let producer = cpu_stream();
    let consumer_a = cpu_stream();
    let consumer_b = cpu_stream();
    let blocker_lhs = Array::ones::<f32>(&[1024, 1024], &producer).unwrap();
    let blocker_rhs = Array::ones::<f32>(&[1024, 1024], &producer).unwrap();
    let blocker = blocker_lhs.matmul(&blocker_rhs, &producer).unwrap();
    async_eval([&blocker]).unwrap();

    let value = Array::ones::<f32>(&[1, 1024], &producer).unwrap();
    let completion = DistributedCompletion::submit(value.clone(), [&value]).unwrap();
    completion.wait_on(&consumer_a).unwrap();
    completion.wait_on(&consumer_b).unwrap();
    let consumed_a = completion.value().square(&consumer_a).unwrap();
    let consumed_b = completion.value().square(&consumer_b).unwrap();
    let completion_a = async_eval_with_event([&consumed_a]).unwrap();
    let completion_b = async_eval_with_event([&consumed_b]).unwrap();

    let value = completion.into_value().unwrap();
    assert_eq!(value.shape(), [1, 1024]);
    completion_a.synchronize().unwrap();
    completion_b.synchronize().unwrap();
    assert_eq!(
        consumed_a.evaluated().unwrap().as_slice::<f32>(),
        &[1.0; 1024]
    );
    assert_eq!(
        consumed_b.evaluated().unwrap().as_slice::<f32>(),
        &[1.0; 1024]
    );
}

#[test]
fn dropping_distributed_completion_preserves_a_queued_cpu_wait() {
    let producer = cpu_stream();
    let consumer = cpu_stream();
    let value = Array::ones::<f32>(&[8, 8], &producer).unwrap();
    let completion = DistributedCompletion::submit(value.clone(), [&value]).unwrap();
    completion.wait_on(&consumer).unwrap();
    let consumed = completion
        .value()
        .add(Array::from_int(1), &consumer)
        .unwrap();
    let consumed_completion = async_eval_with_event([&consumed]).unwrap();
    drop(completion);

    consumed_completion.synchronize().unwrap();
    assert_eq!(consumed.evaluated().unwrap().as_slice::<f32>(), &[2.0; 64]);
}

#[test]
fn received_in_band_boundary_header_is_validated_after_exact_native_completion() {
    let stream = cpu_stream();
    let received = Array::from_slice(&[2_u8, 1, 3, 99], &[4]);
    let received_header = received.try_index_device(0..3, &stream).unwrap();
    let logical_payload = received.try_index_device(3.., &stream).unwrap();
    let completion = MlxCommunicationCompletion::submit(
        [&received, &received_header, &logical_payload],
        vec![
            received.clone(),
            received_header.clone(),
            logical_payload.clone(),
        ],
        Vec::new(),
        Vec::new(),
        Vec::new(),
        vec![stream],
    )
    .unwrap()
    .with_boundary_headers([(received_header, vec![1_u8, 2, 3])]);
    assert_eq!(completion.submitted_outputs(), 3);
    let error = eredu_core::Completion::wait(&completion)
        .expect_err("same-sized reordered role bytes must not complete successfully");
    assert!(error
        .what()
        .contains("differs from the selected route/schema/role contract"));
}

#[test]
#[ignore = "explicit Metal distributed completion test; run on a Metal host"]
fn distributed_completion_metal_wait_does_not_block_the_host() {
    let producer = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let consumer = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let blocker_lhs = Array::ones::<f32>(&[4096, 4096], &producer).unwrap();
    let blocker_rhs = Array::ones::<f32>(&[4096, 4096], &producer).unwrap();
    let blocker = blocker_lhs.matmul(&blocker_rhs, &producer).unwrap();
    async_eval([&blocker]).unwrap();
    let value = Array::ones::<f32>(&[1024, 1024], &producer).unwrap();
    let completion = DistributedCompletion::submit(value.clone(), [&value]).unwrap();

    assert!(!completion.is_complete().unwrap());
    completion.wait_on(&consumer).unwrap();
    assert!(!completion.is_complete().unwrap());
    let consumed = completion.value().square(&consumer).unwrap();
    let consumed_completion = async_eval_with_event([&consumed]).unwrap();
    drop(completion);
    consumed_completion.synchronize().unwrap();
}
