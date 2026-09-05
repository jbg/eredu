fn singleton_communication() -> (crate::backend::runtime::distributed::Group, safemlx::Stream) {
    let native = safemlx::distributed::init(false, safemlx::distributed::Backend::Ring).unwrap();
    let group = crate::backend::runtime::distributed::Group::uncontracted(&native);
    let stream = safemlx::Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    (group, stream)
}

#[test]
fn local_dependency_submission_retains_exact_outputs_and_executor_under_bound() {
    use eredu_core::BoundedCompletion as _;

    let (_, stream) = singleton_communication();
    let values = [
        MlxTensor::from_array(Array::from_slice(&[1.0_f32, 2.0], &[1, 2])),
        MlxTensor::from_array(Array::from_slice(&[3.0_f32], &[1, 1])),
    ];
    crate::backend::runtime::distributed::completion::force_next_communication_pending();
    let submission = <MlxNeuralBackend as CommunicationBackend>::submit_local_dependencies(
        values.iter(),
        &stream,
    )
    .unwrap();
    assert_eq!(submission.completion.submitted_outputs(), values.len());
    assert_eq!(submission.completion.retained_arrays(), values.len());
    assert_eq!(submission.completion.retained_streams(), 1);
    assert_eq!(submission.completion.retained_groups(), 0);
    assert_eq!(submission.completion.retained_routes(), 0);
    let policy = eredu_core::BoundedCompletionWait::new(
        std::time::Duration::from_millis(1),
        eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
    )
    .unwrap();
    assert_eq!(
        submission.completion.wait_bounded(policy).unwrap(),
        eredu_core::BoundedCompletionOutcome::DeadlineExceeded {
            cancellation: eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
        }
    );
    crate::backend::runtime::distributed::completion::release_forced_pending_orphans();
}

#[test]
fn fine_grained_collectives_retain_exact_singleton_resources() {
    let (group, stream) = singleton_communication();

    let reduced = <MlxNeuralBackend as SumReductionBackend>::all_reduce_sum(
        MlxTensor::from_array(Array::from_slice(&[1.0_f32, 2.0], &[1, 2])),
        &group,
        &stream,
    )
    .unwrap();
    assert_eq!(reduced.completion.retained_arrays(), 2);
    assert_eq!(reduced.completion.retained_count_buffers(), 0);
    assert_eq!(reduced.completion.retained_groups(), 1);
    assert_eq!(reduced.completion.retained_routes(), 0);
    assert_eq!(reduced.completion.retained_streams(), 1);
    assert_eq!(
        reduced
            .wait()
            .unwrap()
            .as_array()
            .evaluated()
            .unwrap()
            .as_slice::<f32>(),
        &[1.0, 2.0]
    );

    let gathered = <MlxNeuralBackend as EvenGatherBackend>::all_gather_even(
        MlxTensor::from_array(Array::from_slice(&[3.0_f32, 4.0], &[1, 2])),
        1,
        &group,
        &stream,
    )
    .unwrap();
    assert_eq!(gathered.completion.retained_arrays(), 2);
    let gathered = gathered.wait().unwrap();
    assert_eq!(gathered.as_array().shape(), &[1, 2]);

    let uneven = <MlxNeuralBackend as UnevenGatherBackend>::all_gather_uneven(
        MlxTensor::from_array(Array::from_slice(&[5.0_f32, 6.0], &[1, 2])),
        &[2],
        1,
        &group,
        &stream,
    )
    .unwrap();
    assert_eq!(uneven.completion.retained_count_buffers(), 1);
    let uneven = uneven.wait().unwrap();
    assert_eq!(uneven.as_array().shape(), &[1, 2]);

    let counts = CommunicationPeerCounts::new(vec![2], vec![2], 1).unwrap();
    let exchanged = <MlxNeuralBackend as VariableAllToAllBackend>::variable_all_to_all(
        MlxTensor::from_array(Array::from_slice(&[7.0_f32, 8.0, 9.0, 10.0], &[2, 2])),
        &counts,
        1,
        &group,
        &stream,
    )
    .unwrap();
    assert_eq!(exchanged.completion.retained_arrays(), 2);
    assert_eq!(exchanged.completion.retained_count_buffers(), 2);
    assert_eq!(
        exchanged
            .wait()
            .unwrap()
            .as_array()
            .evaluated()
            .unwrap()
            .as_slice::<f32>(),
        &[7.0, 8.0, 9.0, 10.0]
    );

    let broadcast = <MlxNeuralBackend as BroadcastBackend>::broadcast(
        MlxTensor::from_array(Array::from_slice(&[11.0_f32, 12.0], &[2])),
        0,
        &group,
        &stream,
    )
    .unwrap();
    assert_eq!(broadcast.completion.retained_arrays(), 3);
    assert_eq!(broadcast.completion.retained_groups(), 1);
    assert_eq!(broadcast.completion.retained_streams(), 1);
    assert_eq!(
        broadcast
            .wait()
            .unwrap()
            .as_array()
            .evaluated()
            .unwrap()
            .as_slice::<f32>(),
        &[11.0, 12.0]
    );

    let barrier = <MlxNeuralBackend as BarrierBackend>::barrier(&group, &stream).unwrap();
    assert_eq!(barrier.retained_arrays(), 2);
    assert_eq!(barrier.retained_groups(), 1);
    assert_eq!(barrier.retained_streams(), 1);
    barrier.wait().unwrap();
}

#[test]
fn fine_grained_collectives_admit_i32_only_for_count_gather_and_variable_exchange() {
    let (group, stream) = singleton_communication();

    let gathered = <MlxNeuralBackend as EvenGatherBackend>::all_gather_even(
        MlxTensor::from_array(Array::from_slice(&[2_i32, 0, 3], &[3])),
        0,
        &group,
        &stream,
    )
    .unwrap();
    assert_eq!(gathered.completion.retained_arrays(), 2);
    assert_eq!(
        gathered
            .wait()
            .unwrap()
            .as_array()
            .evaluated()
            .unwrap()
            .as_slice::<i32>(),
        &[2, 0, 3]
    );

    let reduction_error = <MlxNeuralBackend as SumReductionBackend>::all_reduce_sum(
        MlxTensor::from_array(Array::from_slice(&[1_i32], &[1])),
        &group,
        &stream,
    )
    .expect_err("integer reduction is not advertised");
    assert!(reduction_error.what().contains("does not advertise dtype"));

    let uneven_error = <MlxNeuralBackend as UnevenGatherBackend>::all_gather_uneven(
        MlxTensor::from_array(Array::from_slice(&[1_i32], &[1])),
        &[1],
        0,
        &group,
        &stream,
    )
    .expect_err("integer uneven gather is not advertised");
    assert!(uneven_error.what().contains("does not advertise dtype"));

    let counts = CommunicationPeerCounts::new(vec![1], vec![1], 1).unwrap();
    let exchanged = <MlxNeuralBackend as VariableAllToAllBackend>::variable_all_to_all(
        MlxTensor::from_array(Array::from_slice(&[1_i32], &[1])),
        &counts,
        0,
        &group,
        &stream,
    )
    .unwrap();
    assert_eq!(exchanged.completion.retained_arrays(), 2);
    assert_eq!(exchanged.completion.retained_count_buffers(), 2);
    assert_eq!(exchanged.completion.retained_groups(), 1);
    assert_eq!(exchanged.completion.retained_streams(), 1);
    assert_eq!(
        exchanged
            .wait()
            .unwrap()
            .as_array()
            .evaluated()
            .unwrap()
            .as_slice::<i32>(),
        &[1]
    );

    let unsigned_error = <MlxNeuralBackend as EvenGatherBackend>::all_gather_even(
        MlxTensor::from_array(Array::from_slice(&[1_u32], &[1])),
        0,
        &group,
        &stream,
    )
    .expect_err("unsigned count gather is outside the exact admitted set");
    assert!(unsigned_error.what().contains("does not advertise dtype"));

    let unsigned_exchange = <MlxNeuralBackend as VariableAllToAllBackend>::variable_all_to_all(
        MlxTensor::from_array(Array::from_slice(&[1_u32], &[1])),
        &counts,
        0,
        &group,
        &stream,
    )
    .expect_err("unsigned variable exchange is outside the exact admitted set");
    assert!(unsigned_exchange
        .what()
        .contains("does not advertise dtype"));
}
