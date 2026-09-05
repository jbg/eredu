use super::*;

#[test]
fn selected_setup_deadline_rejects_busy_runtime_before_native_submission() {
    use std::sync::{Arc, Barrier};

    let native = distributed::init(false, Backend::Ring).unwrap();
    let policy = eredu_runtime::CommunicationCompletionPolicy::new(
        Duration::from_millis(5),
        eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
    )
    .unwrap();
    let descriptor = CommunicationGroupDescriptor::new(
        CollectiveGroupId::new(41),
        0,
        vec![0],
        Some(0),
        CommunicationGroupRequirements::new([CommunicationOperationRequirement::barrier(true)])
            .unwrap(),
    )
    .unwrap();
    let group = Group::uncontracted(&native)
        .with_manifest_contract(&descriptor, policy)
        .unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    crate::backend::runtime::distributed::group::reset_native_collective_submissions();
    std::thread::scope(|scope| {
        let worker_entered = Arc::clone(&entered);
        let worker_release = Arc::clone(&release);
        scope.spawn(move || {
            let _guard = safemlx::RuntimeCallDeadline::new(Duration::from_secs(1))
                .unwrap()
                .enter()
                .unwrap();
            worker_entered.wait();
            worker_release.wait();
        });
        entered.wait();
        let error = <MlxNeuralBackend as BarrierBackend>::barrier(&group, &stream)
            .expect_err("busy runtime must fail before collective graph submission");
        assert!(error.what().contains("selected deadline"));
        assert_eq!(
            crate::backend::runtime::distributed::group::native_collective_submissions(),
            0
        );
        release.wait();
    });
}

#[test]
fn setup_deadline_poisons_exact_authority_and_retry_makes_no_native_call() {
    use std::sync::{Arc, Barrier};

    let native = distributed::init(false, Backend::Ring).unwrap();
    let policy = eredu_runtime::CommunicationCompletionPolicy::new(
        Duration::from_millis(5),
        eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
    )
    .unwrap();
    let id = CollectiveGroupId::new(42);
    let descriptor = CommunicationGroupDescriptor::new(
        id,
        0,
        vec![0],
        Some(0),
        CommunicationGroupRequirements::new([
            CommunicationOperationRequirement::failure_agreement(true),
        ])
        .unwrap(),
    )
    .unwrap();
    let group = Group::uncontracted(&native)
        .with_manifest_contract(&descriptor, policy)
        .unwrap();
    let manifest = CommunicationManifest::new(1, 0, vec![descriptor], Vec::new())
        .unwrap()
        .with_completion_policy(policy);
    let communication = PartitionCommunication::<MlxNeuralBackend, _, _, _>::new(
        manifest,
        vec![RealizedCommunicationGroup::new(id, group)],
        Vec::<
            eredu_runtime::RealizedCommunicationRoute<
                super::super::topology::CommunicationRouteRealization,
            >,
        >::new(),
        MlxCommunicationTensorMetadata,
    )
    .unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    crate::backend::runtime::distributed::group::reset_native_collective_submissions();
    std::thread::scope(|scope| {
        let worker_entered = Arc::clone(&entered);
        let worker_release = Arc::clone(&release);
        scope.spawn(move || {
            let _guard = safemlx::RuntimeCallDeadline::new(Duration::from_secs(1))
                .unwrap()
                .enter()
                .unwrap();
            worker_entered.wait();
            worker_release.wait();
        });
        entered.wait();
        let first = OpaqueFailureAgreement
            .agree_phase(
                &communication,
                id,
                DistributedExecutionPhase::Execution,
                true,
                &stream,
            )
            .expect_err("setup deadline must poison the selected communication authority");
        let retry = OpaqueFailureAgreement
            .agree_phase(
                &communication,
                id,
                DistributedExecutionPhase::Execution,
                true,
                &stream,
            )
            .expect_err("poisoned communication must reject retry before backend entry");
        assert!(matches!(
            first,
            eredu_runtime::PartitionExecutionError::CommunicationSubmissionFailed { .. }
        ));
        assert!(matches!(
            retry,
            eredu_runtime::PartitionExecutionError::CommunicationPoisoned { .. }
        ));
        assert_eq!(
            crate::backend::runtime::distributed::group::native_collective_submissions(),
            0
        );
        release.wait();
    });
}
