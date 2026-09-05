use super::*;
use eredu_core::{
    consensus::BoundedConsensusTransport as _, BoundedCompletionWait, BoundedSubmissionOutcome,
    CompletionCancellationMode,
};
use safemlx::{Device, DeviceType};

fn singleton_data_manifest() -> (
    eredu_runtime::CommunicationManifest,
    eredu_core::CollectiveGroupId,
) {
    let id = eredu_core::CollectiveGroupId::new(41);
    let requirement = eredu_runtime::CommunicationOperationRequirement::tensors(
        eredu_runtime::CommunicationOperation::AllReduceSum,
        [eredu_core::checkpoint::TensorDtype::F32],
        eredu_runtime::CommunicationTensorLimits::new(1, 1, 8, None).unwrap(),
        true,
    )
    .unwrap();
    let group = eredu_runtime::CommunicationGroupDescriptor::new(
        id,
        0,
        vec![0],
        Some(0),
        eredu_runtime::CommunicationGroupRequirements::new([requirement]).unwrap(),
    )
    .unwrap();
    let policy = eredu_runtime::CommunicationCompletionPolicy::new(
        std::time::Duration::from_secs(1),
        CompletionCancellationMode::QuarantineUntilComplete,
    )
    .unwrap();
    (
        eredu_runtime::CommunicationManifest::new(1, 0, vec![group], Vec::new())
            .unwrap()
            .with_completion_policy(policy),
        id,
    )
}

#[test]
fn bounded_consensus_submission_retains_exact_native_work_until_resolution() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let world = safemlx::distributed::init(false, safemlx::distributed::Backend::Ring)
        .expect("singleton Ring world should initialize");
    let manifest = eredu_runtime::CommunicationManifest::new(1, 0, Vec::new(), Vec::new())
        .unwrap()
        .with_completion_policy(
            eredu_runtime::CommunicationCompletionPolicy::new(
                std::time::Duration::from_secs(1),
                CompletionCancellationMode::QuarantineUntilComplete,
            )
            .unwrap(),
        );
    let session = MlxDistributedSession::from_manifest(&manifest, &world, &stream).unwrap();
    let submission = session
        .submit_all_gather_words(&[0, 1, u32::MAX, 0x8000_0000])
        .unwrap();
    assert_eq!(submission.completion.retained_arrays(), 2);
    assert_eq!(submission.completion.retained_groups(), 1);
    assert_eq!(submission.completion.retained_streams(), 1);
    let wait = BoundedCompletionWait::new(
        std::time::Duration::from_secs(1),
        CompletionCancellationMode::QuarantineUntilComplete,
    )
    .unwrap();
    let output = match submission.wait_bounded(wait).unwrap() {
        BoundedSubmissionOutcome::Completed(output) => output,
        BoundedSubmissionOutcome::DeadlineExceeded { cancellation } => {
            panic!("singleton consensus timed out with {cancellation:?}")
        }
    };
    assert_eq!(
        session.resolve_all_gather_words(output).unwrap(),
        [0, 1, u32::MAX, 0x8000_0000]
    );

    let completion = eredu_runtime::CommunicationCompletionPolicy::new(
        std::time::Duration::from_secs(1),
        CompletionCancellationMode::QuarantineUntilComplete,
    )
    .unwrap();
    let manifest = eredu_runtime::CommunicationManifest::new(1, 0, Vec::new(), Vec::new())
        .unwrap()
        .with_completion_policy(completion);
    let manifest_session = MlxDistributedSession::from_manifest(&manifest, &world, &stream)
        .expect("manifest session should retain a control-only world handle");
    assert!(!DistributedSession::capabilities(&manifest_session).world_collectives());
    assert!(manifest_session.group(CollectiveScope::World).is_err());
    let output = manifest_session
        .submit_all_gather_words(&[17, u32::MAX])
        .unwrap()
        .wait_bounded(wait)
        .unwrap();
    let output = match output {
        BoundedSubmissionOutcome::Completed(output) => output,
        BoundedSubmissionOutcome::DeadlineExceeded { cancellation } => {
            panic!("manifest consensus timed out with {cancellation:?}")
        }
    };
    assert_eq!(
        manifest_session.resolve_all_gather_words(output).unwrap(),
        [17, u32::MAX]
    );
}

#[test]
fn retained_public_view_and_partition_runtime_share_one_poison_authority() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let world = safemlx::distributed::init(false, safemlx::distributed::Backend::Ring)
        .expect("singleton Ring world should initialize");
    let (manifest, group) = singleton_data_manifest();
    let session = MlxDistributedSession::from_manifest(&manifest, &world, &stream).unwrap();
    let retained = session.clone();
    let (communication, _, _, _) = session
        .into_partition_communication(manifest, None, group)
        .unwrap();

    let _ = retained.authority.completion_error(
        "injected public completion failure",
        eredu_runtime::CommunicationOperation::AllReduceSum,
        eredu_runtime::DistributedExecutionPhase::Execution,
        None,
    );
    assert!(communication.authority().ensure_active().is_err());
}

#[test]
fn public_collective_completion_retains_inputs_group_and_stream_after_session_drop() {
    use eredu_core::Completion as _;

    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let world = safemlx::distributed::init(false, safemlx::distributed::Backend::Ring)
        .expect("singleton Ring world should initialize");
    let (manifest, group) = singleton_data_manifest();
    let session = MlxDistributedSession::from_manifest(&manifest, &world, &stream).unwrap();
    let input = MlxTensor::from_array(Array::from_slice(&[3.0_f32], &[1]));
    let submission = session
        .all_reduce_sum(CollectiveScope::Group(group), &input)
        .unwrap();
    assert_eq!(submission.completion.retained_resources(), 2);
    assert_eq!(
        submission.completion.retained_native_resources(),
        (0, 1, 0, 1)
    );
    drop(session);
    drop(input);
    submission.completion.wait().unwrap();
}

#[test]
fn public_collective_wait_uses_manifest_deadline_and_shared_poison() {
    use eredu_core::Completion as _;

    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let world = safemlx::distributed::init(false, safemlx::distributed::Backend::Ring)
        .expect("singleton Ring world should initialize");
    let (manifest, group) = singleton_data_manifest();
    let manifest = manifest.with_completion_policy(
        eredu_runtime::CommunicationCompletionPolicy::new(
            std::time::Duration::from_millis(1),
            CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap(),
    );
    let session = MlxDistributedSession::from_manifest(&manifest, &world, &stream).unwrap();
    crate::backend::runtime::distributed::completion::force_next_communication_pending();
    let input = MlxTensor::from_array(Array::from_slice(&[3.0_f32], &[1]));
    let submission = session
        .all_reduce_sum(CollectiveScope::Group(group), &input)
        .unwrap();
    assert!(submission.completion.wait().is_err());
    assert!(submission.completion.wait_on(&stream).is_err());
    assert!(session.authority.ensure_active().is_err());
    assert_eq!(
        crate::backend::runtime::distributed::completion::distributed_completion_orphan_count(),
        1
    );
    crate::backend::runtime::distributed::completion::release_forced_pending_orphans();
    assert_eq!(
        crate::backend::runtime::distributed::completion::distributed_completion_orphan_count(),
        0
    );
}
