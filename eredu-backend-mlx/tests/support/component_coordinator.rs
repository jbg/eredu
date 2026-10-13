//! Explicit bounded tensor and failure authority for compact component fixtures.
use eredu_backend_mlx::native::MlxReplicatedCoordinator;
use eredu_core::{checkpoint::TensorDtype, CollectiveGroupId, CompletionCancellationMode};
use eredu_runtime::{
    CommunicationCompletionPolicy, CommunicationGroupDescriptor, CommunicationGroupRequirements,
    CommunicationManifest, CommunicationOperation, CommunicationOperationRequirement,
    CommunicationTensorLimits,
};

#[allow(dead_code)] // Shared fixture; individual integration targets use different constructors.
pub fn coordinator(
    world: &safemlx::distributed::Group,
    stream: &safemlx::Stream,
) -> MlxReplicatedCoordinator {
    with_gather(world, stream, false)
}

/// Includes bounded F32 vocabulary gathering for independently leased heads.
#[allow(dead_code)] // Shared fixture; individual integration targets use different constructors.
pub fn coordinator_with_gather(
    world: &safemlx::distributed::Group,
    stream: &safemlx::Stream,
) -> MlxReplicatedCoordinator {
    with_gather(world, stream, true)
}

fn with_gather(
    world: &safemlx::distributed::Group,
    stream: &safemlx::Stream,
    gather: bool,
) -> MlxReplicatedCoordinator {
    let id = CollectiveGroupId::new(71);
    let mut operations = vec![
        CommunicationOperationRequirement::failure_agreement(true),
        CommunicationOperationRequirement::tensors(
            CommunicationOperation::AllReduceSum,
            [TensorDtype::F32],
            CommunicationTensorLimits::new(1, 3, 8192, None).unwrap(),
            true,
        )
        .unwrap(),
        CommunicationOperationRequirement::tensors(
            CommunicationOperation::Broadcast,
            [TensorDtype::I32],
            CommunicationTensorLimits::new(1, 1, 1, None).unwrap(),
            true,
        )
        .unwrap(),
    ];
    if gather {
        operations.push(
            CommunicationOperationRequirement::tensors(
                CommunicationOperation::AllGatherUneven,
                [TensorDtype::F32],
                CommunicationTensorLimits::new(1, 3, 8192, None)
                    .unwrap()
                    .with_output_tensor_elements(8192)
                    .unwrap(),
                true,
            )
            .unwrap(),
        );
    }
    let group = CommunicationGroupDescriptor::new(
        id,
        0,
        (0..world.size()).collect(),
        Some(world.rank()),
        CommunicationGroupRequirements::new(operations).unwrap(),
    )
    .unwrap();
    let manifest = CommunicationManifest::new(world.size(), world.rank(), vec![group], vec![])
        .unwrap()
        .with_completion_policy(
            CommunicationCompletionPolicy::new(
                std::time::Duration::from_secs(5),
                CompletionCancellationMode::QuarantineUntilComplete,
            )
            .unwrap(),
        );
    MlxReplicatedCoordinator::from_manifest(manifest, id, world.size() - 1, world, stream).unwrap()
}
