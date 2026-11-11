//! Communication and publication for the Audio language pipeline.

use super::PreparedAudioLanguagePartition;
use eredu_core::{
    checkpoint::TensorDtype, BackendFailure, CollectiveGroupId, CompletionCancellationMode,
    ParallelRankTopology,
};
use eredu_runtime::{
    project_communication_manifest, ArchitectureBoundary, ArchitectureGroupKind,
    BoundaryDimensionContract, BoundaryRoleContract, CommunicationCompletionPolicy,
    CommunicationGroupRequirements, CommunicationManifest, CommunicationOperation,
    CommunicationOperationRequirement, CommunicationRouteDescriptor, CommunicationRouteId,
    CommunicationTensorLimits, LayeredPartitionDriver, PartitionBoundaryRoute,
    PartitionOutputPublication, PartitionedExecutionPlan, PipelineActivationDtype,
    PipelineWireContract, RoleExactBoundaryContract, TopologyCommunicationPlan,
};
use std::time::Duration;

fn failure(error: impl std::error::Error + Send + Sync + 'static) -> BackendFailure {
    BackendFailure::from_error(error)
}

/// Retained communication selection for one Audio language stage.
pub struct AudioPipelineCommunication {
    /// Native realization consumes this exact manifest.
    pub manifest: CommunicationManifest,
    /// Shared pipeline driver and all-rank publication authority.
    pub plan: PartitionedExecutionPlan,
    /// Collective for ranks that share this pipeline stage.
    pub tensor_group: Option<CollectiveGroupId>,
}

impl PreparedAudioLanguagePartition {
    /// Selects F32 activation routes and all-rank completion for one admitted stage.
    pub fn pipeline_communication(&self) -> Result<AudioPipelineCommunication, BackendFailure> {
        let rank = self.rank();
        let topology = rank.topology();
        let args = self.args();
        let partition = self.partition();
        let hidden = args.hidden_size as usize;
        let context = args.max_position_embeddings as usize;
        let vocab = args.vocab_size as usize;
        let max_hidden = hidden.checked_mul(context).ok_or_else(|| {
            failure(eredu_nn::Error::backend(
                "Audio pipeline hidden extent overflow",
            ))
        })?;
        let tensor_op = |op, elements, output| {
            CommunicationTensorLimits::new(1, 3, elements, None)
                .and_then(|limits| limits.with_output_tensor_elements(output))
                .and_then(|limits| {
                    CommunicationOperationRequirement::tensors(op, [TensorDtype::F32], limits, true)
                })
                .map_err(failure)
        };
        let agree = CommunicationOperationRequirement::failure_agreement(true);
        let broadcast = tensor_op(
            CommunicationOperation::Broadcast,
            max_hidden.max(vocab),
            max_hidden.max(vocab),
        )?;
        let policy = CommunicationCompletionPolicy::new(
            Duration::from_secs(120),
            CompletionCancellationMode::QuarantineUntilComplete,
        )
        .map_err(failure)?;
        let mut projection = TopologyCommunicationPlan::new()
            .with_completion_policy(policy)
            .with_session_group(
                CommunicationGroupRequirements::new([broadcast, agree.clone()]).map_err(failure)?,
            );
        if topology.tensor() > 1 {
            let reduce = tensor_op(CommunicationOperation::AllReduceSum, max_hidden, max_hidden)?;
            let gather = tensor_op(
                CommunicationOperation::AllGatherUneven,
                vocab.div_ceil(topology.tensor()),
                vocab,
            )?;
            projection = projection.with_tensor_groups(
                CommunicationGroupRequirements::new([reduce, gather, agree]).map_err(failure)?,
            );
        }
        let projected =
            project_communication_manifest(topology, rank, &projection).map_err(failure)?;
        let tensor_group = projection
            .tensor_group_id(topology, rank)
            .map_err(failure)?;
        let session = projection
            .session_group_id()
            .expect("declared Audio session group");
        let output_rank = topology.world_size() - topology.tensor();
        let schema = partition.boundary_schema().wire_schema().map_err(failure)?;
        let contract = RoleExactBoundaryContract::new(
            schema.identity(),
            [BoundaryRoleContract::symbolic(
                schema.primary().role(),
                TensorDtype::F32,
                vec![
                    BoundaryDimensionContract::Fixed(1),
                    BoundaryDimensionContract::Variable { maximum: context },
                    BoundaryDimensionContract::Fixed(hidden),
                ],
            )
            .map_err(failure)?],
        )
        .map_err(failure)?;
        let requirement = tensor_op(CommunicationOperation::SendReceive, max_hidden, max_hidden)?;
        let mut descriptors = Vec::new();
        let mut routes = Vec::new();
        for source in 0..topology.world_size() {
            let coordinates = ParallelRankTopology::new(topology, source).map_err(failure)?;
            if let Some(destination) = coordinates.pipeline_successor().map_err(failure)? {
                let order = routes.len();
                let id = CommunicationRouteId::new(order as u64);
                let descriptor = CommunicationRouteDescriptor::new(
                    id,
                    order,
                    source,
                    destination,
                    requirement.clone(),
                )
                .and_then(|route| route.with_boundary_contract(contract.clone()))
                .map_err(failure)?;
                descriptors.push(if topology.pipeline() > 2 && topology.tensor() > 1 {
                    descriptor.with_world_broadcast()
                } else {
                    descriptor
                });
                routes.push(PartitionBoundaryRoute {
                    source_group: 0,
                    destination_group: 0,
                    source_rank: source,
                    destination_rank: destination,
                    route: id,
                });
            }
        }
        let manifest = CommunicationManifest::new(
            topology.world_size(),
            rank.global_rank(),
            projected.groups().to_vec(),
            descriptors,
        )
        .map_err(failure)?
        .with_completion_policy(policy);
        let driver =
            LayeredPartitionDriver::new(partition, 0, partition.groups()[0].global_units())
                .map_err(failure)?;
        let plan = PartitionedExecutionPlan::new(
            partition.graph().clone(),
            vec![(ArchitectureGroupKind::Decoder, false)],
            vec![Some(driver)],
            routes,
            Some(PartitionOutputPublication {
                group: session,
                owner_rank: output_rank,
            }),
            Some(session),
            PipelineWireContract::new(PipelineActivationDtype::Float32),
        )
        .map_err(failure)?;
        plan.validate_manifest(&manifest).map_err(failure)?;
        Ok(AudioPipelineCommunication {
            manifest,
            plan,
            tensor_group,
        })
    }
}
