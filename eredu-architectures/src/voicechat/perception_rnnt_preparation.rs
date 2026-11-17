//! Cold composition of perception and transcription sources and distributed authority.
use super::{
    perception_distribution::PreparedPerceptionPipelineBindings,
    perception_layered::PerceptionRuntimeState,
    perception_rnnt_pipeline::{
        PerceptionRnntMetadata, PerceptionRnntPipeline, PerceptionRnntResidency,
    },
    perception_stage::PerceptionPipelineArchitecture,
    rnnt_layered::RnntUnit,
    rnnt_predictor::RnntPredictorRuntimeState,
    rnnt_predictor_distribution::PreparedRnntPredictorPipelineBindings,
    rnnt_predictor_stage::RnntPredictorPipelineArchitecture,
    rnnt_preparation::RnntPreparationError,
    rnnt_projection_layered::RnntProjection,
    rnnt_projection_preparation::PreparedRnntProjectionBindings,
};
use eredu_core::{checkpoint::TensorDtype, BackendFailure, CollectiveGroupId};
use eredu_nn::{DistributedNeuralBackend, NeuralBackend, Tensor};
use eredu_runtime::{
    replicated_execution::{validate_coordination, CoordinationError, ReplicatedCoordinator},
    CommunicationCapabilities, CommunicationCapabilityError, CommunicationGroupDescriptor,
    CommunicationManifest, CommunicationManifestError, CommunicationOperation,
    CommunicationOperationRequirement, CommunicationTensorLimits, ComponentRankMechanisms,
    LayeredTensorCohort, SubmissionBackend, WeightBinding,
};
use std::collections::BTreeSet;

/// Inconsistent connected geometry or retained distributed authority.
#[derive(Debug, thiserror::Error)]
pub enum PerceptionRnntPreparationError {
    /// A projection selection or recurrent accounting contract is invalid.
    #[error(transparent)]
    Rnnt(#[from] RnntPreparationError),
    /// The decision authority cannot coordinate every selected participant.
    #[error(transparent)]
    Coordination(#[from] CoordinationError),
    /// A concrete decision or projection operation exceeds admitted capabilities.
    #[error(transparent)]
    Capability(#[from] CommunicationCapabilityError),
    /// The requested communication descriptor is invalid.
    #[error(transparent)]
    Manifest(#[from] CommunicationManifestError),
    /// Prepared component ownership or geometry differs.
    #[error("invalid perception/RNNT pipeline composition: {0}")]
    Contract(&'static str),
}

/// Exact rank-local materialized parameters and fully initialized F32 cache payloads.
/// Parameter replicas are charged separately on every participant. These logical
/// bytes exclude conversion temporaries, pool capacities and transport workspace,
/// which remain part of each selected component's ordinary residency report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PerceptionRnntPipelineResources {
    /// Parameters of this rank's selected perception units.
    pub perception_parameter_bytes: u64,
    /// Parameters of this rank's selected recurrent layers.
    pub predictor_parameter_bytes: u64,
    /// Embedding and joint parameters, including selected vocabulary shards.
    pub projection_parameter_bytes: u64,
    /// Full local Conformer caches for the single conversational stream.
    pub perception_cache_bytes: u64,
    /// Local hidden/cell carry plus separately retained published readout when needed.
    pub prediction_cache_bytes: u64,
}

/// Already prepared component sources plus explicit paired-frame decision authority.
/// Construction is metadata-only and preserves each component's restricted source.
pub struct PreparedPerceptionRnntPipelineBindings {
    perception: PreparedPerceptionPipelineBindings,
    predictor: PreparedRnntPredictorPipelineBindings,
    projections: PreparedRnntProjectionBindings,
    coordination: CommunicationManifest,
    decision_group: CollectiveGroupId,
    owner: usize,
    projection: Option<(CollectiveGroupId, usize)>,
    resources: PerceptionRnntPipelineResources,
}
fn group(
    manifest: &CommunicationManifest,
    id: CollectiveGroupId,
) -> Result<&CommunicationGroupDescriptor, PerceptionRnntPreparationError> {
    manifest
        .groups()
        .iter()
        .find(|group| group.id() == id)
        .ok_or(PerceptionRnntPreparationError::Contract(
            "selected participant group is absent",
        ))
}
fn parameter_bytes<'a>(
    bindings: impl Iterator<Item = &'a WeightBinding>,
) -> Result<u64, PerceptionRnntPreparationError> {
    bindings
        .filter(|binding| !binding.is_alias())
        .try_fold(0u64, |total, binding| {
            total.checked_add(binding.expected_bytes()).ok_or(
                PerceptionRnntPreparationError::Contract("parameter accounting overflow"),
            )
        })
}
impl PreparedPerceptionRnntPipelineBindings {
    /// Validates connected widths, participant identities, concrete decision limits
    /// and optional projection sharding before any checkpoint payload is read.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        perception: PreparedPerceptionPipelineBindings,
        predictor: PreparedRnntPredictorPipelineBindings,
        projections: PreparedRnntProjectionBindings,
        coordination: CommunicationManifest,
        decision_group: CollectiveGroupId,
        owner: usize,
        projection_group: Option<CollectiveGroupId>,
    ) -> Result<Self, PerceptionRnntPreparationError> {
        use PerceptionRnntPreparationError::Contract;
        let dimensions = perception.bindings().dimensions();
        let rnnt = predictor.bindings().dimensions();
        if dimensions.block.width != rnnt.encoder_width || rnnt != projections.dimensions() {
            return Err(Contract(
                "perception encoder and RNNT projection geometry differ",
            ));
        }
        validate_coordination(&coordination, decision_group, owner)?;
        let decision = group(&coordination, decision_group)?;
        let participants = decision.members().iter().copied().collect::<BTreeSet<_>>();
        for (manifest, plan) in [
            (perception.manifest(), perception.plan()),
            (predictor.manifest(), predictor.plan()),
        ] {
            if manifest.world_size() != coordination.world_size()
                || manifest.rank() != coordination.rank()
            {
                return Err(Contract("component world size or local rank differs"));
            }
            let publication = plan
                .publication()
                .ok_or(Contract("component output publication is absent"))?;
            if group(manifest, publication.group)?
                .members()
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                != participants
            {
                return Err(Contract(
                    "component output and decision participants differ",
                ));
            }
        }
        let capabilities =
            CommunicationCapabilities::new(decision.requirements().operations().iter().cloned())?;
        capabilities
            .validate_requirement(&CommunicationOperationRequirement::failure_agreement(true))?;
        capabilities.validate_requirement(&CommunicationOperationRequirement::tensors(
            CommunicationOperation::Broadcast,
            [TensorDtype::I32],
            CommunicationTensorLimits::new(1, 1, 1, None)?,
            true,
        )?)?;
        let projection_owner = projection_group
            .map(|id| group(&coordination, id).map(|cohort| cohort.members()[0]))
            .transpose()?;
        let projections = match projection_group {
            Some(id) => {
                let cohort = group(&coordination, id)?;
                validate_coordination(&coordination, id, projection_owner.unwrap())?;
                if !cohort
                    .members()
                    .iter()
                    .all(|rank| participants.contains(rank))
                {
                    return Err(Contract(
                        "projection cohort exceeds paired-frame participants",
                    ));
                }
                projections.select_parallel(&coordination, id)?
            }
            None if projections.local_layout().is_none() => projections,
            None => {
                return Err(Contract(
                    "tensor-local projections require an explicit cohort",
                ))
            }
        };
        let layers = perception
            .bindings()
            .unit_bindings()
            .filter(|(address, _)| address.group() == 1)
            .count() as u64;
        let perception_cache_bytes = (dimensions.block.left_context as u64)
            .checked_add((dimensions.block.conv_kernel - 1) as u64)
            .and_then(|rows| rows.checked_mul(dimensions.block.width as u64))
            .and_then(|elements| elements.checked_mul(4))
            .and_then(|bytes| bytes.checked_mul(layers))
            .ok_or(Contract("perception cache accounting overflow"))?;
        let resources = PerceptionRnntPipelineResources {
            perception_parameter_bytes: parameter_bytes(
                perception
                    .bindings()
                    .unit_bindings()
                    .flat_map(|(_, unit)| unit),
            )?,
            predictor_parameter_bytes: parameter_bytes(
                predictor
                    .bindings()
                    .unit_bindings()
                    .flat_map(|(_, unit)| unit),
            )?,
            projection_parameter_bytes: parameter_bytes(
                projections.unit_bindings().iter().flatten(),
            )?,
            perception_cache_bytes,
            prediction_cache_bytes: predictor.bindings().prediction_state_bytes()?,
        };
        Ok(Self {
            perception,
            predictor,
            projections,
            coordination,
            decision_group,
            owner,
            projection: projection_group.zip(projection_owner),
            resources,
        })
    }
    /// Exact local parameter and cache payload census for this retained selection.
    pub fn resources(&self) -> PerceptionRnntPipelineResources {
        self.resources
    }
    /// Binds all three components and the retained decision authority.
    ///
    /// The coordinator factory realizes one manifest and returns handles in
    /// selection order. The projection retains the selected tensor coordinator.
    pub fn bind<B, C, H, D>(
        self,
        residency: PerceptionRnntResidency,
        communication: &C,
        context: &<B::Tensor as Tensor>::Context,
        mechanisms: &H,
        bind_coordinators: impl FnOnce(
            CommunicationManifest,
            &[(CollectiveGroupId, usize)],
        ) -> Result<Vec<D>, BackendFailure>,
    ) -> Result<
        (
            PerceptionRnntPipeline<
                B,
                <H as ComponentRankMechanisms<B, PerceptionRuntimeState<B::Tensor>, C>>::Pipeline<
                    PerceptionPipelineArchitecture<B>,
                >,
                <H as ComponentRankMechanisms<B, RnntPredictorRuntimeState<B::Tensor>, C>>::Pipeline<
                    RnntPredictorPipelineArchitecture<B>,
                >,
                RnntProjection<B, H::UnitPolicy<RnntUnit<B>>, D>,
                D,
            >,
            PerceptionRnntMetadata,
        ),
        BackendFailure,
    >
    where
        B: DistributedNeuralBackend + SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context> + 'static,
        B::Tensor: 'static,
        H: ComponentRankMechanisms<B, PerceptionRuntimeState<B::Tensor>, C>
            + ComponentRankMechanisms<B, RnntPredictorRuntimeState<B::Tensor>, C>,
        D: ReplicatedCoordinator<B::Tensor, <B::Tensor as Tensor>::Context> + LayeredTensorCohort<B>,
{
        for policy in [
            residency.perception,
            residency.predictor,
            residency.projections,
        ] {
            policy.offload().map_err(BackendFailure::from_error)?;
        }
        <H as ComponentRankMechanisms<B, PerceptionRuntimeState<B::Tensor>, C>>::validate_pipeline_communication(
            mechanisms, self.perception.manifest(), self.perception.tensor_group(),
        )?;
        <H as ComponentRankMechanisms<B, RnntPredictorRuntimeState<B::Tensor>, C>>::validate_pipeline_communication(
            mechanisms, self.predictor.manifest(), self.predictor.tensor_group(),
        )?;

        let mut selections = vec![(self.decision_group, self.owner)];
        selections.extend(self.projection);
        let coordinators = bind_coordinators(self.coordination, &selections)?;
        if coordinators.len() != selections.len() {
            return Err(BackendFailure::from_error(CoordinationError(
                "coordinator count differs from the retained selection",
            )));
        }
        let mut coordinators = coordinators.into_iter();
        let decision = coordinators
            .next()
            .expect("decision coordinator was selected");
        let projection_coordinator = coordinators.next();
        let (perception, perception_metadata) =
            self.perception
                .bind::<B, _, _>(residency.perception, communication, mechanisms)?;
        let (predictor, predictor_metadata) =
            self.predictor
                .bind::<B, _, _>(residency.predictor, communication, mechanisms)?;
        let (projections, projection_metadata) = self.projections.bind::<B, _, _>(
            residency.projections,
            context,
            mechanisms,
            projection_coordinator,
        )?;
        Ok((
            PerceptionRnntPipeline {
                perception,
                predictor,
                projections,
                decision,
                resources: self.resources,
            },
            PerceptionRnntMetadata {
                perception: perception_metadata,
                predictor: predictor_metadata,
                projections: projection_metadata,
            },
        ))
    }
}
