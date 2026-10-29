//! Cold composition of perception and transcription sources and distributed authority.
use super::{
    perception_distribution::PreparedPerceptionPipelineBindings,
    rnnt_predictor_distribution::PreparedRnntPredictorPipelineBindings,
    rnnt_preparation::RnntPreparationError,
    rnnt_projection_preparation::{
        validate_projection_parallel_communication, PreparedRnntProjectionBindings,
    },
};
use eredu_core::{
    checkpoint::TensorDtype, CollectiveGroupId, ParallelRankTopology, ParallelTopology,
};
use eredu_runtime::{
    replicated_execution::{validate_coordination, CoordinationError},
    CommunicationCapabilities, CommunicationCapabilityError, CommunicationGroupDescriptor,
    CommunicationManifest, CommunicationManifestError, CommunicationOperation,
    CommunicationOperationRequirement, CommunicationTensorLimits, WeightBinding,
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
    projection_group: Option<CollectiveGroupId>,
    projection_owner: Option<usize>,
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
                validate_projection_parallel_communication(rnnt, &coordination, id)?;
                let cohort = group(&coordination, id)?;
                validate_coordination(&coordination, id, projection_owner.unwrap())?;
                let capabilities = CommunicationCapabilities::new(
                    cohort.requirements().operations().iter().cloned(),
                )?;
                capabilities.validate_requirement(
                    &CommunicationOperationRequirement::failure_agreement(true),
                )?;
                capabilities.validate_requirement(&CommunicationOperationRequirement::tensors(
                    CommunicationOperation::Broadcast,
                    [TensorDtype::I32],
                    CommunicationTensorLimits::new(1, 1, 1, None)?,
                    true,
                )?)?;
                if !cohort
                    .members()
                    .iter()
                    .all(|rank| participants.contains(rank))
                {
                    return Err(Contract(
                        "projection cohort exceeds paired-frame participants",
                    ));
                }
                let topology = ParallelTopology::new(cohort.members().len(), 1, 1, 1)
                    .and_then(|topology| {
                        ParallelRankTopology::new(topology, cohort.local_index().unwrap())
                    })
                    .map_err(|_| Contract("invalid projection tensor topology"))?;
                projections.select_parallel(topology)?.0
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
            projection_group,
            projection_owner,
            resources,
        })
    }
    /// Exact local parameter and cache payload census for this retained selection.
    pub fn resources(&self) -> PerceptionRnntPipelineResources {
        self.resources
    }
    /// Architecture-declared authority for the optional projection tensor cohort.
    /// This owner is retained at preparation rather than selected during binding.
    pub fn projection_owner(&self) -> Option<usize> {
        self.projection_owner
    }
    /// Retains every source and communication object for ordinary component materializers.
    pub fn into_parts(
        self,
    ) -> (
        PreparedPerceptionPipelineBindings,
        PreparedRnntPredictorPipelineBindings,
        PreparedRnntProjectionBindings,
        CommunicationManifest,
        CollectiveGroupId,
        usize,
        Option<CollectiveGroupId>,
    ) {
        (
            self.perception,
            self.predictor,
            self.projections,
            self.coordination,
            self.decision_group,
            self.owner,
            self.projection_group,
        )
    }
}
