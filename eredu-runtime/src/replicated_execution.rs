//! Exact descriptor agreement and owner publication for replicated operations.
//!
//! This composes the ordinary partition communication authority. It owns no
//! model state, random stream, scheduler or commit boundary.

use std::{borrow::Borrow, cell::Cell};

use eredu_core::{BackendFailure, CollectiveGroupId};

use crate::{
    DistributedExecutionPhase, OpaqueFailureAgreement, OpaqueOutputPublisher,
    PartitionCommitAgreement, PartitionCommunication, PartitionOutputPublication,
    PartitionOutputPublisher,
};
use eredu_core::communication::CommunicationManifest;
use eredu_core::communication::CommunicationOperation;
use eredu_core::communication::{BroadcastBackend, FailureAgreementBackend, RankCommunication};

/// Mechanisms used by an architecture's replicated transition.
pub trait ReplicatedCoordinator<T, C: ?Sized> {
    /// Whether this rank alone produces authoritative decisions.
    fn is_owner(&self) -> bool;
    /// Completes exact descriptor and local-success agreement on every rank.
    fn agree(
        &mut self,
        descriptor: [u64; 8],
        success: bool,
        context: &C,
    ) -> Result<bool, BackendFailure>;
    /// Completes an owner broadcast; other ranks supply matching destination storage.
    fn publish(&mut self, value: T, context: &C) -> Result<T, BackendFailure>;
}

/// Invalid cold coordination selection.
#[derive(Debug, thiserror::Error)]
#[error("invalid replicated coordination: {0}")]
pub struct CoordinationError(pub &'static str);

/// Invalid cold authority for failure-coordinated tensor collectives.
#[derive(Debug, thiserror::Error)]
pub enum CollectiveCoordinationError {
    /// Membership or bounded completion is absent.
    #[error("invalid collective coordination: {0}")]
    Selection(&'static str),
    /// The retained operation declarations are malformed.
    #[error(transparent)]
    Manifest(#[from] eredu_core::communication::CommunicationManifestError),
    /// An exact operation, dtype, payload bound or completion requirement is absent.
    #[error(transparent)]
    Capability(#[from] eredu_core::communication::CommunicationCapabilityError),
}

/// Checks a selected group's failure authority and exact data requirements.
/// Families supply their required tensor geometry; this mechanism admits local
/// membership, bounded completion and failure agreement without native resources.
/// Actual invocation extents remain checked by the retained communication owner.
pub fn validate_collective_coordination(
    manifest: &CommunicationManifest,
    group: CollectiveGroupId,
    requirements: impl IntoIterator<Item = eredu_core::communication::CommunicationOperationRequirement>,
) -> Result<&eredu_core::communication::CommunicationGroupDescriptor, CollectiveCoordinationError> {
    let selected = manifest
        .groups()
        .iter()
        .find(|item| item.id() == group)
        .ok_or(CollectiveCoordinationError::Selection(
            "selected group is absent",
        ))?;
    if selected.local_index().is_none() || manifest.completion_policy().is_none() {
        return Err(CollectiveCoordinationError::Selection(
            "local membership and bounded completion are required",
        ));
    }
    let capabilities = eredu_core::communication::CommunicationCapabilities::new(
        selected.requirements().operations().iter().cloned(),
    )?;
    capabilities.validate_requirement(
        &eredu_core::communication::CommunicationOperationRequirement::failure_agreement(true),
    )?;
    for requirement in requirements {
        capabilities.validate_requirement(&requirement)?;
    }
    Ok(selected)
}

/// Checks membership, required operations and bounded completion without native work.
pub fn validate_coordination(
    manifest: &CommunicationManifest,
    group: CollectiveGroupId,
    owner: usize,
) -> Result<(), CoordinationError> {
    let descriptor = manifest
        .groups()
        .iter()
        .find(|item| item.id() == group)
        .ok_or(CoordinationError("selected group is absent"))?;
    if descriptor.local_index().is_none() || !descriptor.members().contains(&owner) {
        return Err(CoordinationError(
            "local rank or decision owner is not a member",
        ));
    }
    if manifest.completion_policy().is_none() {
        return Err(CoordinationError("bounded completion policy is required"));
    }
    for operation in [
        CommunicationOperation::FailureAgreement,
        CommunicationOperation::Broadcast,
    ] {
        if !descriptor
            .requirements()
            .operations()
            .iter()
            .any(|item| item.operation() == operation)
        {
            return Err(CoordinationError(
                "agreement and broadcast must both be admitted",
            ));
        }
    }
    Ok(())
}

/// Existing bounded agreement/publication policies over one retained authority.
pub struct OpaqueReplicatedCoordinator<C: RankCommunication> {
    communication: PartitionCommunication<C::Backend, C::Group, C::Route, C::Metadata>,
    parallel: C::Parallel,
    executor: <C::Backend as eredu_core::SubmissionBackend>::OwnedExecutor,
    group: CollectiveGroupId,
    owner: usize,
    selection_agreed: Cell<bool>,
}

impl<C: RankCommunication> OpaqueReplicatedCoordinator<C> {
    /// Exact retained operation and completion contract.
    pub fn manifest(&self) -> &CommunicationManifest {
        self.communication.manifest()
    }
    /// Selected participant group.
    pub fn group(&self) -> CollectiveGroupId {
        self.group
    }
    /// Exact native context for readiness-ordered tensor operations.
    pub fn parallel_context(&self) -> &C::Parallel {
        &self.parallel
    }

    /// Retains native rank resources selected by the portable coordinator.
    pub fn bind(
        rank: C,
        manifest: CommunicationManifest,
        group: CollectiveGroupId,
        owner: usize,
    ) -> Result<Self, BackendFailure> {
        validate_coordination(&manifest, group, owner).map_err(BackendFailure::from_error)?;
        let parallel = rank.coordinated_tensor_context(group).ok_or_else(|| {
            BackendFailure::from_error(CoordinationError(
                "the selected group cannot realize independently coordinated tensor operations",
            ))
        })?;
        let (communication, _, executor) = PartitionCommunication::bind_rank(rank, manifest, None)?;
        Ok(Self {
            communication,
            parallel,
            executor,
            group,
            owner,
            selection_agreed: Cell::new(false),
        })
    }
}

type Backend<C> = <C as RankCommunication>::Backend;
type Exec<C> = <Backend<C> as eredu_core::SubmissionBackend>::Executor;
type Value<C> = <Backend<C> as eredu_nn::NeuralBackend>::Tensor;

impl<C> ReplicatedCoordinator<Value<C>, Exec<C>> for OpaqueReplicatedCoordinator<C>
where
    C: RankCommunication<Backend: BroadcastBackend + FailureAgreementBackend>,
{
    fn is_owner(&self) -> bool {
        self.communication.manifest().rank() == self.owner
    }
    fn agree(
        &mut self,
        descriptor: [u64; 8],
        success: bool,
        _: &Exec<C>,
    ) -> Result<bool, BackendFailure> {
        self.agree_phase(descriptor, success)
    }
    fn publish(&mut self, value: Value<C>, _: &Exec<C>) -> Result<Value<C>, BackendFailure> {
        if !self.selection_agreed.get() {
            return Err(BackendFailure::from_error(CoordinationError(
                "decision owner has not been agreed",
            )));
        }
        OpaqueOutputPublisher
            .publish(
                &self.communication,
                value,
                PartitionOutputPublication {
                    group: self.group,
                    owner_rank: self.owner,
                },
                DistributedExecutionPhase::SamplingSynchronization,
                self.executor.borrow(),
            )
            .map_err(BackendFailure::from_error)
    }
}

impl<C: RankCommunication<Backend: FailureAgreementBackend>> OpaqueReplicatedCoordinator<C> {
    fn agree_phase(&self, descriptor: [u64; 8], success: bool) -> Result<bool, BackendFailure> {
        if !self.selection_agreed.get() {
            let accepted = OpaqueFailureAgreement
                .agree_descriptor(
                    &self.communication,
                    self.group,
                    DistributedExecutionPhase::InputPreparation,
                    true,
                    [
                        0x5245504c00010000,
                        u64::from(self.group.value()),
                        self.owner as u64,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                    self.executor.borrow(),
                )
                .map_err(BackendFailure::from_error)?;
            if !accepted {
                return Ok(false);
            }
            self.selection_agreed.set(true);
        }
        OpaqueFailureAgreement
            .agree_descriptor(
                &self.communication,
                self.group,
                DistributedExecutionPhase::Execution,
                success,
                descriptor,
                self.executor.borrow(),
            )
            .map_err(BackendFailure::from_error)
    }
}

impl<B, C> crate::LayeredTensorCohort<B> for OpaqueReplicatedCoordinator<C>
where
    B: FailureAgreementBackend<
        Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
    >,
    C: RankCommunication<Backend = B, Parallel = B::ParallelContext>,
    B::ParallelContext: Sized,
{
    fn selection(&self) -> (&CommunicationManifest, CollectiveGroupId) {
        (self.manifest(), self.group())
    }

    fn parallel_context(&self) -> &B::ParallelContext {
        &self.parallel
    }

    fn agree(
        &self,
        descriptor: [u64; 8],
        ready: bool,
        _: &B::Executor,
    ) -> Result<bool, BackendFailure> {
        self.agree_phase(descriptor, ready)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::communication::{
        CommunicationCapabilityError, CommunicationCompletionPolicy, CommunicationGroupDescriptor,
        CommunicationGroupRequirements, CommunicationOperationRequirement,
        CommunicationTensorLimits,
    };
    use eredu_core::{checkpoint::TensorDtype, CompletionCancellationMode};

    #[test]
    fn coordinated_collectives_require_failure_completion_and_exact_payload_capacity() {
        let id = CollectiveGroupId::new(9);
        let sum = |elements| {
            CommunicationOperationRequirement::tensors(
                CommunicationOperation::AllReduceSum,
                [TensorDtype::F32],
                CommunicationTensorLimits::new(1, 3, elements, None).unwrap(),
                true,
            )
            .unwrap()
        };
        let manifest = |elements, failures, bounded| {
            let mut operations = vec![sum(elements)];
            if failures {
                operations.push(CommunicationOperationRequirement::failure_agreement(true));
            }
            let group = CommunicationGroupDescriptor::new(
                id,
                0,
                vec![0, 1],
                Some(0),
                CommunicationGroupRequirements::new(operations).unwrap(),
            )
            .unwrap();
            let manifest = CommunicationManifest::new(2, 0, vec![group], vec![]).unwrap();
            if bounded {
                manifest.with_completion_policy(
                    CommunicationCompletionPolicy::new(
                        std::time::Duration::from_secs(1),
                        CompletionCancellationMode::QuarantineUntilComplete,
                    )
                    .unwrap(),
                )
            } else {
                manifest
            }
        };
        let admitted = manifest(8, true, true);
        assert_eq!(
            validate_collective_coordination(&admitted, id, [sum(8)])
                .unwrap()
                .members(),
            &[0, 1],
        );
        assert!(matches!(
            validate_collective_coordination(&manifest(7, true, true), id, [sum(8)]),
            Err(CollectiveCoordinationError::Capability(
                CommunicationCapabilityError::InsufficientLimits { .. }
            )),
        ));
        assert!(matches!(
            validate_collective_coordination(&manifest(8, false, true), id, [sum(8)]),
            Err(CollectiveCoordinationError::Capability(
                CommunicationCapabilityError::MissingOperation {
                    operation: CommunicationOperation::FailureAgreement,
                }
            )),
        ));
        assert!(matches!(
            validate_collective_coordination(&manifest(8, true, false), id, [sum(8)]),
            Err(CollectiveCoordinationError::Selection(_)),
        ));
    }
}
