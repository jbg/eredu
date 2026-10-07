//! Native resources for shared replicated agreement and owner publication.

use super::{Group, topology::CommunicationRouteRealization};
use crate::{
    MlxTensor,
    backend::{
        distributed::MlxDistributedSession,
        nn::shared::{MlxCommunicationTensorMetadata, MlxNeuralBackend},
    },
};
use eredu_core::{BackendFailure, CollectiveGroupId};
use eredu_runtime::{
    CommunicationManifest, LayeredLocalPhase, LayeredTraversalHook,
    replicated_execution::{
        OpaqueReplicatedCoordinator, ReplicatedCoordinator, validate_coordination,
    },
};
use safemlx::Stream;
use std::{cell::RefCell, rc::Rc};

/// Retained native groups and the ordinary bounded communication authority.
pub struct MlxReplicatedCoordinator {
    inner: Rc<
        RefCell<
            OpaqueReplicatedCoordinator<
                MlxNeuralBackend,
                Group,
                CommunicationRouteRealization,
                MlxCommunicationTensorMetadata,
            >,
        >,
    >,
    manifest: CommunicationManifest,
    group: CollectiveGroupId,
    stream: Stream,
    parallel: Group,
}
impl MlxReplicatedCoordinator {
    /// Exact retained operation and completion contract for cold component checks.
    pub fn manifest(&self) -> &CommunicationManifest {
        &self.manifest
    }
    /// Selected participant group identity.
    pub fn group(&self) -> CollectiveGroupId {
        self.group
    }
    /// Realizes an explicitly selected manifest without inferring model policy.
    pub fn from_manifest(
        manifest: CommunicationManifest,
        group: CollectiveGroupId,
        owner: usize,
        world: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Self, BackendFailure> {
        validate_coordination(&manifest, group, owner).map_err(BackendFailure::from_error)?;
        let distributed = MlxDistributedSession::from_manifest(&manifest, world, stream)
            .map_err(BackendFailure::from_error)?;
        Self::from_retained_session(distributed, manifest, group, owner)
    }

    /// Realizes the manifest once, then retains independent logical coordinator
    /// handles over the same completion authority and native resources. The
    /// number of local selections need not match other ranks' selections.
    pub(crate) fn from_manifest_selections(
        manifest: CommunicationManifest,
        selections: &[(CollectiveGroupId, usize)],
        world: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Vec<Self>, BackendFailure> {
        for &(group, owner) in selections {
            validate_coordination(&manifest, group, owner).map_err(BackendFailure::from_error)?;
            let selected = manifest
                .groups()
                .iter()
                .find(|item| item.id() == group)
                .expect("validated coordination group exists");
            if !super::independent_status_members(manifest.world_size(), selected.members()) {
                return Err(BackendFailure::from_error(
                    eredu_runtime::replicated_execution::CoordinationError(
                        "native coordinator requires status agreement independent of inactive ranks",
                    ),
                ));
            }
        }
        let distributed = MlxDistributedSession::from_manifest(&manifest, world, stream)
            .map_err(BackendFailure::from_error)?;
        selections
            .iter()
            .map(|&(group, owner)| {
                Self::from_retained_session(distributed.clone(), manifest.clone(), group, owner)
            })
            .collect()
    }

    fn from_retained_session(
        distributed: MlxDistributedSession,
        manifest: CommunicationManifest,
        group: CollectiveGroupId,
        owner: usize,
    ) -> Result<Self, BackendFailure> {
        let (communication, parallel, stream) = distributed
            .into_component_communication(manifest.clone(), Some(group))
            .map_err(BackendFailure::from_error)?;
        let inner = Rc::new(RefCell::new(
            OpaqueReplicatedCoordinator::new(communication, group, owner)
                .map_err(BackendFailure::from_error)?,
        ));
        let authority = Rc::clone(&inner);
        let agreement_stream = stream.clone();
        // The authority retains the original callback-free groups. Only the
        // executable clone points back to it, so no ownership cycle is formed.
        let parallel = parallel
            .expect("selected communication group was realized")
            .with_local_phase_agreement(Rc::new(move |descriptor, success, _| {
                authority
                    .borrow_mut()
                    .agree(descriptor, success, &agreement_stream)
            }));
        Ok(Self {
            inner,
            manifest,
            group,
            stream,
            parallel,
        })
    }

    /// Borrows the selected native context without dropping its bounded failure
    /// and completion authority. Low-level architecture bindings use this group
    /// for numerical collectives and this coordinator for phase agreements.
    pub fn parallel_group(&self) -> &Group {
        &self.parallel
    }
}
impl ReplicatedCoordinator<MlxTensor, Stream> for MlxReplicatedCoordinator {
    fn is_owner(&self) -> bool {
        self.inner.borrow().is_owner()
    }
    fn agree(
        &mut self,
        descriptor: [u64; 8],
        success: bool,
        _: &Stream,
    ) -> Result<bool, BackendFailure> {
        self.inner
            .borrow_mut()
            .agree(descriptor, success, &self.stream)
    }
    fn publish(&mut self, value: MlxTensor, _: &Stream) -> Result<MlxTensor, BackendFailure> {
        self.inner.borrow_mut().publish(value, &self.stream)
    }
}

/// Shares the selected communicator's retained failure and completion authority
/// across the model-independent layered lifecycle.
pub(crate) struct MlxLayeredAgreement<'a>(&'a Group);
impl<'a> MlxLayeredAgreement<'a> {
    /// Borrows a group carrying the selected bounded failure authority.
    pub(crate) fn new(group: &'a Group) -> Self {
        Self(group)
    }
}
impl<C, E> LayeredTraversalHook<MlxNeuralBackend, C, E> for MlxLayeredAgreement<'_>
where
    E: From<BackendFailure>,
{
    fn local_phase(
        &mut self,
        phase: LayeredLocalPhase,
        success: bool,
        context: &Stream,
    ) -> Result<(), E> {
        let (kind, group, index) = match phase {
            LayeredLocalPhase::Initialization => (0, 0, 0),
            LayeredLocalPhase::PolicyBegin => (1, 0, 0),
            LayeredLocalPhase::GroupIngress { group } => (2, group, 0),
            LayeredLocalPhase::BeforeUnit { group, index } => (3, group, index),
            LayeredLocalPhase::Acquisition { group, index } => (4, group, index),
            LayeredLocalPhase::UnitExecution { group, index } => (5, group, index),
            LayeredLocalPhase::UnitCompletion { group, index } => (6, group, index),
            LayeredLocalPhase::GroupCompletion { group } => (7, group, 0),
            LayeredLocalPhase::Finish => (8, 0, 0),
        };
        let accepted = self
            .0
            .agree_local_phase(
                [
                    0x4c415945524544,
                    kind,
                    group as u64,
                    index as u64,
                    0,
                    0,
                    0,
                    0,
                ],
                success,
                context,
            )
            .map_err(E::from)?;
        if accepted {
            Ok(())
        } else {
            Err(BackendFailure::from_error(eredu_nn::Error::backend(
                format!(
                    "a participant failed local layered preparation at {phase:?} (local success: {success})"
                ),
            ))
            .into())
        }
    }
}
