//! Exact inputs to native communication realization.

use std::sync::Arc;

use super::{
    CommunicationAuthority, CommunicationCompletionPolicy, CommunicationError,
    CommunicationGroupDescriptor, CommunicationManifest, CommunicationRouteDescriptor,
    CommunicationSessionIdentity,
};

/// A selected native binding is missing a realization choice for a descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("communication realization choices do not cover the manifest")]
pub struct CommunicationBindingGeometryError;

/// Invalid inputs at the boundary between communication planning and realization.
#[derive(Debug, thiserror::Error)]
pub enum CommunicationBindingError {
    /// The world requires an explicit completion bound.
    #[error("communication manifest requires an explicit bounded completion policy")]
    MissingCompletion,
    /// Could not establish the shared authority for the selected manifest.
    #[error(transparent)]
    Authority(#[from] CommunicationError),
}

/// Exact local group and route bindings in the order selected for realization.
#[derive(Debug, Clone)]
pub struct CommunicationBindings {
    manifest: CommunicationManifest,
    group_world_waves: Vec<bool>,
    route_world_waves: Vec<bool>,
}

impl CommunicationBindings {
    /// Couples each selected descriptor to its exact native world-wave choice.
    ///
    /// The caller has already agreed on the manifest and selected reachable operations.
    pub fn new(
        manifest: CommunicationManifest,
        group_world_waves: Vec<bool>,
        route_world_waves: Vec<bool>,
    ) -> Result<Self, CommunicationBindingGeometryError> {
        if manifest.groups().len() != group_world_waves.len()
            || manifest.routes().len() != route_world_waves.len()
        {
            return Err(CommunicationBindingGeometryError);
        }
        Ok(Self {
            manifest,
            group_world_waves,
            route_world_waves,
        })
    }

    /// Exact local manifest used to bind the native world.
    pub const fn manifest(&self) -> &CommunicationManifest {
        &self.manifest
    }

    /// Whether one group uses a complete-world collective wave.
    pub fn group_world_wave(&self, creation_order: usize) -> Option<bool> {
        self.group_world_waves.get(creation_order).copied()
    }

    /// Whether one route uses a complete-world collective wave.
    pub fn route_world_wave(&self, submission_order: usize) -> Option<bool> {
        self.route_world_waves.get(submission_order).copied()
    }

    /// Borrows exact group bindings in creation order.
    pub fn groups(&self) -> impl ExactSizeIterator<Item = (&CommunicationGroupDescriptor, bool)> {
        self.manifest
            .groups()
            .iter()
            .zip(self.group_world_waves.iter().copied())
    }

    /// Borrows exact route bindings in submission order.
    pub fn routes(&self) -> impl ExactSizeIterator<Item = (&CommunicationRouteDescriptor, bool)> {
        self.manifest
            .routes()
            .iter()
            .zip(self.route_world_waves.iter().copied())
    }
}

/// Native world transport and agreed bindings ready for resource creation.
pub struct WorldCommunicationBinding<T> {
    transport: T,
    bindings: CommunicationBindings,
    completion: CommunicationCompletionPolicy,
    descriptors: Arc<[CommunicationGroupDescriptor]>,
    identity: CommunicationSessionIdentity,
    authority: CommunicationAuthority,
}

impl<T> WorldCommunicationBinding<T> {
    /// Couples agreed descriptors and world identity to the exact native transport.
    pub fn new(
        transport: T,
        bindings: CommunicationBindings,
        descriptors: Arc<[CommunicationGroupDescriptor]>,
        identity: CommunicationSessionIdentity,
    ) -> Result<Self, CommunicationBindingError> {
        let manifest = bindings.manifest();
        let completion = manifest
            .completion_policy()
            .ok_or(CommunicationBindingError::MissingCompletion)?;
        let authority = CommunicationAuthority::from_manifest(manifest)?;
        Ok(Self {
            transport,
            bindings,
            completion,
            descriptors,
            identity,
            authority,
        })
    }

    /// Exact local manifest used to realize the native resources.
    pub const fn manifest(&self) -> &CommunicationManifest {
        self.bindings.manifest()
    }

    /// Releases the agreed bindings and retained authority to native realization.
    #[allow(clippy::type_complexity)]
    pub fn into_parts(
        self,
    ) -> (
        T,
        CommunicationBindings,
        CommunicationCompletionPolicy,
        Arc<[CommunicationGroupDescriptor]>,
        CommunicationSessionIdentity,
        CommunicationAuthority,
    ) {
        (
            self.transport,
            self.bindings,
            self.completion,
            self.descriptors,
            self.identity,
            self.authority,
        )
    }
}

/// A native group paired with its agreed opaque identity.
pub struct RealizedCommunicationGroup<G> {
    /// Opaque group identity.
    pub id: crate::CollectiveGroupId,
    /// Native group retained by the execution owner.
    pub resource: G,
}

impl<G> RealizedCommunicationGroup<G> {
    /// Pairs an exact group with its manifest identity.
    pub const fn new(id: crate::CollectiveGroupId, resource: G) -> Self {
        Self { id, resource }
    }
}

/// A native directed route paired with its agreed opaque identity.
pub struct RealizedCommunicationRoute<R> {
    /// Opaque route identity.
    pub id: super::CommunicationRouteId,
    /// Native route retained by the execution owner.
    pub resource: R,
}

impl<R> RealizedCommunicationRoute<R> {
    /// Pairs an exact route with its manifest identity.
    pub const fn new(id: super::CommunicationRouteId, resource: R) -> Self {
        Self { id, resource }
    }
}

/// Owned native handles in manifest order with their shared completion authority.
pub struct RankCommunicationResources<B: super::CommunicationBackend, G, R, I> {
    /// Realized groups with stable selected identities.
    pub groups: Vec<RealizedCommunicationGroup<G>>,
    /// Realized directed routes with stable selected identities.
    pub routes: Vec<RealizedCommunicationRoute<R>>,
    /// Native tensor metadata for wire validation.
    pub metadata: I,
    /// Completion and failure authority shared with adjacent operations.
    pub authority: CommunicationAuthority,
    /// Native executor for communication submissions.
    pub executor: B::OwnedExecutor,
}

/// Native access to the exact resources realized for one rank.
pub trait RankCommunication: Sized {
    /// Backend submitting operations on these resources.
    type Backend: super::CommunicationBackend;
    /// Owned communicator.
    type Group: std::borrow::Borrow<<Self::Backend as super::CommunicationBackend>::CommunicationGroup>
        + Clone
        + 'static;
    /// Owned directed route.
    type Route: std::borrow::Borrow<<Self::Backend as super::CommunicationBackend>::CommunicationRoute>
        + 'static;
    /// Native context for tensor-partitioned operators.
    type Parallel;
    /// Native tensor metadata reader.
    type Metadata: super::CommunicationTensorMetadata<Self::Backend> + 'static;
    /// Failure to supply the retained native resources.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Borrows one exact group by manifest identity, including singleton groups.
    fn bound_group(&self, id: crate::CollectiveGroupId) -> Option<&Self::Group>;

    /// Borrows or clones the native operator context for one already realized group.
    fn tensor_context(&self, id: crate::CollectiveGroupId) -> Option<Self::Parallel>;

    /// Realizes an operator context with exact readiness and data completion.
    /// Returns no context when the selected group cannot agree independently.
    fn coordinated_tensor_context(&self, id: crate::CollectiveGroupId) -> Option<Self::Parallel>;

    /// Releases exactly the native resources retained for the supplied manifest.
    #[allow(clippy::type_complexity)]
    fn into_rank_resources(
        self,
        manifest: &CommunicationManifest,
    ) -> Result<
        RankCommunicationResources<Self::Backend, Self::Group, Self::Route, Self::Metadata>,
        Self::Error,
    >;
}
