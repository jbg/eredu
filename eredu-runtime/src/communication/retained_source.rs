//! Immutable realization lineage retained by actual native group/route owners.
use super::*;
use std::{mem::size_of, sync::Arc};

#[derive(Debug)]
struct Source {
    realization: PreparedCommunicationRealization,
    session: CommunicationSessionIdentity,
}

/// The same checked setup source, retained after native realization and moves
/// into partition runtimes. This is descriptive provenance, never a native
/// allocation, completed communication, or permission to submit an operation.
#[derive(Clone, Debug)]
pub struct RetainedCommunicationSource(Arc<Source>, eredu_core::HostPreparationAuthority);

/// A retained realization cannot acquire a different setup's world geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("communication setup identity and retained realization have different world geometry")]
pub struct RetainedCommunicationSourceError;

impl PreparedCommunicationRealization {
    /// Move the checked source into its immutable owner during initial native
    /// construction. Consumers retain this owner instead of reconstructing the
    /// manifest or inferring source identity from equal descriptor contents.
    pub fn into_retained_source(self, session: CommunicationSessionIdentity)
        -> Result<RetainedCommunicationSource, RetainedCommunicationSourceError> {
        if self.manifest.world_size != session.participant_count() {
            return Err(RetainedCommunicationSourceError);
        }
        Ok(RetainedCommunicationSource(Arc::new(Source { realization: self, session }),
            eredu_core::HostPreparationAuthority::unmanaged()))
    }
}
impl RetainedCommunicationSource {
    /// Retain already-paid, payload-free lifetime custody on a new alias of
    /// this exact source. No source identity, storage bound, native ownership,
    /// registration or submission authority is created by this attachment.
    ///
    /// Refuses to replace existing custody. The caller owns and prices the
    /// authority's construction and retains it through any refusal. Cloning
    /// the result carries that same custody through later escaped results.
    pub fn with_host_preparation(&self, authority: &eredu_core::HostPreparationAuthority)
        -> Option<Self> {
        self.1.is_unmanaged().then(|| Self(self.0.clone(), authority.clone()))
    }

    /// Exact checked source identity; independently prepared equal manifests
    /// remain different sources even when they name the same setup transcript.
    pub fn same_source(&self, other: &Self) -> bool { Arc::ptr_eq(&self.0, &other.0) }
    /// The immutable source used by the existing native construction worker.
    pub fn realization(&self) -> &PreparedCommunicationRealization { &self.0.realization }
    /// Exact rank-local declarations and completion policy.
    pub fn manifest(&self) -> &CommunicationManifest { self.realization().manifest() }
    /// Descriptive setup transcript identity, distinct from source ownership.
    pub fn session_identity(&self) -> CommunicationSessionIdentity { self.0.session }
    /// Borrow one actual group declaration and its checked world-wave proof.
    pub fn group(&self, order: usize) -> Option<(&CommunicationGroupDescriptor, bool)> {
        Some((self.manifest().groups.get(order)?, self.realization().group_world_wave(order)?))
    }
    /// Borrow one actual route declaration and its checked world-wave proof.
    pub fn route(&self, order: usize) -> Option<(&CommunicationRouteDescriptor, bool)> {
        Some((self.manifest().routes.get(order)?, self.realization().route_world_wave(order)?))
    }
    /// Exact retained Rust source allocation capacities, including the Arc
    /// control header. Handles/caller frames and every native communicator,
    /// stream, transport, operation and completion allocation are separate.
    /// Attached custody has its own already-paid shell and is not counted here.
    /// Clones share these bytes; this query creates no registration or credit.
    pub fn host_storage_bytes(&self) -> Option<usize> {
        let prepared = self.realization();
        let manifest = &prepared.manifest;
        let mut bytes = size_of::<Source>().checked_add(2 * size_of::<usize>())?
            .checked_add(vec_bytes(&manifest.groups)?)?
            .checked_add(vec_bytes(&manifest.routes)?)?
            .checked_add(vec_bytes(&prepared.group_world_waves)?)?
            .checked_add(vec_bytes(&prepared.route_world_waves)?)?;
        for group in &manifest.groups {
            bytes = bytes.checked_add(vec_bytes(&group.members)?)?
                .checked_add(vec_bytes(&group.requirements.operations)?)?;
            for requirement in &group.requirements.operations {
                bytes = bytes.checked_add(vec_bytes(&requirement.dtypes)?)?;
            }
        }
        for route in &manifest.routes {
            bytes = bytes.checked_add(vec_bytes(&route.requirement.dtypes)?)?;
            if let Some(boundary) = &route.boundary {
                bytes = bytes.checked_add(boundary.schema.capacity())?
                    .checked_add(vec_bytes(&boundary.roles)?)?;
                for role in &boundary.roles {
                    bytes = bytes.checked_add(role.role.capacity())?
                        .checked_add(vec_bytes(&role.shape)?)?;
                }
            }
        }
        Some(bytes)
    }
}
fn vec_bytes<T>(value: &Vec<T>) -> Option<usize> { value.capacity().checked_mul(size_of::<T>()) }
