//! H-paid retention of actual native communicator owners, with unknown domains explicit.
use super::*;
use safemlx::distributed::GroupPersistentStorage;

pub(crate) struct OriginalCommunicatorPersistent<'a> {
    native: GroupPersistentStorage<'a>,
    source: RetainedCommunicationSource,
    // Account outlives source/native loan, including every admitted C++ owner.
    _funding: HostMetadataFunding,
}
/// Pure owner-query refusal; allocation permission stays with the retained source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PersistentOwnerControlError {
    Overflow,
    Resource,
}
impl OriginalCommunicatorPersistent<'_> {
    /// Exact known retained owners from the actual native group. Registered
    /// buffer credit requires the same closed table proof as the runtime loan.
    pub(crate) fn retained_owner_control_bytes(
        native: &GroupPersistentStorage<'_>,
        group: &Group,
        registered_buffers: bool,
    ) -> Result<usize, PersistentOwnerControlError> {
        let bytes = native
            .retained_owner_bytes()
            .ok_or(PersistentOwnerControlError::Overflow)?;
        if !registered_buffers {
            return Ok(bytes);
        }
        let buffer = group
            .retained_buffer()
            .ok_or(PersistentOwnerControlError::Resource)?;
        if native.has_unqualified_storage()
            || !buffer.is_for(group.native_group())
            || buffer.bytes() != native.vector_bytes().0
        {
            return Err(PersistentOwnerControlError::Resource);
        }
        bytes
            .checked_sub(buffer.bytes())
            .ok_or(PersistentOwnerControlError::Resource)
    }

    pub(crate) fn native(&self) -> &GroupPersistentStorage<'_> {
        &self.native
    }
    pub(crate) fn source(&self) -> &RetainedCommunicationSource {
        &self.source
    }
}
impl<'a> OriginalCommunicationSource<'a> {
    /// Same persistent query transports, excluding the separately retained owners.
    pub(crate) fn persistent_query_control_bytes(native_bytes: usize) -> Option<usize> {
        let controls = [
            size_of::<OriginalCommunicatorPersistent<'a>>(),
            size_of::<Result<OriginalCommunicatorPersistent<'a>, Error>>(),
            size_of::<Result<Option<OriginalCommunicatorPersistent<'a>>, Error>>(),
            size_of::<(&Self, Option<&Group>)>(),
            size_of::<usize>() * 4,
            size_of::<Option<&safemlx::distributed::RetainedGroupBuffer>>(),
            size_of::<(
                &safemlx::distributed::RetainedGroupBuffer,
                &safemlx::distributed::Group,
            )>(),
            size_of::<(usize, usize, usize)>(),
            size_of::<bool>() * 3,
            size_of::<Option<(&Group, &CommunicationGroupDescriptor, bool)>>(),
            size_of::<
                Option<(
                    &CommunicationRouteRealization,
                    &CommunicationRouteDescriptor,
                    bool,
                )>,
            >(),
            failure_control_bytes()?,
            native_bytes,
        ];
        controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
    }
    fn persistent_controls(&self, group: Option<&Group>) -> Result<(), Error> {
        let native_bytes = group
            .map(|g| g.native_group().persistent_storage_control_bytes())
            .unwrap_or(Some(0))
            .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?;
        self.funding
            .reserve_metadata(
                Self::persistent_query_control_bytes(native_bytes)
                    .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
            )
            .map_err(Error::WorkspacePlanning)
    }
    fn retain_persistent(
        &self,
        group: &'a Group,
    ) -> Result<OriginalCommunicatorPersistent<'a>, Error> {
        self.validate()?;
        let native = group
            .native_group()
            .persistent_storage()
            .map_err(|_| failure(Cause::Resource, &self.source, &self.funding))?;
        let bytes = OriginalCommunicatorPersistent::retained_owner_control_bytes(
            &native,
            group,
            self.registered_buffers.is_some(),
        )
        .map_err(|cause| match cause {
            PersistentOwnerControlError::Overflow => {
                Error::WorkspacePlanning(HostMetadataFundingError::Overflow)
            }
            PersistentOwnerControlError::Resource => {
                failure(Cause::Resource, &self.source, &self.funding)
            }
        })?;
        // The existing immutable native source is borrowed, not reconstructed.
        // Its exact known retained owners are paid before the loan can escape.
        self.funding
            .reserve_metadata(bytes)
            .map_err(Error::WorkspacePlanning)?;
        Ok(OriginalCommunicatorPersistent {
            native,
            source: self.source.clone(),
            _funding: self.funding.clone(),
        })
    }
    pub(crate) fn world_persistent(&self) -> Result<OriginalCommunicatorPersistent<'a>, Error> {
        self.persistent_controls(Some(self.world()))?;
        self.retain_persistent(self.world())
    }
    pub(crate) fn group_persistent(
        &self,
        order: usize,
    ) -> Result<OriginalCommunicatorPersistent<'a>, Error> {
        // Pay the lookup/failure path first; native query controls are paid only
        // after the exact retained declaration resolves its actual group.
        self.persistent_controls(None)?;
        let (group, _, _) = self
            .group(order)
            .ok_or_else(|| failure(Cause::Resource, &self.source, &self.funding))?;
        self.persistent_controls(Some(group))?;
        self.retain_persistent(group)
    }
    pub(crate) fn route_persistent(
        &self,
        order: usize,
    ) -> Result<Option<OriginalCommunicatorPersistent<'a>>, Error> {
        self.persistent_controls(None)?;
        let (route, _, _) = self
            .route(order)
            .ok_or_else(|| failure(Cause::Resource, &self.source, &self.funding))?;
        self.validate()?;
        route
            .group
            .as_ref()
            .map(|group| {
                self.persistent_controls(Some(group))?;
                self.retain_persistent(group)
            })
            .transpose()
    }
}
