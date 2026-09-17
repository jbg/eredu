//! H-paid retention of actual native communicator owners, with unknown domains explicit.
use super::*;
use safemlx::distributed::GroupPersistentStorage;

pub(crate) struct OriginalCommunicatorPersistent<'a>{
    native:GroupPersistentStorage<'a>,
    source:RetainedCommunicationSource,
    // Account outlives source/native loan, including every admitted C++ owner.
    _funding:WorkspaceMetadataFunding,
}
impl OriginalCommunicatorPersistent<'_>{
    pub(crate) fn native(&self)->&GroupPersistentStorage<'_>{&self.native}
    pub(crate) fn source(&self)->&RetainedCommunicationSource{&self.source}
}
impl<'a> OriginalCommunicationSource<'a>{
    fn persistent_controls(&self,group:Option<&Group>)->Result<(),Error>{
        let controls=[size_of::<OriginalCommunicatorPersistent<'a>>(),
            size_of::<Result<OriginalCommunicatorPersistent<'a>,Error>>(),
            size_of::<Result<Option<OriginalCommunicatorPersistent<'a>>,Error>>(),
            size_of::<(&Self,Option<&Group>)>(),size_of::<usize>()*4,
            size_of::<Option<&safemlx::distributed::RetainedGroupBuffer>>(),
            size_of::<(&safemlx::distributed::RetainedGroupBuffer,&safemlx::distributed::Group)>(),
            size_of::<(usize,usize,usize)>(),size_of::<bool>()*3,
            size_of::<Option<(&Group,&CommunicationGroupDescriptor,bool)>>(),
            size_of::<Option<(&CommunicationRouteRealization,&CommunicationRouteDescriptor,bool)>>(),
            failure_control_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?,
            group.map(|g|g.native_group().persistent_storage_control_bytes()).unwrap_or(Some(0))
                .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?];
        self.funding.reserve_metadata(controls.into_iter().try_fold(size_of_val(&controls),usize::checked_add)
            .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?).map_err(Error::WorkspacePlanning)
    }
    fn retain_persistent(&self,group:&'a Group)->Result<OriginalCommunicatorPersistent<'a>,Error>{
        self.validate()?;
        let native=group.native_group().persistent_storage()
            .map_err(|_|failure(Cause::Resource,&self.source,&self.funding))?;
        let bytes=native.retained_owner_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?;
        let bytes=if self.registered_buffers.is_some() {
            // Only this closed table proof can exclude the existing buffer.
            // The source alias below retains its pin, and the actual query
            // must still match that table's immutable native allocation.
            let buffer=group.retained_buffer()
                .ok_or_else(||failure(Cause::Resource,&self.source,&self.funding))?;
            if native.has_unqualified_storage() || !buffer.is_for(group.native_group())
                || buffer.bytes()!=native.vector_bytes().0 {
                return Err(failure(Cause::Resource,&self.source,&self.funding));
            }
            bytes.checked_sub(buffer.bytes())
                .ok_or_else(||failure(Cause::Resource,&self.source,&self.funding))?
        } else {bytes};
        // The existing immutable native source is borrowed, not reconstructed.
        // Its exact known retained owners are paid before the loan can escape.
        self.funding.reserve_metadata(bytes).map_err(Error::WorkspacePlanning)?;
        Ok(OriginalCommunicatorPersistent{native,source:self.source.clone(),_funding:self.funding.clone()})
    }
    pub(crate) fn world_persistent(&self)->Result<OriginalCommunicatorPersistent<'a>,Error>{
        self.persistent_controls(Some(self.world()))?;self.retain_persistent(self.world())
    }
    pub(crate) fn group_persistent(&self,order:usize)->Result<OriginalCommunicatorPersistent<'a>,Error>{
        // Pay the lookup/failure path first; native query controls are paid only
        // after the exact retained declaration resolves its actual group.
        self.persistent_controls(None)?;
        let (group,_,_)=self.group(order).ok_or_else(||failure(Cause::Resource,&self.source,&self.funding))?;
        self.persistent_controls(Some(group))?;self.retain_persistent(group)
    }
    pub(crate) fn route_persistent(&self,order:usize)->Result<Option<OriginalCommunicatorPersistent<'a>>,Error>{
        self.persistent_controls(None)?;
        let (route,_,_)=self.route(order).ok_or_else(||failure(Cause::Resource,&self.source,&self.funding))?;
        self.validate()?;
        route.group.as_ref().map(|group|{self.persistent_controls(Some(group))?;self.retain_persistent(group)}).transpose()
    }
}
