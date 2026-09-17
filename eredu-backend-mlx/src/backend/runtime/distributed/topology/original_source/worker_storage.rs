//! Paid exact group/input loans for the partial native worker producer.
use super::*;
use safemlx::{Array, distributed::{GroupWorkerOperation, GroupWorkerStorage}};

/// Both native owners are borrowed; the exact retained neutral source and
/// query funding remain held until the result retires. No complete fit/grant.
pub(crate) struct OriginalCommunicationWorkers<'a> {
    native: GroupWorkerStorage<'a>,
    source: RetainedCommunicationSource,
    _funding: WorkspaceMetadataFunding,
}
impl<'a> OriginalCommunicationWorkers<'a> {
    pub(crate) fn native(&self) -> &GroupWorkerStorage<'a> { &self.native }
    pub(super) fn funding(&self)->&WorkspaceMetadataFunding { &self._funding }
    pub(crate) fn source(&self) -> &RetainedCommunicationSource { &self.source }
}
impl<'a> OriginalCommunicationSource<'a> {
    fn worker_storage<'b>(&'b self, group: &'b Group, input: &'b Array,
        operation: GroupWorkerOperation) -> Result<OriginalCommunicationWorkers<'b>, Error> {
        let controls = [size_of::<OriginalCommunicationWorkers<'b>>(),
            size_of::<Result<OriginalCommunicationWorkers<'b>, Error>>(),
            size_of::<(&Self,&Group,&Array,GroupWorkerOperation)>(),
            size_of::<Result<GroupWorkerStorage<'b>,safemlx::distributed::GroupStorageUnavailable>>(),
            failure_control_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?,
            group.native_group().worker_storage_control_bytes()
                .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?];
        self.funding.reserve_metadata(controls.into_iter().try_fold(size_of_val(&controls),usize::checked_add)
            .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?)
            .map_err(Error::WorkspacePlanning)?;
        self.validate()?;
        let native = group.native_group().worker_storage(input,operation)
            .map_err(|_|failure(Cause::Resource,&self.source,&self.funding))?;
        Ok(OriginalCommunicationWorkers {native, source:self.source.clone(), _funding:self.funding.clone()})
    }
    pub(crate) fn world_worker_storage<'b>(&'b self, input:&'b Array, operation:GroupWorkerOperation)
        -> Result<OriginalCommunicationWorkers<'b>,Error> {
        self.worker_storage(self.world(),input,operation)
    }
    pub(crate) fn group_worker_storage<'b>(&'b self, order:usize, input:&'b Array, operation:GroupWorkerOperation)
        -> Result<OriginalCommunicationWorkers<'b>,Error> {
        self.funding.reserve_metadata(size_of::<Option<(&Group,&CommunicationGroupDescriptor,bool)>>() +
            size_of::<usize>() + size_of::<(&Self,&Array,GroupWorkerOperation)>() +
            failure_control_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?)
            .map_err(Error::WorkspacePlanning)?;
        let (group,_,_) = self.group(order).ok_or_else(||failure(Cause::Resource,&self.source,&self.funding))?;
        self.worker_storage(group,input,operation)
    }
    pub(crate) fn route_worker_storage<'b>(&'b self, order:usize, input:&'b Array, operation:GroupWorkerOperation)
        -> Result<Option<OriginalCommunicationWorkers<'b>>,Error> {
        self.funding.reserve_metadata(size_of::<Option<(&CommunicationRouteRealization,&CommunicationRouteDescriptor,bool)>>() +
            size_of::<Result<Option<OriginalCommunicationWorkers<'b>>,Error>>() + size_of::<usize>() +
            size_of::<(&Self,&Array,GroupWorkerOperation)>() +
            failure_control_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?)
            .map_err(Error::WorkspacePlanning)?;
        self.validate()?;
        let (route,_,_) = self.route(order).ok_or_else(||failure(Cause::Resource,&self.source,&self.funding))?;
        route.group.as_ref().map(|group| self.worker_storage(group,input,operation)).transpose()
    }
}
