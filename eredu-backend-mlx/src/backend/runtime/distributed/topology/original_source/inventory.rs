//! Paid loans of actual native source geometry. Partial facts never open admission.
use super::*;
use safemlx::distributed::GroupStorageInventory;

pub(crate) struct OriginalCommunicatorInventory<'a> {
    native: GroupStorageInventory<'a>,
    source: RetainedCommunicationSource,
    _funding: WorkspaceMetadataFunding,
}
impl OriginalCommunicatorInventory<'_> {
    pub(crate) fn native(&self) -> &GroupStorageInventory<'_> { &self.native }
    pub(crate) fn source(&self) -> &RetainedCommunicationSource { &self.source }
}
impl<'a> OriginalCommunicationSource<'a> {
    fn inventory_controls(&self) -> Result<(), Error> {
        let controls = [size_of::<OriginalCommunicatorInventory<'a>>(),
            size_of::<Result<OriginalCommunicatorInventory<'a>, Error>>(),
            size_of::<(&Self, &Group)>(), size_of::<usize>(),
            size_of::<Result<Option<OriginalCommunicatorInventory<'a>>, Error>>(),
            size_of::<Option<(&Group, &CommunicationGroupDescriptor, bool)>>(),
            size_of::<Option<(&CommunicationRouteRealization, &CommunicationRouteDescriptor, bool)>>(),
            size_of::<Result<(), Error>>(), size_of::<Option<&Group>>(),
            failure_control_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?,
            NativeGroup::storage_inventory_control_bytes()
                .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?];
        self.funding.reserve_metadata(controls.into_iter().try_fold(size_of_val(&controls), usize::checked_add)
            .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?)
            .map_err(Error::WorkspacePlanning)
    }
    fn inventory(&self, group: &'a Group) -> Result<OriginalCommunicatorInventory<'a>, Error> {
        self.validate()?;
        let native = group.native_group().storage_inventory()
            .map_err(|_| failure(Cause::Resource, &self.source, &self.funding))?;
        Ok(OriginalCommunicatorInventory { native, source: self.source.clone(), _funding: self.funding.clone() })
    }
    pub(crate) fn world_inventory(&self) -> Result<OriginalCommunicatorInventory<'a>, Error> {
        self.inventory_controls()?;
        self.inventory(self.world())
    }
    pub(crate) fn group_inventory(&self, order: usize) -> Result<OriginalCommunicatorInventory<'a>, Error> {
        self.inventory_controls()?;
        let (group, _, _) = self.group(order)
            .ok_or_else(|| failure(Cause::Resource, &self.source, &self.funding))?;
        self.inventory(group)
    }
    pub(crate) fn route_inventory(&self, order: usize) -> Result<Option<OriginalCommunicatorInventory<'a>>, Error> {
        self.inventory_controls()?;
        self.validate()?;
        let (route, _, _) = self.route(order)
            .ok_or_else(|| failure(Cause::Resource, &self.source, &self.funding))?;
        route.group.as_ref().map(|group| self.inventory(group)).transpose()
    }
}
