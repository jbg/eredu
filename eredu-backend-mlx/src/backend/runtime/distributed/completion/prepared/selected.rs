//! Paid source selection and deep host retention before any native submission.
use super::*;
impl PreparedCompletionResources<'_, '_> {
    /// Fixed source-selection transports used by the retained group/route
    /// worker. Source validation and the selected deep copy are separate.
    pub(crate) fn selection_control_bytes()->Option<usize> {
        let parts=[size_of::<(&Self,usize)>(),size_of::<Result<(),Error>>(),
            size_of::<Option<(&Group,&eredu_runtime::CommunicationGroupDescriptor,bool)>>(),
            size_of::<Option<(&CommunicationRouteRealization,&eredu_runtime::CommunicationRouteDescriptor,bool)>>(),
            size_of::<Result<Group,TryReserveError>>(),
            size_of::<Result<CommunicationRouteRealization,TryReserveError>>(),
            size_of::<Option<usize>>(),error_control_bytes()?];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
    fn selection_controls(&self)->Result<(),Error> {
        self.custody.funding.reserve_metadata(Self::selection_control_bytes().ok_or_else(overflow)?)
            .map_err(Error::WorkspacePlanning)?;
        self.source.validate()
    }
    fn reserve_copy(&self,bytes:Option<usize>)->Result<(),Error> {
        self.custody.funding.reserve_metadata(bytes.ok_or_else(overflow)?)
            .map_err(|cause|error(Cause::Funding(cause),&self.custody))
    }
    /// Retains the exact selected control world, with all host copy destinations
    /// paid before birth. This neither clones arrays nor submits collective work.
    pub(crate) fn retain_control_world(&mut self)->Result<(),Error> {
        self.selection_controls()?;
        let value=self.source.world();
        if self.groups.len()==self.limits[1] || !self.source.matches_world(value) {
            return Err(error(Cause::Identity,&self.custody));
        }
        self.reserve_copy(value.retention_copy_bytes())?;
        let value=value.try_copy_for_retention().map_err(|cause|error(Cause::Capacity(cause),&self.custody))?;
        self.groups.push(value);
        Ok(())
    }
    /// Copies the actual selected group, preserving its native wrapper, logical
    /// rank order, route/wave proof, exact requirements and source identity.
    pub(crate) fn retain_group(&mut self,order:usize)->Result<(),Error> {
        self.selection_controls()?;
        let (value,_,_)=self.source.group(order).ok_or_else(||error(Cause::Identity,&self.custody))?;
        if self.groups.len()==self.limits[1] || !self.source.matches_group(order,value) {
            return Err(error(Cause::Identity,&self.custody));
        }
        self.reserve_copy(value.retention_copy_bytes())?;
        let value=value.try_copy_for_retention().map_err(|cause|error(Cause::Capacity(cause),&self.custody))?;
        self.groups.push(value);
        Ok(())
    }
    /// Retains the complete source-selected route, including nonlocal endpoint
    /// absence and exact role/schema declarations, under the same completion H.
    pub(crate) fn retain_route(&mut self,order:usize)->Result<(),Error> {
        self.selection_controls()?;
        let (value,_,_)=self.source.route(order).ok_or_else(||error(Cause::Identity,&self.custody))?;
        if self.routes.len()==self.limits[2] || !self.source.matches_route(order,value) {
            return Err(error(Cause::Identity,&self.custody));
        }
        self.reserve_copy(value.retention_copy_bytes())?;
        let value=value.try_copy_for_retention().map_err(|cause|error(Cause::Capacity(cause),&self.custody))?;
        self.routes.push(value);
        Ok(())
    }
}
