//! Same exact worker source, with separately paid outer dispatch inspection.
use super::*;
use safemlx::distributed::GroupDispatchStorage;

pub(crate) struct OriginalCommunicationDispatch<'a> {
    native:GroupDispatchStorage<'a>,
    source:RetainedCommunicationSource,
    _funding:WorkspaceMetadataFunding,
}
impl OriginalCommunicationDispatch<'_> {
    pub(crate) fn native(&self)->&GroupDispatchStorage<'_> {&self.native}
    pub(crate) fn source(&self)->&RetainedCommunicationSource {&self.source}
}
impl<'a> OriginalCommunicationWorkers<'a> {
    pub(crate) fn dispatch_storage(&self)->Result<OriginalCommunicationDispatch<'a>,Error> {
        let controls=[size_of::<OriginalCommunicationDispatch<'a>>(),size_of::<Result<OriginalCommunicationDispatch<'a>,Error>>(),
            size_of::<&Self>(),failure_control_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?,
            self.native().dispatch_storage_control_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?];
        self.funding().reserve_metadata(controls.into_iter().try_fold(size_of_val(&controls),usize::checked_add)
            .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?).map_err(Error::WorkspacePlanning)?;
        let native=self.native().dispatch_storage().map_err(|_|failure(Cause::Resource,self.source(),self.funding()))?;
        Ok(OriginalCommunicationDispatch {native,source:self.source().clone(),_funding:self.funding().clone()})
    }
}
