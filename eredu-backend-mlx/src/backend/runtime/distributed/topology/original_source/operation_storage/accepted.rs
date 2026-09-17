//! Constructor-minted handoff; no raw-output/identity-only acceptance shortcut.
use super::*;
use crate::backend::runtime::distributed::completion::{OriginalCommunicationCompletion,
    prepared::ReadyCompletionResources};

/// Exists only after an actual completed-operation constructor succeeds inside
/// the exact retained original scope. The borrowed stream is its real execution
/// stream, and the value owns the source and actual primitive/input graph.
pub(crate) struct AcceptedCommunicationSource<'stream> {
    value:OriginalCommunicationConstructed,
    observer:OriginalScopeObserver,
    stream:&'stream Stream,
    traversal:safemlx::OperationEvalTraversalLayout,
    _funding:WorkspaceMetadataFunding,
}
impl OriginalCommunicationCompletedOperation<'_> {
    pub(crate) fn construct_accepted<'stream>(self,source:&OriginalCommunicationSource<'_>,
        observer:&OriginalScopeObserver,stream:&'stream Stream)->Result<AcceptedCommunicationSource<'stream>,Error> {
        let parts=[size_of::<AcceptedCommunicationSource<'stream>>(),
            size_of::<Result<AcceptedCommunicationSource<'stream>,Error>>(),
            size_of::<Result<(OriginalCommunicationConstructed,OriginalCommunicationCompletion),Error>>(),
            size_of::<(&OriginalCommunicationSource<'_>,&OriginalScopeObserver,&Stream)>(),
            size_of::<WorkspaceMetadataFunding>(),size_of::<safemlx::OperationEvalTraversalLayout>(),
            safemlx::OriginalScopeObserver::control_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?];
        self.funding.reserve_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
            .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?).map_err(Error::WorkspacePlanning)?;
        let traversal=self.native.traversal();
        let funding=self.funding.clone();
        let value=self.construct(source,observer,stream)?;
        Ok(AcceptedCommunicationSource{value,observer:observer.clone(),stream,traversal,_funding:funding})
    }
}
impl AcceptedCommunicationSource<'_> {
    pub(crate) fn source(&self)->&RetainedCommunicationSource{self.value.source()}
    pub(crate) fn observer(&self)->&OriginalScopeObserver{&self.observer}
    pub(crate) fn stream(&self)->&Stream{self.stream}
    pub(crate) fn traversal(&self)->safemlx::OperationEvalTraversalLayout{self.traversal}
    pub(crate) fn outputs(&self)->&[Array]{std::slice::from_ref(self.value.value())}
    pub(crate) fn submit(self,prepared:ReadyCompletionResources)
        ->Result<(OriginalCommunicationConstructed,OriginalCommunicationCompletion),Error>{
        let completion=prepared.submit_accepted(&self)?;
        Ok((self.value,completion))
    }
}
