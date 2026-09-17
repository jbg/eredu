//! One accepted constructor source enters the same event/recovery worker.
use super::*;
use crate::backend::runtime::distributed::topology::AcceptedCommunicationSource;
impl ReadyCompletionResources {
    pub(crate) fn submit_accepted(self,accepted:&AcceptedCommunicationSource<'_>)
        ->Result<OriginalCommunicationCompletion,Error>{
        let parts=[size_of::<(&Self,&AcceptedCommunicationSource<'_>)>(),
            size_of::<Result<OriginalCommunicationCompletion,Error>>(),
            size_of::<safemlx::OperationEvalTraversalLayout>(),size_of::<bool>(),
            error_control_bytes().ok_or_else(overflow)?];
        self.custody.funding.reserve_metadata(parts.into_iter()
            .try_fold(size_of_val(&parts),usize::checked_add).ok_or_else(overflow)?)
            .map_err(Error::WorkspacePlanning)?;
        let exact=match &self.recovery {
            PreparedCompletionRecovery::Original{traversal,..}=>*traversal==accepted.traversal(),
            PreparedCompletionRecovery::Owned(_)=>false,
        };
        if !exact || !self.custody.source.same_source(accepted.source()) {
            return Err(error(Cause::Identity,&self.custody));
        }
        // No fresh availability query here: the closed token proves that the
        // actual operation was already constructed under this observer. The
        // same private worker revalidates current scope/stream and still keeps
        // every native failure under its existing Record/Recovery quarantine.
        self.submit_for_source(accepted.observer(),accepted.stream(),accepted.outputs())
    }
}

impl ReadyCompletionResources {
    pub(crate) fn submit_accepted_route(self,
        accepted:&crate::backend::runtime::distributed::topology::original_source::AcceptedRouteSource<'_>)
        ->Result<OriginalCommunicationCompletion,Error>
    {
        let parts=[size_of::<(&Self,&crate::backend::runtime::distributed::topology::original_source::AcceptedRouteSource<'_>)>(),
            size_of::<Result<OriginalCommunicationCompletion,Error>>(),
            size_of::<safemlx::OperationEvalTraversalLayout>(),size_of::<bool>(),
            error_control_bytes().ok_or_else(overflow)?];
        self.custody.funding.reserve_metadata(parts.into_iter()
            .try_fold(size_of_val(&parts),usize::checked_add).ok_or_else(overflow)?)
            .map_err(Error::WorkspacePlanning)?;
        if !self.custody.source.same_source(accepted.source()) {
            return Err(error(Cause::Identity,&self.custody));
        }
        let PreparedCompletionRecovery::Original{traversal,..}=&self.recovery else {
            return Err(error(Cause::Identity,&self.custody));
        };
        if *traversal!=accepted.traversal(){return Err(error(Cause::Identity,&self.custody));}
        // Actual same-source two-edge acceptance is sufficient for completion
        // even after new submissions become unavailable. Recovery is shared.
        self.submit_for_source(accepted.observer(),accepted.stream(),accepted.outputs())
    }
}
