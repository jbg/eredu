//! The model's actual nested bank enters the same completion/recovery owner.
use super::*;
impl ReadyCompletionResources {
    pub(crate) fn submit_model_nested<'a,C,I>(self,
        source:&OriginalCommunicationSource<'_>,observer:&safemlx::OriginalScopeObserver,
        stream:&Stream,roots:&mut safemlx::PreparedNestedRoots<C>,outputs:I,count:usize)
        ->Result<OriginalCommunicationCompletion,Error>
    where I:IntoIterator<Item=&'a Array>{
        let parts=[size_of::<I>(),size_of::<I::IntoIter>(),size_of::<C>(),
            size_of::<(&mut safemlx::PreparedNestedRoots<C>,usize)>(),
            size_of::<Result<OriginalCommunicationCompletion,Error>>(),
            safemlx::PreparedNestedRoots::<C>::submission_control_bytes::<I>().ok_or_else(overflow)?,
            error_control_bytes().ok_or_else(overflow)?];
        self.custody.funding.reserve_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
            .ok_or_else(overflow)?).map_err(Error::WorkspacePlanning)?;
        source.validate()?;
        if !self.custody.source.same_source(source.source()) || !self.custody.funding.same_account(source.funding()) {
            return Err(error(Cause::Identity,&self.custody));
        }
        // The lexical model-root projection has authenticated both observer and
        // exact admitted traversal. Native consumes its original nested attempt.
        // The shared event publisher seals Recovery on success and failure.
        self.submit_event(observer,count,||roots.submit(outputs,observer,stream))
    }
}
