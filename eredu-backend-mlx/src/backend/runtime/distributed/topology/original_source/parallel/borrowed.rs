//! Borrowed context for an existing driver that needs no mutable Group adapter.
use super::*;
impl OriginalParallelInvocation {
    pub(super) fn bind_control_bytes()->Option<usize> {
        let parts=[size_of::<(&Self,&OriginalScopeObserver,&Stream)>(),
            size_of::<Result<&Group,Error>>(),size_of::<StreamCopyPlan<()>>(),
            size_of::<Result<StreamCopyPlan<()>,safemlx::StreamCopyCause>>(),
            size_of::<Result<(),Bound>>(),failure_control_bytes()?,
            OriginalScopeObserver::control_bytes()?,Stream::device_type_control_bytes()?];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
    pub(super) fn finish_control_bytes()->Option<usize> {
        let parts=[size_of::<&Self>(),size_of::<Result<(),Error>>(),failure_control_bytes()?];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
    fn borrowed_control_bytes<T>()->Option<usize> {
        let parts=[size_of::<(&Self,&OriginalScopeObserver,&Stream)>(),
            size_of::<&mut dyn FnMut(&Group)->Result<T,Error>>(),
            size_of::<Result<T,Error>>(),failure_control_bytes()?];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
    pub(crate) fn borrowed_context_control_bytes<T>()->Option<usize> {
        Self::borrowed_control_bytes::<T>()?.checked_add(Self::bind_control_bytes()?)?
            .checked_add(Self::finish_control_bytes()?)
    }
    /// The caller retains this invocation in its existing native Recovery.
    /// Returning or dropping this short loan does not retire any occurrence.
    pub(crate) fn with_borrowed_context<T>(&self,observer:&OriginalScopeObserver,stream:&Stream,
        run:&mut dyn FnMut(&Group)->Result<T,Error>)->Result<T,Error> {
        self.funding.reserve_metadata(Self::borrowed_control_bytes::<T>().ok_or_else(overflow)?)
            .map_err(Error::WorkspacePlanning)?;
        if !self.has_neural_context() {
            return Err(failure(Cause::Identity,&self.state().source,&self.funding));
        }
        let context=self.bind(observer,stream)?;
        let output=run(context);
        // Callback failure retains the original error and leaves the same
        // source under Recovery until completion or quarantine.
        match output {Err(cause)=>Err(cause),Ok(value)=>{self.finish_construction()?;Ok(value)}}
    }
}
