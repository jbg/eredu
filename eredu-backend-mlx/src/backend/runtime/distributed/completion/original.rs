//! Closed original event submission reuses the same completion and quarantine.
use super::*;
use crate::backend::submission_recovery::observed::{PreparedObservedRecovery, operation::OperationRecovery};
use safemlx::{OperationEvent, OriginalScopeObserver, error::Exception};

/// Exact ordinary/original event; neither variant implies outer-role retirement.
#[derive(Debug)]
pub(super) enum NativeEvent { Ordinary(Rc<Event>), Original(Rc<OperationEvent>) }
impl NativeEvent {
    pub(super) fn original_observer(&self)->Option<&OriginalScopeObserver> {
        match self { Self::Ordinary(_)=>None,Self::Original(event)=>event.original_observer() }
    }
    pub(super) fn is_complete(&self)->Result<bool,Exception> {
        match self { Self::Ordinary(event)=>event.is_complete(),Self::Original(event)=>event.is_complete() }
    }
    pub(super) fn wait_on(&self,stream:&Stream)->Result<(),Exception> {
        match self { Self::Ordinary(event)=>event.wait_on(stream),Self::Original(event)=>event.wait_on(stream) }
    }
    #[cfg(test)]
    pub(super) fn synchronize(&self)->Result<(),Exception> {
        match self { Self::Ordinary(event)=>event.synchronize(),Self::Original(event)=>event.synchronize() }
    }
    pub(super) fn try_with_complete<T>(&self,read:impl FnOnce()->Result<T,Exception>)->Result<Option<T>,Exception> {
        match self { Self::Ordinary(event)=>event.try_with_complete(read),Self::Original(event)=>event.try_with_complete(read) }
    }
    pub(super) fn refusal(&self,ordinary:&'static str)->Exception {
        match self.original_observer() { Some(observer)=>observer.invalid_input_error(),None=>Exception::custom(ordinary) }
    }
}

/// Only the status view is shared. Ordinary Recovery and borrowed-role Recovery
/// retain their own exact probes, terminal cleanup and quarantine behavior.
pub(super) trait CompletionState {
    fn completion_status(&self)->Result<Status,Exception>;
    fn resources(&self)->&NativeOwner;
    fn observer(&self)->Option<&OriginalScopeObserver> { None }
}
impl<P:Probe> CompletionState for Recovery<NativeOwner,P> {
    fn completion_status(&self)->Result<Status,Exception> { Ok(self.progress()) }
    fn resources(&self)->&NativeOwner { self.retention() }
}
impl<T:CompletionState> CompletionState for Rc<T> {
    fn completion_status(&self)->Result<Status,Exception> { self.as_ref().completion_status() }
    fn resources(&self)->&NativeOwner { self.as_ref().resources() }
    fn observer(&self)->Option<&OriginalScopeObserver> { self.as_ref().observer() }
}
pub(super) struct CompletionRecovery(OperationRecovery<NativeOwner,prepared::ResourceCustody>);
impl std::fmt::Debug for CompletionRecovery {
    fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result {
        f.debug_struct("CompletionRecovery").field("original",&self.is_original()).finish_non_exhaustive()
    }
}
impl CompletionRecovery {
    pub(super) fn ordinary(value:Recovery<NativeOwner>)->Self { Self(OperationRecovery::ordinary(value)) }
    pub(super) fn original(pending:PreparedObservedRecovery<NativeOwner,prepared::ResourceCustody>,
        resources:NativeOwner,observer:OriginalScopeObserver)->Self {
        Self(OperationRecovery::original(pending,resources,observer))
    }
    pub(super) fn retention(&self)->&NativeOwner { self.0.retention() }
    pub(super) fn seal(&mut self) { self.0.seal(); }
    pub(super) fn is_original(&self)->bool { self.0.original_observer().is_some() }
    pub(super) fn settled(&self)->bool { self.0.progress().is_ok_and(|observed|observed.can_retire()) }
}
impl CompletionState for CompletionRecovery {
    fn completion_status(&self)->Result<Status,Exception> {
        self.0.progress().map(|observed|Status { settled:observed.can_retire(),..observed.status })
    }
    fn resources(&self)->&NativeOwner { self.retention() }
    fn observer(&self)->Option<&OriginalScopeObserver> { self.0.original_observer() }
}

/// A submitted event with the original worker's finite destinations. The private
/// wrapper prevents calling ordinary allocating host-result builders. Prepared
/// scalar, word and header adapters attach through the same private resolver.
/// Consumer-stream waits use only the finite prepared bank attached at birth.
#[derive(Debug)]
#[must_use="retain or wait for the submitted original communication completion"]
pub(crate) struct OriginalCommunicationCompletion(pub(super) MlxCommunicationCompletion);
impl OriginalCommunicationCompletion {
    /// Retain this exact successful completion for its caller after the same
    /// bounded worker has settled it. Deadline/error disposition is unchanged.
    pub(crate) fn wait_bounded_retaining(self, policy: eredu_core::BoundedCompletionWait)
        ->Result<eredu_core::BoundedSubmissionOutcome<Self>, Exception> {
        match self.0.wait_bounded_retaining(policy)? {
            eredu_core::BoundedSubmissionOutcome::Completed(value) =>
                Ok(eredu_core::BoundedSubmissionOutcome::Completed(Self(value))),
            eredu_core::BoundedSubmissionOutcome::DeadlineExceeded { cancellation } =>
                Ok(eredu_core::BoundedSubmissionOutcome::DeadlineExceeded { cancellation }),
        }
    }
    /// Consume one existing finite slot under the supplied current original
    /// role. The retained stream inventory authenticates the native handle.
    pub(crate) fn wait_on(&self,stream:&Stream,observer:&OriginalScopeObserver)->Result<(),Exception> {
        let consumers=self.0.consumers.as_ref().ok_or_else(||observer.capacity_error())?;
        consumers.wait(&self.0.event,self.0.recovery.retention(),stream,observer)
    }
}
impl eredu_core::Completion for OriginalCommunicationCompletion {
    type Error=Exception;
    fn resources_releasable(&self)->bool { eredu_core::Completion::resources_releasable(&self.0) }
    fn is_complete(&self)->Result<bool,Exception> { eredu_core::Completion::is_complete(&self.0) }
    fn wait(&self)->Result<(),Exception> { eredu_core::Completion::wait(&self.0) }
}
impl eredu_core::BoundedCompletion for OriginalCommunicationCompletion {
    fn wait_bounded(self,policy:eredu_core::BoundedCompletionWait)->Result<eredu_core::BoundedCompletionOutcome,Exception> {
        eredu_core::BoundedCompletion::wait_bounded(self.0,policy)
    }
}

pub(super) fn reap_original_communication() {
    crate::backend::submission_recovery::reap();
    communication::reap_communication_orphans();
}

/// Named shared observation/wait/quarantine frames; dynamic host readout and
/// consumer-wait destinations are not implied by this event-only constructor.
pub(super) fn completion_controls()->Option<usize> {
    use std::mem::{size_of,size_of_val};
    let frames=[super::neural::control_bytes()?,size_of::<NativeEvent>(),size_of::<CompletionRecovery>(),size_of::<OriginalCommunicationCompletion>(),
        size_of::<(&MlxCommunicationCompletion,&NativeEvent,&CompletionRecovery)>(),
        size_of::<Option<&OriginalScopeObserver>>(),size_of::<Result<bool,Exception>>(),
        size_of::<Option<Result<bool,Exception>>>(),size_of::<Result<(),Exception>>(),
        size_of::<Option<Exception>>(),size_of::<(&str,bool)>(),size_of::<Status>(),
        size_of::<eredu_core::BoundedCompletionWait>(),size_of::<eredu_core::BoundedCompletionOutcome>(),
        size_of::<Result<eredu_core::BoundedCompletionOutcome,Exception>>(),
        size_of::<eredu_core::BoundedSubmissionOutcome<MlxCommunicationCompletion>>(),
        size_of::<eredu_core::BoundedSubmissionOutcome<OriginalCommunicationCompletion>>(),
        size_of::<Result<eredu_core::BoundedSubmissionOutcome<MlxCommunicationCompletion>,Exception>>(),
        size_of::<Result<eredu_core::BoundedSubmissionOutcome<OriginalCommunicationCompletion>,Exception>>(),
        size_of::<eredu_core::CompletionCancellationMode>(),size_of::<std::time::Instant>(),
        size_of::<Option<std::time::Instant>>(),size_of::<std::time::Duration>(),
        size_of::<Option<destinations::Destination<MlxCommunicationCompletion>>>(),
        size_of::<std::cell::RefMut<'_,communication::CommunicationOrphanQuarantine>>(),
        size_of::<std::cell::BorrowMutError>(),size_of::<std::thread::AccessError>(),
        size_of::<Result<bool,std::thread::AccessError>>(),
    ];
    frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
}
