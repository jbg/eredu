//! Each reached gather owns its original scope until exact bounded completion.
use super::*;
use crate::backend::submission_recovery::{PreparedRecovery,Recovery,Retention,Status};
use eredu_core::{Completion,BoundedCompletion,BoundedCompletionOutcome,BoundedCompletionWait,Submission};
use eredu_runtime::working_memory::{CommunicationPreparationProducer,CommunicationPreparationCustody,
    PreparedCommunicationPreparation,WorkingMemoryPool,WorkingMemoryError};
use safemlx::{OriginalBufferBudget,PreparedOriginalBufferBudget,PreparedSubmissionGraphQuota,
    PreparedSubmissionRecordQuota,PreparedPrefillFailure,RetainedPrefillFailure,RegisteredThreadRuntimeHousekeeping};
use std::cell::RefCell;

#[derive(Clone,Debug)]
struct Custody {source:RetainedCommunicationSource,native:SharedPreparationCustody,funding:HostMetadataFunding}
#[derive(Debug,thiserror::Error)]
enum RoleCause {
    #[error(transparent)] Scope(safemlx::SubmissionScopeOwnerCause),
    #[error(transparent)] Graph(safemlx::SubmissionGraphQuotaCause),
    #[error(transparent)] Record(safemlx::SubmissionRecordQuotaCause),
    #[error(transparent)] Failure(safemlx::PrefillFailureCause),
    #[error(transparent)] Buffer(safemlx::OriginalBufferCause),
    #[error(transparent)] Housekeeping(safemlx::HousekeepingRegistrationCause),
    #[error(transparent)] Native(safemlx::error::Exception),
    #[error(transparent)] Control(safemlx::OriginalNativeControlError),
    #[error("readiness scope observation unavailable: {0:?}")] Observation(safemlx::ScopedSubmissionProgress),
    #[error("readiness native completion is not terminal")] Incomplete,
    #[error("readiness completion deadline cannot be represented or its cancellation is unavailable")] Deadline,
}
#[derive(Debug,thiserror::Error)]
#[error("{cause}")]
struct RoleFailure {#[source] cause:RoleCause,custody:Custody}
fn fail(cause:RoleCause,custody:&Custody)->Error {
    Error::with_original_control_source(eredu_core::BackendFailure::new(eredu_core::BackendFailureKind::Other,
        RoleFailure{cause,custody:custody.clone()}),false)
}
struct Retained {
    budget:OriginalBufferBudget,
    failure:RetainedPrefillFailure,
    housekeeping:RegisteredThreadRuntimeHousekeeping,
    custody:Custody,
}
impl Retention for Retained {fn observe(&self,_:Status){}}
struct Started {
    words:Option<OriginalCommunicationU32Words>,
    completion:NativeCompletion,
}
struct NativeCompletion {
    inner:Option<MlxNeuralCommunicationCompletion>,
    recovery:RefCell<Option<Recovery<Retained>>>,
    observer:OriginalScopeObserver,
    custody:Custody,
}
/// Same communication completion and recovery, with its independent early-stage
/// storage account. No raw native event or additional inference engine escapes.
pub(crate) struct PreparationCompletion(PreparedCommunicationPreparation<Started>);
impl std::fmt::Debug for PreparationCompletion {
    fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result{f.debug_struct("PreparationCompletion").finish_non_exhaustive()}
}
struct Producer<'a,'native>{
    operation:PreparedOriginalPreparationGather<'a>,source:&'a OriginalCommunicationSource<'native>,
    words:&'a [u32],
    stream:&'a Stream,pool:&'a WorkingMemoryPool,bytes:usize,
    policy:Option<&'a eredu_runtime::working_memory::SessionResetPreparationFunding>,
}
impl PreparedOriginalPreparationGather<'_> {
    pub(crate) fn start(self,source:&OriginalCommunicationSource<'_>,
        words:&[u32],stream:&Stream,pool:&WorkingMemoryPool,
        policy:Option<&eredu_runtime::working_memory::SessionResetPreparationFunding>)
        ->Result<Submission<OriginalCommunicationU32Words,PreparationCompletion>,Error>{
        reserve(&self.funding,&[
            size_of::<Producer<'_, '_>>(),size_of::<Started>(),size_of::<NativeCompletion>(),size_of::<PreparationCompletion>(),
            size_of::<Custody>(),size_of::<RoleCause>(),size_of::<RoleFailure>(),size_of::<RetiredFailure>(),
            size_of::<Result<Submission<OriginalCommunicationU32Words,PreparationCompletion>,Error>>(),
            size_of::<Result<PreparedCommunicationPreparation<Started>,eredu_runtime::working_memory::CommunicationPreparationError<Producer<'_, '_>>>>(),
            size_of::<(OriginalCommunicationU32Words,PreparationCompletion)>(),size_of::<[Option<usize>;7]>(),
            size_of::<(&OriginalCommunicationSource<'_>,&[u32],&Stream,&WorkingMemoryPool,Option<&eredu_runtime::working_memory::SessionResetPreparationFunding>)>(),
            eredu_core::BackendFailure::source_retention_peak_bytes::<RoleFailure>().ok_or_else(overflow)?,
            eredu_core::BackendFailure::source_retention_peak_bytes::<RetiredFailure>().ok_or_else(overflow)?,
            failure_control_bytes().ok_or_else(overflow)?,
            size_of::<std::cell::RefMut<'_,Option<Recovery<Retained>>>>(),
            size_of::<Option<&Recovery<Retained>>>(),size_of::<Status>(),
            size_of::<safemlx::SubmissionRetirement>(),
            size_of::<Result<safemlx::SubmissionRetirement,safemlx::error::Exception>>(),
            size_of::<Result<bool,Error>>(),size_of::<Option<Result<bool,Error>>>(),
            size_of::<(&NativeCompletion,bool)>(),
            size_of::<Option<MlxNeuralCommunicationCompletion>>(),
            size_of::<std::time::Instant>(),size_of::<Option<std::time::Instant>>(),
            size_of::<std::time::Duration>(),size_of::<Option<std::time::Duration>>(),
            size_of::<Result<BoundedCompletionWait,eredu_core::BoundedCompletionWaitError>>(),
            size_of::<(&mut NativeCompletion,BoundedCompletionWait)>(),
            size_of::<(&NativeCompletion,BoundedCompletionWait)>(),
            size_of::<BoundedCompletionWait>(),size_of::<BoundedCompletionOutcome>(),
            size_of::<Result<BoundedCompletionOutcome,Error>>(),
            size_of::<eredu_core::CompletionCancellationMode>(),
        ])?;
        let error=|cause|failure(cause,&self.source,&self.funding);
        let graph=PreparedSubmissionGraphQuota::<Custody>::layout(self.graph_capacity())
            .map_err(|cause|error(Cause::SourceGraph(cause)))?;
        let record=PreparedSubmissionRecordQuota::<Custody>::layout(self.record_capacity())
            .map_err(|_|error(Cause::Resource))?;
        let failure_layout=PreparedPrefillFailure::<Custody>::layout().map_err(|_|error(Cause::Resource))?;
        let buffer=PreparedOriginalBufferBudget::<Custody>::layout(self.runtime(),self.backing_capacity())
            .map_err(|cause|error(Cause::Buffer(cause)))?;
        let native=safemlx::OriginalNativeControlLayout::inspect().map_err(|_|error(Cause::Resource))?;
        let shared=SharedPreparationCustody::allocation_bytes().ok_or_else(overflow)?;
        let sizes=[graph.total_bytes(),record.total_bytes(),failure_layout.total_bytes(),buffer.total_owner_bytes(),
            PreparedRecovery::<Retained,Custody>::control_bytes().and_then(|n|usize::try_from(n).ok()),
            safemlx::PreparedThreadRuntimeHousekeeping::<Custody>::control_bytes(),OriginalScopeObserver::control_bytes()];
        let bytes=sizes.into_iter().try_fold(size_of_val(&sizes),|n,value|n.checked_add(value?))
            .and_then(|n|n.checked_add(native.fixed_control_bytes)).and_then(|n|n.checked_add(shared))
            .and_then(|n|n.checked_add(self.backing_capacity())).ok_or_else(overflow)?;
        let funding=self.funding.clone();let retained=self.source.clone();
        let producer=Producer{operation:self,source,words,stream,pool,bytes,policy};
        let mut prepared=pool.prepare_communication(producer)
            .map_err(|cause|retired_error(cause.retire(),&retained,&funding))?;
        let output=prepared.with_output(|value|value.words.take().expect("fresh readiness output"));
        Ok(Submission{output,completion:PreparationCompletion(prepared)})
    }
}
impl CommunicationPreparationProducer for Producer<'_, '_>{
    type Output=Started;type Error=Error;
    fn source(&self)->&RetainedCommunicationSource{&self.operation.source}
    fn frame(&self)->&[u32]{self.words}
    fn validate_pool(&self,pool:&WorkingMemoryPool)->Result<(),WorkingMemoryError>{
        if self.pool.same_domain(pool){Ok(())}else{Err(WorkingMemoryError::IdentityMismatch)}
    }
    fn required_storage_bytes(&self)->Result<usize,WorkingMemoryError>{Ok(self.bytes)}
    fn check_preparation_policy(&self,pool:&WorkingMemoryPool,bytes:u64)->Result<(),WorkingMemoryError>{
        self.policy.map_or(Ok(()),|policy|policy.charge_communication(pool,bytes))
    }
    fn produce(self,raw:CommunicationPreparationCustody)->Result<Started,Error>{
        let custody=Custody{source:self.operation.source.clone(),native:SharedPreparationCustody::new(raw),funding:self.operation.funding.clone()};
        let error=|cause|fail(cause,&custody);
        let budget=PreparedOriginalBufferBudget::try_new(self.operation.runtime(),self.operation.backing_capacity(),custody.clone())
            .map_err(|error_|{let(cause,owner)=error_.into_parts();drop(owner);error(RoleCause::Buffer(cause))})?
            .try_allocate().map_err(|error_|{let(cause,owner)=error_.into_parts();drop(owner);error(RoleCause::Buffer(cause))})?;
        let graph=PreparedSubmissionGraphQuota::try_new(self.operation.graph_capacity(),custody.clone())
            .map_err(|error_|{let(cause,owner)=error_.into_parts();drop(owner);error(RoleCause::Graph(cause))})?
            .try_allocate().map_err(|error_|{let(cause,owner)=error_.into_parts();drop(owner);error(RoleCause::Graph(cause))})?;
        let records=PreparedSubmissionRecordQuota::try_new(self.operation.record_capacity(),custody.clone())
            .map_err(|error_|{let(cause,owner)=error_.into_parts();drop(owner);error(RoleCause::Record(cause))})?
            .try_allocate().map_err(|error_|{let(cause,owner)=error_.into_parts();drop(owner);error(RoleCause::Record(cause))})?;
        let failure=PreparedPrefillFailure::try_new(custody.clone())
            .map_err(|error_|{let(cause,owner)=error_.into_parts();drop(owner);error(RoleCause::Failure(cause))})?
            .try_allocate().map_err(|error_|{let(cause,owner)=error_.into_parts();drop(owner);error(RoleCause::Failure(cause))})?;
        let housekeeping=safemlx::PreparedThreadRuntimeHousekeeping::new(crate::backend::submission_recovery::reap,custody.clone())
            .try_register().map_err(|error_|{let(cause,owner)=error_.into_parts();drop(owner);error(RoleCause::Housekeeping(cause))})?;
        let retained=Retained{budget,failure,housekeeping,custody:custody.clone()};
        let prepared=PreparedRecovery::new(retained,custody.clone())
            .map_err(|error_|{let cause=error_.cause;drop(error_.retention);drop(error_.custody);error(RoleCause::Scope(cause))})?
            .with_graph_quota(Some(graph)).with_record_quota(Some(records));
        let mut recovery=prepared.try_begin().map_err(|error_|{let cause=error_.cause;drop(error_.pending);error(RoleCause::Scope(cause))})?;
        let submitted=recovery.configure_scope_with_retention(|scope,retained|{
            scope.enable_scoped_observation().map_err(|cause|error(RoleCause::Observation(cause)))?;
            scope.require_original_native_controls().map_err(|cause|error(RoleCause::Control(cause)))?;
            retained.failure.bind_original_scope(scope).map_err(|cause|error(RoleCause::Control(cause)))?;
            scope.enable_original_native_controls().map_err(|cause|error(RoleCause::Control(cause)))?;
            scope.bind_original_buffer_budget(&retained.budget).map_err(|cause|error(RoleCause::Buffer(cause)))?;
            let observer=OriginalScopeObserver::require_current().map_err(|cause|error(RoleCause::Native(cause)))?;
            let(words,completion)=self.operation.submit(self.source,&observer,self.stream)?;
            Ok::<_,Error>((words,completion,observer))
        });
        recovery.seal();
        let(words,completion,observer)=submitted?;
        Ok(Started{words:Some(words),completion:NativeCompletion{inner:Some(completion),
            recovery:RefCell::new(Some(recovery)),observer,custody}})
    }
}
impl NativeCompletion {
    /// The exact event can be ready before its sealed enclosing scope. Runtime
    /// contention and healthy pending work are retryable observations, not failures.
    fn try_finish(&self)->Result<bool,Error>{
        safemlx::try_with_submission_retirement(|| {
            let mut slot=self.recovery.borrow_mut();
            let Some(recovery)=slot.as_ref() else{return Ok(true);};
            let status=recovery.progress();
            if status.failed||status.blocked{return Err(fail(RoleCause::Incomplete,&self.custody));}
            if !status.settled{return Ok(false);}
            match self.observer.retire_completed_records()
                .map_err(|cause|fail(RoleCause::Native(cause),&self.custody))? {
                safemlx::SubmissionRetirement::CompleteSnapshot=>{},
                _=>return Ok(false),
            }
            let recovery=slot.take().expect("observed readiness recovery");drop(slot);
            // Keep the SAME reentrant no-hooks runtime guard through destruction.
            // This concrete sealed SubmissionScope cannot acquire new work after
            // terminal proof, and its default Probe::retire_terminal cannot defer.
            // Thus the shared finish worker has no pending/lock-wait iteration.
            let status=recovery.finish()?;
            if !status.settled||status.failed||status.blocked{return Err(fail(RoleCause::Incomplete,&self.custody));}
            Ok(true)
        }).unwrap_or(Ok(false))
    }
    fn expired(&self,wait:BoundedCompletionWait)->Result<BoundedCompletionOutcome,Error>{
        match wait.cancellation() {
            eredu_core::CompletionCancellationMode::QuarantineUntilComplete=>
                Ok(BoundedCompletionOutcome::DeadlineExceeded {
                    cancellation:eredu_core::CompletionCancellationMode::QuarantineUntilComplete }),
            _=>Err(fail(RoleCause::Deadline,&self.custody)),
        }
    }
    fn bounded(&mut self,wait:BoundedCompletionWait)->Result<BoundedCompletionOutcome,Error>{
        // One deadline covers both the existing event waiter and outer retirement.
        // On timeout/error the existing Recovery Drop keeps unresolved custody.
        let deadline=std::time::Instant::now().checked_add(wait.timeout())
            .ok_or_else(||fail(RoleCause::Deadline,&self.custody))?;
        let Some(remaining)=deadline.checked_duration_since(std::time::Instant::now())
            .filter(|remaining|!remaining.is_zero()) else{return self.expired(wait);};
        let inner_wait=BoundedCompletionWait::new(remaining,wait.cancellation())
            .map_err(|_|fail(RoleCause::Deadline,&self.custody))?;
        let inner=self.inner.take().ok_or_else(||fail(RoleCause::Incomplete,&self.custody))?;
        let outcome=inner.wait_bounded(inner_wait)?;
        if !matches!(outcome,BoundedCompletionOutcome::Completed){return Ok(outcome);}
        loop {
            if self.try_finish()?{return Ok(BoundedCompletionOutcome::Completed);}
            if std::time::Instant::now()>=deadline {
                return self.expired(wait);
            }
            std::thread::yield_now();
        }
    }
}
impl Completion for PreparationCompletion {
    type Error=Error;
    fn resources_releasable(&self)->bool{
        let value=&self.0.output().completion;
        value.inner.as_ref().is_none_or(Completion::resources_releasable)
            && value.recovery.borrow().as_ref().is_none_or(|recovery|recovery.progress().settled)
    }
    fn is_complete(&self)->Result<bool,Error>{
        let value=&self.0.output().completion;
        let completed=match &value.inner{Some(inner)=>inner.is_complete()?,None=>value.recovery.borrow().is_none()};
        if completed{value.try_finish()}else{Ok(false)}
    }
    fn wait(&self)->Result<(),Error>{
        let value=&self.0.output().completion;
        if let Some(inner)=&value.inner{inner.wait()?;}
        while !value.try_finish()?{std::thread::yield_now();}
        Ok(())
    }
}
impl BoundedCompletion for PreparationCompletion {
    fn wait_bounded(mut self,policy:BoundedCompletionWait)->Result<BoundedCompletionOutcome,Error>{
        self.0.with_output(|value|value.completion.bounded(policy))
    }
}

#[cfg(test)]
mod tests;
