//! Every scheduler exchange uses the original finite-word source and completion.
use super::*;
use eredu_core::{BackendFailure, BackendFailureKind, SharedBackendFailure,
    Completion, BoundedCompletion, BoundedCompletionOutcome, BoundedCompletionWait,
    consensus::{ConsensusTransport, BoundedConsensusTransport}};
use eredu_nn::workspace::{WorkspaceContext, WorkspaceMetadataFunding, WorkspaceMetadataFundingError};
use eredu_runtime::{RetainedCommunicationSource,
    working_memory::{WorkingMemoryPool, InferenceExecutionIdentity}};
use crate::backend::runtime::distributed::{
    topology::{OriginalCommunicationOwner, original_source::PreparationCompletion},
    completion::{OriginalCommunicationU32Words, CompletedCommunicationU32Words}};
use std::{cell::RefCell, mem::{size_of, size_of_val}, rc::Rc};

#[derive(Debug,thiserror::Error)]
#[error("original realtime consensus: {cause}")]
struct Failure {
    #[source] cause:Error,
    _source:RetainedCommunicationSource,
    _funding:WorkspaceMetadataFunding,
}
#[derive(Debug)]
struct FailureState {
    first:RefCell<Option<SharedBackendFailure>>,
    source:RetainedCommunicationSource,
    funding:WorkspaceMetadataFunding,
}
/// One prepaid first-cause cell shared with exact outstanding completions.
#[derive(Clone,Debug)]
struct Failures(Option<Rc<FailureState>>);
impl Drop for Failures {
    fn drop(&mut self) {if let Some(state)=self.0.take(){drop(Rc::into_inner(state));}}
}
impl Failures {
    fn state(&self)->&FailureState {self.0.as_deref().expect("live consensus failure source")}
    fn record(&self,cause:Error)->Error {
        if let Some(first)=self.state().first.borrow().as_ref() {
            return Error::retained_original(first.retained(),false);
        }
        // The one possible shared diagnostic allocation was included in setup,
        // before native work. A later failure never replaces the first cause.
        let first=SharedBackendFailure::new(BackendFailureKind::Other,Failure{cause,
            _source:self.state().source.clone(),_funding:self.state().funding.clone()});
        *self.state().first.borrow_mut()=Some(first.retained());
        Error::retained_original(first,false)
    }
    fn take(&self)->Option<BackendFailure> {
        self.state().first.borrow().as_ref().map(|first|first.retained().into_failure())
    }
    fn same(&self,other:&Self)->bool {
        Rc::ptr_eq(self.0.as_ref().expect("live source"),other.0.as_ref().expect("live source"))
    }
}
/// Exact loaded communicator/source owner, independently admitted before any
/// scheduler branch or word exchange. It supplies no frame execution authority.
pub(crate) struct PreparedRealtimeConsensusTransport {
    owner:OriginalCommunicationOwner,
    pool:WorkingMemoryPool,
    execution:InferenceExecutionIdentity,
    capacity:u64,
    failures:Failures,
    funding:WorkspaceMetadataFunding,
}
/// Output stays on its actual paid Host destination through neutral validation.
pub(crate) struct RealtimeConsensusOutput {
    words:OriginalCommunicationU32Words,
    expected:usize,
    failures:Failures,
    funding:WorkspaceMetadataFunding,
}
#[derive(Debug)]
pub(crate) struct RealtimeConsensusCompletion {
    inner:PreparationCompletion,
    failures:Failures,
}
impl PreparedRealtimeConsensusTransport {
    pub(crate) fn prepare(session:&MlxDistributedSession,pool:&WorkingMemoryPool,
        execution:&InferenceExecutionIdentity,capacity:u64)->Result<Self,Error> {
        let funding=pool.prepare_workspace_metadata(execution,capacity).map_err(Error::WorkspacePlanning)?;
        reserve(&funding,&[
            size_of::<Self>(),size_of::<Result<Self,Error>>(),size_of::<Failures>(),
            size_of::<FailureState>(),size_of::<Failure>(),size_of::<Error>(),
            size_of::<Option<SharedBackendFailure>>(),size_of::<Option<BackendFailure>>(),
            size_of::<std::cell::Ref<'_,Option<SharedBackendFailure>>>(),
            size_of::<std::cell::RefMut<'_,Option<SharedBackendFailure>>>(),
            size_of::<(&MlxDistributedSession,&WorkingMemoryPool,&InferenceExecutionIdentity,u64)>(),
            WorkspaceContext::metadata_rc_bytes::<FailureState>().ok_or_else(overflow)?,
            SharedBackendFailure::control_bytes::<Failure>().ok_or_else(overflow)?,
        ])?;
        let owner=session.original_communication_owner(&session.manifest,session.native_world(),&funding)?
            .pin_registered_buffers(pool)?;
        let failures=Failures(Some(Rc::new(FailureState{first:RefCell::new(None),
            source:owner.source().clone(),funding:funding.clone()})));
        let inspected=(|| {
            let source=owner.borrow()?;
            if source.world().retained_transport_stream().is_none()
                || source.world().has_original_parallel() || source.world().has_original_control()
                || source.world().original_control_request().is_some() {
                return Err(identity());
            }
            source.world_persistent()?;
            Ok(())
        })();
        inspected.map_err(|cause|failures.record(cause))?;
        Ok(Self{owner,pool:pool.clone(),execution:execution.clone(),capacity,failures,funding})
    }
    pub(crate) fn funding(&self)->&WorkspaceMetadataFunding {&self.funding}
    /// Returns an allocation-free alias; the cell stays retained while any
    /// completion can report, so cleanup cannot replace the first native cause.
    pub(crate) fn take_failure(&self)->Option<BackendFailure> {self.failures.take()}
    fn submit(&self,local:&[u32])->Result<Submission<RealtimeConsensusOutput,RealtimeConsensusCompletion>,Error> {
        let funding=self.pool.prepare_workspace_metadata(&self.execution,self.capacity)
            .map_err(Error::WorkspacePlanning)?;
        reserve(&funding,&[
            size_of::<RealtimeConsensusOutput>(),size_of::<RealtimeConsensusCompletion>(),
            size_of::<Submission<RealtimeConsensusOutput,RealtimeConsensusCompletion>>(),
            size_of::<Result<Submission<RealtimeConsensusOutput,RealtimeConsensusCompletion>,Error>>(),
            size_of::<(&Self,&[u32])>(),size_of::<Failures>(),size_of::<usize>(),
        ])?;
        let expected=local.len().checked_mul(self.participant_count()).ok_or_else(overflow)?;
        let source=self.owner.borrow_funded(&funding)?;
        let frame=source.prepare_preparation_frame(local,&self.pool)?;
        let operation=frame.prepare(&source)?;
        let stream=source.world().retained_transport_stream().ok_or_else(identity)?;
        let Submission{output,completion}=operation.start(&source,local,stream,&self.pool)?;
        Ok(Submission{output:RealtimeConsensusOutput{words:output,expected,
            failures:self.failures.clone(),funding},
            completion:RealtimeConsensusCompletion{inner:completion,failures:self.failures.clone()}})
    }
}
impl ConsensusTransport for PreparedRealtimeConsensusTransport {
    type Error=Error;
    fn participant_count(&self)->usize {self.owner.source().manifest().world_size()}
    fn all_gather_words(&self,_:&[u32])->Result<Vec<u32>,Error> {
        Err(self.failures.record(Error::InvalidOperation("original realtime consensus requires bounded borrowed resolution")))
    }
}
impl BoundedConsensusTransport for PreparedRealtimeConsensusTransport {
    type Completion=RealtimeConsensusCompletion;
    type GatherOutput=RealtimeConsensusOutput;
    fn metadata_funding(&self)->Option<&eredu_core::HostMetadataFunding> {Some(&self.funding)}
    fn submit_all_gather_words(&self,local:&[u32])
        ->Result<Submission<Self::GatherOutput,Self::Completion>,Error> {
        self.submit(local).map_err(|cause|self.failures.record(cause))
    }
    fn resolve_all_gather_words(&self,_:Self::GatherOutput)->Result<Vec<u32>,Error> {
        Err(self.failures.record(Error::InvalidOperation("original realtime consensus requires borrowed Host custody")))
    }
    fn with_resolved_all_gather_words<T,E,F>(&self,output:Self::GatherOutput,validate:F)
        ->Result<Result<T,E>,Error> where F:FnOnce(&[u32])->Result<T,E> {
        let result=(|| {
            reserve(&output.funding,&[
                size_of::<Self::GatherOutput>(),size_of::<CompletedCommunicationU32Words>(),
                size_of::<T>(),size_of::<E>(),size_of::<F>(),size_of::<Result<T,E>>(),
                size_of::<Result<Result<T,E>,Error>>(),size_of::<(&Self,&[u32])>(),
            ])?;
            if !self.failures.same(&output.failures) {return Err(identity());}
            let completed=output.words.resolve()?;
            if completed.as_slice().len()!=output.expected {return Err(identity());}
            Ok(validate(completed.as_slice()))
        })();
        result.map_err(|cause|self.failures.record(cause))
    }
}
impl Completion for RealtimeConsensusCompletion {
    type Error=Error;
    fn resources_releasable(&self)->bool {self.inner.resources_releasable()}
    fn is_complete(&self)->Result<bool,Error> {
        self.inner.is_complete().map_err(|cause|self.failures.record(cause))
    }
    fn wait(&self)->Result<(),Error> {
        self.inner.wait().map_err(|cause|self.failures.record(cause))
    }
}
impl BoundedCompletion for RealtimeConsensusCompletion {
    fn wait_bounded(self,policy:BoundedCompletionWait)->Result<BoundedCompletionOutcome,Error> {
        let Self{inner,failures}=self;
        inner.wait_bounded(policy).map_err(|cause|failures.record(cause))
    }
}
fn overflow()->Error {Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow)}
fn identity()->Error {Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch)}
fn reserve(funding:&WorkspaceMetadataFunding,parts:&[usize])->Result<(),Error> {
    funding.reserve_metadata(parts.iter().copied().try_fold(size_of_val(parts),usize::checked_add)
        .ok_or_else(overflow)?).map_err(Error::WorkspacePlanning)
}
