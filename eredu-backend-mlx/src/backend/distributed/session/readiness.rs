//! Explicit original request source for the existing readiness coordinator.
use super::*;
use eredu_core::{BackendFailure,TextStepContext,consensus::{ConsensusTransport,BoundedConsensusTransport},
    run_preparation::{TextPreparationStage,TextPreparationStatus,TextPreparationOutcome}};
use eredu_nn::workspace::{HostMetadataFunding,HostMetadataFundingError};
use eredu_runtime::{RetainedCommunicationSource,run_preparation::{TextPreparationCoordinator,TextPreparationTransport,
    TextPreparationAgreementError,TEXT_PREPARATION_WORDS},working_memory::{WorkingMemoryPool,InferenceExecutionIdentity}};
use crate::backend::runtime::distributed::{topology::OriginalCommunicationOwner,
    completion::OriginalCommunicationU32Words};
use std::{alloc::Layout,mem::{size_of,size_of_val}};

enum Origin { Text(TextStepContext), Reset(eredu_runtime::working_memory::SessionResetPreparationFunding) }
struct Body {
    owner:OriginalCommunicationOwner,
    coordinator:Arc<TextPreparationCoordinator>,
    pool:WorkingMemoryPool,
    execution:InferenceExecutionIdentity,
    origin:Origin,
    capacity:u64,
    funding:HostMetadataFunding,
}
/// Backend-owned preparation source lent explicitly by the shared text driver.
/// Clones keep the same coordinator/spending lineage; no byte grant is cloned.
pub struct MlxTextPreparationControl {
    body:Option<Rc<Body>>,
    funding:HostMetadataFunding,
}
impl Clone for MlxTextPreparationControl {
    fn clone(&self)->Self{Self{body:self.body.clone(),funding:self.funding.clone()}}
}
impl Drop for MlxTextPreparationControl {
    fn drop(&mut self){if let Some(body)=self.body.take(){drop(Rc::into_inner(body));}}
}
impl std::fmt::Debug for MlxTextPreparationControl {
    fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result{
        f.debug_struct("MlxTextPreparationControl").finish_non_exhaustive()
    }
}
#[derive(Debug,thiserror::Error)]
enum Cause {
    #[error("readiness source differs from the actual loaded session/request")] Identity,
    #[error(transparent)] Backend(Error),
    #[error(transparent)] Protocol(TextPreparationAgreementError),
    #[error(transparent)] Boundary(BackendFailure),
}
#[derive(Debug,thiserror::Error)]
#[error("{cause}")]
struct Failure {#[source] cause:Cause,source:RetainedCommunicationSource,funding:HostMetadataFunding}
fn failure(cause:Cause,source:&RetainedCommunicationSource,funding:&HostMetadataFunding)->BackendFailure{
    BackendFailure::new(eredu_core::BackendFailureKind::Other,Failure{cause,source:source.clone(),funding:funding.clone()})
}
fn overflow()->Error{Error::WorkspacePlanning(HostMetadataFundingError::Overflow)}
fn reserve(funding:&HostMetadataFunding,parts:&[usize])->Result<(),Error>{
    funding.reserve_metadata(parts.iter().copied().try_fold(size_of_val(parts),usize::checked_add)
        .ok_or_else(overflow)?).map_err(Error::WorkspacePlanning)
}
impl MlxDistributedSession {
    pub(crate) fn fence_reset_publication(&self) {
        self.authority.fence_protocol_failure(eredu_runtime::CommunicationOperation::FailureAgreement,
            eredu_runtime::DistributedExecutionPhase::InputPreparation,None);
    }
    pub(crate) fn prepare_original_readiness(&self,selected:&eredu_runtime::CommunicationManifest,
        world:&NativeGroup,pool:&WorkingMemoryPool,execution:&InferenceExecutionIdentity,
        capacity:u64,context:&TextStepContext)->Result<MlxTextPreparationControl,Error>{
        let funding=pool.prepare_workspace_metadata(execution,capacity).map_err(Error::WorkspacePlanning)?;
        self.prepare_readiness_source(selected,world,pool,execution,capacity,Origin::Text(context.clone()),funding)
    }
    pub(crate) fn prepare_reset_readiness(&self,selected:&eredu_runtime::CommunicationManifest,
        world:&NativeGroup,pool:&WorkingMemoryPool,execution:&InferenceExecutionIdentity,
        capacity:u64,preparation:eredu_runtime::working_memory::SessionResetPreparationFunding)
        ->Result<MlxTextPreparationControl,Error>{
        let funding=preparation.metadata().clone();
        self.prepare_readiness_source(selected,world,pool,execution,capacity,Origin::Reset(preparation),funding)
    }
    fn prepare_readiness_source(&self,selected:&eredu_runtime::CommunicationManifest,
        world:&NativeGroup,pool:&WorkingMemoryPool,execution:&InferenceExecutionIdentity,
        capacity:u64,origin:Origin,funding:HostMetadataFunding)->Result<MlxTextPreparationControl,Error>{
        reserve(&funding,&[
            size_of::<Origin>(),size_of::<Body>(),size_of::<MlxTextPreparationControl>(),size_of::<Option<Rc<Body>>>(),
            size_of::<InferenceExecutionIdentity>(),size_of::<TextStepContext>(),
            size_of::<Result<MlxTextPreparationControl,Error>>(),size_of::<Option<Arc<TextPreparationCoordinator>>>(),
            Layout::new::<[usize;2]>().extend(Layout::new::<Body>()).map_err(|_|overflow())?.0.pad_to_align().size(),
            BackendFailure::source_retention_peak_bytes::<Failure>().ok_or_else(overflow)?,
            size_of::<Failure>(),size_of::<Cause>(),
        ])?;
        let owner=self.original_communication_owner(selected,world,&funding)?.pin_registered_buffers(pool)?;
        let error=||Error::with_original_control_source(failure(Cause::Identity,owner.source(),&funding),false);
        let coordinator=self.preparation.as_ref().ok_or_else(error)?;
        // Source inspection retains the actual CPU transport stream and original
        // native implementation. A model tensor/stream cannot substitute for it.
        let source=owner.borrow()?;
        if source.world().retained_transport_stream().is_none() || source.world().has_original_parallel()
            || source.world().has_original_control() || source.world().original_control_request().is_some() {
            return Err(error());
        }
        source.world_persistent()?;
        drop(source);
        Ok(MlxTextPreparationControl{body:Some(Rc::new(Body{owner,coordinator:coordinator.clone(),pool:pool.clone(),
            execution:execution.clone(),origin,capacity,funding:funding.clone()})),funding})
    }
}
impl MlxTextPreparationControl {
    fn body(&self)->&Body{self.body.as_deref().expect("live preparation source")}
    // agree prices the shared neutral boundary error and this exact enclosure
    // before the protocol. The returned cause keeps the source/account alive.
    pub(crate) fn retain_reset_rejection(&self,cause:BackendFailure)->BackendFailure {
        let boundary=std::error::Error::source(&cause).is_some_and(|source|
            source.is::<eredu_core::run_preparation::TextPreparationRejected>()
                || source.is::<eredu_core::run_preparation::TextPreparationCancelled>());
        if boundary {
            BackendFailure::new(cause.kind(),Failure {cause:Cause::Boundary(cause),
                source:self.body().owner.source().clone(),funding:self.body().funding.clone()})
        } else { cause } // typed native/fixed refusal already has its own custody
    }
    pub(crate) fn agree(&self,session:&MlxDistributedSession,pool:&WorkingMemoryPool,execution:&InferenceExecutionIdentity,
        stage:TextPreparationStage,status:TextPreparationStatus)->Result<TextPreparationOutcome,BackendFailure>{
        let body=self.body();
        let funding=match &body.origin {
            Origin::Text(_)=>pool.prepare_workspace_metadata(execution,body.capacity).map_err(|cause|Error::WorkspacePlanning(cause).into_backend_failure())?,
            Origin::Reset(preparation)=>preparation.metadata().clone(),
        };
        reserve(&funding,&[
            size_of::<Transport<'_>>(),size_of::<TextPreparationStage>(),size_of::<TextPreparationStatus>(),
            BackendFailure::source_retention_peak_bytes::<eredu_core::run_preparation::TextPreparationRejected>()
                .ok_or_else(overflow).map_err(Error::into_backend_failure)?,
            BackendFailure::source_retention_peak_bytes::<eredu_core::run_preparation::TextPreparationCancelled>()
                .ok_or_else(overflow).map_err(Error::into_backend_failure)?,
            size_of::<TextPreparationOutcome>(),size_of::<Result<TextPreparationOutcome,TextPreparationAgreementError>>(),
            size_of::<Result<TextPreparationOutcome,BackendFailure>>(),size_of::<Failure>(),size_of::<Cause>(),
            BackendFailure::source_retention_peak_bytes::<Failure>().ok_or_else(overflow).map_err(Error::into_backend_failure)?,
            BackendFailure::source_retention_peak_bytes::<Error>().ok_or_else(overflow).map_err(Error::into_backend_failure)?,
            size_of::<(&Self,&MlxDistributedSession,&WorkingMemoryPool,&InferenceExecutionIdentity)>(),
        ]).map_err(Error::into_backend_failure)?;
        let reset_stage=matches!(stage,TextPreparationStage::SessionReset|TextPreparationStage::SessionResetPublication);
        if reset_stage != matches!(body.origin,Origin::Reset(_)) {
            return Err(failure(Cause::Identity,body.owner.source(),&funding));
        }
        if !body.pool.same_domain(pool) || !body.execution.same_execution(execution)
            || !session.preparation.as_ref().is_some_and(|value|Arc::ptr_eq(value,&body.coordinator))
            || !session.communicators.control_world().retained_source().is_some_and(|actual|actual.same_source(body.owner.source())) {
            return Err(failure(Cause::Identity,body.owner.source(),&funding));
        }
        let transport=Transport{source:self,session,funding:&funding};
        body.coordinator.agree(&transport,stage,status)
            .map_err(|cause|failure(Cause::Protocol(cause),body.owner.source(),&funding))
    }
}
struct Transport<'a>{source:&'a MlxTextPreparationControl,session:&'a MlxDistributedSession,funding:&'a HostMetadataFunding}
impl Transport<'_>{
    fn error(&self,cause:Cause)->Error{
        Error::with_original_control_source(failure(cause,self.source.body().owner.source(),self.funding),false)
    }
}
impl ConsensusTransport for Transport<'_>{
    type Error=Error;
    fn participant_count(&self)->usize{self.session.manifest.world_size()}
    fn all_gather_words(&self,_:&[u32])->Result<Vec<u32>,Error>{Err(self.error(Cause::Identity))}
}
impl BoundedConsensusTransport for Transport<'_>{
    type Completion=crate::backend::runtime::distributed::topology::original_source::PreparationCompletion;
    type GatherOutput=OriginalCommunicationU32Words;
    fn submit_all_gather_words(&self,local:&[u32])->Result<Submission<Self::GatherOutput,Self::Completion>,Error>{
        reserve(self.funding,&[
            size_of::<Result<Submission<Self::GatherOutput,Self::Completion>,Error>>(),
            size_of::<Submission<Self::GatherOutput,Self::Completion>>(),size_of::<&[u32;TEXT_PREPARATION_WORDS]>(),
            size_of::<std::array::TryFromSliceError>(),size_of::<[u32;TEXT_PREPARATION_WORDS]>(),
            BackendFailure::source_retention_peak_bytes::<Failure>().ok_or_else(overflow)?,
            BackendFailure::source_retention_peak_bytes::<Error>().ok_or_else(overflow)?,
        ])?;
        let words:&[u32;TEXT_PREPARATION_WORDS]=local.try_into().map_err(|_|self.error(Cause::Identity))?;
        let source=self.source.body().owner.borrow_funded(self.funding)?;
        let policy=match &self.source.body().origin { Origin::Reset(value)=>Some(value), Origin::Text(_)=>None };
        let frame=source.prepare_preparation_frame(words,&self.source.body().pool,policy)?;
        let operation=frame.prepare(&source)?;
        let stream=source.world().retained_transport_stream().ok_or_else(||self.error(Cause::Identity))?;
        operation.start(&source,words,stream,&self.source.body().pool,policy)
    }
    fn resolve_all_gather_words(&self,_:Self::GatherOutput)->Result<Vec<u32>,Error>{Err(self.error(Cause::Identity))}
    fn with_resolved_all_gather_words<T,E,F>(&self,output:Self::GatherOutput,validate:F)->Result<Result<T,E>,Error>
    where F:FnOnce(&[u32])->Result<T,E>{
        reserve(self.funding,&[size_of::<T>(),size_of::<E>(),size_of::<F>(),size_of::<Result<T,E>>(),
            size_of::<Result<Result<T,E>,Error>>(),size_of::<Self::GatherOutput>(),
            size_of::<crate::backend::runtime::distributed::completion::CompletedCommunicationU32Words>()])?;
        let completed=output.resolve()?;
        Ok(validate(completed.as_slice()))
    }
}
impl TextPreparationTransport for Transport<'_>{
    fn preparation_rank(&self)->usize{self.session.manifest.rank()}
    fn ensure_preparation_active(&self)->Result<(),BackendFailure>{
        self.source.body().owner.borrow_funded(self.funding).map(|_|()).map_err(Error::into_backend_failure)
    }
    fn fail_preparation(&self,_:&TextPreparationAgreementError){
        self.session.authority.fence_protocol_failure(eredu_runtime::CommunicationOperation::AllGatherEven,
            eredu_runtime::DistributedExecutionPhase::InputPreparation,None);
    }
}
