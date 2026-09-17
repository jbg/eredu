//! Public admission and submission using the singular scheduler/frame driver.
use super::*;
use super::{original_branch::{OriginalRealtimeSessionState,RealtimeBranchSource},
    original_execution::{FrameNativeLayout,PreparedOriginalRealtimeFrame},
    retained_frame_source::{compile_frame_source,RealtimeNativeFrameSource,CompiledRealtimeFrameSource}};
use crate::backend::{OriginalCopyEnvironment,
    nn::workspace::{ResidentExecutionMechanisms,MlxMetalWorkspaceMechanisms}};
use eredu_core::{BackendFailure,scheduler::{SchedulerError,SchedulerProgress,WorkId}};
use eredu_nn::workspace::{WorkspaceContext,WorkspaceMetadataFunding};
use eredu_runtime::{RealtimePayloadState,RealtimePayloadContract,RealtimeSessionScheduler,
    working_memory::{RealtimeFrameRequirements,WorkingMemoryError}};
use std::{cell::RefCell,mem::{size_of,size_of_val},time::Instant};

/// Unpublished scheduler branch with one accepted native frame preparation.
pub type MlxManagedFrameSessionBranch=MlxFrameSessionBranch<PreparedOriginalRealtimeFrame>;
/// Existing fair scheduler with original frame admission before state branching.
pub type MlxManagedRealtimeScheduler=RealtimeSessionScheduler<
    RealtimePayloadState<MlxKeyValueState,MlxTensor>,GenerationSampler,RandomState,
    MlxRealtimeCompletion,MlxPrepublicationFrame,PreparedOriginalRealtimeFrame>;
type BorrowedModel<'a>=RefCell<&'a mut MoshiRealtimeExecution<MlxRealtimeExecution>>;
type FirstFrameFailure=RefCell<Option<BackendFailure>>;
fn keep_first<T>(result:Result<T,BackendFailure>,slot:Option<&FirstFrameFailure>)->Result<T,BackendFailure> {
    match (result,slot) {
        (Err(cause),Some(slot))=>{
            if slot.borrow().is_none(){*slot.borrow_mut()=Some(cause);}
            Err(eredu_core::HostMetadataFundingError::Unavailable.into_backend_failure())
        }
        (result,_)=>result,
    }
}
type FrameCompletionFailure=eredu_runtime::RealtimePrepublicationError<Error,Error>;
#[derive(Debug,thiserror::Error)]
#[error("realtime frame completion: {cause}")]
struct FundedCompletionFailure {
    #[source] cause:FrameCompletionFailure,
    _funding:WorkspaceMetadataFunding,
}
fn completion_failure_callback<'a>(first:Option<&'a FirstFrameFailure>,funding:Option<&'a WorkspaceMetadataFunding>)
    ->impl FnMut(WorkId,FrameCompletionFailure)+'a + use<'a> {
    move |_,cause| {
        let first=first.expect("actual completion callback retains its prepaid first source slot");
        if first.borrow().is_none() {
            *first.borrow_mut()=Some(BackendFailure::from_error(FundedCompletionFailure{
                cause,_funding:funding.expect("completion source paid before polling").clone()}));
        }
    }
}

#[derive(Debug,thiserror::Error)]
#[error("{cause}")]
struct FundedSchedulerFailure {#[source] cause:SchedulerError,_funding:WorkspaceMetadataFunding}

#[derive(Debug,thiserror::Error)]
#[error("realtime frame {stage}: {cause}")]
struct FundedFailure {stage:&'static str,#[source] cause:Error,_funding:WorkspaceMetadataFunding}
fn funded(cause:Error,funding:WorkspaceMetadataFunding,stage:&'static str)->BackendFailure {
    match cause.take_retained_backend_failure() {
        Ok(cause)=>cause,
        Err(cause)=>BackendFailure::from_error(FundedFailure{stage,cause,_funding:funding}),
    }
}
fn overflow()->Error {Error::PrefillControl(WorkingMemoryError::Overflow)}
fn identity()->Error {Error::PrefillControl(WorkingMemoryError::IdentityMismatch)}
fn u64_bytes(value:usize)->Result<u64,Error>{u64::try_from(value).map_err(|_|overflow())}

// Both the public wrapper and its cold source use these exact borrowed
// callback representations. No model is erased or put in a process side table.
fn scheduler_callbacks<'a,'model>(context:Option<&'a MlxRealtimeExecutionContext>,
    model:Option<&'a BorrowedModel<'model>>,first:Option<&'a FirstFrameFailure>,capacity:u64,limit:Option<u64>)
    ->(impl FnMut(WorkId,&RealtimeInputFrame,&OriginalRealtimeSessionState)
            ->Result<MlxManagedFrameSessionBranch,BackendFailure>+'a + use<'a,'model>,
       impl FnMut(WorkId,&RealtimeInputFrame,&mut MlxManagedFrameSessionBranch)
            ->Result<MlxPrepublicationFrame,BackendFailure>+'a + use<'a,'model>)
where 'model:'a {
    (move |_,frame,state| {
        let context=context.ok_or_else(||identity().into_backend_failure())?;
        let model=model.ok_or_else(||identity().into_backend_failure())?
            .try_borrow().map_err(|_|identity().into_backend_failure())?;
        keep_first(context.prepare_realtime_frame(&model,frame,state,capacity,limit),first)
    },move |_,frame,branch| {
        let context=context.ok_or_else(||identity().into_backend_failure())?;
        let mut model=model.ok_or_else(||identity().into_backend_failure())?
            .try_borrow_mut().map_err(|_|identity().into_backend_failure())?;
        keep_first(context.submit_prepared_realtime_frame(&mut model,frame,branch),first)
    })
}
fn planning_control_bytes()->Option<usize> {
    let callbacks=scheduler_callbacks(None,None,None,0,None);
    let parts=[size_of::<&'static str>(),size_of_val(&callbacks),size_of::<BorrowedModel<'_>>(),
        size_of::<std::cell::Ref<'_,&mut MoshiRealtimeExecution<MlxRealtimeExecution>>>(),
        size_of::<std::cell::RefMut<'_,&mut MoshiRealtimeExecution<MlxRealtimeExecution>>>(),
        size_of::<RealtimeBranchSource<'_>>(),size_of::<CompiledRealtimeFrameSource>(),
        size_of::<FrameNativeLayout>(),size_of::<RealtimeFrameRequirements>(),
        size_of::<RealtimePayloadContract>(),size_of::<WorkspaceContext>(),
        size_of::<RealtimeNativeFrameSource<'_>>(),size_of::<ResidentExecutionMechanisms>(),
        size_of::<Result<MlxManagedFrameSessionBranch,BackendFailure>>(),size_of::<FundedFailure>(),
        size_of::<(&MlxRealtimeExecutionContext,&MoshiRealtimeExecution<MlxRealtimeExecution>,
            &RealtimeInputFrame,&OriginalRealtimeSessionState,u64,Option<u64>)>(),
        size_of::<Arc<neutral_moshi::SelectedRealtimeResources>>(),
        OriginalCopyEnvironment::control_bytes()?,
        BackendFailure::source_retention_peak_bytes::<FundedFailure>()?];
    parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
}
/// Source of the public submit wrapper, debited once before borrowing the
/// original runtime. The retained model/stream allocations have separate owners.
pub(super) fn submission_control_bytes()->Option<usize> {
    let parts=[size_of::<FundedFailure>(),size_of::<WorkspaceMetadataFunding>(),
        size_of::<Arc<neutral_moshi::SelectedRealtimeResources>>(),
        size_of::<PreparedOriginalRealtimeFrame>(),
        size_of::<Result<MlxPrepublicationFrame,BackendFailure>>(),
        size_of::<(&MlxRealtimeExecutionContext,&mut MoshiRealtimeExecution<MlxRealtimeExecution>,
            &RealtimeInputFrame,&mut MlxManagedFrameSessionBranch)>(),
        OriginalCopyEnvironment::control_bytes()?,
        BackendFailure::source_retention_peak_bytes::<FundedFailure>()?];
    parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
}
#[cfg(all(target_vendor="apple",feature="metal",not(feature="cuda")))]
fn mechanisms(stream:&Stream,funding:&WorkspaceMetadataFunding)->Result<ResidentExecutionMechanisms,Error> {
    ResidentExecutionMechanisms::from_stream(MlxMetalWorkspaceMechanisms::current_host().map_err(Error::Neural)?,
        stream,funding).map_err(Error::Neural)
}
#[cfg(not(all(target_vendor="apple",feature="metal",not(feature="cuda"))))]
fn mechanisms(_stream:&Stream,_funding:&WorkspaceMetadataFunding)->Result<ResidentExecutionMechanisms,Error> {
    Err(Error::PrefillControl(WorkingMemoryError::UnknownBound))
}
impl MlxRealtimeExecutionContext {
    /// Admits one complete frame against an explicit application-selected
    /// shared-domain capacity and optional per-frame limit before cloning state.
    /// Native preparation copies complete before this method returns. The
    /// returned branch owns its one-use source through submission or cancellation.
    pub fn prepare_realtime_frame(&self,model:&MoshiRealtimeExecution<MlxRealtimeExecution>,
        frame:&RealtimeInputFrame,canonical:&OriginalRealtimeSessionState,
        capacity_bytes:u64,frame_limit_bytes:Option<u64>)
        ->Result<MlxManagedFrameSessionBranch,BackendFailure> {
        let resources=model.executor().completion_resources();
        resources.validate_loaded_source().map_err(Error::into_backend_failure)?;
        // Original source installation belongs to this actual loaded backend,
        // not merely a replacement stream on an equal device.
        if !std::ptr::eq(self.backend(),resources.backend()) {
            return Err(identity().into_backend_failure());
        }
        let funding:WorkspaceMetadataFunding=self.backend().memory_pool()
            .prepare_workspace_metadata(resources.execution_identity(),capacity_bytes)
            .map_err(|cause|Error::WorkspacePlanning(cause).into_backend_failure())?.into();
        let mut stage="planning controls";
        let result=(|| {
            funding.reserve_metadata(planning_control_bytes().ok_or_else(overflow)?)
                .map_err(Error::WorkspacePlanning)?;
            stage="original source environment";
            let environment=self.original_copy_environment().map_err(|cause|Error::Neural(funding.metadata_source(cause)))?;
            let runtime=environment.input_runtime().map_err(|cause|Error::Neural(funding.metadata_source(cause)))?;
            let stream=environment.stream();
            stage="selected numerical mechanism";
            let mechanism=mechanisms(stream,&funding)?;
            let context=mechanism.context(funding.clone()).map_err(|cause|Error::Neural(cause.into()))?;
            stage="schedule occurrence";
            let occurrence=canonical.frame_occurrence_for(frame.batch())
                .map_err(|cause|Error::Neural(context.metadata_source(cause)))?;
            stage="ingress source";
            let ingress=resources.ingress_contract().inspect(frame)
                .map_err(|cause|Error::Neural(context.metadata_source(cause)))?;
            stage="payload contract";
            let payload=canonical.payload_contract_for_frame(resources.ingress_contract(),frame.batch(),&funding)
                .map_err(|cause|Error::Neural(context.metadata_source(cause)))?;
            let generation=canonical.generation();
            let native=RealtimeNativeFrameSource{
                state:generation.model_state().model_state(),history:generation.model_state().payload_history(),
                schedule:generation.schedule_state(),sampling:generation.sampling(),
                samplers:generation.samplers(),random:generation.random_state(),
            };
            stage="cold equation source";
            let source=compile_frame_source(model.executor(),native,ingress,&payload,&runtime,stream,environment.pool(),mechanism,&context)?;
            stage="native frame layout";
            let layout=FrameNativeLayout::inspect(&source,&runtime,ingress)?;
            stage="state branch source";
            let branch_source=RealtimeBranchSource::inspect(canonical,&runtime,stream)?;
            stage="frame requirements";
            let controls=layout.control_bytes().checked_add(branch_source.host_bytes().ok_or_else(overflow)?)
                .ok_or_else(overflow)?;
            // Numerical storage already includes every newly constructed state
            // root. The separate preparation phase covers copied prior state.
            let mut requirements=RealtimeFrameRequirements::new(occurrence,
                Some(u64_bytes(layout.capacity.backing)?),Some(0),
                Some(u64_bytes(layout.capacity.graph)?),Some(u64_bytes(layout.capacity.records)?),
                Some(u64_bytes(controls)?),source.source_program.facts()).map_err(Error::PrefillControl)?;
            if let Some((copy,controls))=branch_source.preparation()? {
                requirements=requirements.with_preparation(copy,Some(u64_bytes(controls)?)).map_err(Error::PrefillControl)?;
            }
            stage="frame admission";
            let account=environment.pool().reserve_realtime_frame(resources.execution_identity(),requirements,
                capacity_bytes,frame_limit_bytes).map_err(|cause|Error::Neural(context.metadata_source(cause)))?;
            stage="accepted source installation";
            let mut preparation=PreparedOriginalRealtimeFrame::accept(account,source,layout,payload,model.executor())?;
            stage="prepared state branch";
            let mut branch=branch_source.prepare(&mut preparation,Some(model.selected().completion().timeout()))
                .map_err(Error::StorageSource)?;
            branch.install_preparation(preparation).map_err(|_|identity())?;
            Ok(branch)
        })();
        result.map_err(|cause|funded(cause,funding,stage))
    }

    /// Consumes the branch's actual admitted preparation and runs the ordinary
    /// coordinator/equations under completion-held native ownership. No source
    /// is recreated on a repeated call or after a failed submission.
    pub fn submit_prepared_realtime_frame(&self,model:&mut MoshiRealtimeExecution<MlxRealtimeExecution>,
        frame:&RealtimeInputFrame,branch:&mut MlxManagedFrameSessionBranch)
        ->Result<MlxPrepublicationFrame,BackendFailure> {
        let preparation=branch.take_preparation().ok_or_else(||identity().into_backend_failure())?;
        let funding=preparation.funding().clone();
        let result=(|| {
            funding.reserve_metadata(submission_control_bytes().ok_or_else(overflow)?)
                .map_err(Error::WorkspacePlanning)?;
            let resources=model.executor().completion_resources();
            resources.validate_loaded_source()?;
            if !std::ptr::eq(self.backend(),resources.backend()){return Err(identity());}
            let environment=self.original_copy_environment().map_err(|cause|Error::Neural(funding.metadata_source(cause)))?;
            let runtime=environment.input_runtime().map_err(|cause|Error::Neural(funding.metadata_source(cause)))?;
            let timeout=Some(model.selected().completion().timeout());
            preparation.run(model,branch,frame,resources.execution_identity(),&runtime,environment.stream(),timeout)
        })();
        result.map_err(|cause|funded(cause,funding,"submission"))
    }

    /// Runs the existing local fair scheduler with admission before every
    /// branch. Capacity is application policy; it is not a hardware-memory
    /// observation. Polling, cancellation, publication and limits remain in
    /// the shared scheduler.
    pub fn run_realtime_bounded(&self,model:&mut MoshiRealtimeExecution<MlxRealtimeExecution>,
        scheduler:&mut MlxManagedRealtimeScheduler,now:Instant,maximum_frames:usize,
        capacity_bytes:u64,frame_limit_bytes:Option<u64>)
        ->Result<SchedulerProgress<RealtimeInputFrame,MlxPrepublicationFrame>,SchedulerError> {
        let model=RefCell::new(model);
        let (prepare,execute)=scheduler_callbacks(Some(self),Some(&model),None,capacity_bytes,frame_limit_bytes);
        scheduler.run_local_bounded_with_preparation(now,maximum_frames,prepare,execute)
    }

    /// Runs topology-wide source preparation, schedule and completion agreement
    /// with the retained loaded world's independently admitted word transport.
    /// Model and scheduler sources remain separate owners in the same pool.
    #[allow(clippy::too_many_arguments)]
    pub fn run_realtime_distributed_bounded(
        &self, model:&mut MoshiRealtimeExecution<MlxRealtimeExecution>,
        scheduler:&mut MlxManagedRealtimeScheduler,protocol:u64,now:Instant,
        maximum_frames:usize,capacity_bytes:u64,frame_limit_bytes:Option<u64>,
    )->Result<SchedulerProgress<RealtimeInputFrame,MlxPrepublicationFrame>,BackendFailure> {
        let resources=model.executor().completion_resources();
        resources.validate_loaded_source().map_err(Error::into_backend_failure)?;
        if !std::ptr::eq(self.backend(),resources.backend()){return Err(identity().into_backend_failure());}
        let (session,_)=model.executor().parallel_communication()
            .ok_or_else(||identity().into_backend_failure())?;
        let transport=crate::backend::distributed::PreparedRealtimeConsensusTransport::prepare(
            session,self.backend().memory_pool(),resources.execution_identity(),capacity_bytes)
            .map_err(Error::into_backend_failure)?;
        let callback_shape=scheduler_callbacks(None,None,None,0,None);
        let completion_shape=completion_failure_callback(None,None);
        let parts=[size_of_val(&callback_shape),size_of_val(&completion_shape),size_of::<FirstFrameFailure>(),
            size_of::<FrameCompletionFailure>(),size_of::<FundedCompletionFailure>(),
            size_of::<(WorkId,FrameCompletionFailure)>(),
            size_of::<(Option<&FirstFrameFailure>,Option<&WorkspaceMetadataFunding>)>(),
            size_of::<std::cell::Ref<'_,Option<BackendFailure>>>(),
            size_of::<std::cell::RefMut<'_,Option<BackendFailure>>>(),
            BackendFailure::source_retention_peak_bytes::<FundedCompletionFailure>().ok_or_else(||overflow().into_backend_failure())?,
            size_of::<BorrowedModel<'_>>(),size_of::<FundedSchedulerFailure>(),
            size_of::<Result<SchedulerProgress<RealtimeInputFrame,MlxPrepublicationFrame>,BackendFailure>>(),
            size_of::<(&Self,u64,usize,u64,Option<u64>)>(),
            BackendFailure::source_retention_peak_bytes::<FundedSchedulerFailure>().ok_or_else(||overflow().into_backend_failure())?];
        let bytes=parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
            .ok_or_else(||overflow().into_backend_failure())?;
        transport.funding().reserve_metadata(bytes).map_err(BackendFailure::from)?;
        let first=RefCell::new(None);
        let model=RefCell::new(model);
        let (prepare,execute)=scheduler_callbacks(Some(self),Some(&model),Some(&first),capacity_bytes,frame_limit_bytes);
        let completion_failure=completion_failure_callback(Some(&first),Some(transport.funding()));
        let result=scheduler.run_distributed_bounded_with_preparation_and_errors(protocol,&transport,now,
            maximum_frames,prepare,execute,completion_failure);
        if let Some(cause)=first.into_inner(){return Err(cause);}
        if let Some(cause)=transport.take_failure(){return Err(cause);}
        result.map_err(|cause|BackendFailure::from_error(FundedSchedulerFailure{
            cause,_funding:transport.funding().clone()}))
    }
}
