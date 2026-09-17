//! Existing original agreement source/role lent to a capture member vote.
use super::*;
use eredu_runtime::capture::partition::PartitionCaptureHookTransport;

impl OriginalCaptureTransport {
    fn complete_member_vote(&self,members:&[usize],wait:BoundedCompletionWait,success:bool)
        ->Result<bool,Error> {
        let owner=self.owner.owner();let c=&owner.custody;
        controls(c)?;
        reserve(&c.funding,&[
            size_of::<(&Self,&[usize],BoundedCompletionWait,bool)>(),
            size_of::<OriginalParallelControlInvocation>(),size_of::<ParallelControlEvent>(),
            size_of::<Result<OriginalParallelControlInvocation,Error>>(),size_of::<Result<bool,Error>>(),
            size_of::<Submission<crate::backend::runtime::distributed::completion::OriginalCommunicationBool,MlxNeuralCommunicationCompletion>>(),
            size_of::<Result<bool,eredu_core::BackendFailure>>(),size_of::<Result<Result<bool,Error>,eredu_core::BackendFailure>>(),
            size_of::<(&Group,&Stream)>(),size_of::<Option<&CommunicationGroupDescriptor>>(),
            failure_control_bytes().ok_or_else(overflow)?,
        ])?;
        self.estimate_capture_hook(members).map_err(|cause|fail(CaptureCause::Protocol(cause.into()),c))?;
        if members.len()<2||!members.contains(&self.capture_rank()) {
            return Err(fail(CaptureCause::Identity,c));
        }
        let source=owner.request.source.communication_source()?;
        source.validate()?;
        if !self.authority.same_authority(source.authority)||!c.source.same_source(source.source())
            ||self.authority.completion_policy().is_none_or(|policy|policy.bounded_wait()!=wait) {
            return Err(fail(CaptureCause::Identity,c));
        }
        let descriptor=MlxDistributedSession::capture_hook_source(source.source().manifest().groups(),members)
            .ok_or_else(||fail(CaptureCause::Identity,c))?;
        let selected=source.source().manifest().select_group_operation(descriptor.id(),CommunicationOperation::FailureAgreement)
            .map_err(|cause|failure(Cause::Rank(cause),&c.source,&c.funding))?;
        let group=source.group(selected.order()).ok_or_else(||fail(CaptureCause::Identity,c))?.0;
        let stream=group.retained_transport_stream().ok_or_else(||fail(CaptureCause::Identity,c))?;
        let phase=DistributedExecutionPhase::ObservationCoordination;
        let event=ParallelControlEvent::Phase(phase);
        // Existing request issuance consumes this occurrence before any native
        // role can start. Its selected Group comes from this same actual source.
        let invocation=match owner.request.prepare(event,Some(group)) {
            Ok(invocation)=>invocation,
            Err(cause)=>{owner.failed.set(true);return Err(cause);}
        };
        if owner.failed.get()||owner.running.replace(true) {
            owner.failed.set(true);return Err(fail(CaptureCause::Identity,c));
        }
        let _running=Running {running:&owner.running,failed:&owner.failed};
        let result=run_role(invocation,&owner.bank,&owner.controls,c,stream,|group,_funding| {
            let binding=group.original_control().ok_or_else(||fail(CaptureCause::Identity,c))?;
            let (output,completion)=binding.agree(group,success,stream).map_err(|cause|{
                source.authority.fence_protocol_failure(CommunicationOperation::FailureAgreement,phase,None);cause
            })?;
            let output=source.authority.wait_with_error(Submission {output,completion},
                CommunicationOperation::FailureAgreement,phase,None,|cause:Error|
                    PartitionExecutionError::PreparedCommunication {operation:CommunicationOperation::FailureAgreement,
                        phase,completion:true,source:cause.into_backend_failure()})
                .map_err(|cause|fail(CaptureCause::Communication(cause),c))?;
            output.resolve()
        }).map_err(|cause|Error::with_original_control_source(cause,false)).and_then(|result|result);
        if result.is_err() {owner.failed.set(true);}
        result
    }
}
impl PartitionCaptureHookTransport for OriginalCaptureTransport {
    // Original completion remains inside run_role; no synthetic native event or
    // completed output is inserted into the legacy submission-only interface.
    type HookOutput=();
    fn estimate_capture_hook(&self,members:&[usize])->Result<CaptureUsage,CaptureError> {
        let usage=MlxDistributedSession::capture_hook_usage(self.participant_count(),members)?;
        if members.len()>1&&MlxDistributedSession::capture_hook_source(&self.groups,members).is_none() {
            return Err(CaptureError::Unsupported("capture hook group has no selected exact failure agreement".into()));
        }
        Ok(usage)
    }
    fn complete_capture_hook(&self,members:&[usize],wait:BoundedCompletionWait,success:bool)
        ->Result<bool,PartitionCaptureExchangeError> {
        self.complete_member_vote(members,wait,success)
            .map_err(|cause|PartitionCaptureExchangeError::Backend(cause.into_backend_failure()))
    }
    fn submit_capture_hook(&self,_members:&[usize],_success:bool)
        ->Result<Submission<Self::HookOutput,MlxNeuralCommunicationCompletion>,Error> {
        let c=&self.owner.owner().custody;controls(c)?;Err(fail(CaptureCause::Identity,c))
    }
    fn resolve_capture_hook(&self,_output:Self::HookOutput)->Result<bool,Error> {
        let c=&self.owner.owner().custody;controls(c)?;Err(fail(CaptureCause::Identity,c))
    }
}
