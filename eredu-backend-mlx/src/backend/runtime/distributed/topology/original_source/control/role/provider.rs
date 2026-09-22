//! A reached provider vote borrows one architecture-declared control source.
use super::*;
use super::super::super::parallel::RetainedExpertVote;
use super::super::super::inputs::CompletedCommunicationInput;
use crate::backend::runtime::distributed::completion::PreparedCommunicationWords;
use safemlx::{Array,PreparedInputPlan,distributed::GroupWorkerOperation};
use eredu_runtime::DistributedExecutionPhase;
fn invalid(c:&Custody)->Error{control_error(ControlCause::Identity,&c.source,&c.funding)}
impl OriginalParallelControlProjection {
    pub(crate) fn complete_provider_vote(&self,success:bool,group:&Group,phase:DistributedExecutionPhase,
        stream:&Stream,context:&Group)->Result<Option<bool>,Error>{
        if phase!=DistributedExecutionPhase::Execution{return Ok(None);}
        let Some(binding)=self.expert_binding(context)else{return Ok(None);};
        let retained=self.upgrade().map_err(|cause|Error::with_original_control_source(cause,false))?;
        let owner=retained.owner();let c=&self.custody;
        let Some(model)=owner.request.source.model()else{return Ok(None);};
        reserve(&c.funding,&[size_of::<RetainedExpertVote>(),size_of::<Option<RetainedExpertVote>>(),
            size_of::<Result<Option<bool>,Error>>(),size_of::<OriginalParallelControlOwner>(),
            size_of::<PhysicalVote>(),size_of::<ParallelControlClaim>(),size_of::<CompletedCommunicationInput>(),
            size_of::<PreparedInputPlan<'_>>(),size_of::<[i32;1]>(),size_of::<[usize;1]>(),
            size_of::<Result<bool,Error>>(),failure_control_bytes().ok_or_else(overflow)?])?;
        let Some(vote)=binding.claim_expert_vote(model,group,stream)
            .map_err(|cause|{owner.failed.set(true);cause})? else{return Ok(None);};
        let claim=owner.request.cursor.try_borrow_mut().map_err(|_|invalid(c))?
            .claim(ParallelControlEvent::Phase(phase)).map_err(|cause|
                control_error(ControlCause::WorkingMemory(cause),&c.source,&c.funding))?;
        if owner.failed.get()||owner.running.replace(true){owner.failed.set(true);return Err(invalid(c));}
        let _running=Running{running:&owner.running,failed:&owner.failed};
        let result=(||{
            let (order,peers,maximum)=vote.selection().ok_or_else(||invalid(c))?;
            let source=owner.request.source.communication_source()?;
            if !source.matches_group(order,group){return Err(invalid(c));}
            let runtime=owner.request.source.agreement_inputs().ok_or_else(||invalid(c))?.runtime();
            let data=[i32::from(success)];let shape=[1usize];
            let plan=runtime.i32(&data,&shape).map_err(|cause|failure(Cause::Input(cause),&c.source,&c.funding))?;
            let [input]=super::super::super::inputs::construct_retained(&source,[plan])?;
            let completed=if let Some(quote)=vote.logical(){
                let finish=|output:Array,observer:&OriginalScopeObserver|{
                    let words=PreparedCommunicationWords::read_completed(&source,&output,observer)?;
                    match words.as_slice(){[sum]=>Ok(*sum==i32::try_from(peers).map_err(|_|invalid(c))?),_=>Err(invalid(c))}
                };
                reserve(&c.funding,&[size_of_val(&finish)])?;
                logical::execute_source(&retained,claim,group,&input,stream,quote,None,finish)?
            }else{
                let operation=source.group_cpu_operation_storage(order,input.value(),GroupWorkerOperation::Sum)?;
                let backing=operation.backing_storage(runtime)?.capacity();
                let operation=operation.with_completion()?;
                if backing>maximum{return Err(invalid(c));}
                let capacity=AgreementCapacity{graph:operation.graph_capacity(),records:operation.record_capacity(),backing};
                drop(operation);drop(source);
                let physical=PhysicalVote{input,claim,vote,capacity,owner:OriginalParallelControlOwner(retained.0.clone()),custody:c.clone()};
                return run_native_role(physical,capacity,&owner.native,c,|physical,observer|{
                    Ok(physical.run(observer).and_then(|value|{
                        binding.complete_expert_vote(model,&physical.vote,stream)?;Ok(value)
                    }))
                }).map_err(|cause|Error::with_original_control_source(cause,false))?;
            };
            binding.complete_expert_vote(model,&vote,stream)?;
            Ok(completed)
        })();
        if result.is_err(){owner.failed.set(true);}
        result.map(Some)
    }
}
struct PhysicalVote {
    input:CompletedCommunicationInput,
    claim:ParallelControlClaim,
    vote:RetainedExpertVote,
    capacity:AgreementCapacity,
    owner:OriginalParallelControlOwner,
    custody:Custody,
}
impl PhysicalVote {
    fn run(&self,observer:&OriginalScopeObserver)->Result<bool,Error>{
        let owner=self.owner.owner();let c=&self.custody;
        reserve(&c.funding,&[size_of::<Self>(),size_of::<PreparedCommunicationWords>(),
            size_of::<Result<bool,Error>>(),failure_control_bytes().ok_or_else(overflow)?])?;
        if self.claim.identity()!=owner.request.cursor.borrow().identity()
            ||self.claim.event()!=ParallelControlEvent::Phase(DistributedExecutionPhase::Execution){return Err(invalid(c));}
        let (order,peers,maximum)=self.vote.selection().ok_or_else(||invalid(c))?;
        let source=owner.request.source.communication_source()?;
        if !self.input.belongs_to(&source){return Err(invalid(c));}
        let runtime=owner.request.source.agreement_inputs().ok_or_else(||invalid(c))?.runtime();
        let operation=source.group_cpu_operation_storage(order,self.input.value(),GroupWorkerOperation::Sum)?;
        let backing=operation.backing_storage(runtime)?.capacity();
        let operation=operation.with_completion()?;
        if backing>maximum||!self.capacity.covers(AgreementCapacity{graph:operation.graph_capacity(),
            records:operation.record_capacity(),backing}){return Err(invalid(c));}
        let words=PreparedCommunicationWords::prepare_operation(&source,&operation)?;
        let ready=operation.prepare_resources(&source,Some(order))?;
        let group=source.group(order).ok_or_else(||invalid(c))?.0;
        let stream=group.retained_transport_stream().ok_or_else(||invalid(c))?;
        let accepted=operation.construct_accepted(&source,observer,stream)?;
        let (output,completion)=words.submit_accepted(accepted,ready)?;
        let kind=CommunicationOperation::FailureAgreement;let phase=DistributedExecutionPhase::Execution;
        let output=source.authority.wait_with_error(eredu_core::Submission{output,completion},kind,phase,None,
            |cause|eredu_runtime::PartitionExecutionError::PreparedCommunication{operation:kind,phase,completion:true,
                source:failure(Cause::Native(cause),&c.source,&c.funding).into_backend_failure()})
            .map_err(|cause|Error::with_original_control_source(eredu_core::BackendFailure::from_error(cause),false))?;
        let output=output.resolve()?;
        match output.as_slice(){[sum]=>Ok(*sum==i32::try_from(peers).map_err(|_|invalid(c))?),_=>Err(invalid(c))}
    }
}
