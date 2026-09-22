//! Same accepted logical pair and bounded completion, retaining the actual
//! settled owner for the shared public transfer completion.
use super::*;
use eredu_core::{Completion,BoundedCompletion,BoundedCompletionWait,BoundedCompletionOutcome,BoundedSubmissionOutcome};

struct ReturnWait<'a> {
    completion:Option<OriginalCommunicationCompletion>,
    destination:&'a mut Option<OriginalCommunicationCompletion>,
}
impl Completion for ReturnWait<'_> {
    type Error=safemlx::error::Exception;
    fn is_complete(&self)->Result<bool,Self::Error>{self.completion.as_ref().expect("owned completion").is_complete()}
    fn wait(&self)->Result<(),Self::Error>{self.completion.as_ref().expect("owned completion").wait()}
    fn resources_releasable(&self)->bool{self.completion.as_ref().is_some_and(Completion::resources_releasable)}
}
impl BoundedCompletion for ReturnWait<'_> {
    fn wait_bounded(mut self,policy:BoundedCompletionWait)->Result<BoundedCompletionOutcome,Self::Error>{
        match self.completion.take().expect("owned completion").wait_bounded_retaining(policy)? {
            BoundedSubmissionOutcome::Completed(completion)=>{
                *self.destination=Some(completion);Ok(BoundedCompletionOutcome::Completed)
            }
            BoundedSubmissionOutcome::DeadlineExceeded{cancellation}=>
                Ok(BoundedCompletionOutcome::DeadlineExceeded{cancellation}),
        }
    }
}
pub(super) fn wait_retaining(source:&OriginalCommunicationSource<'_>,completion:OriginalCommunicationCompletion,
    phase:DistributedExecutionPhase,route:CommunicationRouteId,c:&Custody)->Result<OriginalCommunicationCompletion,Error>{
    reserve(&c.funding,&[size_of::<ReturnWait<'_>>(),size_of::<Option<OriginalCommunicationCompletion>>(),
        size_of::<Result<OriginalCommunicationCompletion,Error>>(),size_of::<BoundedSubmissionOutcome<OriginalCommunicationCompletion>>(),
        size_of::<Result<BoundedSubmissionOutcome<OriginalCommunicationCompletion>,safemlx::error::Exception>>(),
        failure_bytes().ok_or_else(overflow)?])?;
    let mut settled=None;
    source.authority.wait_with_error(eredu_core::Submission{output:(),completion:ReturnWait{
        completion:Some(completion),destination:&mut settled}},CommunicationOperation::SendReceive,phase,Some(route),
        |cause|PartitionExecutionError::PreparedCommunication{operation:CommunicationOperation::SendReceive,
            phase,completion:true,source:fail(FrameCause::Native(cause),c).into_backend_failure()})
        .map_err(|cause|fail(FrameCause::Communication(cause),c))?;
    settled.ok_or_else(||fail(FrameCause::Identity,c))
}
struct Exchange {
    quote:RetainedPipelineBoundary,
    input:Array,
    order:usize,
    round:usize,
    claim:ParallelControlClaim,
    capacity:AgreementCapacity,
    owner:OriginalParallelControlOwner,
    custody:Custody,
}
impl OriginalParallelControlProjection {
    pub(super) fn exchange_boundary_frame(&self,input:Array,route:CommunicationRouteId,quote:RetainedPipelineBoundary)
        ->Result<(Array,OriginalCommunicationCompletion),Error>{
        reserve(&self.custody.funding,&[size_of::<Exchange>(),size_of::<ParallelControlClaim>(),
            size_of::<OriginalParallelControlOwner>(),size_of::<AgreementCapacity>(),
            size_of::<Result<AgreementCapacity,Error>>(),size_of::<Result<(Array,OriginalCommunicationCompletion),Error>>(),
            failure_bytes().ok_or_else(overflow)?])?;
        let owner=self.upgrade().map_err(|cause|Error::with_original_control_source(cause,false))?;
        let c=&self.custody;
        let first_claim=owner.owner().request.cursor.try_borrow_mut().map_err(|_|fail(FrameCause::Identity,c))?
            .claim(ParallelControlEvent::Phase(DistributedExecutionPhase::Execution))
            .map_err(|_|fail(FrameCause::Identity,c))?;
        if !self.has_model_boundary() || owner.owner().failed.get() || owner.owner().running.replace(true){
            owner.owner().failed.set(true);return Err(fail(FrameCause::Identity,c));
        }
        let _running=Running{running:&owner.owner().running,failed:&owner.owner().failed};
        let result=(||{
            let source=owner.owner().request.source.communication_source()?;
            let order=source.source().manifest().routes().iter().position(|item|item.id()==route)
                .ok_or_else(||fail(FrameCause::Identity,c))?;
            if quote.value().route!=route || quote.value().wire_shape()!=input.shape() || input.dtype()!=safemlx::Dtype::Uint8 {
                return Err(fail(FrameCause::Identity,c));
            }
            let selected=source.framed_route_exchange(order)?.ok_or_else(||fail(FrameCause::Identity,c))?;
            if quote.value().rounds.is_empty() || selected.rounds()!=quote.value().rounds.len(){
                return Err(fail(FrameCause::Identity,c));
            }
            drop(selected);
            drop(source);
            let mut input=input;
            let mut completion=None;
            let mut first_claim=Some(first_claim);
            for (round,selected) in quote.value().rounds.iter().enumerate(){
                reserve(&c.funding,&[size_of::<Exchange>(),size_of::<RetainedPipelineBoundary>(),
                    size_of::<Option<ParallelControlClaim>>(),size_of::<Option<OriginalCommunicationCompletion>>(),
                    size_of::<(Array,OriginalCommunicationCompletion)>(),size_of::<usize>(),failure_bytes().ok_or_else(overflow)?])?;
                let claim=match first_claim.take(){
                    Some(claim)=>claim,
                    None=>owner.owner().request.cursor.try_borrow_mut().map_err(|_|fail(FrameCause::Identity,c))?
                        .claim(ParallelControlEvent::Phase(DistributedExecutionPhase::Execution))
                        .map_err(|_|fail(FrameCause::Identity,c))?,
                };
                let capacity=AgreementCapacity{graph:selected.graph_capacity(),records:selected.record_capacity(),backing:selected.backing_capacity()};
                // The prior round has completed; the actual forwarded Array
                // retains its storage. Each next round receives a fresh finite
                // child role rather than reopening a terminal scope.
                drop(completion.take());
                let invocation=Exchange{quote:quote.clone(),input,order,round,claim,capacity,
                    owner:OriginalParallelControlOwner(owner.0.clone()),custody:c.clone()};
                let (received,settled)=run_native_role(invocation,capacity,&owner.owner().native,c,
                    |value,observer|Ok(value.run(observer))).map_err(|cause|Error::with_original_control_source(cause,false))??;
                input=received;completion=Some(settled);
            }
            Ok((input,completion.ok_or_else(||fail(FrameCause::Identity,c))?))
        })();
        if result.is_err(){owner.owner().failed.set(true);}
        result
    }
}
impl Exchange {
    fn run(&self,observer:&OriginalScopeObserver)->Result<(Array,OriginalCommunicationCompletion),Error>{
        let c=&self.custody;let owner=self.owner.owner();
        reserve(&c.funding,&[size_of::<Self>(),size_of::<[Array;2]>(),
            size_of::<(Array,OriginalCommunicationCompletion)>(),size_of::<Result<(Array,OriginalCommunicationCompletion),Error>>(),
            safemlx::OperationEvent::traversal_leaf_control_bytes().and_then(|n|n.checked_mul(3)).ok_or_else(overflow)?,
            failure_bytes().ok_or_else(overflow)?])?;
        let source=owner.request.source.communication_source()?;
        let route=source.framed_route_exchange(self.order)?.ok_or_else(||fail(FrameCause::Identity,c))?;
        let id=route.route().descriptor().id();
        let phase=DistributedExecutionPhase::Execution;
        if self.claim.identity()!=owner.request.cursor.borrow().identity()
            || self.claim.event()!=ParallelControlEvent::Phase(phase)
            || self.quote.value().route!=id || route.rounds()!=self.quote.value().rounds.len() {
            return Err(fail(FrameCause::Identity,c));
        }
        let runtime=owner.request.source.agreement_inputs().ok_or_else(||fail(FrameCause::Identity,c))?.runtime();
        let stream=route.route().group().and_then(Group::retained_transport_stream)
            .ok_or_else(||fail(FrameCause::Identity,c))?;
        let selected=self.quote.value().rounds.get(self.round).ok_or_else(||fail(FrameCause::Identity,c))?;
        let round=selected.bind_actual(&source,&self.input)?.prepare(&source,runtime)?;
        if !self.capacity.covers(AgreementCapacity{graph:round.graph_capacity(),records:round.record_capacity(),backing:round.backing_capacity()}){
            return Err(fail(FrameCause::Identity,c));
        }
        let (outputs,completion)=round.construct_accepted(&source,observer,stream)?.submit()?;
        let completion=wait_retaining(&source,completion,phase,id,c)?;
        for output in outputs.outputs(){
            safemlx::OperationEvent::validate_traversal_leaf(output,observer).map_err(|cause|fail(cause.into(),c))?;
        }
        let([sent,received],source,funding)=outputs.into_parts();
        drop((sent,source,funding));
        Ok((received,completion))
    }
}
