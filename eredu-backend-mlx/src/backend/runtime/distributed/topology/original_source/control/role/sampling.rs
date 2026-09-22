//! The actual selected sampling rank, with request-owned protocol scopes.
//! Immutable scalar sources are independent of the already closed model loan.
use super::*;
use eredu_runtime::{DistributedExecutionPhase,PartitionExecutionError};
use eredu_runtime::generation::SamplingSynchronizationPlan;
use safemlx::{Array,PreparedInputPlan,distributed::GroupWorkerOperation};

pub(super) mod token;

pub(crate) struct OriginalSamplingSource {
    attempt:u64,
    owner:OriginalParallelControlOwner,
}
impl std::fmt::Debug for OriginalSamplingSource {
    fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result {
        f.debug_struct("OriginalSamplingSource").field("attempt",&self.attempt).finish_non_exhaustive()
    }
}
pub(crate) struct BoundSamplingSource {
    order:usize,
    plan:SamplingSynchronizationPlan,
    next:Cell<u8>,
    source:OriginalSamplingSource,
}
impl OriginalParallelControlOwner {
    pub(crate) fn sampling_source(&self,step:&eredu_runtime::working_memory::InferenceTextStep)
        ->Result<OriginalSamplingSource,Error> {
        let owner=self.owner();
        reserve(&owner.custody.funding,&[size_of::<OriginalSamplingSource>(),
            size_of::<Result<OriginalSamplingSource,Error>>(),size_of::<Self>(),
            size_of::<(&Self,&eredu_runtime::working_memory::InferenceTextStep)>(),
            failure_control_bytes().ok_or_else(overflow)?])?;
        owner.native.text_execution()?.validate_same_request(step.request()).map_err(|cause|
            control_error(ControlCause::WorkingMemory(cause),&owner.custody.source,&owner.custody.funding))?;
        Ok(OriginalSamplingSource{attempt:step.attempt(),owner:Self(self.0.clone())})
    }
}
impl OriginalSamplingSource {
    pub(crate) fn bind(self,group:&Group,authority:&PartitionCommunicationAuthority,root:usize)
        ->Result<BoundSamplingSource,Error> {
        let owner=self.owner.owner();let custody=&owner.custody;
        reserve(&custody.funding,&[size_of::<BoundSamplingSource>(),
            size_of::<Result<BoundSamplingSource,Error>>(),size_of::<SamplingSynchronizationPlan>(),
            size_of::<Result<SamplingSynchronizationPlan,eredu_runtime::generation::SamplingSynchronizationError>>(),
            size_of::<Option<usize>>(),size_of::<(&Self,&Group,&PartitionCommunicationAuthority,usize)>(),
            CommunicationManifest::group_operation_control_bytes().ok_or_else(overflow)?,
            failure_control_bytes().ok_or_else(overflow)?])?;
        let source=owner.request.source.communication_source()?;
        source.validate()?;
        let order=source.source().manifest().groups().iter().enumerate()
            .find_map(|(order,_)|source.matches_group(order,group).then_some(order))
            .ok_or_else(||failure(Cause::SamplingIdentity,&custody.source,&custody.funding))?;
        let (_,descriptor,_)=source.group(order).ok_or_else(||failure(Cause::SamplingIdentity,&custody.source,&custody.funding))?;
        let selected=source.source().manifest().select_group_operation(descriptor.id(),CommunicationOperation::Broadcast)
            .map_err(|cause|failure(Cause::Rank(cause),&custody.source,&custody.funding))?;
        let plan=SamplingSynchronizationPlan::new(group.size(),group.rank(),root,1)
            .map_err(|cause|failure(Cause::SamplingPlan(cause),&custody.source,&custody.funding))?;
        if !source.authority.same_authority(authority) || selected.order()!=order
            || descriptor.local_index()!=Some(group.rank()) || !selected.requirement().exact_completion()
            || group.retained_transport_stream().is_none() || owner.failed.get() {
            return Err(failure(Cause::SamplingIdentity,&custody.source,&custody.funding));
        }
        drop(source);
        Ok(BoundSamplingSource{order,plan,next:Cell::new(0),source:self})
    }
}
impl BoundSamplingSource {
    fn owner(&self)->&Owner{self.source.owner.owner()}
    pub(crate) fn plan(&self)->SamplingSynchronizationPlan{self.plan}
    pub(crate) fn fund(&self,bytes:usize)->Result<(),Error>{
        reserve(&self.owner().custody.funding,&[bytes])
    }
    pub(crate) fn token_source(&self,token:u32)->Result<Array,Error>{
        let owner=self.owner();let c=&owner.custody;
        reserve(&c.funding,&[size_of::<[u32;1]>(),size_of::<[usize;1]>(),
            size_of::<PreparedInputPlan<'_>>(),size_of::<Result<PreparedInputPlan<'_>,safemlx::PreparedInputCause>>(),
            size_of::<Result<Array,Error>>(),failure_control_bytes().ok_or_else(overflow)?])?;
        let source=owner.request.source.communication_source()?;
        let inputs=owner.request.source.agreement_inputs().ok_or_else(||failure(Cause::SamplingIdentity,&c.source,&c.funding))?;
        let data=[token];let shape=[1];
        let plan=inputs.runtime().u32(&data,&shape).map_err(|cause|failure(Cause::Input(cause),&c.source,&c.funding))?;
        token::construct(owner, plan)

    }
    pub(crate) fn contribution(&self,value:f32,token:bool)->Result<Array,Error>{
        let owner=self.owner();let c=&owner.custody;
        reserve(&c.funding,&[size_of::<[f32;1]>(),size_of::<[usize;2]>(),size_of::<&[usize]>(),
            size_of::<PreparedInputPlan<'_>>(),size_of::<Result<PreparedInputPlan<'_>,safemlx::PreparedInputCause>>(),
            size_of::<Result<Array,Error>>(),failure_control_bytes().ok_or_else(overflow)?])?;
        if self.next.get()!=u8::from(!token) || (!self.plan.is_sampling_rank() && value!=0.0)
            || (!token && value!=0.0 && value!=1.0) {
            return Err(failure(Cause::SamplingIdentity,&c.source,&c.funding));
        }
        let source=owner.request.source.communication_source()?;
        let inputs=owner.request.source.agreement_inputs().ok_or_else(||failure(Cause::SamplingIdentity,&c.source,&c.funding))?;
        let data=[value];let token_shape=[1,1];let shape=if token{&token_shape[..]}else{&[]};
        let plan=inputs.runtime().f32(&data,shape).map_err(|cause|failure(Cause::Input(cause),&c.source,&c.funding))?;
        let [output]=super::super::super::inputs::construct(&source,[plan])?;
        Ok(output)
    }
    pub(crate) fn exchange(&self,input:Array,token:bool)->Result<f32,Error>{
        let owner=self.owner();let c=&owner.custody;
        // The actual event is attempted before source lookup or quota setup;
        // saved model/sampler state never contains or resets this cursor.
        let claim=owner.request.cursor.try_borrow_mut()
            .map_err(|_|Error::with_original_control_source(owner.request.fallback.retained().into_failure(),false))?
            .claim(ParallelControlEvent::Phase(DistributedExecutionPhase::SamplingSynchronization))
            .map_err(|_|Error::with_original_control_source(owner.request.fallback.retained().into_failure(),false))?;
        reserve(&c.funding,&[size_of::<Invocation>(),size_of::<ParallelControlClaim>(),
            size_of::<OriginalParallelControlOwner>(),size_of::<AgreementCapacity>(),
            size_of::<Result<AgreementCapacity,Error>>(),size_of::<Result<f32,Error>>(),
            size_of::<Result<Result<f32,Error>,eredu_core::BackendFailure>>(),
            failure_control_bytes().ok_or_else(overflow)?])?;
        if self.next.get()!=u8::from(!token) || owner.failed.get() || owner.running.replace(true) {
            owner.failed.set(true);
            return Err(failure(Cause::SamplingIdentity,&c.source,&c.funding));
        }
        self.next.set(self.next.get()+1);
        let _running=Running{running:&owner.running,failed:&owner.failed};
        let result=(||{
            let source=owner.request.source.communication_source()?;
            if input.dtype()!=safemlx::Dtype::Float32 || input.shape()!=if token{&[1,1][..]}else{&[][..]} {
                return Err(failure(Cause::SamplingIdentity,&c.source,&c.funding));
            }
            let capacity=capacity(&source,&input,self.order,owner)?;
            drop(source);
            let invocation=Invocation{input,claim,order:self.order,capacity,
                owner:OriginalParallelControlOwner(self.source.owner.0.clone())};
            run_native_role(invocation,capacity,&owner.native,c,|invocation,observer|{
                Ok(invocation.run(observer))
            }).map_err(|cause|Error::with_original_control_source(cause,false))?
        })();
        if result.is_err(){owner.failed.set(true);}
        result
    }
}
struct Invocation {
    input:Array,
    claim:ParallelControlClaim,
    order:usize,
    capacity:AgreementCapacity,
    owner:OriginalParallelControlOwner,
}
fn capacity(source:&OriginalCommunicationSource<'_>,input:&Array,order:usize,owner:&Owner)
    ->Result<AgreementCapacity,Error>{
    let c=&owner.custody;
    reserve(&c.funding,&[size_of::<AgreementCapacity>(),size_of::<Result<AgreementCapacity,Error>>(),
        size_of::<super::super::super::OriginalCommunicationCompletedOperation<'_>>(),
        CommunicationManifest::group_operation_control_bytes().ok_or_else(overflow)?,
        eredu_runtime::CommunicationOperationRequirement::tensor_metadata_control_bytes()
            .and_then(|n|n.checked_mul(2)).ok_or_else(overflow)?,failure_control_bytes().ok_or_else(overflow)?])?;
    let (_,descriptor,_)=source.group(order).ok_or_else(||failure(Cause::SamplingIdentity,&c.source,&c.funding))?;
    let selected=source.source().manifest().select_group_operation(descriptor.id(),CommunicationOperation::Broadcast)
        .map_err(|cause|failure(Cause::Rank(cause),&c.source,&c.funding))?;
    let req=selected.requirement();
    req.validate_tensor_metadata(&TensorDtype::F32,input.ndim(),Some(input.size()),false)
        .map_err(|cause|failure(Cause::Tensor(cause),&c.source,&c.funding))?;
    let operation=source.group_cpu_operation_storage(order,input,GroupWorkerOperation::Sum)?;
    let (rank,elements)=operation.native().constructor().output_geometry();
    req.validate_tensor_metadata(&TensorDtype::F32,rank,Some(elements),true)
        .map_err(|cause|failure(Cause::Tensor(cause),&c.source,&c.funding))?;
    let inputs=owner.request.source.agreement_inputs().ok_or_else(||failure(Cause::SamplingIdentity,&c.source,&c.funding))?;
    let backing=operation.backing_storage(inputs.runtime())?.capacity();
    let operation=operation.with_completion()?;
    Ok(AgreementCapacity{graph:operation.graph_capacity(),records:operation.record_capacity(),backing})
}
impl Invocation {
    fn run(&self,observer:&OriginalScopeObserver)->Result<f32,Error>{
        let owner=self.owner.owner();let c=&owner.custody;
        reserve(&c.funding,&[size_of::<Self>(),size_of::<PartitionExecutionError>(),
            size_of::<Result<f32,Error>>(),size_of::<Result<&[f32],safemlx::error::AsSliceError>>(),
            size_of::<Result<safemlx::EvaluatedArray<'_>,safemlx::error::Exception>>(),
            safemlx::EvaluatedArray::iteration_control_bytes::<f32>().ok_or_else(overflow)?,
            failure_control_bytes().and_then(|n|n.checked_mul(3)).ok_or_else(overflow)?])?;
        let source=owner.request.source.communication_source()?;
        if self.claim.identity()!=owner.request.cursor.borrow().identity()
            || self.claim.event()!=ParallelControlEvent::Phase(DistributedExecutionPhase::SamplingSynchronization)
            || !self.capacity.covers(capacity(&source,&self.input,self.order,owner)?) {
            return Err(failure(Cause::SamplingIdentity,&c.source,&c.funding));
        }
        let operation=source.group_cpu_operation_storage(self.order,&self.input,GroupWorkerOperation::Sum)?.with_completion()?;
        let ready=operation.prepare_resources(&source,Some(self.order))?;
        let group=source.group(self.order).ok_or_else(||failure(Cause::SamplingIdentity,&c.source,&c.funding))?.0;
        let stream=group.retained_transport_stream().ok_or_else(||failure(Cause::SamplingIdentity,&c.source,&c.funding))?;
        let phase=DistributedExecutionPhase::SamplingSynchronization;
        let submitted=operation.construct_accepted(&source,observer,stream).and_then(|accepted|accepted.submit(ready));
        let (output,completion)=submitted.map_err(|cause|{
            source.authority.fence_protocol_failure(CommunicationOperation::Broadcast,phase,None);
            cause
        })?;
        let output=source.authority.wait_with_error(eredu_core::Submission{output,completion},
            CommunicationOperation::Broadcast,phase,None,|cause|PartitionExecutionError::PreparedCommunication{
                operation:CommunicationOperation::Broadcast,phase,completion:true,source:failure(Cause::Native(cause),&c.source,&c.funding).into_backend_failure(),
            }).map_err(|cause|failure(Cause::SamplingCommunication(cause),&c.source,&c.funding))?;
        let evaluated=output.value().completed_in_original_scope(observer)
            .map_err(|cause|failure(Cause::Native(cause),&c.source,&c.funding))?;
        let values=evaluated.try_as_slice::<f32>().map_err(|cause|failure(Cause::SamplingData(cause),&c.source,&c.funding))?;
        if values.len()!=1 || !values[0].is_finite(){return Err(failure(Cause::SamplingWord,&c.source,&c.funding));}
        Ok(values[0])
    }
}
