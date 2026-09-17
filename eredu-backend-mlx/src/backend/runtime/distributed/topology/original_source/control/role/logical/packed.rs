//! Consume each selected pack/world/extraction stage through existing child roles.
use super::*;
pub(super) fn execute_finish<T,F>(owner:&OriginalParallelControlOwner,quote:RetainedLogicalCollective,
    input:&Array,stream:&Stream,first:ParallelControlClaim,c:&Custody,world_observer:Option<WorldObserver<'_>>,finish:F)->Result<T,Error>
where F:FnOnce(Array,&OriginalScopeObserver)->Result<T,Error>{
    reserve(&c.funding,&[size_of::<Option<WorldObserver<'_>>>(),size_of::<F>(),size_of::<T>(),size_of::<Result<T,Error>>(),size_of::<World>(),size_of::<Result<Array,Error>>(),
        size_of::<[Array;2]>(),size_of::<Array>(),size_of::<ParallelControlClaim>(),
        size_of::<(&OriginalParallelControlOwner,&RetainedLogicalCollective,&Array,&Stream,&Custody)>(),
        failure_bytes().ok_or_else(overflow)?])?;
    let selected=quote.value().packed().ok_or_else(||fail(LogicalCause::Identity,c))?;
    let packed=run_arithmetic_claim(owner,quote.clone(),ArithmeticStage::Pack,
        [retained_array(input,c)?,retained_array(input,c)?],stream,first,c)?;
    let capacity=AgreementCapacity{graph:selected.world.graph_capacity(),records:selected.world.record_capacity(),backing:selected.world.backing_capacity()};
    let world=World{input:packed,quote:quote.clone(),claim:claim(owner.owner(),c)?,
        owner:OriginalParallelControlOwner(owner.0.clone()),custody:c.clone()};
    let output=run_native_role(world,capacity,&owner.owner().bank,&owner.owner().controls,c,
        |world,observer|Ok(world.run(observer,world_observer)))
        .map_err(|cause|Error::with_original_control_source(cause,false))??;
    // Unary stages use the existing pair-worker frame with two paid aliases.
    let alias=retained_array(&output,c)?;
    run_arithmetic_finish(owner,quote.clone(),ArithmeticStage::Unpack,
        [output,alias],stream,c,|output,observer|{
    let expected=quote.value();
    let shape=match expected.kind {
        LogicalCollectiveKind::Sum=>output.shape()==input.shape(),
        LogicalCollectiveKind::Gather if input.shape().is_empty()=>output.shape()==[i32::try_from(selected.members).map_err(|_|overflow())?],
        LogicalCollectiveKind::Gather=>output.shape().len()==input.shape().len()
            && input.shape()[0].checked_mul(i32::try_from(selected.members).map_err(|_|overflow())?)==output.shape().first().copied()
            && output.shape().iter().skip(1).eq(input.shape().iter().skip(1)),
    };
    if !shape || output.dtype()!=expected.native_dtype{return Err(fail(LogicalCause::Identity,c));}
    finish(output,observer)
    })
}
struct World {
    input:Array,
    quote:RetainedLogicalCollective,
    claim:ParallelControlClaim,
    owner:OriginalParallelControlOwner,
    custody:Custody,
}
impl World {
    fn run(&self,observer:&OriginalScopeObserver,world_observer:Option<WorldObserver<'_>>)->Result<Array,Error>{
        let c=&self.custody;
        reserve(&c.funding,&[size_of::<Self>(),size_of::<OriginalCommunicationCompletion>(),
            size_of::<Result<Array,Error>>(),safemlx::OperationEvent::traversal_leaf_control_bytes().ok_or_else(overflow)?,
            failure_bytes().ok_or_else(overflow)?])?;
        validate_claim(&self.claim,self.owner.owner(),c)?;
        let source=self.owner.owner().request.source.communication_source()?;
        let selected=self.quote.value().packed().ok_or_else(||fail(LogicalCause::Identity,c))?;
        let operation=selected.world.bind_actual(&source,&self.input)?;
        let prepared=operation.prepare_resources(&source,None)?;
        let stream=source.world().retained_transport_stream().ok_or_else(||fail(LogicalCause::Identity,c))?;
        let (value,completion)=operation.construct_accepted(&source,observer,stream)?.submit(prepared)?;
        let value=wait(&source,value,completion,self.quote.value().protocol,c)?;
        safemlx::OperationEvent::validate_traversal_leaf(value.value(),observer).map_err(|cause|fail(cause.into(),c))?;
        if let Some(observe) = world_observer { observe(value.value(),observer)?; }
        // Retain a paid descriptor before the accepted constructor wrapper retires.
        retained_array(value.value(),c)
    }
}
