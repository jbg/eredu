//! Exact selected local-route stages, with descriptive native pair envelopes.
use super::*;
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub(crate) enum ExpertLocalStage { Slice{value:usize},Count{value:usize,step:usize},Data{value:usize,step:usize},Receive{value:usize,step:usize},Join }
#[derive(Clone,Copy,Debug)]
pub(crate) struct ExpertLocalStageBound {
    pub(crate) kind:ExpertLocalStage,
    pub(crate) numerical:Option<ExpertReorderEnvelope>,
    capacity:BoundaryStageCapacity,
    births:usize,
    rows:usize,
    width:i32,
    dtype:safemlx::Dtype,
}
impl ExpertLocalStageBound {
    pub(crate) fn covers_pair(self,send:&[i32],receive:&[i32],dtype:safemlx::Dtype,
        graph:usize,records:usize,backing:usize,births:usize)->bool{
        let shapes=match self.kind{
            ExpertLocalStage::Count{..}=>send==[1]&&receive==[1]&&dtype==safemlx::Dtype::Int32,
            ExpertLocalStage::Data{..}=>[send,receive].into_iter().all(|shape|shape.len()==2&&shape[1]==self.width
                &&usize::try_from(shape[0]).is_ok_and(|rows|rows<=self.rows)),
            _=>false,
        };
        self.numerical.is_none()&&shapes&&dtype==self.dtype&&graph<=self.capacity.graph
            &&records<=self.capacity.records&&backing<=self.capacity.backing&&births<=self.births
    }
}
pub(crate) struct ExpertLocalTransport {
    stages:Vec<ExpertLocalStageBound>,
    pub(crate) capacity:BoundaryStageCapacity,
    pub(crate) births:usize,
}
impl ExpertLocalTransport {
    pub(super) fn prepare(source:&OriginalParallelSource,mechanism:ResidentExecutionMechanisms,order:usize,width:i32,dtype:safemlx::Dtype,
        send_rows:usize,receive_rows:usize)->Result<Self,Error>{
        let context=WorkspaceContext::new_with_metadata_funding(mechanism,source.funding().clone())?;
        context.charge_metadata(size_of::<(Self,Result<Self,Error>,ExpertLocalStageBound,WorkspaceContext,
            [usize;3],[i32;2],[i32;2])>())?;
        let invalid=||context.metadata_error(format_args!("expert local transport has no exact route source"));
        let actual=source.communication_source().map_err(|cause|source.neural_error(cause))?;
        let (group,descriptor,_)=actual.group(order).ok_or_else(invalid)?;
        let plan=group.logical_variable_route_plan().map_err(|_|invalid())?.ok_or_else(invalid)?;
        if !std::ptr::eq(plan.group(),group)||plan.len()!=descriptor.members().len()
            ||descriptor.local_index()!=Some(group.rank())||width<=0{return Err(invalid());}
        let count=plan.values().try_fold(plan.len().checked_add(1).ok_or_else(invalid)?,|count,route|{
            let hops=(0..route.steps()).filter(|&step|route.exchange(step).is_some()).count();
            count.checked_add(hops.checked_mul(3)? )
        }).ok_or_else(invalid)?;
        let mut stages=context.metadata_vec(count)?;
        let mut capacity=BoundaryStageCapacity{graph:0,records:0,backing:0};let mut births=0usize;
        let mut push=|stage:ExpertLocalStageBound|->Result<(),Error>{
            if stages.len()==count{return Err(invalid());}
            capacity.graph=capacity.graph.max(stage.capacity.graph);
            capacity.records=capacity.records.max(stage.capacity.records);
            capacity.backing=capacity.backing.checked_add(stage.capacity.backing).ok_or_else(invalid)?;
            births=births.checked_add(stage.births).ok_or_else(invalid)?;
            stages.push(stage);Ok(())
        };
        context.charge_metadata(std::mem::size_of_val(&push))?;
        // A single matrix edge is bounded by both its global sender and receiver.
        // The pair worker branches only on zero versus positive extent; include
        // singleton and upper geometry independently in both directions.
        let edge_rows=send_rows.min(receive_rows);
        for (value,route) in plan.values().enumerate(){
            let slice=ExpertReorderEnvelope::prepare(source,mechanism,send_rows,width,dtype,plan.len())?;
            push(ExpertLocalStageBound{kind:ExpertLocalStage::Slice{value},numerical:Some(slice),
                capacity:slice.capacity,births:slice.births,rows:send_rows,width,dtype})?;
            for step in 0..route.steps(){
                if route.exchange(step).is_none(){continue;}
                for count in [true,false]{
                    let kind=if count{ExpertLocalStage::Count{value,step}}else{ExpertLocalStage::Data{value,step}};
                    let mut bound=ExpertLocalStageBound{kind,numerical:None,
                        capacity:BoundaryStageCapacity{graph:0,records:0,backing:0},births:0,
                        rows:edge_rows,width,dtype:if count{safemlx::Dtype::Int32}else{dtype}};
                    let candidates=if count{[1,1,1]}else{[0,edge_rows.min(1),edge_rows]};
                    for (a,&sent) in candidates.iter().enumerate(){
                        if candidates[..a].contains(&sent){continue;}
                        for (b,&received) in candidates.iter().enumerate(){
                            if candidates[..b].contains(&received){continue;}
                            let send=[i32::try_from(sent).map_err(|_|invalid())?,width];
                            let receive=[i32::try_from(received).map_err(|_|invalid())?,width];
                            let (send,receive)=if count{(&send[..1],&receive[..1])}else{(&send[..],&receive[..])};
                            let queried=actual.group_variable_route_layout(order,value,step,send,receive,bound.dtype)
                                .map_err(|cause|source.neural_error(cause))?.ok_or_else(invalid)?;
                            let owned=queried.try_into_owned(source).map_err(|cause|source.neural_error(cause))?;
                            bound.capacity.graph=bound.capacity.graph.max(owned.graph_capacity());
                            bound.capacity.records=bound.capacity.records.max(owned.record_capacity());
                            bound.capacity.backing=bound.capacity.backing.max(owned.backing_capacity());
                            bound.births=bound.births.max(owned.maximum_backing_births().ok_or_else(invalid)?);
                        }
                    }
                    push(bound)?;
                    if count{
                        let empty=ExpertReorderEnvelope::prepare_empty(source,mechanism,edge_rows,width,dtype)?;
                        push(ExpertLocalStageBound{kind:ExpertLocalStage::Receive{value,step},numerical:Some(empty),
                            capacity:empty.capacity,births:empty.births,rows:edge_rows,width,dtype})?;
                    }
                }
            }
        }
        let join=ExpertReorderEnvelope::prepare_join(source,mechanism,receive_rows,width,dtype,plan.len())?;
        push(ExpertLocalStageBound{kind:ExpertLocalStage::Join,numerical:Some(join),capacity:join.capacity,
            births:join.births,rows:receive_rows,width,dtype})?;
        drop(push);
        if stages.len()!=count{return Err(invalid());}
        Ok(Self{stages,capacity,births})
    }
    pub(crate) fn len(&self)->usize{self.stages.len()}
    pub(crate) fn stage(&self,index:usize)->Option<ExpertLocalStageBound>{self.stages.get(index).copied()}
}
