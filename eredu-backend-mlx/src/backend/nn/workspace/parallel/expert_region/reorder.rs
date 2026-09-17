//! Finite branch census of the shared leading-block permutation/join worker.
use super::*;
use crate::backend::nn::logical_collective::{self,blocks};
use crate::backend::nn::workspace::byte_view::Dtype;
use eredu_nn::workspace::WorkspaceRepresentation;
#[derive(Clone,Copy,Debug)]
pub(crate) struct ExpertReorderEnvelope {
    maximum_rows:usize,
    width:i32,
    dtype:safemlx::Dtype,
    members:usize,
    join:bool,
    empty:bool,
    pub(crate) capacity:BoundaryStageCapacity,
    pub(crate) births:usize,
    controls:u64,
    kernels:usize,
}
impl ExpertReorderEnvelope {
    pub(super) fn prepare(source:&OriginalParallelSource,mechanism:ResidentExecutionMechanisms,maximum_rows:usize,width:i32,dtype:safemlx::Dtype,members:usize)
        ->Result<Self,Error>{ Self::prepare_kind(source,mechanism,maximum_rows,width,dtype,members,false,false) }
    pub(super) fn prepare_join(source:&OriginalParallelSource,mechanism:ResidentExecutionMechanisms,maximum_rows:usize,width:i32,dtype:safemlx::Dtype,members:usize)
        ->Result<Self,Error>{ Self::prepare_kind(source,mechanism,maximum_rows,width,dtype,members,true,false) }
    pub(super) fn prepare_empty(source:&OriginalParallelSource,mechanism:ResidentExecutionMechanisms,
        maximum_rows:usize,width:i32,dtype:safemlx::Dtype)->Result<Self,Error>{
        Self::prepare_kind(source,mechanism,maximum_rows,width,dtype,1,false,true)
    }
    fn prepare_kind(source:&OriginalParallelSource,mechanism:ResidentExecutionMechanisms,maximum_rows:usize,width:i32,dtype:safemlx::Dtype,members:usize,join:bool,empty:bool)
        ->Result<Self,Error>{
        let context=WorkspaceContext::new_with_metadata_funding(mechanism,source.funding().clone())?;
        context.charge_metadata(size_of::<(Self,Result<Self,Error>,WorkspaceContext,WorkspaceTraceReport,
            SpeculativeNumericalRecipe,[usize;3],[usize;2],Vec<usize>,Vec<WorkspaceTensor>)>())?;
        let invalid=||context.metadata_error(format_args!("variable block worker has no complete finite source"));
        let scalar=Dtype::from_native(dtype).ok_or_else(invalid)?;
        if members==0||width<=0{return Err(invalid());}
        let mut result=Self{maximum_rows,width,dtype,members,join,empty,capacity:BoundaryStageCapacity{graph:0,records:0,backing:0},
            births:0,controls:0,kernels:0};
        // A positive interval retains a rank-two view. Concatenation prices its
        // fixed input arity and total rows componentwise; zero/alias branches
        // are distinct. A join includes all peers even if some have zero rows.
        let candidates=[0,maximum_rows.min(1),maximum_rows];
        for (index,&rows) in candidates.iter().enumerate(){
            if index>0 && candidates[..index].contains(&rows){continue;}
            let arities=if empty||rows==0{[0,0]}else{[1,members.min(rows)]};
            for (at,&arity) in arities.iter().enumerate(){
                if at>0&&arities[..at].contains(&arity){continue;}
                let mut counts=context.metadata_vec(members)?;counts.resize(members,0usize);
                if arity>0{
                    for count in &mut counts[..arity]{*count=1;}
                    counts[arity-1]=counts[arity-1].checked_add(rows-arity).ok_or_else(invalid)?;
                }
                // These layouts are guaranteed by the selected Gather/Scatter
                // or accepted pair output. They do not invent a count matrix.
                let existing=|rows:usize|->Result<WorkspaceTensor,Error>{
                    let shape=[i32::try_from(rows).map_err(|_|invalid())?,width];
                    let layout=context.layout(&shape,scalar.logical())?.with_representation(
                        scalar.floating().map(|dtype|WorkspaceRepresentation::new(dtype,true)));
                    WorkspaceTensor::existing(layout,&context).map_err(Into::into)
                };
                context.charge_metadata(std::mem::size_of_val(&existing))?;
                if empty{counts[0]=rows;}
                let output=if join{
                    let mut inputs=context.metadata_vec(members)?;
                    for &count in &counts{inputs.push(existing(count)?);}
                    context.begin_span();blocks::join(&logical_collective::Workspace(&context),&inputs)?
                }else{
                    let input=existing(rows)?;
                    context.begin_span();
                    if empty{blocks::concatenate(&logical_collective::Workspace(&context),&input,&counts,std::iter::empty())?}
                    else{blocks::concatenate(&logical_collective::Workspace(&context),&input,&counts,(0..members).rev())?}
                };
                let report=context.finish_report(&[output])?;
                let recipe=super::super::numerical(&report,1,mechanism,&context)?;
                let capacity=boundary::capacity(recipe,source.initialized_runtime(),&context)?;
                result.capacity.graph=result.capacity.graph.max(capacity.graph);
                result.capacity.records=result.capacity.records.max(capacity.records);
                result.capacity.backing=result.capacity.backing.max(capacity.backing);
                result.births=result.births.max(recipe.storage.maximum_births());
                result.controls=result.controls.max(recipe.controls);
                result.kernels=result.kernels.max(recipe.kernels);
            }
        }
        Ok(result)
    }
    fn resources(self,graph:usize,records:usize,backing:usize,recipe:&SpeculativeNumericalRecipe)->bool{
        graph<=self.capacity.graph&&records<=self.capacity.records&&backing<=self.capacity.backing
            &&recipe.storage.maximum_births()<=self.births&&recipe.controls<=self.controls&&recipe.kernels<=self.kernels
    }
    pub(crate) fn covers(self,input:&safemlx::Array,members:usize,graph:usize,records:usize,backing:usize,
        recipe:&SpeculativeNumericalRecipe)->bool{
        !self.join&&(!self.empty||self.members==1)&&input.dtype()==self.dtype&&input.shape().len()==2&&input.shape()[1]==self.width&&members==self.members
            &&usize::try_from(input.shape()[0]).is_ok_and(|rows|rows<=self.maximum_rows)
            &&self.resources(graph,records,backing,recipe)
    }
    pub(crate) fn covers_join(self,inputs:&[safemlx::Array],graph:usize,records:usize,backing:usize,
        recipe:&SpeculativeNumericalRecipe)->bool{
        if !self.join||inputs.len()!=self.members{return false;}
        let rows=inputs.iter().try_fold(0usize,|sum,input|{
            if input.dtype()!=self.dtype||input.shape().len()!=2||input.shape()[1]!=self.width{return None;}
            sum.checked_add(usize::try_from(input.shape()[0]).ok()?)
        });
        rows.is_some_and(|rows|rows<=self.maximum_rows)&&self.resources(graph,records,backing,recipe)
    }
}
