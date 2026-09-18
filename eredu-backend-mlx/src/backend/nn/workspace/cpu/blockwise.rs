//! Source-visible CPU execution of the shared paged attention recurrence.
use super::*;
use safemlx::Dtype;
use crate::backend::nn::workspace::attention::blockwise::descriptor::{self,Descriptor,Stage};
mod source;
#[cfg(test)]mod tests;

pub(super) fn inspect(operation:WorkspaceOperationView<'_>,mechanism:MlxCpuWorkspaceMechanisms)
    ->facts::FactResult<Option<OperationPlan>> {
    let Some(Descriptor{policy,stage})=descriptor::decode(operation)? else{return Ok(None)};
    if policy.output_type!=Some(WorkspaceFloatingType::Float32){return Ok(None);}
    // The retained physical scalar, rather than the neutral four-byte class,
    // selects this exact F32 CPU worker. Boolean mask views are separate inputs.
    for value in operation.inputs.iter() {
        if value.dtype()==WorkspaceDtype::Float32&&!value.representation().is_some_and(|r|
            r.dtype()==WorkspaceFloatingType::Float32&&r.last_axis_contiguous()) {return Ok(None);}
    }
    if let Stage::Accumulate{..}=stage {
        // Paged copies carry dense K/V rows. Replication below consumes that
        // exact source; an unknown strided cache never borrows these facts.
        if [1,2].into_iter().any(|i|!operation.inputs.get(i).is_some_and(|v|
            v.representation().is_some_and(|r|r.row_contiguous()))) {return Ok(None);}
    }
    let mut query_copies=0;
    if matches!(stage,Stage::Accumulate{..})
        &&policy.options.arithmetic==eredu_nn::AttentionArithmetic::InputScores {
        let query=operation.inputs.get(0).expect("validated query");
        let representation=query.representation().expect("validated floating source");
        let Some(strides)=super::views::physical_strides(query,representation) else{return Ok(None)};
        // Same two matrix interiors as the unchanged CPU check_transpose worker.
        // Batch/head transposes may preserve the final stride while requiring
        // an actual compact query copy for its first InputScores product.
        // Layout facts intentionally canonicalize unit-axis strides. Native
        // check_transpose still reads those raw strides, so a singleton matrix
        // axis permits one real query copy even when its logical layout is dense.
        query_copies=usize::from(query.shape()[2]==1||query.shape()[3]==1
            ||!((strides[2]==i64::from(query.shape()[3])&&strides[3]==1)
                ||(strides[2]==1&&strides[3]==i64::from(query.shape()[2]))));
    }
    let source=match stage {
        Stage::Begin{q,mask,..}=>source::begin(q,mask),
        Stage::Accumulate{..}=>source::accumulate(mechanism,policy,stage,query_copies),
        Stage::Finish{shape,input_scores}=>source::finish(shape,input_scores),
    };
    let Some(mut source)=source else{return Ok(None)};
    let alias=match stage {Stage::Begin{..}|Stage::Finish{input_scores:true,..}=>true,_=>false};
    let mut output_bytes=0;
    if !alias {for output in operation.outputs.iter() {
        output_bytes=facts::add(output_bytes,mechanism.allocation.fixed_buffer_capacity(output.bytes()?)?)?;
    }}
    let maximum=mechanism.allocation.fixed_buffer_capacity(facts::mul(u64::try_from(source.maximum)?,4)?)?;
    let total=facts::add(facts::mul(maximum,u64::try_from(source.native.births)?)?,
        facts::mul(mechanism.allocation.fixed_buffer_capacity(4)?,u64::try_from(source.seeds)?)?)?;
    let scratch_bytes=total.checked_sub(output_bytes).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    let shells=match stage {Stage::Begin{sink,..}=>usize::from(sink.is_some()),
        Stage::Accumulate{q,k,..}=>2*usize::from(q[1]==k[1]),_=>0};
    let shared=super::super::resident_recipe::blockwise_control_bytes(operation)
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    let frames=[shared,super::views::physical_stride_control_bytes().ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,size_of::<source::Source>()*3,size_of::<Option<source::Source>>(),
        size_of::<OperationPlan>(),size_of::<Option<OperationPlan>>(),
        size_of::<WorkspaceOperationView<'_>>(),size_of::<WorkspaceLayoutView<'_>>()*4,
        size_of::<Descriptor<'_>>(),size_of::<Stage<'_>>(),size_of::<MlxCpuWorkspaceMechanisms>(),
        size_of::<eredu_nn::workspace::WorkspaceBlockwisePolicy>(),
        size_of::<eredu_nn::operation_geometry::AbsoluteAttentionMaskGeometry>(),
        size_of::<(WorkspaceOperationView<'_>,MlxCpuWorkspaceMechanisms)>(),
        size_of::<(MlxCpuWorkspaceMechanisms,eredu_nn::workspace::WorkspaceBlockwisePolicy,Stage<'_>,usize)>(),
        size_of::<(&mut source::Source,CpuBinaryOperation,Dtype,usize,usize,(usize,usize),(usize,usize))>(),
        size_of::<(&mut source::Source,CpuBinaryOperation,usize,usize,usize)>(),
        size_of::<(&mut source::Source,safemlx::CpuUnaryOperation,Dtype,usize,usize)>(),
        size_of::<(&mut source::Source,Dtype,Dtype,usize,usize)>(),
        size_of::<(&mut source::Source,usize,usize,(usize,usize),(usize,usize),(usize,usize))>(),
        size_of::<(&mut source::Source,MlxCpuWorkspaceMechanisms,usize,usize,usize,usize,usize)>(),
        size_of::<(&mut CpuPopulation,MlxCpuWorkspaceMechanisms,Dtype,usize,usize,usize,usize,usize,usize)>(),
        size_of::<(&mut source::Source,usize,usize,usize,usize,usize)>(),
        size_of::<(&mut source::Source,CpuCopyEvalLayout,usize,usize)>(),
        size_of::<(&mut source::Source,bool,usize,usize)>(),
        size_of::<(&mut source::Source,eredu_nn::operation_geometry::AbsoluteAttentionMaskGeometry)>(),
        size_of::<CpuCopyEvalLayout>()*2,size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<safemlx::CpuBinaryEvalLayout>(),size_of::<Option<safemlx::CpuBinaryEvalLayout>>(),
        size_of::<safemlx::CpuUnaryEvalLayout>(),size_of::<Option<safemlx::CpuUnaryEvalLayout>>(),
        size_of::<[i64;4]>(),size_of::<WorkspaceRepresentation>(),size_of::<Option<WorkspaceRepresentation>>(),
        size_of::<[i32;5]>(),size_of::<[i64;5]>(),size_of::<[i32;4]>()*4,
        size_of::<[(usize,usize);3]>(),size_of::<std::array::IntoIter<(usize,usize),3>>(),
        size_of::<usize>()*24,size_of::<u64>()*4,size_of::<bool>()*4,
        size_of::<Option<()>>(),size_of::<Option<usize>>(),
        size_of::<std::ops::Range<usize>>(),size_of::<std::slice::Iter<i32>>(),
        size_of::<eredu_nn::workspace::WorkspaceLayoutIter<'_>>()];
    source.native.controls=frames.into_iter().try_fold(source.native.controls.checked_add(size_of_val(&frames))
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,|n,b|n.checked_add(b).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW))?;
    let rank=match stage {Stage::Accumulate{q,k,..} if q[1]!=k[1]=>5,_=>4};
    Ok(Some(OperationPlan{dtype:WorkspaceFloatingType::Float32,population:source.native,
        alias_input:alias.then_some(0),output_bytes,scratch_bytes,rank,parameter_shells:shells,
        seeds:source.seeds,validations:0}))
}

pub(super) fn emit_outputs(operation:WorkspaceOperationView<'_>,mechanism:MlxCpuWorkspaceMechanisms,
    sink:&mut facts::Emitter<'_>)->facts::FactResult<()> {
    let Descriptor{stage,..}=descriptor::decode(operation)?.expect("selected blockwise operation");
    match stage {
        Stage::Begin{mask,..}=>{
            sink.output(facts::Output::AliasInput(0))?;
            if mask.is_some(){sink.output(facts::Output::AliasInput(1))?;}
        }
        Stage::Finish{input_scores:true,..}=>sink.output(facts::Output::AliasInput(0))?,
        _=>for output in operation.outputs.iter(){sink.output(facts::Output::Allocate(
            mechanism.allocation.fixed_buffer_capacity(output.bytes()?)?))?;},
    }
    Ok(())
}
pub(super) fn representation(operation:WorkspaceOperationView<'_>,output:usize)->Option<WorkspaceRepresentation> {
    let Descriptor{stage,..}=descriptor::decode(operation).ok()??;
    let result=operation.outputs.get(output)?;
    if result.dtype()!=WorkspaceDtype::Float32{return None;}
    match stage {
        Stage::Begin{..} if output==0=>operation.inputs.get(0)?.representation(),
        Stage::Begin{..}=>operation.inputs.get(1)?.representation().map(|r|
            WorkspaceRepresentation::new(r.dtype(),false).with_last_axis_contiguous(r.last_axis_contiguous())),
        Stage::Finish{input_scores:true,..}=>operation.inputs.get(0)?.representation(),
        _=>Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true)),
    }
}
pub(in crate::backend::nn::workspace) fn nested_completions(operation:WorkspaceOperationView<'_>)->usize {
    usize::from(matches!(operation.kind,WorkspaceOperationKindView::BlockwiseAttention{
        stage:eredu_nn::workspace::WorkspaceBlockwiseStage::Accumulate{..},..}))
}
