//! CPU sources for the existing checked axis-zero take and additive row scatter.
//! These are tensor operations; no expert/family identity qualifies their inputs.
use super::*;
use safemlx::Dtype;

pub(super) fn inspect(operation:WorkspaceOperationView<'_>,mechanism:MlxCpuWorkspaceMechanisms)
    ->facts::FactResult<Option<OperationPlan>> {
    match operation.kind {
        WorkspaceOperationKindView::Gather{axis:0}=>gather(operation,mechanism),
        WorkspaceOperationKindView::IndexedRowAdd=>add(operation,mechanism),
        _=>Ok(None),
    }
}

fn floating(layout:WorkspaceLayoutView<'_>)->Option<(WorkspaceFloatingType,Dtype)> {
    if layout.dtype()!=WorkspaceDtype::Float32{return None;}
    let dtype=layout.representation()?.dtype();
    Some((dtype,match dtype {
        WorkspaceFloatingType::Float32=>Dtype::Float32,
        WorkspaceFloatingType::Float16=>Dtype::Float16,
        WorkspaceFloatingType::Bfloat16=>Dtype::Bfloat16,
    }))
}

fn gather(operation:WorkspaceOperationView<'_>,mechanism:MlxCpuWorkspaceMechanisms)
    ->facts::FactResult<Option<OperationPlan>> {
    let (Some([input,indices]),Some([output]))=(operation.inputs.array(),operation.outputs.array())
        else{return Err(MlxWorkspaceFactError::descriptor("CPU take-axis population differs"));};
    let rank=input.shape().len();
    if !(1..=2).contains(&rank)||indices.shape().len()!=1||indices.dtype()!=WorkspaceDtype::Int32
        ||input.shape().iter().any(|&n|n<0)||indices.shape()[0]<0
        ||(rank==2&&input.shape()[1]<=0){return Ok(None);}
    if output.dtype()!=input.dtype()||!output.shape().iter().eq(indices.shape().iter().chain(&input.shape()[1..])) {
        return Err(MlxWorkspaceFactError::descriptor("CPU take-axis output geometry differs"));
    }
    let Some((dtype,native))=floating(input)else{return Ok(None)};
    let count=usize::try_from(indices.elements()?)?;
    let input_elements=usize::try_from(input.elements()?)?;
    let output_elements=usize::try_from(output.elements()?)?;
    if [count,input_elements,output_elements].into_iter().any(|n|n>i32::MAX as usize)
        ||(count!=0&&input.shape()[0]==0){return Ok(None);}
    let width=if rank==1{1}else{input.shape()[1] as usize};
    let mut population=CpuPopulation::default();
    let mut scratch_bytes=0;
    let (seeds,validations)=if count==0{(0,0)}else{
        let Some(prefix)=embedding::checked_take_indices(1,count,mechanism)?else{return Ok(None)};
        population.add(prefix.population).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
        scratch_bytes=facts::add(prefix.output_bytes,prefix.scratch_bytes)?;
        (prefix.seeds,prefix.validations)
    };
    // gather_owned preserves the one I32 index's dtype. Empty validation and
    // identity AsType retain handles, but construct no native copy primitive.
    let Some(source)=OperationEvent::cpu_gather_layout(native,Dtype::Int32,rank,1,
        input_elements,count,width,false)else{return Ok(None)};
    if source.backing_births()!=usize::from(output_elements!=0)
        ||population.copy(source,2).is_none(){return Ok(None);}
    let Some(squeeze)=OperationEvent::cpu_squeeze_layout(rank+1,false)else{return Ok(None)};
    if squeeze.backing_births()!=0||population.copy(squeeze,1).is_none(){return Ok(None);}
    // The identity index AsType is a frontend constructor candidate only.
    population.construction_entries=population.construction_entries.checked_add(1)
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    population.maximum_captures=population.maximum_captures.max(4);
    let frames=[size_of::<(WorkspaceOperationView<'_>,MlxCpuWorkspaceMechanisms)>(),
        size_of::<[WorkspaceLayoutView<'_>;3]>(),size_of::<Option<[WorkspaceLayoutView<'_>;2]>>(),
        size_of::<Option<[WorkspaceLayoutView<'_>;1]>>(),size_of::<(WorkspaceFloatingType,Dtype)>(),
        size_of::<Option<(WorkspaceFloatingType,Dtype)>>(),size_of::<[usize;5]>(),size_of::<u64>(),
        size_of::<(usize,usize)>(),size_of::<CpuPopulation>(),size_of::<OperationPlan>(),
        size_of::<Option<OperationPlan>>(),size_of::<facts::FactResult<Option<OperationPlan>>>(),
        size_of::<CpuCopyEvalLayout>()*2,size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<std::slice::Iter<'_,i32>>(),size_of::<std::iter::Chain<std::slice::Iter<'_,i32>,std::slice::Iter<'_,i32>>>(),
        size_of::<std::array::IntoIter<usize,3>>(),size_of::<Result<usize,std::num::TryFromIntError>>(),
        size_of::<(&crate::MlxTensor,&crate::MlxTensor,i32,&safemlx::Stream)>(),
        size_of::<Result<crate::MlxTensor,Error>>(),size_of::<safemlx::Array>(),
        size_of::<Result<safemlx::Array,safemlx::error::Exception>>(),size_of::<i32>()*2,
        size_of::<Result<i32,std::num::TryFromIntError>>(),size_of::<(&safemlx::Array,&safemlx::Array,i32,&safemlx::Stream)>(),
        size_of::<(&safemlx::Array,i32,&safemlx::Stream)>(),size_of::<bool>(),
        crate::backend::nn::expert_movement::control_bytes::<crate::backend::nn::expert_movement::Native<'_>>()
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?];
    population.controls=frames.into_iter().try_fold(population.controls.checked_add(size_of_val(&frames))
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,usize::checked_add)
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    Ok(Some(OperationPlan{dtype,population,alias_input:None,
        output_bytes:mechanism.allocation.fixed_buffer_capacity(output.bytes()?)?,scratch_bytes,
        rank:rank+1,parameter_shells:usize::from(count==0),seeds,validations}))
}

fn add(operation:WorkspaceOperationView<'_>,mechanism:MlxCpuWorkspaceMechanisms)
    ->facts::FactResult<Option<OperationPlan>> {
    if !super::super::basic::indexed_row_add(operation){
        return Err(MlxWorkspaceFactError::descriptor("CPU additive rows geometry differs"));
    }
    let base=operation.inputs.get(0).expect("validated base");
    let indices=operation.inputs.get(1).expect("validated indices");
    let updates=operation.inputs.get(2).expect("validated updates");
    let output=operation.outputs.get(0).expect("validated output");
    if base.shape().iter().chain(updates.shape()).any(|&n|n<0)||base.shape()[1]<=0
        ||floating(base)!=Some((WorkspaceFloatingType::Float32,Dtype::Float32))
        ||floating(updates)!=Some((WorkspaceFloatingType::Float32,Dtype::Float32)){return Ok(None);}
    let index=match indices.dtype(){WorkspaceDtype::Int32=>Dtype::Int32,
        WorkspaceDtype::Uint32=>Dtype::Uint32,_=>return Ok(None)};
    let output_elements=usize::try_from(output.elements()?)?;
    let update_elements=usize::try_from(updates.elements()?)?;
    if [output_elements,update_elements].into_iter().any(|n|n>i32::MAX as usize){return Ok(None);}
    let mut population=CpuPopulation::default();
    if output_elements!=0 {
        // All update values already have the base dtype. Only the [K,1]
        // index requires a changed-shape Broadcast when the row width is >1.
        // The three ignoring-axis broadcasts preserve these exact shapes.
        if base.shape()[1]!=1 {
            let broadcast=if update_elements==0 {OperationEvent::cpu_empty_broadcast_layout(2,2,false)}
                else {OperationEvent::cpu_broadcast_alias_layout(2,2,false)};
            let Some(broadcast)=broadcast else{return Ok(None)};
            if broadcast.backing_births()!=0||population.copy(broadcast,1).is_none(){return Ok(None);}
        }
        let Some(scatter)=OperationEvent::cpu_scatter_add_rows_layout(index,output_elements,update_elements,false)
            else{return Ok(None)};
        if scatter.backing_births()!=1||population.copy(scatter,3).is_none(){return Ok(None);}
        population.maximum_captures=6;
        // One AsType, two element broadcasts, three ignoring-axis
        // broadcasts and Scatter share the canonical frontend graph bank.
        // Only changed-shape Broadcast and Scatter enter the actual Eval tape.
        population.construction_entries=7;
    }
    let frames=[size_of::<(WorkspaceOperationView<'_>,MlxCpuWorkspaceMechanisms)>(),
        size_of::<[WorkspaceLayoutView<'_>;4]>(),size_of::<Dtype>(),size_of::<usize>()*2,
        size_of::<CpuPopulation>(),size_of::<Option<CpuCopyEvalLayout>>(),size_of::<CpuCopyEvalLayout>(),
        size_of::<OperationPlan>(),size_of::<facts::FactResult<Option<OperationPlan>>>(),
        size_of::<std::iter::Chain<std::slice::Iter<'_,i32>,std::slice::Iter<'_,i32>>>(),
        size_of::<std::array::IntoIter<usize,2>>(),size_of::<Result<usize,std::num::TryFromIntError>>(),
        size_of::<(&safemlx::Array,&safemlx::Array,&safemlx::Array,i32,&safemlx::Stream)>(),
        size_of::<Result<safemlx::Array,safemlx::error::Exception>>(),
        crate::backend::nn::expert_movement::control_bytes::<crate::backend::nn::expert_movement::Native<'_>>()
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?];
    population.controls=frames.into_iter().try_fold(population.controls.checked_add(size_of_val(&frames))
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,usize::checked_add)
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    Ok(Some(OperationPlan{dtype:WorkspaceFloatingType::Float32,population,
        alias_input:(output_elements==0).then_some(0),
        output_bytes:if output_elements==0{0}else{mechanism.allocation.fixed_buffer_capacity(output.bytes()?)?},
        scratch_bytes:0,rank:2,parameter_shells:usize::from(output_elements==0),seeds:0,validations:0}))
}

#[cfg(all(test,target_vendor="apple",feature="metal",not(feature="cuda")))]
mod tests;
