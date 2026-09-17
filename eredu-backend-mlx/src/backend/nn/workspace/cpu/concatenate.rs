//! Same-type concatenation over the exact nonempty input population.
use super::*;
use safemlx::Dtype;
pub(super) fn inspect(operation:WorkspaceOperationView<'_>,mechanism:MlxCpuWorkspaceMechanisms)
    ->facts::FactResult<Option<OperationPlan>> {
    if !matches!(operation.kind,WorkspaceOperationKindView::Concatenate) {return Ok(None);}
    if operation.inputs.len()==1&&operation.outputs.len()==1 {
        let input=operation.inputs.get(0).expect("one concatenate source");
        let output=operation.outputs.get(0).expect("one concatenate result");
        if input.shape()!=output.shape()||input.dtype()!=output.dtype(){
            return Err(MlxWorkspaceFactError::descriptor("CPU singleton concatenate changes its source"));
        }
        // The shared safemlx worker takes the same MLX singleton early return
        // through one existing fallible C handle clone. No vector, AsType,
        // Concatenate, CPU task or new backing is constructed on this path.
        let controls=[size_of::<WorkspaceOperationView<'_>>(),size_of::<WorkspaceLayoutView<'_>>()*2,
            size_of::<OperationPlan>(),size_of::<Option<OperationPlan>>(),size_of::<CpuPopulation>(),
            size_of::<Option<WorkspaceRepresentation>>(),size_of::<(&[crate::MlxTensor],i32,&safemlx::Stream)>(),
            size_of::<Result<crate::MlxTensor,eredu_nn::Error>>(),
            safemlx::ops::concatenate_axis_control_bytes().ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?];
        let controls=controls.into_iter().try_fold(size_of_val(&controls),|n,b|n.checked_add(b)
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW))?;
        return Ok(Some(OperationPlan{dtype:input.representation().map_or(WorkspaceFloatingType::Float32,|r|r.dtype()),
            population:CpuPopulation{controls,..CpuPopulation::default()},alias_input:Some(0),output_bytes:0,scratch_bytes:0,
            rank:input.shape().len(),parameter_shells:1,seeds:0,validations:0}));
    }
    let inputs=operation.inputs.len();
    if inputs<2||operation.outputs.len()!=1{return Ok(None);}
    let first=operation.inputs.get(0).expect("nonempty concatenation");
    let output=operation.outputs.get(0).expect("one concatenation output");
    let rank=first.shape().len();
    if !(1..=4).contains(&rank)||output.shape().len()!=rank{return Ok(None);}
    let Some(dtype)=first.representation().map(|r|r.dtype()) else{return Ok(None)};
    let native=match dtype {WorkspaceFloatingType::Float32=>Dtype::Float32,
        WorkspaceFloatingType::Float16=>Dtype::Float16,WorkspaceFloatingType::Bfloat16=>Dtype::Bfloat16};
    if output.dtype()!=WorkspaceDtype::Float32||output.shape().iter().any(|&n|n<=0){return Ok(None);}
    let mut axis=None;
    for (index,(&before,&after)) in first.shape().iter().zip(output.shape()).enumerate() {
        if before!=after {
            if axis.is_some()||after<=before{return Err(MlxWorkspaceFactError::descriptor("CPU concatenate joined axis differs"));}
            axis=Some(index);
        }
    }
    let Some(axis)=axis else{return Err(MlxWorkspaceFactError::descriptor("CPU concatenate omitted its joined axis"))};
    let mut extent=0i32;let mut elements=0u64;
    for input in operation.inputs.iter() {
        if input.shape().len()!=rank||input.dtype()!=WorkspaceDtype::Float32||input.shape().iter().any(|&n|n<=0)||
            !input.representation().is_some_and(|r|r.dtype()==dtype&&r.last_axis_contiguous()){return Ok(None);}
        for index in 0..rank {
            if index!=axis&&input.shape()[index]!=output.shape()[index]{return Err(MlxWorkspaceFactError::descriptor("CPU concatenate nonjoined geometry differs"));}
        }
        extent=extent.checked_add(input.shape()[axis]).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
        elements=elements.checked_add(input.elements()?).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    }
    if extent!=output.shape()[axis]||elements!=output.elements()?{
        return Err(MlxWorkspaceFactError::descriptor("CPU concatenate output extent differs"));
    }
    if elements>i32::MAX as u64{return Ok(None);}
    let Some(source)=OperationEvent::cpu_concatenate_many_layout(native,rank,inputs,usize::try_from(elements)?,false) else{return Ok(None)};
    let mut population=CpuPopulation::default();
    if source.backing_births()!=1||population.copy(source,inputs).is_none(){return Ok(None);}
    // Equal scalar types make ordinary AsType identity for every operand.
    // One native Concatenate owns all destination slices/jobs and one backing.
    let frames=[size_of::<WorkspaceOperationView<'_>>(),size_of::<WorkspaceLayoutView<'_>>()*3,
        size_of::<OperationPlan>(),size_of::<Option<OperationPlan>>(),size_of::<CpuPopulation>(),
        size_of::<CpuCopyEvalLayout>(),size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<MlxCpuWorkspaceMechanisms>(),size_of::<WorkspaceFloatingType>(),size_of::<Dtype>(),
        size_of::<Option<usize>>(),size_of::<std::ops::Range<usize>>(),size_of::<usize>()*6,size_of::<u64>(),
        size_of::<i32>(),size_of::<Option<i32>>(),size_of::<Option<u64>>(),
        size_of::<std::iter::Enumerate<std::iter::Zip<std::slice::Iter<'_,i32>,std::slice::Iter<'_,i32>>>>(),
        size_of::<(&[crate::MlxTensor],i32,&safemlx::Stream)>(),
        size_of::<Result<crate::MlxTensor,eredu_nn::Error>>(),
        safemlx::ops::concatenate_axis_control_bytes().ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?];
    population.controls=frames.into_iter().try_fold(population.controls.checked_add(size_of_val(&frames))
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,usize::checked_add).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    Ok(Some(OperationPlan{dtype,population,alias_input:None,
        output_bytes:mechanism.allocation.fixed_buffer_capacity(output.bytes()?)?,scratch_bytes:0,rank,
        parameter_shells:0,seeds:0,validations:0}))
}

#[cfg(all(test,target_vendor="apple",feature="metal",not(feature="cuda")))]
mod tests {
    use super::*;
    use eredu_nn::Tensor;
    use crate::backend::nn::logical_collective;
    #[test]
    fn cpu_ordered_logical_members_share_one_exact_concat_and_completion(){
        let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let choice=MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),choice);
        for count in [2,8] {
            for dtype in [WorkspaceFloatingType::Float32,WorkspaceFloatingType::Float16,WorkspaceFloatingType::Bfloat16] {
                let context=WorkspaceContext::new(cpu);
                let world=WorkspaceTensor::existing(context.layout(&[count,1,2,19],WorkspaceDtype::Float32).unwrap()
                    .with_representation(Some(WorkspaceRepresentation::new(dtype,true))),&context).unwrap();
                let members:Vec<_>=(0..count as usize).rev().collect();
                context.begin_state_span([&world]).unwrap();
                let ops=logical_collective::Workspace(&context);
                let stacked=logical_collective::packed::gather_stacked(&ops,&world,&members).unwrap();
                let output=logical_collective::packed::flatten(&ops,&stacked,&[1,2,19],count as usize).unwrap();
                assert_eq!(output.shape(),[count,2,19]);
                let report=context.report(&[output]).unwrap();
                assert!(report.operations.iter().all(|op|cpu.plan(op.as_view()).unwrap().is_some()));
                let joined=report.operations.iter().find(|op|matches!(op.kind,WorkspaceOperationKind::Concatenate)).unwrap();
                let plan=cpu.plan(joined.as_view()).unwrap().unwrap();
                assert_eq!(plan.population.primitives,1);assert_eq!(plan.population.births,1);
                assert_eq!(plan.population.input_edges,count as usize);assert_eq!(plan.scratch_bytes,0);
                super::super::super::SpeculativeNumericalRecipe::inspect_cpu_equations(&report,ordinary,cpu,&context).unwrap();
            }
        }
    }
}
