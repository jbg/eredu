//! Same-type concatenation over the exact input population, including empty rows.
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
    let floating=first.representation().map(|r|r.dtype());
    let native=match (first.dtype(),floating) {
        (WorkspaceDtype::Float32,Some(WorkspaceFloatingType::Float32))=>Dtype::Float32,
        (WorkspaceDtype::Float32,Some(WorkspaceFloatingType::Float16))=>Dtype::Float16,
        (WorkspaceDtype::Float32,Some(WorkspaceFloatingType::Bfloat16))=>Dtype::Bfloat16,
        (WorkspaceDtype::Int32,None)=>Dtype::Int32,
        _=>return Ok(None),
    };
    if output.dtype()!=first.dtype()||output.shape().iter().any(|&n|n<0){return Ok(None);}
    let mut elements=0u64;
    for input in operation.inputs.iter() {
        if input.shape().len()!=rank||input.dtype()!=first.dtype()||input.shape().iter().any(|&n|n<0)
            ||match floating {
                Some(dtype)=>!input.representation().is_some_and(|r|r.dtype()==dtype&&r.last_axis_contiguous()),
                None=>input.representation().is_some(),
            } {return Ok(None);}
        elements=elements.checked_add(input.elements()?).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    }
    // The descriptor omits the axis. Prove one common joined dimension from
    // every real operand, including empty rows. Several axes can coincide only
    // for empty geometry; the same rank/arity worker covers those choices.
    let mut axis=None;
    for candidate in 0..rank {
        let mut extent=0i32;let mut valid=true;
        for input in operation.inputs.iter() {
            for index in 0..rank {
                if index!=candidate&&input.shape()[index]!=output.shape()[index]{valid=false;}
            }
            extent=extent.checked_add(input.shape()[candidate]).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
        }
        if valid&&extent==output.shape()[candidate]{axis=Some(candidate);break;}
    }
    if axis.is_none()||elements!=output.elements()?{
        return Err(MlxWorkspaceFactError::descriptor("CPU concatenate output extent differs"));
    }
    if elements>i32::MAX as u64{return Ok(None);}
    // OperationPlan's private floating slot is not representation evidence.
    // Integral output layouts remain Int32 with no floating representation.
    let dtype=floating.unwrap_or(WorkspaceFloatingType::Float32);
    let Some(source)=OperationEvent::cpu_concatenate_many_layout(native,rank,inputs,usize::try_from(elements)?,false) else{return Ok(None)};
    let mut population=CpuPopulation::default();
    if source.backing_births()!=usize::from(elements!=0)||population.concatenate(source,inputs).is_none(){return Ok(None);}
    // Equal scalar types make ordinary AsType identity for every operand.
    // One native Concatenate owns all destination slices/jobs and one backing.
    let frames=[size_of::<WorkspaceOperationView<'_>>(),size_of::<WorkspaceLayoutView<'_>>()*3,
        size_of::<OperationPlan>(),size_of::<Option<OperationPlan>>(),size_of::<CpuPopulation>(),
        size_of::<CpuCopyEvalLayout>(),size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<MlxCpuWorkspaceMechanisms>(),size_of::<WorkspaceFloatingType>(),size_of::<Dtype>(),
        size_of::<Option<WorkspaceFloatingType>>(),size_of::<WorkspaceDtype>(),size_of::<bool>(),
        size_of::<eredu_nn::workspace::WorkspaceLayoutIter<'_>>(),size_of::<usize>()*2,
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
                assert_eq!(plan.population.maximum_captures,1+2*count as usize);
                let mut repeated=plan.population;
                repeated.add(plan.population).unwrap();
                assert_eq!(repeated.maximum_captures,plan.population.maximum_captures);
                let recipe=super::super::super::SpeculativeNumericalRecipe::inspect_cpu_equations(&report,ordinary,cpu,&context).unwrap();
                assert_eq!(recipe.completion.traversal.limits().captures,8.max(1+2*count as usize));
            }
        }
    }
}

#[cfg(all(test,target_vendor="apple",feature="metal",not(feature="cuda")))]
mod empty_integer_tests {
    use super::*;
    use crate::backend::nn::logical_collective::{self,blocks};
    use eredu_nn::Tensor;
    fn mechanisms()->(MlxMetalWorkspaceMechanisms,MlxCpuWorkspaceMechanisms) {
        let native=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let cpu=MlxCpuWorkspaceMechanisms::new(native.allocation(),
            MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap());
        (native,cpu)
    }
    fn value(context:&WorkspaceContext,rows:i32,dtype:WorkspaceDtype)->WorkspaceTensor {
        let representation=(dtype==WorkspaceDtype::Float32)
            .then_some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true));
        WorkspaceTensor::existing(context.layout(&[rows,3],dtype).unwrap()
            .with_representation(representation),context).unwrap()
    }
    #[test]
    fn cpu_expert_block_empty_and_integer_joins_quote_actual_shared_workers() {
        let (native,cpu)=mechanisms();
        for dtype in [WorkspaceDtype::Int32,WorkspaceDtype::Float32] {
            for counts in [[0usize,0,0],[0,3,0],[2,0,3]] {
                for join in [false,true] {
                    let context=WorkspaceContext::new(cpu);
                    let inputs=counts.map(|rows|value(&context,rows as i32,dtype));
                    let source=value(&context,counts.iter().sum::<usize>() as i32,dtype);
                    context.begin_span();
                    let ops=logical_collective::Workspace(&context);
                    let output=if join {blocks::join(&ops,&inputs).unwrap()}
                        else {blocks::concatenate(&ops,&source,&counts,(0..3).rev()).unwrap()};
                    assert_eq!(output.shape(),source.shape());assert_eq!(output.layout().dtype(),dtype);
                    if dtype==WorkspaceDtype::Int32 {assert_eq!(output.layout().representation(),None);}
                    let report=context.finish_report(&[output]).unwrap();
                    for operation in &report.operations {
                        let plan=cpu.plan(operation.as_view()).unwrap().unwrap_or_else(||
                            panic!("missing CPU source: dtype={dtype:?} counts={counts:?} join={join} operation={operation:?}"));
                        if matches!(operation.kind,WorkspaceOperationKind::Concatenate) {
                            let operands=operation.inputs.len();
                            assert_eq!(plan.population.primitives,1);
                            assert_eq!(plan.population.input_edges,operands);
                            assert_eq!(plan.population.maximum_captures,1+2*operands);
                            assert_eq!(plan.population.births,usize::from(counts.iter().sum::<usize>()!=0));
                            assert_eq!(plan.output_bytes==0,counts.iter().sum::<usize>()==0);
                        } else if super::super::super::zero_fill::dtype(operation.as_view()).is_some() {
                            assert_eq!(plan.population.primitives,2); // Broadcast and Full Evals.
                            assert_eq!(plan.population.input_edges,2);
                            assert_eq!(plan.population.births,0);
                            assert_eq!(plan.seeds,1);assert_eq!(plan.output_bytes,0);
                            assert!(plan.scratch_bytes>0); // The actual typed scalar seed.
                        }
                    }
                    let recipe=SpeculativeNumericalRecipe::inspect_cpu_outputs(&report,1,native,cpu,&context).unwrap();
                    assert_eq!(recipe.kernels,0);
                }
            }
        }
    }
    #[test]
    fn cpu_integer_join_rejects_mixed_dtype_and_false_join_geometry() {
        let (_,cpu)=mechanisms();
        let a=WorkspaceLayoutView::new(&[0,3],WorkspaceDtype::Int32).unwrap();
        let b=WorkspaceLayoutView::new(&[2,3],WorkspaceDtype::Int32).unwrap();
        let bad=WorkspaceLayoutView::new(&[2,4],WorkspaceDtype::Int32).unwrap();
        let mixed=WorkspaceLayoutView::new(&[2,3],WorkspaceDtype::Uint32).unwrap();
        for (right,output) in [(b,bad),(mixed,b)] {
            let inputs=[a,right];let outputs=[output];
            let operation=WorkspaceOperationView {kind:WorkspaceOperationKindView::Concatenate,
                inputs:WorkspaceLayoutList::Views(&inputs),outputs:WorkspaceLayoutList::Views(&outputs)};
            assert!(cpu.plan(operation).map_or(true,|plan|plan.is_none()));
        }
    }
}
