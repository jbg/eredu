//! Existing typed scalar seed, optional Broadcast and Full at logical boundaries.
use super::*;
use super::super::zero_fill as source;

pub(super) fn inspect(operation:WorkspaceOperationView<'_>,mechanism:MlxCpuWorkspaceMechanisms)
    ->facts::FactResult<Option<OperationPlan>> {
    let Some((native,floating,seed_bytes))=source::dtype(operation) else{return Ok(None)};
    let output=operation.outputs.get(0).expect("one declared zero output");
    let rank=output.shape().len();let count=usize::try_from(output.elements()?)?;
    if rank>4||count>i32::MAX as usize{return Ok(None);}
    let mut population=CpuPopulation::default();
    // Empty Broadcast constructs Data instead of aliasing. Full still owns
    // its copy Eval, but neither zero-byte output has a physical backing birth.
    if rank!=0 {
        let broadcast=if count==0 {OperationEvent::cpu_empty_broadcast_layout(0,rank,false)}
            else {OperationEvent::cpu_broadcast_alias_layout(0,rank,false)};
        let Some(broadcast)=broadcast else{return Ok(None)};
        if broadcast.backing_births()!=0||population.copy(broadcast,1).is_none(){return Ok(None);}
    }
    let Some(full)=OperationEvent::cpu_scalar_full_layout(native,rank,count,false) else{return Ok(None)};
    if full.backing_births()!=usize::from(count!=0)||population.copy(full,1).is_none(){return Ok(None);}
    let frames=[source::control_bytes().ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
        size_of::<WorkspaceOperationView<'_>>(),size_of::<WorkspaceLayoutView<'_>>(),
        size_of::<MlxCpuWorkspaceMechanisms>(),size_of::<CpuPopulation>(),size_of::<OperationPlan>(),
        size_of::<facts::FactResult<Option<OperationPlan>>>(),size_of::<Option<OperationPlan>>(),
        size_of::<CpuCopyEvalLayout>(),size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<Option<(safemlx::Dtype,Option<WorkspaceFloatingType>,u64)>>(),
        size_of::<safemlx::Dtype>(),size_of::<Option<WorkspaceFloatingType>>(),size_of::<u64>(),
        size_of::<usize>()*2,size_of::<Result<usize,std::num::TryFromIntError>>()];
    population.controls=frames.into_iter().try_fold(population.controls.checked_add(size_of_val(&frames))
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,usize::checked_add).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    Ok(Some(OperationPlan{dtype:floating.unwrap_or(WorkspaceFloatingType::Float32),population,alias_input:None,
        output_bytes:mechanism.allocation.fixed_buffer_capacity(output.bytes()?)?,
        scratch_bytes:mechanism.allocation.fixed_buffer_capacity(seed_bytes)?,rank,parameter_shells:0,seeds:1,validations:0}))
}

#[cfg(all(test,target_vendor="apple",feature="metal",not(feature="cuda")))]
mod tests {
    use super::*;
    use crate::backend::nn::logical_collective::{self,Operations};
    use eredu_nn::Tensor;
    #[test]
    fn cpu_logical_zero_pack_and_selected_member_keep_exact_source_chain(){
        let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let choice=MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),choice);
        let context=WorkspaceContext::new(cpu);
        let known=Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true));
        let input=WorkspaceTensor::existing(context.layout(&[1,2,19],WorkspaceDtype::Float32).unwrap().with_representation(known),&context).unwrap();
        context.begin_state_span([&input]).unwrap();
        let ops=logical_collective::Workspace(&context);let zero=ops.zero(&input).unwrap();
        let peer=logical_collective::peer(&ops,&input,&input,&zero).unwrap();
        let packed=logical_collective::packed::pack(&ops,&peer,2,4).unwrap();
        let selected=logical_collective::packed::sum_result(&ops,&packed,2).unwrap();
        assert_eq!(selected.shape(),input.shape());assert_eq!(selected.layout().representation(),known);
        let report=context.report(&[selected]).unwrap();
        assert!(report.operations.iter().all(|op|cpu.plan(op.as_view()).unwrap().is_some()));
        assert!(!report.operations.iter().any(|op|matches!(op.kind,WorkspaceOperationKind::Initialize|WorkspaceOperationKind::InitializeFloating(_)|WorkspaceOperationKind::Index{..})));
        super::super::super::SpeculativeNumericalRecipe::inspect_cpu_equations(&report,ordinary,cpu,&context).unwrap();
        for dtype in [WorkspaceFloatingType::Float32,WorkspaceFloatingType::Float16,WorkspaceFloatingType::Bfloat16] {
            for shape in [&[][..],&[2,3,19][..]] {
                let context=WorkspaceContext::new(cpu);
                let prototype=WorkspaceLayoutView::new(&[],WorkspaceDtype::Float32).unwrap()
                    .with_representation(Some(WorkspaceRepresentation::new(dtype,true)));
                let zero=source::trace(shape,prototype,&context).unwrap();
                assert_eq!(zero.layout().representation(),Some(WorkspaceRepresentation::new(dtype,true)));
                let report=context.report(&[zero]).unwrap();let plan=cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
                assert_eq!(plan.seeds,1);assert_eq!(plan.population.births,1);
                assert_eq!(plan.population.primitives,1+usize::from(!shape.is_empty()));
                assert!(plan.output_bytes>0&&plan.scratch_bytes>0);
            }
        }
    }
}
