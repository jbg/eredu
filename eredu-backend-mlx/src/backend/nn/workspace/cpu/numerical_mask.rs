//! CPU population of the expanded-Bool token mask and optional input-alias path.
use super::*;
use safemlx::Dtype;

pub(in crate::backend::nn::workspace) fn token_filter(rank:usize,elements:usize)->Option<CpuPopulation>{
    if !(2..=3).contains(&rank)||elements==0||elements>i32::MAX as usize{return None;}
    let mut population=CpuPopulation::default();
    // Both eager seeds (the full Bool mask and F32 -infinity) are already
    // named by the shared TokenFilter constructor/physical source. where
    // applies the same three casts and broadcast aliases before Select.
    population.copy(OperationEvent::cpu_cast_layout(Dtype::Bool,Dtype::Bool,rank,elements,false)?,1)?;
    population.copy(OperationEvent::cpu_cast_layout(Dtype::Float32,Dtype::Float32,0,1,false)?,1)?;
    population.copy(OperationEvent::cpu_cast_layout(Dtype::Float32,Dtype::Float32,rank,elements,false)?,1)?;
    population.copy(OperationEvent::cpu_broadcast_alias_layout(rank,rank,false)?,1)?;
    population.copy(OperationEvent::cpu_broadcast_alias_layout(0,rank,false)?,1)?;
    population.copy(OperationEvent::cpu_broadcast_alias_layout(rank,rank,false)?,1)?;
    population.copy(OperationEvent::cpu_select_broadcast_layout(rank,elements,false)?,3)?;
    if population.primitives!=7||population.input_edges!=9{return None;}
    let parts=[size_of::<CpuPopulation>()*2,size_of::<Option<CpuPopulation>>(),
        size_of::<(usize,usize)>(),size_of::<CpuCopyEvalLayout>(),
        size_of::<Option<CpuCopyEvalLayout>>(),size_of::<Option<()>>()];
    population.controls=parts.into_iter().try_fold(population.controls.checked_add(size_of_val(&parts))?,usize::checked_add)?;
    Some(population)
}

/// Compose the actual ordinary mask worker with other CPU equations. Its
/// source is the same seven constructors used by the standalone policy quote.
pub(super) fn inspect(operation:WorkspaceOperationView<'_>,mechanism:MlxCpuWorkspaceMechanisms)
    ->facts::FactResult<Option<OperationPlan>> {
    if !matches!(operation.kind,WorkspaceOperationKindView::Sampling(
        WorkspaceSamplingOperation::TokenFilter | WorkspaceSamplingOperation::OptionalTokenFilter)) {
        return Ok(None);
    }
    let Some([input])=operation.inputs.array() else {return Err(MlxWorkspaceFactError::descriptor("CPU token mask input population differs"));};
    let Some([output])=operation.outputs.array() else {return Err(MlxWorkspaceFactError::descriptor("CPU token mask output population differs"));};
    if input.shape()!=output.shape()||input.dtype()!=WorkspaceDtype::Float32||output.dtype()!=input.dtype() {
        return Err(MlxWorkspaceFactError::descriptor("CPU token mask changes floating geometry"));
    }
    let rank=input.shape().len();
    if !(2..=3).contains(&rank)||input.shape().iter().any(|&n|n<=0)
        ||input.representation()!=Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true)) {
        return Ok(None);
    }
    let elements=usize::try_from(input.elements()?)?;
    let Some(mut population)=token_filter(rank,elements) else{return Ok(None);};
    let parts=[size_of::<WorkspaceOperationView<'_>>(),size_of::<WorkspaceLayoutView<'_>>()*2,
        size_of::<MlxCpuWorkspaceMechanisms>(),size_of::<CpuPopulation>(),size_of::<Option<CpuPopulation>>(),
        size_of::<OperationPlan>(),size_of::<facts::FactResult<Option<OperationPlan>>>(),
        size_of::<(usize,usize,u64,u64)>()];
    population.controls=parts.into_iter().try_fold(population.controls.checked_add(size_of_val(&parts))
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,usize::checked_add)
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    let output_bytes=mechanism.allocation.fixed_buffer_capacity(facts::mul(elements as u64,4)?)?;
    // Seed Bool/scalar storage plus the conservative three AsType copy sources
    // named above remain live alongside Select's output. Same-type aliases do
    // not grant early retirement or reduce the accepted copy envelope.
    let bool_bytes=mechanism.allocation.fixed_buffer_capacity(elements as u64)?;
    let scalar_bytes=mechanism.allocation.fixed_buffer_capacity(4)?;
    let scratch_bytes=facts::add(output_bytes,facts::add(facts::mul(bool_bytes,2)?,facts::mul(scalar_bytes,2)?)?)?;
    Ok(Some(OperationPlan{dtype:WorkspaceFloatingType::Float32,population,alias_input:None,
        output_bytes,scratch_bytes,rank,parameter_shells:0,seeds:2,validations:0}))
}

#[cfg(all(test, target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
mod tests {
    use super::*;

    #[test]
    fn optional_cpu_mask_retains_full_input_alias_and_prices_masked_branch() {
        let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let matmul = MlxCpuMatmulMechanism::select(
            eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), matmul);
        for shape in [&[1, 17][..], &[1, 1, 17][..]] {
            for optional in [false, true] {
                let context = WorkspaceContext::new(cpu);
                let storage = WorkspaceExistingStorage::try_new(Some(4096), &context).unwrap();
                let layout = context.layout(shape, WorkspaceDtype::Float32).unwrap()
                    .with_representation(Some(WorkspaceRepresentation::new(
                        WorkspaceFloatingType::Float32, true)));
                let input = WorkspaceTensor::existing_with_storage(layout, &storage, &context).unwrap();
                context.begin_state_span([&input]).unwrap();
                let kind = if optional { WorkspaceSamplingOperation::OptionalTokenFilter }
                    else { WorkspaceSamplingOperation::TokenFilter };
                let output = context.execute(WorkspaceOperationKind::Sampling(kind), &[&input],
                    vec![input.layout().clone()]).unwrap().remove(0);
                assert_eq!(output.layout().representation(), input.layout().representation());
                let report = context.finish_report(&[output]).unwrap();
                assert!(report.unpriced_operations.is_empty());
                assert!(report.unpriced_host_operations.is_empty());
                assert_eq!(report.host_workspace_bytes, Some(34));
                let output_capacity = ordinary.allocation().fixed_buffer_capacity(17 * 4).unwrap();
                assert_eq!(report.closing_storage.bytes,
                    Some(output_capacity + if optional { 4096 } else { 0 }));
                let plan = cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
                assert_eq!(plan.seeds, 2);
                assert_eq!(plan.population.primitives, 7);
                assert_eq!(plan.population.input_edges, 9);
                let mut missing = report.operations[0].clone();
                missing.inputs[0] = missing.inputs[0].clone().with_representation(None);
                assert!(cpu.plan(missing.as_view()).unwrap().is_none());
            }
        }
    }
}
