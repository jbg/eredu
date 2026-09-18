//! Source-visible immutable Host transfers using the existing General CPU copy.
use super::*;
pub(in crate::backend::nn::workspace) fn is_load(kind:WorkspaceOperationKindView<'_>)->bool {
    matches!(kind,WorkspaceOperationKindView::HostTransferFloating(_)|WorkspaceOperationKindView::HostLoadStoredFloating(..))
}
pub(super) fn inspect(operation:WorkspaceOperationView<'_>,mechanism:MlxCpuWorkspaceMechanisms)
    ->facts::FactResult<Option<OperationPlan>> {
    let store=matches!(operation.kind,WorkspaceOperationKindView::HostStoreFloating(..));
    if !store&&!is_load(operation.kind){return Ok(None);}
    let Some(dtype)=basic::host_transfer_dtype(operation)else{return Ok(None)};
    let value=if store {operation.inputs.get(0)}else{operation.outputs.get(0)}.expect("validated transfer");
    let rank=value.shape().len();
    if value.elements()?>i32::MAX as u64{return Ok(None);}
    let Some(source)=OperationEvent::cpu_host_transfer_layout(dtype,rank,store,false)else{return Ok(None)};
    let mut population=CpuPopulation::default();
    if population.copy(source,1).is_none(){return Ok(None);}
    let frontend=if store {
        // Native copy_to_host_into always calls Contiguous before its transfer.
        // The actual row alias may need less, but no cold flag discounts it.
        let Some(compact)=OperationEvent::cpu_contiguous_layout(rank,false)else{return Ok(None)};
        if population.copy(compact,1).is_none(){return Ok(None);}
        safemlx::PreparedHostCopyDestination::original_layout(rank,dtype)
    } else {
        // This retained Host leaf needs a descriptor/graph seed and traversal
        // slot, but its accepted backing is not a fresh Buffer birth.
        population.hidden_leaves=1;
        safemlx::ImmutableHostTransferBuffer::original_copy_layout(rank,dtype)
    };
    let Some((controls,extent))=frontend else{return Ok(None)};
    let frames=[controls,size_of::<(WorkspaceOperationView<'_>,MlxCpuWorkspaceMechanisms)>(),
        size_of::<WorkspaceLayoutView<'_>>(),size_of::<safemlx::Dtype>(),size_of::<WorkspaceFloatingType>(),
        size_of::<CpuPopulation>(),size_of::<OperationPlan>(),size_of::<Option<OperationPlan>>(),
        size_of::<facts::FactResult<Option<OperationPlan>>>(),size_of::<CpuCopyEvalLayout>()*2,
        size_of::<Option<CpuCopyEvalLayout>>() *2,size_of::<(usize,usize)>(),size_of::<Option<(usize,usize)>>(),
        size_of::<usize>()*3,size_of::<bool>(),size_of::<u64>(),
        if store {crate::backend::submission_recovery::prefill::TransientRootsProjection::control_bytes()
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?}else{0}];
    population.controls=frames.into_iter().try_fold(population.controls.checked_add(size_of_val(&frames))
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,usize::checked_add)
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    population.extents=population.extents.checked_add(extent).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    let bytes=mechanism.allocation.fixed_buffer_capacity(value.bytes()?)?;
    Ok(Some(OperationPlan{dtype:match dtype{safemlx::Dtype::Float32=>WorkspaceFloatingType::Float32,
        safemlx::Dtype::Float16=>WorkspaceFloatingType::Float16,safemlx::Dtype::Bfloat16=>WorkspaceFloatingType::Bfloat16,
        _=>return Ok(None)},population,alias_input:None,output_bytes:if store{0}else{bytes},
        scratch_bytes:if store{bytes}else{0},rank,parameter_shells:0,seeds:0,validations:0}))
}
#[cfg(all(test,target_vendor="apple",feature="metal",not(feature="cuda")))]
mod tests;
