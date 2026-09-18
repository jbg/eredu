//! The shared one-index Gather and general Scatter overwrite on CPU.
use super::*;
use safemlx::Dtype;
pub(super) fn inspect(operation: WorkspaceOperationView<'_>, mechanism: MlxCpuWorkspaceMechanisms)
    -> facts::FactResult<Option<OperationPlan>> {
    let update = match operation.kind {
        WorkspaceOperationKindView::IndexedElementSelect => false,
        WorkspaceOperationKindView::IndexedElementUpdate => true,
        _ => return Ok(None),
    };
    if !super::super::basic::indexed_elements(operation) {
        return Err(MlxWorkspaceFactError::descriptor("CPU sparse element source geometry differs"));
    }
    let base = operation.inputs.get(0).expect("validated base");
    let indices = operation.inputs.get(1).expect("validated indices");
    if !base.representation().is_some_and(|r| r.dtype()==WorkspaceFloatingType::Float32 && r.row_contiguous())
        || (update && !operation.inputs.get(2).expect("validated update").representation()
            .is_some_and(|r| r.dtype()==WorkspaceFloatingType::Float32)) { return Ok(None); }
    let elements = usize::try_from(base.elements()?)?;
    let selected = usize::try_from(indices.elements()?)?;
    if selected == 0 || elements > i32::MAX as usize || selected > i32::MAX as usize { return Ok(None); }
    let plan = (|| -> Option<CpuPopulation> {
        let mut p = CpuPopulation::default();
        if update {
            p.copy(OperationEvent::cpu_broadcast_alias_layout(1,1,false)?,1)?;
            p.copy(OperationEvent::cpu_reshape_alias_layout(1,2,false)?,1)?;
            p.copy(OperationEvent::cpu_cast_layout(Dtype::Int32,Dtype::Int32,1,selected,false)?,1)?;
            p.copy(OperationEvent::cpu_cast_layout(Dtype::Float32,Dtype::Float32,2,selected,false)?,1)?;
            p.copy(OperationEvent::cpu_scatter_layout(Dtype::Float32,Dtype::Int32,1,elements,selected,false)?,3)?;
            p.controls = p.controls.checked_add(safemlx::Array::flat_index_update_control_bytes()?)?;
        } else {
            p.copy(OperationEvent::cpu_cast_layout(Dtype::Int32,Dtype::Int32,1,selected,false)?,1)?;
            p.copy(OperationEvent::cpu_gather_layout(Dtype::Float32,Dtype::Int32,1,1,elements,selected,1,false)?,2)?;
            p.copy(OperationEvent::cpu_squeeze_layout(2,false)?,1)?;
        }
        Some(p)
    })();
    let Some(mut population) = plan else { return Ok(None); };
    let frames = [size_of::<MlxCpuWorkspaceMechanisms>(),size_of::<WorkspaceOperationView<'_>>(),
        size_of::<WorkspaceLayoutView<'_>>() * 3,size_of::<CpuPopulation>() * 2,
        size_of::<Option<CpuPopulation>>(),size_of::<CpuCopyEvalLayout>(),size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<[usize;3]>(),size_of::<bool>(),size_of::<OperationPlan>(),
        size_of::<facts::FactResult<Option<OperationPlan>>>()];
    population.controls = frames.into_iter().try_fold(population.controls.checked_add(size_of_val(&frames))
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,usize::checked_add)
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    Ok(Some(OperationPlan { dtype:WorkspaceFloatingType::Float32, population,
        output_bytes:mechanism.allocation.fixed_buffer_capacity(operation.outputs.get(0).expect("validated output").bytes()?)?,
        scratch_bytes:0,rank:2,parameter_shells:0,seeds:0,validations:0,alias_input:None }))
}
