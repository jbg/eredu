//! Exact saved-copy storage facts for the existing contiguous/eager worker.
use super::*;

pub(super) fn inspect(
    operation: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
) -> facts::FactResult<Option<OperationPlan>> {
    let deep = match operation.kind {
        WorkspaceOperationKindView::Contiguous => false,
        WorkspaceOperationKindView::DeepCopy => true,
        _ => return Ok(None),
    };
    if operation.inputs.len() != 1 || operation.outputs.len() != 1 {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU storage copy requires one source and destination",
        ));
    }
    let input = operation.inputs.get(0).expect("one copy source");
    let output = operation.outputs.get(0).expect("one copy destination");
    if input.dtype() != output.dtype() || input.shape() != output.shape() {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU storage copy changes logical geometry",
        ));
    }
    let rank = input.shape().len();
    if rank > 5 || input.shape().iter().any(|&n| n < 0) || input.elements()? > i32::MAX as u64 {
        return Ok(None);
    }
    let representation = input.representation();
    let dtype = match input.dtype() {
        WorkspaceDtype::Float32 => match representation {
            Some(value) => value.dtype(),
            None => return Ok(None),
        },
        WorkspaceDtype::Int32
        | WorkspaceDtype::Uint32
        | WorkspaceDtype::Uint8
        | WorkspaceDtype::Bool => WorkspaceFloatingType::Float32,
    };
    if output
        .representation()
        .is_some_and(|value| input.dtype() != WorkspaceDtype::Float32 || value.dtype() != dtype)
    {
        return Ok(None);
    }
    let bytes = mechanism
        .allocation
        .fixed_buffer_capacity(output.bytes()?)?;
    // An ordinary arbitrary-stride deep clone first compacts; the isolated
    // saved worker has already produced a row-contiguous value, so that branch
    // is absent for its floating input. Integers carry no layout promise.
    let compaction = !deep || !representation.is_some_and(|value| value.row_contiguous());
    let mut population = CpuPopulation::default();
    if compaction {
        let Some(source) = OperationEvent::cpu_contiguous_layout(rank, false) else {
            return Ok(None);
        };
        if source.backing_births() != 1 || population.copy(source, 1).is_none() {
            return Ok(None);
        }
    }
    let eager = if deep {
        let Some(controls) = safemlx::original_scoped_deep_copy_control_bytes(rank) else {
            return Ok(None);
        };
        controls
    } else {
        0
    };
    let frames = [
        eager,
        size_of::<WorkspaceOperationView<'_>>(),
        size_of::<WorkspaceLayoutView<'_>>() * 2,
        size_of::<MlxCpuWorkspaceMechanisms>(),
        size_of::<CpuPopulation>(),
        size_of::<OperationPlan>(),
        size_of::<facts::FactResult<Option<OperationPlan>>>(),
        size_of::<Option<OperationPlan>>(),
        size_of::<WorkspaceRepresentation>(),
        size_of::<Option<WorkspaceRepresentation>>(),
        size_of::<CpuCopyEvalLayout>(),
        size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<WorkspaceFloatingType>(),
        size_of::<bool>() * 2,
        size_of::<u64>(),
        size_of::<usize>() * 2,
        size_of::<Option<usize>>(),
        size_of::<std::slice::Iter<'_, i32>>(),
        size_of::<(&safemlx::Array, bool, &safemlx::Stream)>(),
        size_of::<Result<safemlx::Array, safemlx::error::Exception>>(),
    ];
    population.controls = frames
        .into_iter()
        .try_fold(
            population
                .controls
                .checked_add(size_of_val(&frames))
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
            usize::checked_add,
        )
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    Ok(Some(OperationPlan {
        dtype,
        population,
        alias_input: None,
        output_bytes: bytes,
        scratch_bytes: if deep && compaction { bytes } else { 0 },
        rank,
        parameter_shells: 0,
        // Eager data construction owns a seed, not a second copy Eval task.
        // The containing OriginalCopyPlan supplies the real completed frontier,
        // graph/record/buffer account and retained recovery aliases separately.
        seeds: usize::from(deep),
        validations: 0,
    }))
}
