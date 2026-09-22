//! The contiguous Bool result consumed by the shared NegLog recipe validator.
use super::*;

pub(super) fn inspect(
    operation: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
) -> facts::FactResult<Option<OperationPlan>> {
    if !matches!(
        operation.kind,
        WorkspaceOperationKindView::Elementwise("recipe_all")
    ) {
        return Ok(None);
    }
    let Some([input]) = operation.inputs.array() else {
        return Err(MlxWorkspaceFactError::descriptor(
            "recipe Bool reduction input differs",
        ));
    };
    let Some([output]) = operation.outputs.array() else {
        return Err(MlxWorkspaceFactError::descriptor(
            "recipe Bool reduction output differs",
        ));
    };
    if input.dtype() != WorkspaceDtype::Bool
        || output.dtype() != WorkspaceDtype::Bool
        || !output.shape().is_empty()
    {
        return Err(MlxWorkspaceFactError::descriptor(
            "recipe validation changes Bool scalar geometry",
        ));
    }
    let rank = input.shape().len();
    let elements = usize::try_from(input.elements()?)?;
    if rank > 4 || elements == 0 || elements > i32::MAX as usize {
        return Ok(None);
    }
    let mut population = CpuPopulation::default();
    if elements > 1 {
        let Some(reduce) = OperationEvent::cpu_boolean_reduce_layout(true, rank, elements, false)
        else {
            return Ok(None);
        };
        if reduce.backing_births() != 1 || population.copy(reduce, 1).is_none() {
            return Ok(None);
        }
    }
    // All-singleton axes elide Reduce. Non-scalar all(false) still squeezes
    // every selected axis; rank zero returns the same Bool value directly.
    if rank != 0 {
        let Some(squeeze) = OperationEvent::cpu_squeeze_layout(rank, false) else {
            return Ok(None);
        };
        if squeeze.backing_births() != 0 || population.copy(squeeze, 1).is_none() {
            return Ok(None);
        }
    }
    let frames = [
        size_of::<WorkspaceOperationView<'_>>(),
        size_of::<MlxCpuWorkspaceMechanisms>(),
        size_of::<[WorkspaceLayoutView<'_>; 1]>() * 2,
        size_of::<WorkspaceLayoutView<'_>>() * 2,
        size_of::<Option<[WorkspaceLayoutView<'_>; 1]>>() * 2,
        size_of::<usize>() * 2,
        size_of::<CpuPopulation>(),
        size_of::<CpuCopyEvalLayout>() * 2,
        size_of::<Option<CpuCopyEvalLayout>>() * 2,
        size_of::<Option<()>>(),
        size_of::<OperationPlan>(),
        size_of::<facts::FactResult<Option<OperationPlan>>>(),
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
        dtype: WorkspaceFloatingType::Float32,
        population,
        alias_input: (elements == 1).then_some(0),
        output_bytes: if elements == 1 {
            0
        } else {
            mechanism.allocation.fixed_buffer_capacity(1)?
        },
        scratch_bytes: 0,
        rank,
        parameter_shells: 0,
        seeds: 0,
        validations: 0,
    }))
}
