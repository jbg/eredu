//! The actual eager F32 result and aliasing Copy of the selected host decoder.
use super::geometry::gguf as source;
use super::*;

pub(super) fn inspect(
    op: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
) -> facts::FactResult<Option<OperationPlan>> {
    let Some(g) = source::inspect(op)? else {
        return Ok(None);
    };
    let Some(copy) = OperationEvent::cpu_copy_alias_layout(g.rank, false) else {
        return Ok(None);
    };
    let mut population = CpuPopulation::default();
    if copy.backing_births() != 0 || population.copy(copy, 1).is_none() {
        return Ok(None);
    }
    let controls = [
        source::control_bytes().ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
        size_of::<CpuPopulation>(),
        size_of::<OperationPlan>(),
        size_of::<Option<OperationPlan>>(),
        size_of::<CpuCopyEvalLayout>(),
    ];
    population.controls = controls
        .into_iter()
        .try_fold(
            population
                .controls
                .checked_add(size_of_val(&controls))
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
            usize::checked_add,
        )
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    Ok(Some(OperationPlan {
        dtype: WorkspaceFloatingType::Float32,
        population,
        alias_input: None,
        output_bytes: mechanism
            .allocation
            .fixed_buffer_capacity(facts::mul(g.values as u64, 4)?)?,
        scratch_bytes: 0,
        rank: g.rank,
        parameter_shells: 1,
        seeds: 1,
        validations: 0,
    }))
}
