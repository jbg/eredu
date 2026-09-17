//! Shared explicit-key state and final token workers in a resident CPU trace.
use super::*;
use eredu_nn::workspace::WorkspaceSamplingOperation as S;

pub(super) fn inspect(operation: WorkspaceOperationView<'_>, mechanism: MlxCpuWorkspaceMechanisms)
    -> facts::FactResult<Option<OperationPlan>>
{
    let WorkspaceOperationKindView::Sampling(kind) = operation.kind else { return Ok(None); };
    if !matches!(kind, S::CreateRandomKey | S::SplitRandomKey | S::SelectRandomKey { .. }
        | S::Greedy | S::Categorical | S::UniformUnitInterval | S::ReadToken) { return Ok(None); }
    let [output] = operation.outputs.array().ok_or_else(|| MlxWorkspaceFactError::descriptor("CPU sampler output differs"))?;
    let source = (|| -> Option<OperationPlan> {
        let mut population = CpuPopulation::default();
        let mut alias_input = None;
        let mut output_bytes = mechanism.allocation.fixed_buffer_capacity(output.bytes().ok()?).ok()?;
        let scalar = mechanism.allocation.fixed_buffer_capacity(4).ok()?;
        let mut scratch_bytes = 0;
        let mut seeds = 0;
        let mut rank = output.shape().len();
        match kind {
            S::CreateRandomKey => {
                if !operation.inputs.is_empty() || output.dtype() != WorkspaceDtype::Uint32 || output.shape() != [2] { return None; }
                seeds = 1;
            }
            S::SplitRandomKey => {
                let [key] = operation.inputs.array()?;
                if key.shape() != [2] || key.dtype() != WorkspaceDtype::Uint32
                    || output.dtype() != WorkspaceDtype::Uint32 || output.shape().len() != 2
                    || output.shape()[0] <= 0 || output.shape()[1] != 2 { return None; }
                let words = usize::try_from(output.elements().ok()?).ok()?;
                if words > i32::MAX as usize { return None; }
                population.copy(OperationEvent::cpu_random_bits_layout(2, words, false)?, 1)?;
            }
            S::SelectRandomKey { index } => {
                let [table] = operation.inputs.array()?;
                if table.dtype() != WorkspaceDtype::Uint32 || table.shape().len() != 2
                    || table.shape()[0] <= 0 || table.shape()[1] != 2
                    || u64::from(*index) >= table.shape()[0] as u64
                    || table.elements().ok()? > i32::MAX as u64
                    || output.dtype() != WorkspaceDtype::Uint32 || output.shape() != [2] { return None; }
                if table.shape()[0] != 1 { population.copy(OperationEvent::cpu_slice_layout(2, false)?, 1)?; }
                population.copy(OperationEvent::cpu_reshape_alias_layout(2, 1, false)?, 1)?;
                rank = 2; alias_input = Some(0); output_bytes = 0;
            }
            S::ReadToken => {
                let [token] = operation.inputs.array()?;
                if token.dtype() != WorkspaceDtype::Uint32 || token.elements().ok()? != 1 || token != output { return None; }
                alias_input = Some(0); output_bytes = 0;
            }
            S::UniformUnitInterval => {
                let [key] = operation.inputs.array()?;
                if key.shape() != [2] || key.dtype() != WorkspaceDtype::Uint32
                    || output.dtype() != WorkspaceDtype::Float32 || output.shape() != [1] { return None; }
                population = numerical_random::uniform_draw(1, 1)?;
                // One scalar range and six draw-shaped results; the final draw
                // is the output. Four actual F32 seeds precede these workers.
                scratch_bytes = scalar.checked_mul(10)?;
                seeds = 4;
            }
            S::Greedy | S::Categorical => {
                let scores = operation.inputs.get(0)?;
                rank = scores.shape().len();
                if !(2..=3).contains(&rank) || scores.dtype() != WorkspaceDtype::Float32
                    || scores.shape()[..rank-1].iter().any(|&n| n != 1)
                    || !scores.representation().is_some_and(|r| r.dtype() == WorkspaceFloatingType::Float32 && r.row_contiguous())
                    || output.dtype() != WorkspaceDtype::Uint32 || output.shape() != &scores.shape()[..rank-1] { return None; }
                let columns = usize::try_from(scores.shape()[rank-1]).ok()?;
                if columns <= 1 || columns > i32::MAX as usize { return None; }
                if matches!(kind, S::Greedy) {
                    if operation.inputs.len() != 1 { return None; }
                    population.copy(OperationEvent::cpu_arg_reduce_layout(rank, columns, 1, false)?, 1)?;
                    population.copy(OperationEvent::cpu_squeeze_layout(rank, false)?, 1)?;
                } else {
                    let key = operation.inputs.get(1)?;
                    if operation.inputs.len() != 2 || key.dtype() != WorkspaceDtype::Uint32 || key.shape() != [2] { return None; }
                    population = numerical_random::categorical_draw(rank, columns)?;
                    // Six uniform row births, four Gumbel unary rows, one logit
                    // addition. Scalar range + four actual seeds stay separate;
                    // ArgReduce's single U32 result is the declared output.
                    let row = mechanism.allocation.fixed_buffer_capacity((columns as u64).checked_mul(4)?).ok()?;
                    scratch_bytes = row.checked_mul(11)?.checked_add(scalar.checked_mul(5)?)?;
                    seeds = 4;
                }
            }
            _ => return None,
        }
        let parts = [size_of::<WorkspaceOperationView<'_>>(), size_of::<WorkspaceLayoutView<'_>>() * 3,
            size_of::<CpuPopulation>(), size_of::<OperationPlan>(), size_of::<Option<OperationPlan>>(),
            size_of::<facts::FactResult<Option<OperationPlan>>>(), size_of::<Option<usize>>(),
            size_of::<(usize, usize, u64, u64, u64, usize)>(),
            size_of::<CpuCopyEvalLayout>(), size_of::<Option<CpuCopyEvalLayout>>(),
            size_of::<Option<WorkspaceRepresentation>>(), size_of::<std::slice::Iter<'_, i32>>(),
            size_of::<(&WorkspaceSamplingOperation, MlxCpuWorkspaceMechanisms)>(),
            crate::backend::random::standard_sampling_control_bytes()?,
        ];
        population.controls = parts.into_iter().try_fold(population.controls.checked_add(size_of_val(&parts))?, usize::checked_add)?;
        Some(OperationPlan { dtype: WorkspaceFloatingType::Float32, population, alias_input,
            output_bytes, scratch_bytes, rank, parameter_shells: 0, seeds, validations: 0 })
    })();
    Ok(source)
}

