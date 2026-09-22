//! Integer-scalar equality follows native promotion before the Bool comparison.
use super::*;

pub(super) fn inspect(
    operation: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
) -> facts::FactResult<Option<OperationPlan>> {
    if !matches!(
        operation.kind,
        WorkspaceOperationKindView::Elementwise("equal_i32")
    ) {
        return Ok(None);
    }
    if operation.inputs.len() != 1 || operation.outputs.len() != 1 {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU scalar equality operand population differs",
        ));
    }
    let input = operation.inputs.get(0).expect("scalar equality input");
    let output = operation.outputs.get(0).expect("scalar equality output");
    let (source, promoted) = match input.dtype() {
        WorkspaceDtype::Int32 => (Dtype::Int32, Dtype::Int32),
        // Native promote_types(U32, I32) is I64, retaining all source integers.
        WorkspaceDtype::Uint32 => (Dtype::Uint32, Dtype::Int64),
        _ => return Ok(None),
    };
    if input.representation().is_some() || output.representation().is_some() {
        return Ok(None);
    }
    if output.dtype() != WorkspaceDtype::Bool || input.shape() != output.shape() {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU scalar equality output geometry differs",
        ));
    }
    let rank = input.shape().len();
    if rank > 4 || input.shape().iter().any(|&n| n <= 0) {
        return Ok(None);
    }
    let count = usize::try_from(input.elements()?)?;
    if count > i32::MAX as usize {
        return Ok(None);
    }
    let source_population = (|| {
        let mut p = CpuPopulation::default();
        p.copy(
            OperationEvent::cpu_cast_layout(source, promoted, rank, count, false)?,
            1,
        )?;
        p.copy(
            OperationEvent::cpu_cast_layout(Dtype::Int32, promoted, 0, 1, false)?,
            1,
        )?;
        if rank != 0 {
            p.copy(
                OperationEvent::cpu_broadcast_alias_layout(0, rank, false)?,
                1,
            )?;
        }
        p.binary(OperationEvent::cpu_binary_layout(
            CpuBinaryOperation::Equal,
            promoted,
            rank,
            count,
            false,
        )?)?;
        Some(p)
    })();
    let Some(mut population) = source_population else {
        return Ok(None);
    };
    let mut scratch_bytes = mechanism.allocation.fixed_buffer_capacity(4)?;
    if source != promoted {
        scratch_bytes = facts::add(
            scratch_bytes,
            mechanism
                .allocation
                .fixed_buffer_capacity(facts::mul(input.elements()?, 8)?)?,
        )?;
        scratch_bytes = facts::add(
            scratch_bytes,
            mechanism.allocation.fixed_buffer_capacity(8)?,
        )?;
    }
    let frames = [
        size_of::<WorkspaceOperationView<'_>>(),
        size_of::<MlxCpuWorkspaceMechanisms>(),
        size_of::<WorkspaceLayoutView<'_>>() * 2,
        size_of::<Option<WorkspaceLayoutView<'_>>>(),
        size_of::<(Dtype, Dtype)>(),
        size_of::<Dtype>() * 2,
        size_of::<WorkspaceDtype>(),
        size_of::<usize>() * 2,
        size_of::<u64>(),
        size_of::<CpuPopulation>() * 2,
        size_of::<Option<CpuPopulation>>(),
        size_of::<CpuCopyEvalLayout>(),
        size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<safemlx::CpuBinaryEvalLayout>(),
        size_of::<Option<safemlx::CpuBinaryEvalLayout>>(),
        size_of::<Option<()>>(),
        size_of::<OperationPlan>(),
        size_of::<facts::FactResult<Option<OperationPlan>>>(),
        size_of::<std::slice::Iter<'_, i32>>(),
        size_of::<Option<&i32>>(),
        size_of::<(&crate::MlxTensor, i32, &safemlx::Stream)>(),
        size_of::<safemlx::Array>(),
        size_of::<Result<safemlx::Array, safemlx::error::Exception>>(),
        size_of::<Result<crate::MlxTensor, eredu_nn::Error>>(),
    ];
    population.controls = frames.into_iter().try_fold(
        population
            .controls
            .checked_add(size_of_val(&frames))
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
        |n, bytes| {
            n.checked_add(bytes)
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)
        },
    )?;
    Ok(Some(OperationPlan {
        dtype: WorkspaceFloatingType::Float32,
        population,
        alias_input: None,
        rank,
        parameter_shells: 0,
        seeds: 1,
        validations: 0,
        scratch_bytes,
        output_bytes: mechanism
            .allocation
            .fixed_buffer_capacity(output.bytes()?)?,
    }))
}

#[cfg(all(test, target_vendor = "apple", not(feature = "cuda")))]
mod tests;
