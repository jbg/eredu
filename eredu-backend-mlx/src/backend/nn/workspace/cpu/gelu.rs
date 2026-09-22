//! Shared exact and approximate GELU sources with their actual scalar promotion.
use super::*;
use safemlx::{CpuUnaryOperation, Dtype};
mod approximate;
#[cfg(all(test, target_vendor = "apple", not(feature = "cuda")))]
mod tests;
pub(super) fn inspect(
    operation: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
) -> facts::FactResult<Option<OperationPlan>> {
    if let Some(plan) = approximate::inspect(operation, mechanism)? {
        return Ok(Some(plan));
    }
    if !matches!(
        operation.kind,
        WorkspaceOperationKindView::Elementwise("gelu")
    ) {
        return Ok(None);
    }
    if operation.inputs.len() != 1 || operation.outputs.len() != 1 {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU GELU population differs",
        ));
    }
    let input = operation.inputs.get(0).expect("GELU input");
    let output = operation.outputs.get(0).expect("GELU output");
    if input.shape() != output.shape() {
        return Err(MlxWorkspaceFactError::descriptor("CPU GELU shape differs"));
    }
    let rank = input.shape().len();
    if rank > 4
        || input.shape().iter().any(|&n| n <= 0)
        || input.dtype() != WorkspaceDtype::Float32
        || output.dtype() != WorkspaceDtype::Float32
        || !input.representation().is_some_and(|r| r.row_contiguous())
    {
        return Ok(None);
    }
    let native_input = match input.representation().map(|r| r.dtype()) {
        Some(WorkspaceFloatingType::Float32) => Dtype::Float32,
        Some(WorkspaceFloatingType::Bfloat16) => Dtype::Bfloat16,
        Some(WorkspaceFloatingType::Float16) => Dtype::Float16,
        _ => return Ok(None),
    };
    let count = usize::try_from(input.elements()?)?;
    if count > i32::MAX as usize {
        return Ok(None);
    }
    let source = (|| {
        let mut p = CpuPopulation::default();
        let mut scalar_births = 0usize;
        // x / sqrt(2), 1 + erf(...), x * (...), (...) / 2. Preserve both
        // divisions and the actual I32 one / F32 sqrt(2) / F32 two sources.
        let steps = [
            (CpuBinaryOperation::Divide, false, true),
            (CpuBinaryOperation::Add, true, false),
            (CpuBinaryOperation::Multiply, false, false),
            (CpuBinaryOperation::Divide, false, true),
        ];
        for (step, (kind, left_scalar, right_scalar)) in steps.into_iter().enumerate() {
            for (operand, scalar) in [left_scalar, right_scalar].into_iter().enumerate() {
                let (r, n) = if scalar { (0, 1) } else { (rank, count) };
                // x / sqrt(2) and x * (1 + erf) each cast the actual original
                // input. Every other floating operand is already F32.
                let from = if step == 1 && scalar {
                    Dtype::Int32
                } else if operand == 0 && (step == 0 || step == 2) {
                    native_input
                } else {
                    Dtype::Float32
                };
                let cast = OperationEvent::cpu_cast_layout(from, Dtype::Float32, r, n, false)?;
                if scalar {
                    scalar_births = scalar_births.checked_add(cast.backing_births())?;
                }
                p.copy(cast, 1)?;
                p.copy(
                    OperationEvent::cpu_broadcast_alias_layout(r, rank, false)?,
                    1,
                )?;
            }
            p.binary(OperationEvent::cpu_binary_layout(
                kind,
                Dtype::Float32,
                rank,
                count,
                false,
            )?)?;
            if step == 0 {
                p.copy(
                    OperationEvent::cpu_cast_layout(
                        Dtype::Float32,
                        Dtype::Float32,
                        rank,
                        count,
                        false,
                    )?,
                    1,
                )?;
                p.unary(OperationEvent::cpu_unary_layout(
                    CpuUnaryOperation::Erf,
                    Dtype::Float32,
                    rank,
                    false,
                )?)?;
            }
        }
        Some((p, scalar_births))
    })();
    let Some((mut population, scalar_births)) = source else {
        return Ok(None);
    };
    let full = mechanism
        .allocation
        .fixed_buffer_capacity(facts::mul(input.elements()?, 4)?)?;
    let scalar = mechanism.allocation.fixed_buffer_capacity(4)?;
    let full_births = population
        .births
        .checked_sub(scalar_births)
        .and_then(|n| n.checked_sub(1))
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    let scratch_bytes = facts::add(
        facts::mul(full, u64::try_from(full_births)?)?,
        facts::mul(
            scalar,
            u64::try_from(
                scalar_births
                    .checked_add(3)
                    .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
            )?,
        )?,
    )?;
    let frames = [
        size_of::<CpuPopulation>() * 3,
        size_of::<Option<(CpuPopulation, usize)>>(),
        size_of::<OperationPlan>(),
        size_of::<Option<OperationPlan>>(),
        size_of::<WorkspaceOperationView<'_>>(),
        size_of::<WorkspaceLayoutView<'_>>() * 2,
        size_of::<MlxCpuWorkspaceMechanisms>(),
        size_of::<[(CpuBinaryOperation, bool, bool); 4]>(),
        size_of::<std::iter::Enumerate<std::array::IntoIter<(CpuBinaryOperation, bool, bool), 4>>>(
        ),
        size_of::<(usize, (CpuBinaryOperation, bool, bool))>(),
        size_of::<[bool; 2]>(),
        size_of::<std::array::IntoIter<bool, 2>>(),
        size_of::<std::iter::Enumerate<std::array::IntoIter<bool, 2>>>(),
        size_of::<(usize, bool)>(),
        size_of::<(usize, usize)>(),
        size_of::<Dtype>() * 2,
        size_of::<CpuCopyEvalLayout>(),
        size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<safemlx::CpuUnaryEvalLayout>(),
        size_of::<Option<safemlx::CpuUnaryEvalLayout>>(),
        size_of::<safemlx::CpuBinaryEvalLayout>(),
        size_of::<Option<safemlx::CpuBinaryEvalLayout>>(),
        size_of::<usize>() * 6,
        size_of::<u64>() * 3,
        size_of::<std::slice::Iter<i32>>(),
        size_of::<(&safemlx::Array, &safemlx::Stream)>(),
        size_of::<safemlx::Array>() * 3,
        size_of::<Result<safemlx::Array, safemlx::error::Exception>>(),
        size_of::<bool>() * 2,
    ];
    population.controls = frames.into_iter().try_fold(
        population
            .controls
            .checked_add(size_of_val(&frames))
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
        |n, b| {
            n.checked_add(b)
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)
        },
    )?;
    Ok(Some(OperationPlan {
        dtype: WorkspaceFloatingType::Float32,
        population,
        output_bytes: full,
        scratch_bytes,
        rank,
        parameter_shells: 0,
        alias_input: None,
        seeds: 3,
        validations: 0,
    }))
}
