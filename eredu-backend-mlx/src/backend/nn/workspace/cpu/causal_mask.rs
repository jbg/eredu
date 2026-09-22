//! Existing exact I32 coordinate and Bool mask worker, including cache offset.
use super::*;
use safemlx::Dtype;
fn binary(
    p: &mut CpuPopulation,
    kind: CpuBinaryOperation,
    dtype: Dtype,
    rank: usize,
    elements: usize,
    left: (usize, usize),
    right: (usize, usize),
) -> Option<()> {
    for (source_rank, count) in [left, right] {
        p.copy(
            OperationEvent::cpu_cast_layout(dtype, dtype, source_rank, count, false)?,
            1,
        )?;
        p.copy(
            OperationEvent::cpu_broadcast_alias_layout(source_rank, rank, false)?,
            1,
        )?;
    }
    p.binary(OperationEvent::cpu_binary_layout(
        kind, dtype, rank, elements, false,
    )?)
}
pub(super) fn inspect(
    operation: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
) -> facts::FactResult<Option<OperationPlan>> {
    let WorkspaceOperationKindView::CausalMask(geometry) = operation.kind else {
        return Ok(None);
    };
    if !operation.inputs.is_empty() || operation.outputs.len() != 1 {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU mask input/output population differs",
        ));
    }
    let output = operation.outputs.get(0).expect("one mask output");
    if output.dtype() != WorkspaceDtype::Bool
        || output.shape() != [geometry.sequence(), geometry.keys()]
    {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU mask differs from declared integer coordinates",
        ));
    }
    if geometry.sequence() <= 0 || geometry.keys() <= 0 {
        return Ok(None);
    }
    let q = usize::try_from(geometry.sequence())?;
    let k = usize::try_from(geometry.keys())?;
    let Some(elements) = q.checked_mul(k).filter(|&n| n <= i32::MAX as usize) else {
        return Ok(None);
    };
    let source = (|| {
        let mut p = CpuPopulation::default();
        p.copy(
            OperationEvent::cpu_arange_int_layout(safemlx::Dtype::Int32, k, false)?,
            0,
        )?;
        p.copy(
            OperationEvent::cpu_arange_int_layout(safemlx::Dtype::Int32, q, false)?,
            0,
        )?;
        p.copy(OperationEvent::cpu_reshape_alias_layout(1, 2, false)?, 1)?;
        p.copy(OperationEvent::cpu_reshape_alias_layout(1, 2, false)?, 1)?;
        binary(
            &mut p,
            CpuBinaryOperation::GreaterEqual,
            Dtype::Int32,
            2,
            elements,
            (2, q),
            (2, k),
        )?;
        if geometry.max_past().is_some() {
            binary(
                &mut p,
                CpuBinaryOperation::Subtract,
                Dtype::Int32,
                2,
                q,
                (2, q),
                (0, 1),
            )?;
            binary(
                &mut p,
                CpuBinaryOperation::GreaterEqual,
                Dtype::Int32,
                2,
                elements,
                (2, k),
                (2, q),
            )?;
            binary(
                &mut p,
                CpuBinaryOperation::LogicalAnd,
                Dtype::Bool,
                2,
                elements,
                (2, elements),
                (2, elements),
            )?;
        }
        Some(p)
    })();
    let Some(mut population) = source else {
        return Ok(None);
    };
    let seeds = usize::from(geometry.max_past().is_some());
    // All I32 coordinates and Bool intermediates are no larger than the actual
    // nonempty mask matrix. Each possible backing survives until completion;
    // its four-byte envelope also covers the one-byte comparison results.
    let capacity = mechanism
        .allocation
        .fixed_buffer_capacity(facts::mul(u64::try_from(elements)?, 4)?)?;
    let output_bytes = mechanism
        .allocation
        .fixed_buffer_capacity(u64::try_from(elements)?)?;
    let total = facts::add(
        facts::mul(capacity, u64::try_from(population.births)?)?,
        facts::mul(
            mechanism.allocation.fixed_buffer_capacity(4)?,
            u64::try_from(seeds)?,
        )?,
    )?;
    let scratch_bytes = total
        .checked_sub(output_bytes)
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    let frames = [
        size_of::<CpuPopulation>() * 2,
        size_of::<Option<CpuPopulation>>(),
        size_of::<OperationPlan>(),
        size_of::<Option<OperationPlan>>(),
        size_of::<WorkspaceOperationView<'_>>(),
        size_of::<WorkspaceLayoutView<'_>>(),
        size_of::<MlxCpuWorkspaceMechanisms>(),
        size_of::<eredu_nn::operation_geometry::CausalMaskGeometry>(),
        size_of::<(WorkspaceOperationView<'_>, MlxCpuWorkspaceMechanisms)>(),
        size_of::<usize>() * 7,
        size_of::<u64>() * 4,
        size_of::<Option<usize>>(),
        size_of::<[i32; 2]>(),
        size_of::<(
            &mut CpuPopulation,
            CpuBinaryOperation,
            Dtype,
            usize,
            usize,
            (usize, usize),
            (usize, usize),
        )>(),
        size_of::<[(usize, usize); 2]>(),
        size_of::<std::array::IntoIter<(usize, usize), 2>>(),
        size_of::<CpuCopyEvalLayout>(),
        size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<safemlx::CpuBinaryEvalLayout>(),
        size_of::<Option<safemlx::CpuBinaryEvalLayout>>(),
        size_of::<Option<()>>(),
        // Shared Rust create_causal_mask worker and its fixed arange/reshape
        // arguments/results. The no-lengths source never builds a host mask.
        size_of::<(
            i32,
            Option<i32>,
            Option<i32>,
            Option<safemlx::Array>,
            &safemlx::Stream,
        )>(),
        size_of::<safemlx::Array>() * 8,
        size_of::<Option<safemlx::Array>>(),
        size_of::<Result<safemlx::Array, safemlx::error::Exception>>(),
        size_of::<Result<eredu_nn::operation_geometry::CausalMaskGeometry, eredu_nn::Error>>(),
        size_of::<[i32; 2]>() * 2,
        size_of::<Option<i32>>() * 3,
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
        output_bytes,
        scratch_bytes,
        rank: 2,
        parameter_shells: 0,
        alias_input: None,
        seeds,
        validations: 0,
    }))
}
