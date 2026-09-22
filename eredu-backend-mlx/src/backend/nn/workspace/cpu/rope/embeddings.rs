//! The shared explicit cosine/sine application, using its actual CPU workers.
use super::*;
use eredu_nn::workspace::WorkspaceLayoutIter;

pub(super) fn inspect(
    operation: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
) -> facts::FactResult<Option<OperationPlan>> {
    let WorkspaceOperationKindView::Rotary(spec, None) = operation.kind else {
        return Ok(None);
    };
    if spec.arithmetic != RotaryArithmetic::Native
        || !matches!(spec.algorithm, RotaryAlgorithm::Default)
        || spec.traditional
    {
        return Ok(None);
    }
    if operation.inputs.len() != 3 || operation.outputs.len() != 1 {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU explicit rotary operand population differs",
        ));
    }
    let input = operation.inputs.get(0).expect("explicit rotary input");
    let output = operation.outputs.get(0).expect("explicit rotary output");
    let shape = input.shape();
    if shape.len() != 4 || shape.iter().any(|&n| n <= 0) {
        return Ok(None);
    }
    if spec.dimensions <= 0
        || spec.dimensions % 2 != 0
        || spec.dimensions != shape[3]
        || input.shape() != output.shape()
        || !spec.base.is_finite()
        || spec.base <= 0.0
    {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU explicit rotary geometry differs",
        ));
    }
    if output.dtype() != WorkspaceDtype::Float32 {
        return Ok(None);
    }
    for value in operation.inputs.iter() {
        if value.dtype() != WorkspaceDtype::Float32
            || !value.representation().is_some_and(|r| {
                r.dtype() == WorkspaceFloatingType::Float32 && r.last_axis_contiguous()
            })
        {
            return Ok(None);
        }
    }
    for embedding in operation
        .inputs
        .slice(1..3)
        .expect("cosine and sine")
        .iter()
    {
        if !embedding
            .representation()
            .is_some_and(|r| r.row_contiguous())
        {
            return Ok(None);
        }
        let s = embedding.shape();
        if !matches!(s.len(), 2 | 3)
            || s.iter().any(|&n| n <= 0)
            || (s.len() == 3 && s[0] != 1 && s[0] != shape[0])
            || (s[s.len() - 2] != 1 && s[s.len() - 2] != shape[2])
            || (s[s.len() - 1] != 1 && s[s.len() - 1] != shape[3])
        {
            return Err(MlxWorkspaceFactError::descriptor(
                "CPU explicit rotary embedding broadcast differs",
            ));
        }
    }
    let count = usize::try_from(input.elements()?)?;
    if count > i32::MAX as usize {
        return Ok(None);
    }
    let source = (|| {
        let mut p = CpuPopulation::default();
        for embedding in operation.inputs.slice(1..3)?.iter() {
            let rank = embedding.shape().len();
            let elements = usize::try_from(embedding.elements().ok()?).ok()?;
            if rank == 2 {
                p.copy(
                    OperationEvent::cpu_expand_dims_alias_layout(2, 3, false)?,
                    1,
                )?;
            }
            cast(&mut p, Dtype::Float32, 3, elements)?;
            // The actual tuple index performs Slice then Reshape to insert H.
            p.copy(OperationEvent::cpu_slice_layout(3, false, false)?, 1)?;
            p.copy(OperationEvent::cpu_reshape_alias_layout(3, 4, false)?, 1)?;
        }
        for _ in 0..2 {
            p.copy(OperationEvent::cpu_slice_layout(4, false, false)?, 1)?;
            p.copy(OperationEvent::cpu_reshape_alias_layout(4, 4, false)?, 1)?;
        }
        binary(
            &mut p,
            CpuBinaryOperation::Multiply,
            4,
            count / 2,
            (4, count / 2, Dtype::Float32),
            (0, 1, Dtype::Float32),
        )?;
        cast(&mut p, Dtype::Float32, 4, count / 2)?;
        cast(&mut p, Dtype::Float32, 4, count / 2)?;
        p.concatenate(
            OperationEvent::cpu_concatenate_layout(Dtype::Float32, 4, count / 2, count / 2, false)?,
            2,
        )?;
        for embedding in operation.inputs.slice(1..3)?.iter() {
            let elements = usize::try_from(embedding.elements().ok()?).ok()?;
            binary(
                &mut p,
                CpuBinaryOperation::Multiply,
                4,
                count,
                (4, count, Dtype::Float32),
                (4, elements, Dtype::Float32),
            )?;
        }
        binary(
            &mut p,
            CpuBinaryOperation::Add,
            4,
            count,
            (4, count, Dtype::Float32),
            (4, count, Dtype::Float32),
        )?;
        p.controls = p
            .controls
            .checked_add(crate::backend::nn::attention::apply_rotary_embeddings_control_bytes()?)?;
        Some(p)
    })();
    let Some(mut population) = source else {
        return Ok(None);
    };
    let full = mechanism
        .allocation
        .fixed_buffer_capacity(facts::mul(input.elements()?, 4)?)?;
    // Casts, products, and concatenation all fit this actual full input extent.
    // Their simultaneous lifetimes receive no intermediate retirement credit.
    let scratch_bytes = facts::add(
        facts::mul(
            full,
            u64::try_from(
                population
                    .births
                    .checked_sub(1)
                    .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
            )?,
        )?,
        mechanism.allocation.fixed_buffer_capacity(4)?,
    )?;
    let frames = [
        size_of::<WorkspaceOperationView<'_>>(),
        size_of::<MlxCpuWorkspaceMechanisms>(),
        size_of::<eredu_nn::RotarySpec>(),
        size_of::<WorkspaceLayoutView<'_>>() * 4,
        size_of::<WorkspaceLayoutIter<'_>>() * 3,
        size_of::<Option<WorkspaceLayoutView<'_>>>(),
        size_of::<std::slice::Iter<'_, i32>>() * 2,
        size_of::<Option<&i32>>(),
        size_of::<CpuPopulation>() * 2,
        size_of::<Option<CpuPopulation>>(),
        size_of::<OperationPlan>(),
        size_of::<Option<OperationPlan>>(),
        size_of::<usize>() * 4,
        size_of::<u64>() * 2,
        size_of::<&[i32]>() * 2,
        size_of::<std::ops::Range<usize>>(),
        size_of::<std::ops::Range<i32>>(),
        size_of::<(
            &mut CpuPopulation,
            CpuBinaryOperation,
            usize,
            usize,
            (usize, usize, Dtype),
            (usize, usize, Dtype),
        )>(),
        size_of::<(&mut CpuPopulation, Dtype, usize, usize)>(),
        size_of::<[(usize, usize, Dtype); 2]>(),
        size_of::<std::array::IntoIter<(usize, usize, Dtype), 2>>(),
        size_of::<CpuCopyEvalLayout>(),
        size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<safemlx::CpuBinaryEvalLayout>(),
        size_of::<Option<safemlx::CpuBinaryEvalLayout>>(),
        size_of::<Option<()>>(),
        size_of::<Option<u64>>(),
        size_of::<Option<usize>>(),
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
        output_bytes: full,
        scratch_bytes,
        rank: 4,
        parameter_shells: 0,
        alias_input: None,
        seeds: 1,
        validations: 0,
    }))
}

#[cfg(all(test, target_vendor = "apple", not(feature = "cuda")))]
mod tests;
