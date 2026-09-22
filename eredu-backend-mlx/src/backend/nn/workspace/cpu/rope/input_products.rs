//! The existing ElementwiseRotary worker selected by InputProducts arithmetic.
use super::*;

pub(super) fn inspect(
    operation: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
    spec: eredu_nn::RotarySpec,
    offset: i32,
) -> facts::FactResult<Option<OperationPlan>> {
    if spec.traditional
        || !matches!(
            spec.algorithm,
            RotaryAlgorithm::Default
                | RotaryAlgorithm::Linear { .. }
                | RotaryAlgorithm::Yarn { .. }
        )
    {
        // Other constructors may retain lazy frequency ancestors; those need
        // their own actual source census before this worker can consume them.
        return Ok(None);
    }
    spec.algorithm.validate_fixed()?;
    if spec.dimensions <= 0
        || spec.dimensions % 2 != 0
        || !spec.base.is_finite()
        || spec.base <= 0.0
    {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU explicit RoPE scalar geometry differs",
        ));
    }
    if offset < 0 || operation.inputs.len() != 1 || operation.outputs.len() != 1 {
        return Ok(None);
    }
    let input = operation.inputs.get(0).expect("explicit RoPE source");
    let output = operation.outputs.get(0).expect("explicit RoPE result");
    if input.shape().len() != 4
        || input.shape().iter().any(|&n| n <= 0)
        || spec.dimensions != input.shape()[3]
        || input.dtype() != WorkspaceDtype::Float32
        || output.dtype() != WorkspaceDtype::Float32
    {
        return Ok(None);
    }
    let Some(representation) = input
        .representation()
        .filter(|r| r.dtype() == WorkspaceFloatingType::Float32 && r.last_axis_contiguous())
    else {
        return Ok(None);
    };
    if input.shape() != output.shape() {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU explicit RoPE output geometry differs",
        ));
    }
    let count = usize::try_from(input.elements()?)?;
    if count > i32::MAX as usize || offset.checked_add(input.shape()[2]).is_none() {
        return Ok(None);
    }
    let positions = usize::try_from(input.shape()[2])?;
    let half = usize::try_from(spec.dimensions)? / 2;
    let theta = positions
        .checked_mul(half)
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    let flattened = [
        input.shape()[0]
            .checked_mul(input.shape()[1])
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
        input.shape()[2],
        input.shape()[3],
    ];
    let source = (|| {
        let mut p = CpuPopulation::default();
        // Exact strides determine whether flattening the two leading axes
        // aliases or copies. Last-axis contiguity alone cannot prove an alias.
        let strides = super::super::views::physical_strides(input, representation)?;
        p.copy(
            OperationEvent::cpu_reshape_layout(input.shape(), &strides, &flattened, false)?,
            1,
        )?;
        // position_rows forms absolute integer coordinates before F32 casting;
        // it never adds an already-rounded floating offset to a zero-based range.
        p.copy(
            OperationEvent::cpu_arange_int_layout(safemlx::Dtype::Int32, positions, false)?,
            0,
        )?;
        cast(&mut p, Dtype::Int32, 1, positions)?;
        p.copy(
            OperationEvent::cpu_expand_dims_alias_layout(1, 2, false)?,
            1,
        )?;
        binary(
            &mut p,
            CpuBinaryOperation::Multiply,
            2,
            theta,
            (2, positions, Dtype::Float32),
            (1, half, Dtype::Float32),
        )?;
        for kind in [CpuUnaryOperation::Cosine, CpuUnaryOperation::Sine] {
            unary(&mut p, kind, 2, theta)?;
            binary(
                &mut p,
                CpuBinaryOperation::Multiply,
                2,
                theta,
                (2, theta, Dtype::Float32),
                (0, 1, Dtype::Float32),
            )?;
            cast(&mut p, Dtype::Float32, 2, theta)?;
        }
        // The original tuple-index worker makes the full rotary view followed
        // by its two half slices. No selected axis or gather exists here.
        for _ in 0..3 {
            p.copy(OperationEvent::cpu_slice_layout(3, false, false)?, 1)?;
        }
        for combine in [CpuBinaryOperation::Subtract, CpuBinaryOperation::Add] {
            for _ in 0..2 {
                binary(
                    &mut p,
                    CpuBinaryOperation::Multiply,
                    3,
                    count / 2,
                    (3, count / 2, Dtype::Float32),
                    (2, theta, Dtype::Float32),
                )?;
            }
            binary(
                &mut p,
                combine,
                3,
                count / 2,
                (3, count / 2, Dtype::Float32),
                (3, count / 2, Dtype::Float32),
            )?;
        }
        cast(&mut p, Dtype::Float32, 3, count / 2)?;
        cast(&mut p, Dtype::Float32, 3, count / 2)?;
        p.concatenate(
            OperationEvent::cpu_concatenate_layout(Dtype::Float32, 3, count / 2, count / 2, false)?,
            2,
        )?;
        p.copy(OperationEvent::cpu_reshape_alias_layout(3, 4, false)?, 1)?;
        p.hidden_leaves = 1; // Exact uploaded inverse-frequency source, already retained by the module.
        p.controls = p
            .controls
            .checked_add(super::super::views::physical_stride_control_bytes()?)?
            .checked_add(crate::backend::nn::rope::rotary_control_bytes(true, 1)?)?;
        Some(p)
    })();
    let Some(mut population) = source else {
        return Ok(None);
    };
    let full = mechanism
        .allocation
        .fixed_buffer_capacity(facts::mul(input.elements()?, 4)?)?;
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
        size_of::<CpuPopulation>() * 2,
        size_of::<Option<CpuPopulation>>(),
        size_of::<OperationPlan>(),
        size_of::<Option<OperationPlan>>(),
        size_of::<WorkspaceOperationView<'_>>(),
        size_of::<WorkspaceLayoutView<'_>>() * 2,
        size_of::<MlxCpuWorkspaceMechanisms>(),
        size_of::<eredu_nn::RotarySpec>(),
        size_of::<i32>(),
        size_of::<usize>() * 4,
        size_of::<[i32; 3]>(),
        size_of::<[i64; 4]>(),
        size_of::<WorkspaceRepresentation>(),
        size_of::<Option<WorkspaceRepresentation>>(),
        size_of::<(
            &mut CpuPopulation,
            CpuBinaryOperation,
            usize,
            usize,
            (usize, usize, Dtype),
            (usize, usize, Dtype),
        )>(),
        size_of::<(&mut CpuPopulation, CpuUnaryOperation, usize, usize)>(),
        size_of::<(&mut CpuPopulation, Dtype, usize, usize)>(),
        size_of::<[(usize, usize, Dtype); 2]>(),
        size_of::<std::array::IntoIter<(usize, usize, Dtype), 2>>(),
        size_of::<[CpuUnaryOperation; 2]>(),
        size_of::<std::array::IntoIter<CpuUnaryOperation, 2>>(),
        size_of::<[CpuBinaryOperation; 2]>(),
        size_of::<std::array::IntoIter<CpuBinaryOperation, 2>>(),
        size_of::<std::ops::Range<i32>>(),
        size_of::<CpuCopyEvalLayout>(),
        size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<safemlx::CpuUnaryEvalLayout>(),
        size_of::<Option<safemlx::CpuUnaryEvalLayout>>(),
        size_of::<safemlx::CpuBinaryEvalLayout>(),
        size_of::<Option<safemlx::CpuBinaryEvalLayout>>(),
        size_of::<Option<()>>(),
        size_of::<std::slice::Iter<i32>>(),
        size_of::<u64>() * 2,
        size_of::<(
            WorkspaceOperationView<'_>,
            MlxCpuWorkspaceMechanisms,
            eredu_nn::RotarySpec,
            i32,
        )>(),
        size_of::<(
            &crate::backend::nn::rope::ElementwiseRotary,
            &safemlx::Array,
            i32,
            &safemlx::Stream,
        )>(),
        size_of::<(&safemlx::Array, &[i32], i32, i32)>(),
        size_of::<[safemlx::Array; 8]>(),
        size_of::<Result<safemlx::Array, safemlx::error::Exception>>(),
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
        alias_input: None,
        output_bytes: full,
        scratch_bytes,
        rank: 4,
        parameter_shells: 0,
        seeds: 1,
        validations: 0,
    }))
}
