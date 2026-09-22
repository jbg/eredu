//! Source census of the ordinary fast::layer_norm CPU fallback.
use super::*;

pub(super) fn inspect(
    operation: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
    weight: bool,
    bias: bool,
) -> facts::FactResult<Option<OperationPlan>> {
    if operation.outputs.len() != 1
        || operation.inputs.len() != 1 + usize::from(weight) + usize::from(bias)
    {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU LayerNorm operand population differs",
        ));
    }
    let input = operation.inputs.get(0).expect("LayerNorm input");
    let output = operation.outputs.get(0).expect("LayerNorm output");
    let rank = input.shape().len();
    if !(1..=4).contains(&rank) || input.shape().iter().any(|&n| n <= 0) {
        return Ok(None);
    }
    let width = usize::try_from(input.shape()[rank - 1])?;
    if input.shape() != output.shape()
        || operation
            .inputs
            .iter()
            .skip(1)
            .any(|v| v.shape() != [input.shape()[rank - 1]])
    {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU LayerNorm affine/output geometry differs",
        ));
    }
    if output.dtype() != WorkspaceDtype::Float32
        || operation.inputs.iter().any(|v| {
            v.dtype() != WorkspaceDtype::Float32
                || !v.representation().is_some_and(|r| r.row_contiguous())
        })
    {
        return Ok(None);
    }
    let Some(dtype) = super::super::super::representation::output(operation, 0).map(|v| v.dtype())
    else {
        return Ok(None);
    };
    let native_dtype = |dtype| match dtype {
        WorkspaceFloatingType::Float32 => Dtype::Float32,
        WorkspaceFloatingType::Float16 => Dtype::Float16,
        WorkspaceFloatingType::Bfloat16 => Dtype::Bfloat16,
    };
    let input_native = native_dtype(input.representation().expect("qualified source").dtype());
    let native = native_dtype(dtype);
    let count = usize::try_from(input.elements()?)?;
    if count > i32::MAX as usize {
        return Ok(None);
    }
    let rows = count / width;
    let source = (|| {
        let mut p = Population::default();
        // Both optional affine values are cast by fast::layer_norm before the
        // fallback. Absent values are the actual eager one/zero scalar inputs.
        for value in operation.inputs.iter().skip(1) {
            p.cast(
                native_dtype(value.representation()?.dtype()),
                native,
                1,
                width,
                Buffer::Vector,
            )?;
        }
        p.cast(input_native, Dtype::Float32, rank, count, Buffer::Full)?;
        mean(&mut p, rank, width, rows, count)?;
        p.binary(
            CpuBinaryOperation::Subtract,
            Dtype::Float32,
            rank,
            count,
            Buffer::Full,
            Buffer::Row,
            rank,
            rows,
            Buffer::Full,
        )?;
        p.unary(CpuUnaryOperation::Square, rank, Buffer::Full)?;
        mean(&mut p, rank, width, rows, count)?;
        p.binary(
            CpuBinaryOperation::Add,
            Dtype::Float32,
            rank,
            rows,
            Buffer::Row,
            Buffer::Scalar,
            0,
            1,
            Buffer::Row,
        )?;
        p.unary(CpuUnaryOperation::Rsqrt, rank, Buffer::Row)?;
        p.binary(
            CpuBinaryOperation::Multiply,
            Dtype::Float32,
            rank,
            count,
            Buffer::Full,
            Buffer::Row,
            rank,
            rows,
            Buffer::Full,
        )?;
        p.cast(Dtype::Float32, native, rank, count, Buffer::Full)?;
        if weight {
            p.binary(
                CpuBinaryOperation::Multiply,
                native,
                rank,
                count,
                Buffer::Full,
                Buffer::Vector,
                1,
                width,
                Buffer::Full,
            )?;
        }
        if bias {
            p.binary(
                CpuBinaryOperation::Add,
                native,
                rank,
                count,
                Buffer::Full,
                Buffer::Vector,
                1,
                width,
                Buffer::Full,
            )?;
        }
        p.native.controls = p
            .native
            .controls
            .checked_add(OperationEvent::cpu_layer_norm_fallback_control_bytes(
                native, rank, width, rows,
            )?)?
            .checked_add(safemlx::Stream::device_type_control_bytes()?)?;
        Some(p)
    })();
    let Some(mut source) = source else {
        return Ok(None);
    };
    if source
        .births
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        != Some(source.native.births)
    {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU LayerNorm source population differs",
        ));
    }
    // The same worker creates two mean divisors, epsilon, and each absent
    // affine default. F32 envelopes also cover reduced-precision destinations.
    let seeds = 3 + usize::from(!weight) + usize::from(!bias);
    let full = mechanism
        .allocation
        .fixed_buffer_capacity(facts::mul(input.elements()?, 4)?)?;
    let row = mechanism
        .allocation
        .fixed_buffer_capacity(facts::mul(u64::try_from(rows)?, 4)?)?;
    let vector = mechanism
        .allocation
        .fixed_buffer_capacity(facts::mul(u64::try_from(width)?, 4)?)?;
    let scalar = mechanism.allocation.fixed_buffer_capacity(4)?;
    let total = source
        .births
        .into_iter()
        .zip([full, row, vector, scalar])
        .try_fold(
            facts::mul(scalar, u64::try_from(seeds)?)?,
            |n, (births, bytes)| facts::add(n, facts::mul(bytes, u64::try_from(births)?)?),
        )?;
    let scratch_bytes = total
        .checked_sub(full)
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    let frames = [
        size_of::<Population>() * 3,
        size_of::<Option<Population>>(),
        size_of::<OperationPlan>(),
        size_of::<Option<OperationPlan>>(),
        size_of::<WorkspaceOperationView<'_>>(),
        size_of::<WorkspaceLayoutView<'_>>() * 3,
        size_of::<MlxCpuWorkspaceMechanisms>(),
        size_of::<WorkspaceFloatingType>(),
        size_of::<Dtype>() * 3,
        size_of::<Option<WorkspaceRepresentation>>(),
        size_of::<WorkspaceRepresentation>(),
        size_of::<(
            CpuBinaryOperation,
            Dtype,
            usize,
            usize,
            Buffer,
            Buffer,
            usize,
            usize,
            Buffer,
        )>(),
        size_of::<(Dtype, Dtype, usize, usize, Buffer)>(),
        size_of::<(CpuUnaryOperation, usize, Buffer)>(),
        size_of::<(&mut Population, CpuCopyEvalLayout, Option<Buffer>)>(),
        size_of::<(&mut Population, Buffer, usize)>(),
        size_of::<(&mut Population, usize, usize)>(),
        size_of::<(&mut Population, usize, usize, usize, usize)>(),
        size_of::<Option<Buffer>>(),
        size_of::<Option<()>>(),
        size_of::<CpuCopyEvalLayout>(),
        size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<safemlx::CpuBinaryEvalLayout>(),
        size_of::<Option<safemlx::CpuBinaryEvalLayout>>(),
        size_of::<safemlx::CpuUnaryEvalLayout>(),
        size_of::<Option<safemlx::CpuUnaryEvalLayout>>(),
        size_of::<(
            &crate::MlxTensor,
            Option<&crate::MlxTensor>,
            Option<&crate::MlxTensor>,
            f32,
            &safemlx::Stream,
        )>(),
        size_of::<(
            &safemlx::Array,
            Option<&safemlx::Array>,
            Option<&safemlx::Array>,
            f32,
            &safemlx::Stream,
        )>(),
        size_of::<safemlx::Array>() * 2,
        size_of::<Result<safemlx::Array, safemlx::error::Exception>>(),
        size_of::<Result<crate::MlxTensor, eredu_nn::Error>>(),
        size_of::<usize>() * 9,
        size_of::<u64>() * 5,
        size_of::<[usize; 4]>(),
        size_of::<[u64; 4]>(),
        size_of::<std::iter::Zip<std::array::IntoIter<usize, 4>, std::array::IntoIter<u64, 4>>>(),
        size_of::<std::slice::Iter<i32>>(),
        size_of::<bool>() * 2,
        size_of::<std::iter::Skip<eredu_nn::workspace::WorkspaceLayoutIter<'_>>>(),
    ];
    source.native.controls = frames.into_iter().try_fold(
        source
            .native
            .controls
            .checked_add(size_of_val(&frames))
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
        |n, bytes| {
            n.checked_add(bytes)
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)
        },
    )?;
    Ok(Some(OperationPlan {
        alias_input: None,
        dtype,
        population: source.native,
        output_bytes: full,
        scratch_bytes,
        rank,
        parameter_shells: usize::from(weight) + usize::from(bias),
        seeds,
        validations: 0,
    }))
}

fn mean(p: &mut Population, rank: usize, width: usize, rows: usize, count: usize) -> Option<()> {
    if width == 1 {
        p.cast(Dtype::Float32, Dtype::Float32, rank, count, Buffer::Full)?;
    } else {
        p.copy(
            OperationEvent::cpu_row_sum_layout(rank, width, rows, false)?,
            Some(Buffer::Row),
        )?;
    }
    p.binary(
        CpuBinaryOperation::Divide,
        Dtype::Float32,
        rank,
        rows,
        Buffer::Row,
        Buffer::Scalar,
        0,
        1,
        Buffer::Row,
    )
}

#[cfg(all(test, target_vendor = "apple", not(feature = "cuda")))]
#[path = "layer/tests.rs"]
mod tests;
