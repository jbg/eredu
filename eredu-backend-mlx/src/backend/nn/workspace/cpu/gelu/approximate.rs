//! Shared tanh GELU source, including the input-precision Power and F32 scalars.
use super::super::program::Program;
use super::*;

fn binary(
    p: &mut Program,
    kind: CpuBinaryOperation,
    dtype: Dtype,
    rank: usize,
    count: usize,
    left: (Dtype, usize, usize),
    right: (Dtype, usize, usize),
) -> Option<()> {
    for (from, source_rank, elements) in [left, right] {
        p.cast(from, dtype, source_rank, elements)?;
        p.broadcast(source_rank, rank)?;
    }
    let source = OperationEvent::cpu_binary_layout(kind, dtype, rank, count, false)?;
    p.bytes = p.bytes.checked_add(
        p.capacity(count, dtype)?
            .checked_mul(source.backing_births() as u64)?,
    )?;
    p.native.binary(source)?;
    p.controls(
        size_of::<(
            &mut Program,
            CpuBinaryOperation,
            Dtype,
            usize,
            usize,
            (Dtype, usize, usize),
            (Dtype, usize, usize),
            safemlx::CpuBinaryEvalLayout,
        )>()
        .checked_add(size_of::<std::array::IntoIter<(Dtype, usize, usize), 2>>())?,
    )
}

pub(super) fn inspect(
    operation: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
) -> facts::FactResult<Option<OperationPlan>> {
    if !matches!(
        operation.kind,
        WorkspaceOperationKindView::Elementwise("gelu_approximate")
    ) {
        return Ok(None);
    }
    if operation.inputs.len() != 1 || operation.outputs.len() != 1 {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU approximate GELU population differs",
        ));
    }
    let input = operation.inputs.get(0).expect("one GELU source");
    let output = operation.outputs.get(0).expect("one GELU result");
    if input.shape() != output.shape() {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU approximate GELU shape differs",
        ));
    }
    let rank = input.shape().len();
    if rank > 4
        || input.shape().iter().any(|&n| n <= 0)
        || input.dtype() != WorkspaceDtype::Float32
        || output.dtype() != WorkspaceDtype::Float32
    {
        return Ok(None);
    }
    let Some(representation) = input.representation().filter(|r| r.row_contiguous()) else {
        return Ok(None);
    };
    let native = match representation.dtype() {
        WorkspaceFloatingType::Float32 => Dtype::Float32,
        WorkspaceFloatingType::Float16 => Dtype::Float16,
        WorkspaceFloatingType::Bfloat16 => Dtype::Bfloat16,
    };
    let count = usize::try_from(input.elements()?)?;
    if count > i32::MAX as usize {
        return Ok(None);
    }
    let source = (|| {
        let mut p = Program::new(mechanism);
        let original = (native, rank, count);
        let full = (Dtype::Float32, rank, count);
        let scalar = (Dtype::Float32, 0, 1);
        // 0.5*x. All original x occurrences retain their own actual cast.
        p.scalar()?;
        binary(
            &mut p,
            CpuBinaryOperation::Multiply,
            Dtype::Float32,
            rank,
            count,
            scalar,
            original,
        )?;
        // sqrt(2/pi) is a native scalar operation, not a host constant fold.
        p.scalar()?;
        p.scalar()?;
        p.cast(Dtype::Float32, Dtype::Float32, 0, 1)?;
        p.unary(CpuUnaryOperation::Sqrt, Dtype::Float32, 0, 1)?;
        p.scalar()?;
        p.scalar()?;
        binary(
            &mut p,
            CpuBinaryOperation::Power,
            native,
            rank,
            count,
            original,
            (Dtype::Int32, 0, 1),
        )?;
        binary(
            &mut p,
            CpuBinaryOperation::Multiply,
            Dtype::Float32,
            rank,
            count,
            scalar,
            original,
        )?;
        binary(
            &mut p,
            CpuBinaryOperation::Add,
            Dtype::Float32,
            rank,
            count,
            original,
            full,
        )?;
        binary(
            &mut p,
            CpuBinaryOperation::Multiply,
            Dtype::Float32,
            rank,
            count,
            scalar,
            full,
        )?;
        p.cast(Dtype::Float32, Dtype::Float32, rank, count)?;
        p.unary(CpuUnaryOperation::Tanh, Dtype::Float32, rank, count)?;
        binary(
            &mut p,
            CpuBinaryOperation::Add,
            Dtype::Float32,
            rank,
            count,
            scalar,
            full,
        )?;
        binary(
            &mut p,
            CpuBinaryOperation::Multiply,
            Dtype::Float32,
            rank,
            count,
            full,
            full,
        )?;
        let frames = [
            size_of::<Program>() * 2,
            size_of::<Option<Program>>(),
            size_of::<OperationPlan>(),
            size_of::<Option<OperationPlan>>(),
            size_of::<WorkspaceOperationView<'_>>(),
            size_of::<WorkspaceLayoutView<'_>>() * 2,
            size_of::<WorkspaceRepresentation>(),
            size_of::<MlxCpuWorkspaceMechanisms>(),
            size_of::<Dtype>(),
            size_of::<usize>() * 2,
            size_of::<(Dtype, usize, usize)>() * 3,
            size_of::<safemlx::Array>() * 12,
            size_of::<(&safemlx::Array, &safemlx::Stream)>(),
            size_of::<Result<safemlx::Array, safemlx::error::Exception>>(),
        ];
        p.controls(
            frames
                .into_iter()
                .try_fold(size_of_val(&frames), usize::checked_add)?,
        )?;
        let output_bytes = p.capacity(count, Dtype::Float32)?;
        Some(OperationPlan {
            dtype: WorkspaceFloatingType::Float32,
            population: p.native,
            output_bytes,
            scratch_bytes: p.bytes.checked_sub(output_bytes)?,
            rank,
            parameter_shells: 0,
            alias_input: None,
            seeds: p.seeds,
            validations: 0,
        })
    })();
    Ok(source)
}
