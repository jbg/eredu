//! Exact source of mlx_weightless_rms_norm's unchanged half equation.
use super::*;

fn promoted_binary(
    p: &mut Population,
    kind: CpuBinaryOperation,
    left_dtype: Dtype,
    rank: usize,
    count: usize,
    left: Buffer,
    right: Buffer,
    right_rank: usize,
    right_count: usize,
    output: Buffer,
) -> Option<()> {
    p.cast(left_dtype, Dtype::Float32, rank, count, left)?;
    p.cast(
        Dtype::Float32,
        Dtype::Float32,
        right_rank,
        right_count,
        right,
    )?;
    p.alias(rank, rank)?;
    p.alias(right_rank, rank)?;
    let native = OperationEvent::cpu_binary_layout(kind, Dtype::Float32, rank, count, false)?;
    p.birth(output, native.backing_births())?;
    p.native.binary(native)
}
pub(super) fn source(
    dtype: Dtype,
    rank: usize,
    width: usize,
    rows: usize,
    count: usize,
) -> Option<Population> {
    if !matches!(dtype, Dtype::Float16 | Dtype::Bfloat16) {
        return None;
    }
    let mut p = Population::default();
    // square and mean stay in the actual input precision. Half Sum uses the
    // ordinary typed SIMD accumulator, not the learned RMS F32 cascade.
    let square = OperationEvent::cpu_unary_layout(CpuUnaryOperation::Square, dtype, rank, false)?;
    p.birth(Buffer::Full, square.backing_births())?;
    p.native.unary(square)?;
    if width == 1 {
        // sum_owned elides this Reduce and returns its same-shape cast alias.
        p.cast(dtype, dtype, rank, count, Buffer::Full)?;
    } else {
        p.copy(
            OperationEvent::cpu_half_row_sum_layout(dtype, rank, width, rows, false)?,
            Some(Buffer::Row),
        )?;
    }
    p.binary(
        CpuBinaryOperation::Divide,
        dtype,
        rank,
        rows,
        Buffer::Row,
        Buffer::Scalar,
        0,
        1,
        Buffer::Row,
    )?;
    // Only the real eager F32 epsilon promotes the half mean. The input is
    // separately promoted by the final multiply, then rounded back to half.
    promoted_binary(
        &mut p,
        CpuBinaryOperation::Add,
        dtype,
        rank,
        rows,
        Buffer::Row,
        Buffer::Scalar,
        0,
        1,
        Buffer::Row,
    )?;
    p.unary(CpuUnaryOperation::Rsqrt, rank, Buffer::Row)?;
    promoted_binary(
        &mut p,
        CpuBinaryOperation::Multiply,
        dtype,
        rank,
        count,
        Buffer::Full,
        Buffer::Row,
        rank,
        rows,
        Buffer::Full,
    )?;
    p.cast(Dtype::Float32, dtype, rank, count, Buffer::Full)?;
    // Same actual mean/count C++ frontends as the source-visible RMS fallback.
    p.native.controls =
        p.native
            .controls
            .checked_add(OperationEvent::cpu_rms_fallback_control_bytes(
                dtype, rank, width, rows,
            )?)?;
    let frames = [
        size_of::<Population>() * 2,
        size_of::<Option<Population>>(),
        size_of::<(Dtype, usize, usize, usize, usize)>(),
        size_of::<(
            &mut Population,
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
        size_of::<safemlx::CpuUnaryEvalLayout>(),
        size_of::<Option<safemlx::CpuUnaryEvalLayout>>(),
        size_of::<safemlx::CpuBinaryEvalLayout>(),
        size_of::<Option<safemlx::CpuBinaryEvalLayout>>(),
        size_of::<(&safemlx::Array, f32, &safemlx::Stream)>(),
        size_of::<safemlx::Array>() * 5,
        size_of::<Result<safemlx::Array, eredu_nn::Error>>(),
        size_of::<Result<Option<safemlx::Array>, safemlx::error::Exception>>(),
        size_of::<Result<safemlx::Array, safemlx::error::Exception>>(),
        size_of::<Dtype>(),
        size_of::<usize>() * 4,
        size_of::<Option<()>>(),
    ];
    p.native.controls = frames.into_iter().try_fold(
        p.native.controls.checked_add(size_of_val(&frames))?,
        |n, bytes| n.checked_add(bytes),
    )?;
    Some(p)
}
