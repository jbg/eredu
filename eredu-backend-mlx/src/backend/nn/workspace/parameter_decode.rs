//! The standalone native dequantizer, with its actual primary/companion geometry.
use super::*;
use eredu_checkpoint::LinearFormat;
pub(super) mod fp8;
pub(super) mod gguf;

pub(super) fn output_dtype(op: WorkspaceOperationView<'_>) -> Option<WorkspaceFloatingType> {
    if fp8::inspect(op).ok()?.is_some() || gguf::inspect(op).ok()?.is_some() {
        return Some(WorkspaceFloatingType::Float32);
    }
    Some(inspect(op).ok()??.dtype())
}

#[derive(Clone, Copy)]
pub(super) struct Geometry {
    pub(super) rank: usize,
    pub(super) rows: usize,
    pub(super) words: usize,
    pub(super) values: usize,
    pub(super) scales: usize,
    pub(super) group: usize,
    pub(super) bits: usize,
    pub(super) affine: bool,
}
impl Geometry {
    pub(super) fn dtype(self) -> WorkspaceFloatingType {
        if self.affine {
            WorkspaceFloatingType::Float32
        } else {
            WorkspaceFloatingType::Bfloat16
        }
    }
}
pub(super) fn inspect(op: WorkspaceOperationView<'_>) -> facts::FactResult<Option<Geometry>> {
    let WorkspaceOperationKindView::ParameterDecode(decoding) = op.kind else {
        return Ok(None);
    };
    let (affine, group, bits) = match decoding.format {
        LinearFormat::Affine(config) => {
            config.validate_fixed().map_err(|_| {
                MlxWorkspaceFactError::descriptor(
                    "invalid standalone affine decoding configuration",
                )
            })?;
            (
                true,
                usize::try_from(config.group_size)?,
                usize::try_from(config.bits)?,
            )
        }
        LinearFormat::MxFp4 => (false, 32, 4),
        _ => return Ok(None),
    };
    let bad =
        || MlxWorkspaceFactError::descriptor("standalone dequantizer source geometry differs");
    if op.inputs.len() != if affine { 3 } else { 2 } || op.outputs.len() != 1 {
        return Err(bad());
    }
    let weight = op.inputs.get(0).ok_or_else(bad)?;
    let scale = op.inputs.get(1).ok_or_else(bad)?;
    let output = op.outputs.get(0).ok_or_else(bad)?;
    let rank = weight.shape().len();
    if !(2..=3).contains(&rank) {
        return Ok(None);
    }
    if weight.dtype() != WorkspaceDtype::Uint32
        || output.dtype() != WorkspaceDtype::Float32
        || weight.shape().iter().any(|&n| n <= 0)
        || scale.shape().len() != rank
        || output.shape().len() != rank
        || weight.shape()[..rank - 1] != scale.shape()[..rank - 1]
        || weight.shape()[..rank - 1] != output.shape()[..rank - 1]
        || scale.shape()[rank - 1] <= 0
        || output.shape()[rank - 1] <= 0
    {
        return Err(bad());
    }
    let row_bits = usize::try_from(weight.shape()[rank - 1])?
        .checked_mul(32)
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    let width = usize::try_from(output.shape()[rank - 1])?;
    if row_bits % bits != 0
        || row_bits / bits != width
        || usize::try_from(scale.shape()[rank - 1])?.checked_mul(group) != Some(width)
    {
        return Err(bad());
    }
    if affine {
        for input in [scale, op.inputs.get(2).ok_or_else(bad)?] {
            if input.shape() != scale.shape() || input.dtype() != WorkspaceDtype::Float32 {
                return Err(bad());
            }
            if !input
                .representation()
                .is_some_and(|r| r.dtype() == WorkspaceFloatingType::Float32)
            {
                return Ok(None);
            }
        }
    } else if scale.dtype() != WorkspaceDtype::Uint8 {
        return Err(bad());
    }
    let rows = weight.shape()[..rank - 1]
        .iter()
        .try_fold(1usize, |n, &v| n.checked_mul(v as usize))
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    let words = usize::try_from(weight.elements()?)?;
    let values = usize::try_from(output.elements()?)?;
    let scales = usize::try_from(scale.elements()?)?;
    let intermediate = if affine && !bits.is_power_of_two() {
        words.checked_mul(32)
    } else if !affine {
        words.checked_mul(4)
    } else {
        Some(values)
    }
    .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    if intermediate > i32::MAX as usize || values > i32::MAX as usize || scales > i32::MAX as usize
    {
        return Ok(None);
    }
    Ok(Some(Geometry {
        rank,
        rows,
        words,
        values,
        scales,
        group,
        bits,
        affine,
    }))
}

pub(super) fn emit(
    op: WorkspaceOperationView<'_>,
    allocation: NativeAllocationFacts,
    sink: &mut facts::Emitter<'_>,
) -> facts::FactResult<Option<WorkspaceOperationFacts>> {
    if let Some(value) = gguf::emit(op, allocation, sink)? {
        return Ok(Some(value));
    }
    if let Some(value) = fp8::emit(op, allocation, sink)? {
        return Ok(Some(value));
    }
    let Some(_geometry) = inspect(op)? else {
        return Ok(None);
    };
    let output = op.outputs.get(0).expect("validated decoded result");
    sink.output(facts::Output::Allocate(
        allocation.fixed_buffer_capacity(output.bytes()?)?,
    ))?;
    // Quantize::eval_gpu calls ensure_row_contiguous independently on the
    // weight and each companion. Every possible temporary remains live until
    // this one native dequantization kernel completes.
    let scratch = op.inputs.iter().try_fold(0u64, |n, input| {
        facts::add(n, allocation.fixed_buffer_capacity(input.bytes()?)?)
    })?;
    sink.finish(scratch,format_args!("MLX standalone fast::Quantize dequantization: one decoded result and one possible row compaction per actual source; physical output precision is carried separately")).map(Some)
}

pub(super) fn control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let parts = [
        size_of::<Geometry>(),
        size_of::<Option<Geometry>>(),
        size_of::<WorkspaceOperationView<'_>>(),
        size_of::<WorkspaceLayoutView<'_>>() * 4,
        size_of::<std::slice::Iter<'_, i32>>(),
        size_of::<std::array::IntoIter<WorkspaceLayoutView<'_>, 2>>(),
        size_of::<usize>() * 12,
        size_of::<(
            &safemlx::Array,
            &safemlx::Array,
            Option<&safemlx::Array>,
            i32,
            i32,
            safemlx::ops::QuantizationMode,
            &safemlx::Stream,
        )>(),
        size_of::<Result<safemlx::Array, safemlx::error::Exception>>(),
        size_of::<eredu_nn::parameter_values::ParameterDecoding>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
