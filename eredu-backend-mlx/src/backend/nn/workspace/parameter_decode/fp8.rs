//! The standalone block-FP8 helper: decode scales, repeat, slice and multiply.
use super::*;
use safemlx::Dtype;

#[derive(Clone, Copy)]
pub(in super::super) struct Geometry {
    pub rank: usize,
    pub work_rank: usize,
    pub partitioned: bool,
    pub values: usize,
    pub scales: usize,
    pub scale_dtype: Dtype,
    pub encoded_scale: bool,
}
impl Geometry {
    pub fn first_repeat(self) -> Option<usize> {
        self.scales.checked_mul(128)
    }
    pub fn second_repeat(self) -> Option<usize> {
        self.first_repeat()?.checked_mul(128)
    }
    pub fn decoded_scale(self) -> Dtype {
        if self.encoded_scale {
            Dtype::Float32
        } else {
            self.scale_dtype
        }
    }
    pub fn births(self) -> Option<usize> {
        // Two repeat copies, weight conversion, result, optional scale cast,
        // table/cast/Gather, and the two independent-partition source reshapes.
        4usize
            .checked_add(usize::from(self.decoded_scale() != Dtype::Float32))?
            .checked_add(if self.encoded_scale { 3 } else { 0 })?
            .checked_add(if self.partitioned { 2 } else { 0 })
    }
}
pub(in super::super) fn inspect(
    op: WorkspaceOperationView<'_>,
) -> facts::FactResult<Option<Geometry>> {
    let WorkspaceOperationKindView::ParameterDecode(decoding) = op.kind else {
        return Ok(None);
    };
    let LinearFormat::E4M3BlockFp8(config) = decoding.format else {
        return Ok(None);
    };
    config
        .validate_fixed()
        .map_err(|_| MlxWorkspaceFactError::descriptor("invalid standalone FP8 configuration"))?;
    if config.block_rows != 128 || config.block_columns != 128 {
        return Ok(None);
    }
    let bad = || MlxWorkspaceFactError::descriptor("standalone FP8 source geometry differs");
    if op.inputs.len() != 2 || op.outputs.len() != 1 {
        return Err(bad());
    }
    let weight = op.inputs.get(0).ok_or_else(bad)?;
    let scale = op.inputs.get(1).ok_or_else(bad)?;
    let output = op.outputs.get(0).ok_or_else(bad)?;
    let rank = weight.shape().len();
    if !(2..=3).contains(&rank) {
        return Ok(None);
    }
    if weight.dtype() != WorkspaceDtype::Uint8
        || output.dtype() != WorkspaceDtype::Float32
        || output.shape() != weight.shape()
        || scale.shape().len() != rank
        || weight.shape().iter().any(|&n| n <= 0)
        || scale.shape().iter().any(|&n| n <= 0)
        || weight.shape()[..rank - 2] != scale.shape()[..rank - 2]
    {
        return Err(bad());
    }
    let rows = usize::try_from(weight.shape()[rank - 2])?;
    let columns = usize::try_from(weight.shape()[rank - 1])?;
    let scale_rows = decoding.row_layout.scale_rows_fixed(rows, 128)?;
    decoding.row_layout.rows_per_partition_fixed(rows)?;
    if usize::try_from(scale.shape()[rank - 2])? != scale_rows
        || usize::try_from(scale.shape()[rank - 1])? != columns.div_ceil(128)
    {
        return Err(bad());
    }
    let (encoded_scale, scale_dtype) = match scale.dtype() {
        WorkspaceDtype::Uint8 => (true, Dtype::Uint8),
        WorkspaceDtype::Float32 => {
            let Some(representation) = scale.representation() else {
                return Ok(None);
            };
            (
                false,
                match representation.dtype() {
                    WorkspaceFloatingType::Float16 => Dtype::Float16,
                    WorkspaceFloatingType::Bfloat16 => Dtype::Bfloat16,
                    WorkspaceFloatingType::Float32 => Dtype::Float32,
                },
            )
        }
        _ => return Err(bad()),
    };
    if encoded_scale
        != matches!(
            config.scale_encoding,
            eredu_checkpoint::BlockFp8ScaleEncoding::Ue8m0
        )
    {
        return Err(bad());
    }
    let partitioned = decoding.row_layout != eredu_nn::LinearRowLayout::Contiguous;
    let geometry = Geometry {
        rank,
        work_rank: if partitioned { 3 } else { rank },
        partitioned,
        values: usize::try_from(weight.elements()?)?,
        scales: usize::try_from(scale.elements()?)?,
        scale_dtype,
        encoded_scale,
    };
    let padded = geometry
        .second_repeat()
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    if padded > i32::MAX as usize || geometry.values > i32::MAX as usize {
        return Ok(None);
    }
    Ok(Some(geometry))
}
fn capacity(allocation: NativeAllocationFacts, n: usize, dtype: Dtype) -> facts::FactResult<u64> {
    let width = match dtype {
        Dtype::Uint8 => 1,
        Dtype::Float16 | Dtype::Bfloat16 => 2,
        Dtype::Float32 | Dtype::Uint32 => 4,
        _ => {
            return Err(MlxWorkspaceFactError::descriptor(
                "invalid standalone FP8 physical precision",
            ));
        }
    };
    allocation.fixed_buffer_capacity(facts::mul(u64::try_from(n)?, width)?)
}
pub(in super::super) fn emit(
    op: WorkspaceOperationView<'_>,
    allocation: NativeAllocationFacts,
    sink: &mut facts::Emitter<'_>,
) -> facts::FactResult<Option<WorkspaceOperationFacts>> {
    let Some(g) = inspect(op)? else {
        return Ok(None);
    };
    let output = capacity(allocation, g.values, Dtype::Float32)?;
    sink.output(facts::Output::Allocate(output))?;
    let mut scratch = output; // converted weight before final multiplication
    if g.partitioned {
        scratch = facts::add(scratch, capacity(allocation, g.values, Dtype::Uint8)?)?;
        scratch = facts::add(scratch, capacity(allocation, g.scales, g.scale_dtype)?)?;
    }
    if g.encoded_scale {
        scratch = facts::add(scratch, capacity(allocation, 256, Dtype::Float32)?)?;
        scratch = facts::add(
            scratch,
            facts::mul(capacity(allocation, g.scales, Dtype::Float32)?, 2)?,
        )?;
    }
    for count in [g.first_repeat(), g.second_repeat()] {
        scratch = facts::add(
            scratch,
            capacity(
                allocation,
                count.ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
                g.decoded_scale(),
            )?,
        )?;
    }
    if g.decoded_scale() != Dtype::Float32 {
        // Slice can retain the padded repeat backing; AsType preserves its
        // actual data extent when the native copy selects a vector branch.
        scratch = facts::add(
            scratch,
            capacity(
                allocation,
                g.second_repeat()
                    .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
                Dtype::Float32,
            )?,
        )?;
    }
    sink.finish(scratch, format_args!("standalone block-FP8 source: actual scale lookup, two repeat copies, F32 weight conversion, static slice and mixed-precision multiply; independent row origins preserve partition reshapes")).map(Some)
}
pub(in super::super) fn control_bytes(g: Geometry) -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let frames = [
        size_of::<Geometry>(),
        size_of::<Option<Geometry>>(),
        size_of::<WorkspaceOperationView<'_>>(),
        size_of::<WorkspaceLayoutView<'_>>() * 3,
        size_of::<eredu_nn::parameter_values::ParameterDecoding>(),
        size_of::<(
            &safemlx::Array,
            &safemlx::Array,
            eredu_nn::LinearRowLayout,
            &safemlx::Stream,
        )>(),
        size_of::<safemlx::Array>() * 7,
        size_of::<Result<safemlx::Array, safemlx::error::Exception>>(),
        size_of::<[i32; 3]>() * 2,
        size_of::<usize>() * 12,
        size_of::<[std::ops::RangeTo<i32>; 3]>(),
        safemlx::ops::indexing::basic_range_index_control_bytes(g.work_rank)?,
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
