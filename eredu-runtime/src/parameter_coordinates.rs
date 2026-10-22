//! Effective parameter coordinates derived from the actual selected layout.
//!
//! These declarations contain no loaded-session, epoch or communication authority.
use crate::{LocalTensorLayout, ReplicatedTextMaterializationTask, TensorPlacement};
use eredu_core::{
    capture::{CaptureReservation, CaptureUsage},
    component::ComponentCoordinateMap,
    parameters::{ParameterCoordinateMap, ParameterError},
    ParallelRankTopology,
};

/// One rank's parameter ownership, including nonlocal pipeline stages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParameterPartitionLayout {
    topology: ParallelRankTopology,
    target: String,
    global_shape: Vec<u64>,
    coordinates: Option<ParameterCoordinateMap>,
}
impl ParameterPartitionLayout {
    /// Exact retained rank topology.
    pub const fn topology(&self) -> ParallelRankTopology {
        self.topology
    }
    /// Canonical module slot, resolved through the selected task's declared aliases.
    pub fn target(&self) -> &str {
        &self.target
    }
    /// Complete effective shape, also available on stages that do not own the slot.
    pub fn global_shape(&self) -> &[u64] {
        &self.global_shape
    }
    /// Local effective coordinates, or no parameter on this pipeline stage.
    pub fn coordinates(&self) -> Option<&ParameterCoordinateMap> {
        self.coordinates.as_ref()
    }
    /// Transfers owned coordinate metadata without duplicating indexed axes.
    pub fn into_coordinates(self) -> Option<ParameterCoordinateMap> {
        self.coordinates
    }
}

/// Converts an exact physical placement to effective logical coordinates.
/// Quantization blocks are indivisible; byte offsets are never interpreted as
/// scalar indices. Independent EP/TP axes and fused indexed segments retain order.
pub fn derive_parameter_coordinates(
    task: &ReplicatedTextMaterializationTask,
    tensor: &LocalTensorLayout,
    rank: usize,
    reservation: &mut impl CaptureReservation,
) -> Result<Option<ParameterCoordinateMap>, ParameterError> {
    derive_coordinates(
        task.logical_shape(),
        task.lowering_descriptor().packed_axis(),
        task.executable(),
        tensor,
        rank,
        reservation,
    )
}

/// Companion outputs are scalar storage arrays with their own retained logical
/// geometry and sharding. Encoded weight block expansion does not apply to them.
fn derive_companion_coordinates(
    logical: &[usize],
    tensor: &LocalTensorLayout,
    rank: usize,
    reservation: &mut impl CaptureReservation,
) -> Result<Option<ParameterCoordinateMap>, ParameterError> {
    derive_coordinates(
        logical,
        None,
        eredu_checkpoint::LinearFormat::Dense,
        tensor,
        rank,
        reservation,
    )
}

fn derive_coordinates(
    logical: &[usize],
    packed_axis: Option<usize>,
    format: eredu_checkpoint::LinearFormat,
    tensor: &LocalTensorLayout,
    rank: usize,
    reservation: &mut impl CaptureReservation,
) -> Result<Option<ParameterCoordinateMap>, ParameterError> {
    let physical = tensor.global_shape();
    if logical.len() > 32
        || logical.len() != physical.len()
        || tensor.local_shape().len() != physical.len()
        || logical.contains(&0)
        || physical.contains(&0)
    {
        return Err(invalid("parameter logical and physical ranks disagree"));
    }
    charge(
        reservation,
        256u64
            .checked_add(
                (logical.len() as u64)
                    .checked_mul(160)
                    .ok_or(ParameterError::Overflow)?,
            )
            .ok_or(ParameterError::Overflow)?,
    )?;
    let blocks = physical
        .iter()
        .zip(logical)
        .enumerate()
        .map(|(axis, (&physical, &logical))| {
            if physical == logical {
                return Ok((1usize, 1usize));
            }
            if Some(axis) != packed_axis {
                return Err(invalid("nonpacking parameter axis changed extent"));
            }
            let quantization = format
                .weight_quantization()
                .ok_or_else(|| invalid("unpacked parameter changed its physical extent"))?;
            quantization
                .validate()
                .map_err(|error| invalid(&error.to_string()))?;
            let block =
                usize::try_from(quantization.group_size()).map_err(|_| ParameterError::Overflow)?;
            if !logical.is_multiple_of(block) || !physical.is_multiple_of(logical / block) {
                return Err(invalid("parameter extent splits an encoded block"));
            }
            let stored = physical / (logical / block);
            // The retained geometry may store microscaling values in bytes or words.
            // GGML blocks include their metadata and must retain their exact byte size.
            let valid = match format {
                eredu_checkpoint::LinearFormat::Affine(format) => {
                    stored.checked_mul(32) == block.checked_mul(format.bits as usize)
                }
                eredu_checkpoint::LinearFormat::MxFp4 => {
                    stored.checked_mul(8) == block.checked_mul(4)
                        || stored.checked_mul(32) == block.checked_mul(4)
                }
                eredu_checkpoint::LinearFormat::GgufIQuant { ggml_type, .. } => {
                    ggml_type
                        .block_and_bytes()
                        .map_err(|error| invalid(&error.to_string()))?
                        .1
                        == stored as u64
                }
                _ => false,
            };
            if !valid {
                return Err(invalid(
                    "selected encoding and physical parameter blocks disagree",
                ));
            }
            Ok((stored, block))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut axes = logical
        .iter()
        .map(|size| ComponentCoordinateMap::range(*size, 0..*size).map_err(coordinate_error))
        .collect::<Result<Vec<_>, _>>()?;
    let mut changed = vec![false; logical.len()];
    let mut local_physical = physical.to_vec();
    for placement in tensor
        .additional_placements()
        .iter()
        .chain(std::iter::once(tensor.placement()))
    {
        let (axis, range, indices) = match placement {
            TensorPlacement::Replicated | TensorPlacement::Local => continue,
            TensorPlacement::Rank { rank: owner } if *owner == rank => continue,
            TensorPlacement::Rank { .. } | TensorPlacement::Omit => return Ok(None),
            TensorPlacement::Range { axis, start, end } => (*axis, Some(*start..*end), None),
            TensorPlacement::Shard { axis, index, parts } => {
                let extent = *physical
                    .get(*axis)
                    .ok_or_else(|| invalid("parameter shard axis is absent"))?;
                if *parts == 0 || *index >= *parts || !extent.is_multiple_of(*parts) {
                    return Err(invalid("parameter shard geometry is invalid"));
                }
                let width = extent / parts;
                (*axis, Some(index * width..(index + 1) * width), None)
            }
            TensorPlacement::Indices { axis, indices } => (*axis, None, Some(indices)),
        };
        if axis >= axes.len() || changed[axis] {
            return Err(invalid("parameter placements repeat or exceed an axis"));
        }
        changed[axis] = true;
        let (stored, scalars) = blocks[axis];
        let convert = |value: usize| -> Result<usize, ParameterError> {
            if value > physical[axis] || !value.is_multiple_of(stored) {
                return Err(invalid("parameter placement splits an encoded block"));
            }
            (value / stored)
                .checked_mul(scalars)
                .ok_or(ParameterError::Overflow)
        };
        axes[axis] = if let Some(range) = range {
            if range.start > range.end {
                return Err(invalid("reversed parameter placement"));
            }
            local_physical[axis] = range.len();
            ComponentCoordinateMap::range(logical[axis], convert(range.start)?..convert(range.end)?)
                .map_err(coordinate_error)?
        } else {
            let indices = indices.expect("range or indices");
            if !indices.len().is_multiple_of(stored) {
                return Err(invalid(
                    "indexed parameter placement splits an encoded block",
                ));
            }
            local_physical[axis] = indices.len();
            let count = (indices.len() / stored)
                .checked_mul(scalars)
                .ok_or(ParameterError::Overflow)?;
            // ComponentCoordinateMap validates uniqueness with a temporary set.
            charge(
                reservation,
                128u64
                    .checked_add(
                        (count as u64)
                            .checked_mul(48)
                            .ok_or(ParameterError::Overflow)?,
                    )
                    .ok_or(ParameterError::Overflow)?,
            )?;
            let mut selected = Vec::with_capacity(count);
            for block in indices.chunks_exact(stored) {
                let first = block[0];
                let start = convert(first)?;
                let end = first.checked_add(stored).ok_or(ParameterError::Overflow)?;
                convert(end)?;
                if !block.iter().copied().eq(first..end) {
                    return Err(invalid(
                        "indexed parameter placement reorders encoded block members",
                    ));
                }
                selected.extend(start..start + scalars);
            }
            ComponentCoordinateMap::indices(logical[axis], selected).map_err(coordinate_error)?
        };
    }
    if local_physical != tensor.local_shape() {
        return Err(invalid(
            "parameter placement differs from its retained local shape",
        ));
    }
    ParameterCoordinateMap::new(logical.iter().map(|n| *n as u64).collect(), axes).map(Some)
}

fn charge(
    reservation: &mut impl CaptureReservation,
    host_bytes: u64,
) -> Result<(), ParameterError> {
    reservation.reserve_quota(CaptureUsage {
        host_bytes,
        ..Default::default()
    })?;
    Ok(())
}
fn invalid(message: &str) -> ParameterError {
    ParameterError::Invalid(message.into())
}
fn coordinate_error(error: eredu_core::component::ComponentCoordinateError) -> ParameterError {
    invalid(&error.to_string())
}

/// Derives one selected parameter's effective rank-local coordinates.
/// Families supply actual ownership and retained or ordinary placement. Runtime
/// lowers only the requested atomic group and prepays all coordinate metadata.
pub fn derive_parameter_member_layout(
    resolved: crate::ResolvedParameterMember<'_>,
    topology: ParallelRankTopology,
    placement: ParallelRankTopology,
    owns: bool,
    retained_layout: Option<&crate::LocalModelLayout>,
    reservation: &mut impl eredu_core::capture::CaptureReservation,
) -> Result<ParameterPartitionLayout, eredu_core::parameters::ParameterError> {
    use eredu_core::parameters::ParameterError;
    let crate::ResolvedParameterMember {
        task,
        companion,
        group,
        member,
    } = resolved;
    let logical = companion.map_or_else(
        || task.logical_shape(),
        |companion| companion.logical_shape(),
    );
    let global_rank = topology.global_rank();
    charge(
        reservation,
        256u64
            .checked_add(member.target().len() as u64)
            .and_then(|n| n.checked_add((logical.len() as u64).checked_mul(8)?))
            .ok_or(ParameterError::Overflow)?,
    )?;
    let coordinates = if owns {
        let mut bytes = 256u64;
        for member in group.members() {
            bytes = bytes
                .checked_add(512)
                .and_then(|n| n.checked_add(member.target().len() as u64))
                .and_then(|n| n.checked_add(group.group().logical_name().len() as u64))
                .and_then(|n| n.checked_add((member.global_shape().len() as u64).checked_mul(64)?))
                .ok_or(ParameterError::Overflow)?;
            if let crate::MemberSharding::Segmented { axis, .. }
            | crate::MemberSharding::PartitionedSegments { axis, .. }
            | crate::MemberSharding::PartitionedChunkSegments { axis, .. } = member.sharding()
            {
                let extent = member
                    .global_shape()
                    .get(*axis)
                    .ok_or_else(|| invalid("segmented parameter axis is absent"))?;
                bytes = bytes
                    .checked_add(
                        (*extent as u64)
                            .checked_mul(32)
                            .ok_or(ParameterError::Overflow)?,
                    )
                    .ok_or(ParameterError::Overflow)?;
            }
        }
        charge(reservation, bytes)?;
        let derived = if retained_layout.is_none() {
            Some(
                crate::derive_partitioned_group_layout(std::iter::once(group), placement)
                    .map_err(|error| invalid(&error))?,
            )
        } else {
            None
        };
        let tensor = retained_layout
            .or(derived.as_ref())
            .expect("one layout authority")
            .tensor(member.target())
            .ok_or_else(|| invalid("selected parameter layout is absent"))?;
        if companion.is_some() {
            derive_companion_coordinates(logical, tensor, global_rank, reservation)?
        } else {
            derive_parameter_coordinates(task, tensor, global_rank, reservation)?
        }
    } else {
        None
    };
    Ok(ParameterPartitionLayout {
        topology,
        target: member.target().into(),
        global_shape: logical.iter().map(|n| *n as u64).collect(),
        coordinates,
    })
}
