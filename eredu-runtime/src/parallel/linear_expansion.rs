//! The one physical-format expansion worker and its actual metadata destination.
use super::construction::{Checked, Destination, Ordinary};
use super::*;

fn collect<T, D: Destination>(
    iter: impl Iterator<Item = Result<T, D::Error>>,
    count: usize,
    destination: D,
) -> Result<Vec<T>, D::Error> {
    destination.controls::<(usize, Vec<T>, Result<T, D::Error>)>()?;
    destination.value_controls(&iter)?;
    let mut out = destination.vector(count)?;
    for value in iter {
        out.push(value?);
    }
    Ok(out)
}
fn values<T, D: Destination, const N: usize>(
    values: [T; N],
    destination: D,
) -> Result<Vec<T>, D::Error> {
    destination.controls::<([T; N], Vec<T>)>()?;
    let mut out = destination.vector(N)?;
    out.extend(values);
    Ok(out)
}
fn copy_values<T: Clone, D: Destination>(source: &[T], destination: D) -> Result<Vec<T>, D::Error> {
    destination.controls::<(&[T], Vec<T>)>()?;
    let mut out = destination.vector(source.len())?;
    out.extend_from_slice(source);
    Ok(out)
}
fn copy_sharding<D: Destination>(
    source: &MemberSharding,
    destination: D,
) -> Result<MemberSharding, D::Error> {
    destination.controls::<(&MemberSharding, MemberSharding)>()?;
    Ok(match source {
        MemberSharding::Segmented { axis, segments } => MemberSharding::Segmented {
            axis: *axis,
            segments: copy_values(segments, destination)?,
        },
        MemberSharding::PartitionedSegments { axis, segments } => {
            MemberSharding::PartitionedSegments {
                axis: *axis,
                segments: copy_values(segments, destination)?,
            }
        }
        MemberSharding::PartitionedChunkSegments {
            axis,
            segments,
            chunk_size,
        } => MemberSharding::PartitionedChunkSegments {
            axis: *axis,
            segments: copy_values(segments, destination)?,
            chunk_size: *chunk_size,
        },
        other => other.clone(),
    })
}
fn member<D: Destination>(
    name: &str,
    shape: Vec<usize>,
    sharding: MemberSharding,
    companion: Option<(eredu_nn::LinearCompanionRole, &str)>,
    destination: D,
) -> Result<ParameterMemberSpec, D::Error> {
    destination.controls::<(
        &str,
        Vec<usize>,
        MemberSharding,
        Option<(eredu_nn::LinearCompanionRole, &str)>,
        ParameterMemberSpec,
    )>()?;
    Ok(ParameterMemberSpec {
        target: destination.string(format_args!("{name}"))?,
        global_shape: shape,
        sharding,
        linear_companion: companion.map(|v| v.0),
        linear_companion_of: companion
            .map(|v| destination.string(format_args!("{}", v.1)))
            .transpose()?,
        linear_row_layout: eredu_nn::LinearRowLayout::Contiguous,
    })
}
fn copy_member<D: Destination>(
    source: &ParameterMemberSpec,
    destination: D,
) -> Result<ParameterMemberSpec, D::Error> {
    let mut result = member(
        source.target(),
        copy_values(source.global_shape(), destination)?,
        copy_sharding(source.sharding(), destination)?,
        source.linear_companion.zip(source.linear_companion_of()),
        destination,
    )?;
    result.linear_row_layout = source.linear_row_layout;
    Ok(result)
}
fn expand<D: Destination>(
    groups: Vec<ParameterGroupSpec>,
    declaration: impl Fn(&ParameterMemberSpec) -> Result<Option<LinearFormatSpec>, D::Error>,
    destination: D,
) -> Result<Vec<ParameterGroupSpec>, D::Error> {
    destination.controls::<(
        Vec<ParameterGroupSpec>,
        ParameterGroupSpec,
        Vec<ParameterMemberSpec>,
        Option<LinearFormatSpec>,
        Option<usize>,
        Result<Vec<ParameterGroupSpec>, D::Error>,
    )>()?;
    destination.value_controls(&declaration)?;
    let mut output = destination.vector(groups.len())?;
    for group in groups {
        let mut members = destination.vector(
            group
                .members
                .len()
                .checked_mul(3)
                .ok_or_else(|| destination.overflow())?,
        )?;
        for source in &group.members {
            match declaration(source)? {
                Some(declaration) => members.extend(expand_linear_format_member(
                    source,
                    &declaration,
                    destination,
                )?),
                None => members.push(copy_member(source, destination)?),
            }
        }
        let units = group
            .partition_units
            .map(|units| {
                construction::preferred_units(&members, units)
                    .map_err(|cause| destination.issue(cause))
            })
            .transpose()?;
        if units == Some(0) {
            return Err(destination.invalid_group(format_args!(
                "parallel logical partition must contain at least one unit"
            )));
        }
        construction::validate_name(&group.logical_name)
            .map_err(|cause| destination.issue(cause))?;
        construction::validate_members(&group.logical_name, units, &members, destination)?;
        output.push(ParameterGroupSpec {
            logical_name: group.logical_name,
            role: group.role,
            partition_units: units,
            members,
        });
    }
    Ok(output)
}
pub(super) fn ordinary(
    groups: Vec<ParameterGroupSpec>,
    declaration: impl Fn(&ParameterMemberSpec) -> Result<Option<LinearFormatSpec>, ParallelPlanError>,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    expand(groups, declaration, Ordinary)
}
pub fn expand_linear_format_parameter_groups_with_metadata(
    groups: Vec<ParameterGroupSpec>,
    declaration: impl Fn(&ParameterMemberSpec) -> Result<Option<LinearFormatSpec>, eredu_nn::Error>,
    context: &eredu_nn::workspace::WorkspaceContext,
) -> Result<Vec<ParameterGroupSpec>, eredu_nn::Error> {
    expand(groups, declaration, Checked(context))
}

fn remap_linear_segments<D: Destination>(
    sharding: &MemberSharding,
    axis: usize,
    divisor: usize,
    name: &str,
    destination: D,
) -> Result<MemberSharding, D::Error> {
    destination.controls::<(
        &ParameterMemberSpec,
        &LinearFormatSpec,
        &MemberSharding,
        &[Range<usize>],
        &str,
        usize,
        usize,
        usize,
        usize,
        usize,
        MemberSharding,
        Vec<usize>,
        Vec<usize>,
        Vec<ParameterMemberSpec>,
        Result<Vec<ParameterMemberSpec>, D::Error>,
        std::fmt::Arguments<'_>,
    )>()?;
    let remap = |segments: &[Range<usize>]| {
        collect(
            segments.iter().map(|segment| {
                if !segment.start.is_multiple_of(divisor) || !segment.end.is_multiple_of(divisor) {
                    return Err(destination.invalid_tensor(format_args!(
                        "packed companion {name} segment {segment:?} is not aligned to {divisor}"
                    )));
                }
                Ok(segment.start / divisor..segment.end / divisor)
            }),
            segments.len(),
            destination,
        )
    };
    destination.value_controls(&remap)?;
    match sharding {
        MemberSharding::PartitionedChunks {
            axis: selected,
            chunk_size,
        } if *selected == axis => {
            if *chunk_size == 0 || !chunk_size.is_multiple_of(divisor) {
                return Err(destination.invalid_tensor(format_args!(
                    "packed companion {name} chunk width {chunk_size} is not aligned to {divisor}"
                )));
            }
            Ok(MemberSharding::PartitionedChunks {
                axis,
                chunk_size: chunk_size / divisor,
            })
        }
        MemberSharding::PartitionedChunkSegments {
            axis: selected,
            segments,
            chunk_size,
        } if *selected == axis => {
            if *chunk_size == 0 || !chunk_size.is_multiple_of(divisor) {
                return Err(destination.invalid_tensor(format_args!(
                    "packed companion {name} chunk width {chunk_size} is not aligned to {divisor}"
                )));
            }
            Ok(MemberSharding::PartitionedChunkSegments {
                axis,
                segments: remap(segments)?,
                chunk_size: chunk_size / divisor,
            })
        }
        MemberSharding::PartitionedSegments {
            axis: selected,
            segments,
        } if *selected == axis => Ok(MemberSharding::PartitionedSegments {
            axis: *selected,
            segments: remap(segments)?,
        }),
        MemberSharding::Segmented {
            axis: selected,
            segments,
        } if *selected == axis => Ok(MemberSharding::Segmented {
            axis: *selected,
            segments: remap(segments)?,
        }),
        other => copy_sharding(other, destination),
    }
}

fn remap_fp8_rows<D: Destination>(
    source: &ParameterMemberSpec,
    layout: eredu_nn::LinearRowLayout,
    block: usize,
    destination: D,
) -> Result<MemberSharding, D::Error> {
    destination.controls::<(
        &ParameterMemberSpec,
        &LinearFormatSpec,
        &MemberSharding,
        &[Range<usize>],
        &str,
        usize,
        usize,
        usize,
        usize,
        usize,
        MemberSharding,
        Vec<usize>,
        Vec<usize>,
        Vec<ParameterMemberSpec>,
        Result<Vec<ParameterMemberSpec>, D::Error>,
        std::fmt::Arguments<'_>,
    )>()?;
    let row_axis = source.global_shape().len() - 2;
    if layout == eredu_nn::LinearRowLayout::Contiguous {
        return remap_linear_segments(
            source.sharding(),
            row_axis,
            block,
            source.target(),
            destination,
        );
    }
    let rows = source.global_shape()[row_axis];
    let remap = |segments: &[Range<usize>]| {
        collect(
            segments.iter().map(|segment| {
                let boundary = |value| {
                    layout
                        .block_boundary(rows, block, value)
                        .map_err(|error| destination.invalid_tensor(format_args!("{error}")))
                };
                Ok(boundary(segment.start)?..boundary(segment.end)?)
            }),
            segments.len(),
            destination,
        )
    };
    destination.value_controls(&remap)?;
    match source.sharding() {
        MemberSharding::PartitionedSegments { axis, segments } if *axis == row_axis => {
            Ok(MemberSharding::PartitionedSegments {
                axis: *axis,
                segments: remap(segments)?,
            })
        }
        MemberSharding::Segmented { axis, segments } if *axis == row_axis => {
            Ok(MemberSharding::Segmented {
                axis: *axis,
                segments: remap(segments)?,
            })
        }
        MemberSharding::PartitionedChunkSegments {
            axis,
            segments,
            chunk_size,
        } if *axis == row_axis => {
            if *chunk_size == 0 || !chunk_size.is_multiple_of(block) {
                return Err(
                    destination.invalid_tensor(format_args!("FP8 row chunk splits a scale block"))
                );
            }
            Ok(MemberSharding::PartitionedChunkSegments {
                axis: *axis,
                segments: remap(segments)?,
                chunk_size: chunk_size / block,
            })
        }
        MemberSharding::Partitioned { axis }
        | MemberSharding::PartitionedChunks { axis, .. }
        | MemberSharding::Equal { axis }
        | MemberSharding::Balanced { axis }
            if *axis == row_axis =>
        {
            Err(destination.invalid_tensor(format_args!(
                "independent FP8 row blocks require explicit segment placement"
            )))
        }
        other => copy_sharding(other, destination),
    }
}

fn expand_linear_format_member<D: Destination>(
    source: &ParameterMemberSpec,
    declaration: &LinearFormatSpec,
    destination: D,
) -> Result<Vec<ParameterMemberSpec>, D::Error> {
    destination.controls::<(
        &ParameterMemberSpec,
        &LinearFormatSpec,
        &MemberSharding,
        &[Range<usize>],
        &str,
        usize,
        usize,
        usize,
        usize,
        usize,
        MemberSharding,
        Vec<usize>,
        Vec<usize>,
        Vec<ParameterMemberSpec>,
        Result<Vec<ParameterMemberSpec>, D::Error>,
        std::fmt::Arguments<'_>,
    )>()?;
    let name = source.target();
    let shape = source.global_shape();
    let format = declaration.encoding();
    if format == LinearFormat::Dense {
        return if declaration.scale().is_none() && declaration.affine_bias().is_none() {
            values([copy_member(source, destination)?], destination)
        } else {
            Err(destination.invalid_group(format_args!(
                "dense linear parameter {name} declares physical companions"
            )))
        };
    }
    if shape.len() < 2 {
        return Err(destination.invalid_tensor(format_args!(
            "encoded linear parameter {name} must have at least two dimensions"
        )));
    }
    let row_axis = shape.len() - 2;
    let column_axis = shape.len() - 1;
    let invalid = |detail: std::fmt::Arguments<'_>| destination.invalid_tensor(detail);
    destination.value_controls(&invalid)?;
    match format {
        LinearFormat::Dense => unreachable!(),
        LinearFormat::E4M3BlockFp8(fp8) => {
            let Some(scale) = declaration.scale() else {
                return Err(destination.invalid_group(format_args!(
                    "block-FP8 linear parameter {name} must declare exactly one scale companion"
                )));
            };
            if declaration.affine_bias().is_some() {
                return Err(destination.invalid_group(format_args!(
                    "block-FP8 linear parameter {name} must not declare an affine-bias companion"
                )));
            }
            fp8.validate()
                .map_err(|error| invalid(format_args!("{error}")))?;
            let rows = usize::try_from(fp8.block_rows)
                .map_err(|_| invalid(format_args!("invalid block rows for {name}")))?;
            let columns = usize::try_from(fp8.block_columns)
                .map_err(|_| invalid(format_args!("invalid block columns for {name}")))?;
            let mut scale_shape = copy_values(shape, destination)?;
            scale_shape[row_axis] = declaration
                .row_layout()
                .scale_rows(shape[row_axis], rows)
                .map_err(|error| invalid(format_args!("{error}")))?;
            scale_shape[column_axis] = scale_shape[column_axis].div_ceil(columns);
            let scale_sharding =
                remap_fp8_rows(source, declaration.row_layout(), rows, destination).and_then(
                    |value| remap_linear_segments(&value, column_axis, columns, name, destination),
                )?;
            values(
                [
                    copy_member(source, destination)?
                        .with_linear_row_layout(declaration.row_layout()),
                    member(
                        scale.id.as_str(),
                        scale_shape,
                        scale_sharding,
                        Some((eredu_nn::LinearCompanionRole::Scale, name)),
                        destination,
                    )?,
                ],
                destination,
            )
        }
        LinearFormat::GgufIQuant { ggml_type, .. } => {
            if declaration.scale().is_some() || declaration.affine_bias().is_some() {
                return Err(destination.invalid_group(format_args!(
                    "GGUF linear parameter {name} must not declare companion tensors"
                )));
            }
            let (block_values, block_bytes) = ggml_type
                .block_and_bytes()
                .map_err(|error| invalid(format_args!("{error}")))?;
            let block_values = usize::try_from(block_values)
                .map_err(|_| invalid(format_args!("GGUF block width for {name} exceeds usize")))?;
            let block_bytes = usize::try_from(block_bytes)
                .map_err(|_| invalid(format_args!("GGUF block bytes for {name} exceeds usize")))?;
            let input = shape[column_axis];
            if !input.is_multiple_of(block_values) {
                return Err(invalid(format_args!(
                    "GGUF matrix {name} input {input} is not aligned to block {block_values}"
                )));
            }
            let mut packed = copy_values(shape, destination)?;
            packed[column_axis] = input / block_values * block_bytes;
            let sharding = remap_linear_segments(
                source.sharding(),
                column_axis,
                block_values,
                name,
                destination,
            )?;
            // Chunk coordinates above are in encoded blocks. GGUF stores byte
            // rows, so retain byte widths and segment offsets in the placement.
            let bytes = |blocks: usize| {
                blocks.checked_mul(block_bytes).ok_or_else(|| {
                    invalid(format_args!("GGUF chunk coordinates for {name} overflow"))
                })
            };
            destination.value_controls(&bytes)?;
            let sharding = match sharding {
                MemberSharding::PartitionedChunks { axis, chunk_size } if axis == column_axis => {
                    MemberSharding::PartitionedChunks {
                        axis,
                        chunk_size: bytes(chunk_size)?,
                    }
                }
                MemberSharding::PartitionedChunkSegments {
                    axis,
                    segments,
                    chunk_size,
                } if axis == column_axis => MemberSharding::PartitionedChunkSegments {
                    axis,
                    segments: {
                        let count = segments.len();
                        collect(
                            segments
                                .into_iter()
                                .map(|segment| Ok(bytes(segment.start)?..bytes(segment.end)?)),
                            count,
                            destination,
                        )?
                    },
                    chunk_size: bytes(chunk_size)?,
                },
                other => other,
            };
            values(
                [member(name, packed, sharding, None, destination)?],
                destination,
            )
        }
        LinearFormat::Affine(_) | LinearFormat::MxFp4 => {
            let quantization = format.weight_quantization().expect("packed format");
            let Some(scale) = declaration.scale() else {
                return Err(destination.invalid_group(format_args!(
                    "packed linear parameter {name} must declare a scale companion"
                )));
            };
            let bias = declaration.affine_bias();
            if quantization.has_biases() != bias.is_some() {
                return Err(destination.invalid_group(format_args!(
                    "packed linear parameter {name} declares companions inconsistent with its format"
                )));
            }
            let bits = usize::try_from(quantization.bits())
                .map_err(|_| invalid(format_args!("packed bit width for {name} exceeds usize")))?;
            let group = usize::try_from(quantization.group_size()).map_err(|_| {
                invalid(format_args!("packed group width for {name} exceeds usize"))
            })?;
            let input = shape[column_axis];
            let packed_bits = input
                .checked_mul(bits)
                .ok_or_else(|| invalid(format_args!("packed matrix {name} overflows")))?;
            if group == 0 || !input.is_multiple_of(group) || !packed_bits.is_multiple_of(32) {
                return Err(invalid(format_args!(
                    "packed matrix {name} input {input} is incompatible with group {group} and {bits} bits"
                )));
            }
            let mut packed = copy_values(shape, destination)?;
            packed[column_axis] = packed_bits / 32;
            let mut companion = copy_values(shape, destination)?;
            companion[column_axis] = input / group;
            let mut members = destination.vector(2 + usize::from(bias.is_some()))?;
            members.push(member(
                name,
                packed,
                remap_linear_segments(
                    source.sharding(),
                    column_axis,
                    32 / bits,
                    name,
                    destination,
                )?,
                None,
                destination,
            )?);
            let companion_sharding =
                remap_linear_segments(source.sharding(), column_axis, group, name, destination)?;
            members.push(member(
                scale.id.as_str(),
                copy_values(&companion, destination)?,
                copy_sharding(&companion_sharding, destination)?,
                Some((eredu_nn::LinearCompanionRole::Scale, name)),
                destination,
            )?);
            if let Some(bias) = bias {
                members.push(member(
                    bias.id.as_str(),
                    companion,
                    companion_sharding,
                    Some((eredu_nn::LinearCompanionRole::AffineBias, name)),
                    destination,
                )?);
            }
            Ok(members)
        }
    }
}

pub(super) fn copy_group(
    source: &ParameterGroupSpec,
    context: &eredu_nn::workspace::WorkspaceContext,
) -> Result<ParameterGroupSpec, eredu_nn::Error> {
    let destination = Checked(context);
    destination.controls::<(
        &ParameterGroupSpec,
        ParameterGroupSpec,
        Vec<ParameterMemberSpec>,
    )>()?;
    let mut members = destination.vector(source.members.len())?;
    for value in &source.members {
        members.push(copy_member(value, destination)?);
    }
    Ok(ParameterGroupSpec {
        logical_name: destination.string(format_args!("{}", source.logical_name))?,
        role: source.role,
        partition_units: source.partition_units,
        members,
    })
}

#[cfg(test)]
mod tests;
