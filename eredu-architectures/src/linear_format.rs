//! Shared physical-format declarations for architecture operators and parallel plans.

use crate::decoder::parameter_metadata::{DeclarationDestination, ParameterGroupError};
use eredu_checkpoint::LinearFormat;
use eredu_nn::{Error, GroupedProjectionSpec, LinearFormatSpec, ParameterSpec};
use eredu_runtime::{ParallelPlanError, ParameterMemberSpec};

pub(crate) fn companion(
    weight_name: &str,
    companion_name: String,
    component: &str,
) -> Result<ParameterSpec, Error> {
    let mut companion = ParameterSpec::trainable(companion_name).map_err(Error::backend)?;
    companion.group = Some(weight_name.to_owned());
    if companion.id.as_str() == weight_name {
        return Err(Error::backend(format!(
            "linear {component} companion reuses weight identity {weight_name:?}"
        )));
    }
    Ok(companion)
}

/// Declares one ordinary matrix's encoding and exact physical companions.
pub(crate) fn standard_linear_format(
    weight_name: &str,
    format: LinearFormat,
) -> Result<LinearFormatSpec, Error> {
    crate::decoder::static_construction::format::standard_linear_format_with(
        weight_name,
        format,
        &mut crate::decoder::static_construction::format::OrdinaryFormat,
    )
}

pub(crate) fn standard_expert_format(
    weight_name: &str,
    format: LinearFormat,
) -> Result<LinearFormatSpec, Error> {
    expert_format_with(
        weight_name,
        format,
        crate::decoder::ModuleMetadata::ordinary(),
    )
}
fn expert_format_with(
    weight_name: &str,
    format: LinearFormat,
    metadata: crate::decoder::ModuleMetadata<'_>,
) -> Result<LinearFormatSpec, Error> {
    metadata.controls::<(
        LinearFormatSpec,
        Option<ParameterSpec>,
        Option<ParameterSpec>,
    )>()?;
    let companion = |suffix: &str| -> Result<ParameterSpec, Error> {
        let mut spec = metadata.named_parameter(format_args!("{weight_name}{suffix}"))?;
        spec.group = Some(metadata.text(format_args!("{weight_name}"))?);
        Ok(spec)
    };
    let (scale, bias) = match format {
        LinearFormat::Dense | LinearFormat::GgufIQuant { .. } => (None, None),
        LinearFormat::MxFp4 | LinearFormat::E4M3BlockFp8(_) => (Some(companion("_scales")?), None),
        LinearFormat::Affine(_) => (Some(companion("_scales")?), Some(companion("_biases")?)),
    };
    let declaration = metadata.declared_format(format, scale, bias)?;
    if matches!(format, LinearFormat::E4M3BlockFp8(_)) && weight_name.ends_with(".gate_up_proj") {
        declaration.with_row_layout(eredu_nn::LinearRowLayout::equal_partitions(2)?)
    } else {
        Ok(declaration)
    }
}

/// Declares one expert projection using the repository's checkpoint convention.
pub(crate) fn standard_expert_projection(
    weight_name: &str,
    bias_name: Option<&str>,
    format: LinearFormat,
) -> Result<GroupedProjectionSpec, Error> {
    standard_expert_projection_with(
        weight_name,
        bias_name,
        format,
        crate::decoder::ModuleMetadata::ordinary(),
    )
}
/// Same names, companions and row geometry, with the caller's funded producer.
pub(crate) fn standard_expert_projection_with(
    weight_name: &str,
    bias_name: Option<&str>,
    format: LinearFormat,
    metadata: crate::decoder::ModuleMetadata<'_>,
) -> Result<GroupedProjectionSpec, Error> {
    metadata.controls::<GroupedProjectionSpec>()?;
    GroupedProjectionSpec::new(
        metadata.plain_parameter(weight_name)?,
        bias_name
            .map(|name| metadata.plain_parameter(name))
            .transpose()?,
        expert_format_with(weight_name, format, metadata)?,
    )
}

/// Declares the canonical dense-matrix and packed-expert companion convention.
pub(crate) fn standard_parallel_linear_format(
    member: &ParameterMemberSpec,
    format: LinearFormat,
) -> Result<Option<LinearFormatSpec>, ParallelPlanError> {
    standard_parallel_linear_format_with(member, format, DeclarationDestination(None))
        .map_err(ParameterGroupError::ordinary)
}

/// The same physical companion declaration with a caller-owned metadata destination.
pub(crate) fn standard_parallel_linear_format_with(
    member: &ParameterMemberSpec,
    format: LinearFormat,
    destination: DeclarationDestination<'_>,
) -> Result<Option<LinearFormatSpec>, ParameterGroupError> {
    destination.controls::<(&ParameterMemberSpec, LinearFormat, Option<LinearFormatSpec>,
        crate::decoder::ModuleMetadata<'_>, Result<LinearFormatSpec, Error>, Error,
        &str, &str, bool)>()?;
    if format == LinearFormat::Dense || member.global_shape().len() < 2 {
        return Ok(None);
    }
    let name = member.target();
    let (prefix, expert_bank) = if let Some(prefix) = name.strip_suffix(".weight") {
        (prefix, false)
    } else if name.ends_with(".gate_up_proj") || name.ends_with(".down_proj") {
        (name, true)
    } else {
        return Ok(None);
    };
    let mut metadata = crate::decoder::ModuleMetadata::destination(destination.0);
    let declaration = if expert_bank {
        expert_format_with(prefix, format, metadata)
    } else {
        crate::decoder::static_construction::format::standard_linear_format_with(
            name, format, &mut metadata)
    }.map_err(|error| match destination.0 {
        Some(_) => ParameterGroupError::Metadata(error),
        None => ParameterGroupError::Ordinary(ParallelPlanError::InvalidGroup(error.to_string())),
    })?;
    Ok(Some(declaration))
}

/// Input-axis block granularity required to retain encoded projection slices.
pub(crate) fn input_partition_alignment(format: LinearFormat) -> i32 {
    match format {
        LinearFormat::Dense => 1,
        LinearFormat::E4M3BlockFp8(block) => block.block_columns,
        _ => format
            .weight_quantization()
            .expect("packed format")
            .group_size(),
    }
}

/// Retains a complete partial FP8 block while requiring aligned encoded splits.
pub(crate) fn input_partition_units(
    name: &str,
    semantic_units: usize,
    elements_per_unit: usize,
    format: LinearFormat,
) -> Result<usize, ParallelPlanError> {
    input_partition_units_with(
        name,
        semantic_units,
        elements_per_unit,
        format,
        DeclarationDestination(None),
    )
    .map_err(ParameterGroupError::ordinary)
}
pub(crate) fn input_partition_units_with(
    name: &str,
    semantic_units: usize,
    elements_per_unit: usize,
    format: LinearFormat,
    destination: DeclarationDestination<'_>,
) -> Result<usize, ParameterGroupError> {
    let alignment = usize::try_from(input_partition_alignment(format))
        .map_err(|error| destination.group_error(format_args!("{error}")))?;
    if let Some(context) = destination.0 {
        return eredu_runtime::aligned_partition_units_with_metadata(
            name,
            semantic_units,
            elements_per_unit,
            alignment,
            matches!(format, LinearFormat::E4M3BlockFp8(_)),
            context,
        )
        .map_err(ParameterGroupError::Metadata);
    }
    match format {
        LinearFormat::E4M3BlockFp8(_) => eredu_runtime::aligned_partition_units_with_tail(
            name,
            semantic_units,
            elements_per_unit,
            alignment,
        ),
        _ => eredu_runtime::aligned_partition_units(
            name,
            semantic_units,
            elements_per_unit,
            alignment,
        ),
    }
    .map_err(ParameterGroupError::Ordinary)
}

/// Retains complete FP8 input blocks and a short final block in a dense FFN.
/// Read rows, output columns and declared scale companions share the same
/// logical chunks. The cold declaration has no companions yet; format
/// expansion applies the same physical coordinate conversion to those members.
pub(crate) fn dense_ffn_partition_tail(
    group: eredu_runtime::ParameterGroupSpec,
    intermediate: usize,
    output_format: LinearFormat,
    format_of: impl Fn(&str) -> LinearFormat,
) -> Result<eredu_runtime::ParameterGroupSpec, ParallelPlanError> {
    dense_ffn_partition_tail_with(
        group,
        intermediate,
        output_format,
        |name| Ok(format_of(name)),
        DeclarationDestination(None),
    )
    .map_err(ParameterGroupError::ordinary)
}
pub(crate) fn dense_ffn_partition_tail_with(
    group: eredu_runtime::ParameterGroupSpec,
    intermediate: usize,
    output_format: LinearFormat,
    format_of: impl Fn(&str) -> Result<LinearFormat, ParameterGroupError>,
    destination: DeclarationDestination<'_>,
) -> Result<eredu_runtime::ParameterGroupSpec, ParameterGroupError> {
    destination.controls::<eredu_runtime::ParameterGroupSpec>()?;
    let LinearFormat::E4M3BlockFp8(block) = output_format else {
        return Ok(group);
    };
    block
        .validate()
        .map_err(|error| destination.group_error(format_args!("{error}")))?;
    let chunk = usize::try_from(block.block_columns)
        .map_err(|error| destination.group_error(format_args!("{error}")))?;
    if intermediate.is_multiple_of(chunk) {
        return Ok(group);
    }
    destination.chunks(group, intermediate.div_ceil(chunk), |member, _source| {
        let Some(owner) = member.linear_companion_of() else {
            return Ok(chunk);
        };
        let LinearFormat::E4M3BlockFp8(format) = format_of(owner)? else {
            // Dense FFN read projections shard output rows. Affine and MXFP4
            // companions retain these rows; only their input axis is packed.
            return Ok(chunk);
        };
        format
            .validate()
            .map_err(|error| destination.group_error(format_args!("{error}")))?;
        let axis = match member.sharding() {
            eredu_runtime::MemberSharding::Partitioned { axis }
            | eredu_runtime::MemberSharding::PartitionedSegments { axis, .. } => *axis,
            _ => {
                return Err(destination
                    .tensor_error(format_args!("dense FFN companion has no partition axis")));
            }
        };
        let divisor = match member.global_shape().len().checked_sub(axis) {
            Some(2) => format.block_rows as usize,
            Some(1) => format.block_columns as usize,
            _ => {
                return Err(destination.tensor_error(format_args!(
                    "dense FFN companion axis is not a matrix axis"
                )));
            }
        };
        if !chunk.is_multiple_of(divisor) {
            return Err(
                destination.tensor_error(format_args!("dense FFN chunk splits a scale block"))
            );
        }
        Ok(chunk / divisor)
    })
}

/// Canonical executable encoding of one physical GGUF matrix.
pub(crate) fn gguf_tensor_format(
    tensor: &eredu_gguf::CatalogTensor,
    endian: eredu_gguf::Endian,
) -> Result<LinearFormat, String> {
    if let Some((bits, group)) = tensor.affine() {
        Ok(LinearFormat::Affine(
            eredu_checkpoint::AffineQuantization::new(
                i32::try_from(group).map_err(|_| "GGUF group size exceeds i32")?,
                i32::from(bits),
            )
            .map_err(|e| e.to_string())?,
        ))
    } else if tensor.is_mxfp4() {
        Ok(LinearFormat::MxFp4)
    } else if tensor.descriptor().ggml_type.block_and_bytes().is_ok()
        && !matches!(
            tensor.descriptor().ggml_type,
            eredu_gguf::GgmlType::F16 | eredu_gguf::GgmlType::F32 | eredu_gguf::GgmlType::Bf16
        )
    {
        Ok(LinearFormat::GgufIQuant {
            ggml_type: tensor.descriptor().ggml_type,
            endian,
        })
    } else {
        Ok(LinearFormat::Dense)
    }
}
