//! Shared physical-format declarations for architecture operators and parallel plans.

use eredu_checkpoint::LinearFormat;
use eredu_nn::{Error, GroupedProjectionSpec, LinearFormatSpec, ParameterSpec};
use eredu_runtime::{ParallelPlanError, ParameterMemberSpec};

/// Packed formats do not establish activation precision from integer storage.
/// Retain the portable floating fallback envelope until a compute contract
/// narrows it. Dense and affine formats obtain precision from their sources.
pub(crate) fn encoded_floating_compute_precisions(
    format: LinearFormat,
) -> &'static [eredu_checkpoint::recipe::RecipeDtype] {
    use eredu_checkpoint::recipe::RecipeDtype;
    match format {
        LinearFormat::Dense | LinearFormat::Affine(_) => &[],
        _ => &[RecipeDtype::F16, RecipeDtype::BF16],
    }
}

/// Floating types that can reach selected projections, including FP32 equation
/// promotion. Packed integer weights are not activation precisions; their
/// admitted companions and conversion-input metadata supply the compute types.
pub(crate) fn floating_compute_precisions(
    tasks: &[eredu_runtime::ReplicatedTextMaterializationTask],
) -> Vec<eredu_core::checkpoint::TensorDtype> {
    use eredu_checkpoint::recipe::RecipeDtype;
    use eredu_core::checkpoint::TensorDtype;
    let mut precisions = Vec::new();
    let mut include = |dtype: RecipeDtype| {
        let dtype = match dtype {
            RecipeDtype::F16 => TensorDtype::F16,
            RecipeDtype::BF16 => TensorDtype::Bf16,
            RecipeDtype::F32 => TensorDtype::F32,
            _ => return,
        };
        if !precisions.contains(&dtype) {
            precisions.push(dtype);
        }
    };
    for task in tasks {
        for dtype in encoded_floating_compute_precisions(task.executable()) {
            include(dtype.clone());
        }
        if let Some(dtype) = task.source_encoding().scalar_dtype().map(Into::into) {
            include(dtype);
        }
        if let Some(metadata) = task.derived_output() {
            include(metadata.dtype.clone());
        }
        for companion in task.output_companions() {
            if let Some(dtype) = companion
                .catalog_source()
                .and_then(|source| source.source_encoding().scalar_dtype())
                .map(Into::into)
            {
                include(dtype);
            }
            if let Some(metadata) = companion.derived_output() {
                include(metadata.dtype.clone());
            }
        }
    }
    include(RecipeDtype::F32);
    // Stable canonical order independent of physical tensor enumeration.
    [TensorDtype::F16, TensorDtype::Bf16, TensorDtype::F32]
        .into_iter()
        .filter(|dtype| precisions.contains(dtype))
        .collect()
}

fn expert_parameter(name: &str) -> Result<ParameterSpec, Error> {
    ParameterSpec::trainable(name).map_err(Error::backend)
}

fn companion(
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
    if matches!(
        format,
        LinearFormat::Dense | LinearFormat::GgufIQuant { .. }
    ) {
        return LinearFormatSpec::unscaled(format);
    }
    let prefix = weight_name.strip_suffix(".weight").ok_or_else(|| {
        Error::backend(format!(
            "encoded ordinary linear parameter {weight_name:?} must end in .weight"
        ))
    })?;
    matrix_linear_format(weight_name, prefix, format)
}

/// Declares a matrix whose companion prefix is explicit, including named codebooks.
/// Parameter identities and companion ownership remain architecture-declared.
pub(crate) fn matrix_linear_format(
    weight_name: &str,
    prefix: &str,
    format: LinearFormat,
) -> Result<LinearFormatSpec, Error> {
    match format {
        LinearFormat::Dense | LinearFormat::GgufIQuant { .. } => LinearFormatSpec::unscaled(format),
        LinearFormat::E4M3BlockFp8(_) => LinearFormatSpec::scaled(
            format,
            companion(weight_name, format!("{prefix}.weight_scale_inv"), "scale")?,
        ),
        LinearFormat::MxFp4 => LinearFormatSpec::scaled(
            format,
            companion(weight_name, format!("{prefix}.scales"), "scale")?,
        ),
        LinearFormat::NvFp4 => LinearFormatSpec::nvfp4(
            companion(weight_name, format!("{prefix}.scales"), "scale")?,
            companion(
                weight_name,
                format!("{prefix}.global_scale"),
                "global-scale",
            )?,
        ),
        LinearFormat::Affine(_) => LinearFormatSpec::affine(
            format,
            companion(weight_name, format!("{prefix}.scales"), "scale")?,
            companion(weight_name, format!("{prefix}.biases"), "affine-bias")?,
        ),
    }
}

pub(crate) fn standard_expert_format(
    weight_name: &str,
    format: LinearFormat,
) -> Result<LinearFormatSpec, Error> {
    let declaration = match format {
        LinearFormat::Dense | LinearFormat::GgufIQuant { .. } => LinearFormatSpec::unscaled(format),
        LinearFormat::MxFp4 | LinearFormat::E4M3BlockFp8(_) => LinearFormatSpec::scaled(
            format,
            companion(weight_name, format!("{weight_name}_scales"), "scale")?,
        ),
        LinearFormat::NvFp4 => LinearFormatSpec::nvfp4(
            companion(weight_name, format!("{weight_name}_scales"), "scale")?,
            companion(
                weight_name,
                format!("{weight_name}_global_scale"),
                "global-scale",
            )?,
        ),
        LinearFormat::Affine(_) => LinearFormatSpec::affine(
            format,
            companion(weight_name, format!("{weight_name}_scales"), "scale")?,
            companion(weight_name, format!("{weight_name}_biases"), "affine-bias")?,
        ),
    }?;
    if matches!(format, LinearFormat::E4M3BlockFp8(_) | LinearFormat::NvFp4)
        && weight_name.ends_with(".gate_up_proj")
    {
        declaration.with_row_layout(eredu_nn::LinearRowLayout::equal_partitions(2)?)
    } else {
        Ok(declaration)
    }
}

/// Declares one expert projection using the repository's checkpoint convention.
///
/// The returned neutral specification contains the complete topology. Backends
/// consume these identities literally and never reconstruct this convention.
pub(crate) fn standard_expert_projection(
    weight_name: &str,
    bias_name: Option<&str>,
    format: LinearFormat,
) -> Result<GroupedProjectionSpec, Error> {
    GroupedProjectionSpec::new(
        expert_parameter(weight_name)?,
        bias_name.map(expert_parameter).transpose()?,
        standard_expert_format(weight_name, format)?,
    )
}

/// Declares the canonical dense-matrix and packed-expert companion convention.
pub(crate) fn standard_parallel_linear_format(
    member: &ParameterMemberSpec,
    format: LinearFormat,
) -> Result<Option<LinearFormatSpec>, ParallelPlanError> {
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
    let declaration = if expert_bank {
        standard_expert_format(prefix, format)
    } else {
        standard_linear_format(name, format)
    }
    .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
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
    let alignment = usize::try_from(input_partition_alignment(format))
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
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
    let LinearFormat::E4M3BlockFp8(block) = output_format else {
        return Ok(group);
    };
    block
        .validate()
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let chunk = usize::try_from(block.block_columns)
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    if intermediate.is_multiple_of(chunk) {
        return Ok(group);
    }
    eredu_runtime::partition_parameter_group_chunks(group, intermediate.div_ceil(chunk), |member| {
        let Some(owner) = member.linear_companion_of() else {
            return Ok(chunk);
        };
        let LinearFormat::E4M3BlockFp8(format) = format_of(owner) else {
            // Dense FFN read projections shard output rows. Affine and MXFP4
            // companions retain these rows; only their input axis is packed.
            return Ok(chunk);
        };
        format
            .validate()
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
        let axis = match member.sharding() {
            eredu_runtime::MemberSharding::Partitioned { axis }
            | eredu_runtime::MemberSharding::PartitionedSegments { axis, .. } => *axis,
            _ => {
                return Err(ParallelPlanError::InvalidTensor(
                    "dense FFN companion has no partition axis".into(),
                ))
            }
        };
        let divisor = match member.global_shape().len().checked_sub(axis) {
            Some(2) => format.block_rows as usize,
            Some(1) => format.block_columns as usize,
            _ => {
                return Err(ParallelPlanError::InvalidTensor(
                    "dense FFN companion axis is not a matrix axis".into(),
                ))
            }
        };
        if !chunk.is_multiple_of(divisor) {
            return Err(ParallelPlanError::InvalidTensor(
                "dense FFN chunk splits a scale block".into(),
            ));
        }
        Ok(chunk / divisor)
    })
}

/// Canonical executable encoding of one physical GGUF matrix.
pub(crate) fn gguf_tensor_format(
    tensor: &eredu_gguf::CatalogTensor,
    endian: eredu_gguf::Endian,
) -> Result<LinearFormat, String> {
    if tensor.is_decoded_f32() {
        return Ok(LinearFormat::Dense);
    }
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
