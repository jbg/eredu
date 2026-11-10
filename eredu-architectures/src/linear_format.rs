//! Shared physical-format declarations for architecture operators and parallel plans.

use eredu_checkpoint::LinearFormat;
use eredu_nn::{Error, GroupedProjectionSpec, LinearFormatSpec, ParameterSpec};
use eredu_runtime::ParallelPlanError;

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

/// Projects ordinary matrix storage while preserving retained operator arithmetic.
pub(crate) fn project_matrix_format(
    weight: &ParameterSpec,
    old: &LinearFormatSpec,
    encoding: LinearFormat,
) -> Result<LinearFormatSpec, Error> {
    let name = weight.id.as_str();
    let mut format =
        matrix_linear_format(name, name.strip_suffix(".weight").unwrap_or(name), encoding)?
            .with_arithmetic(old.arithmetic())?;
    if old.exact_zero_inputs() {
        format = format.with_exact_zero_inputs();
    }
    Ok(format)
}

/// Projects storage from packed gate/up semantics while retaining geometry and arithmetic.
pub(crate) fn project_packed_gated_formats(
    bank: &eredu_nn::GroupedGatedProductSpec,
    encoding: impl Fn(&str) -> LinearFormat,
) -> Result<eredu_nn::GroupedGatedProductSpec, Error> {
    let eredu_nn::GatedProductGroupLayout::Packed { gate_up, .. } = bank.layout() else {
        return Err(Error::backend(
            "storage projection requires retained packed gate/up slots",
        ));
    };
    let fused = &gate_up.weight().id;
    bank.map_projections(|p| {
        let row_layout = if &p.weight().id == fused {
            eredu_nn::LinearRowLayout::equal_partitions(2)?
        } else {
            eredu_nn::LinearRowLayout::Contiguous
        };
        project_expert_format(p, encoding(p.weight().id.as_str()), row_layout)
    })
}

/// Projects expert storage while retaining primary/bias ownership and arithmetic.
pub(crate) fn project_expert_format(
    projection: &GroupedProjectionSpec,
    encoding: LinearFormat,
    row_layout: eredu_nn::LinearRowLayout,
) -> Result<GroupedProjectionSpec, Error> {
    let mut format = standard_expert_format(projection.weight().id.as_str(), encoding, row_layout)?
        .with_arithmetic(projection.format().arithmetic())?;
    if projection.format().exact_zero_inputs() {
        format = format.with_exact_zero_inputs();
    }
    projection.clone().with_format(format)
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

pub(crate) fn standard_expert_companion_name(
    weight: &str,
    role: eredu_nn::LinearCompanionRole,
) -> String {
    let suffix = match role {
        eredu_nn::LinearCompanionRole::Scale => "scales",
        eredu_nn::LinearCompanionRole::AffineBias => "biases",
        eredu_nn::LinearCompanionRole::GlobalScale => "global_scale",
    };
    format!("{weight}_{suffix}")
}

/// Declares bank companion identities with explicit block-scaled row origins.
/// Other encodings retain contiguous rows.
pub(crate) fn standard_expert_format(
    weight_name: &str,
    format: LinearFormat,
    row_layout: eredu_nn::LinearRowLayout,
) -> Result<LinearFormatSpec, Error> {
    let declaration = match format {
        LinearFormat::Dense | LinearFormat::GgufIQuant { .. } => LinearFormatSpec::unscaled(format),
        LinearFormat::MxFp4 | LinearFormat::E4M3BlockFp8(_) => LinearFormatSpec::scaled(
            format,
            companion(
                weight_name,
                standard_expert_companion_name(weight_name, eredu_nn::LinearCompanionRole::Scale),
                "scale",
            )?,
        ),
        LinearFormat::NvFp4 => LinearFormatSpec::nvfp4(
            companion(
                weight_name,
                standard_expert_companion_name(weight_name, eredu_nn::LinearCompanionRole::Scale),
                "scale",
            )?,
            companion(
                weight_name,
                standard_expert_companion_name(
                    weight_name,
                    eredu_nn::LinearCompanionRole::GlobalScale,
                ),
                "global-scale",
            )?,
        ),
        LinearFormat::Affine(_) => LinearFormatSpec::affine(
            format,
            companion(
                weight_name,
                standard_expert_companion_name(weight_name, eredu_nn::LinearCompanionRole::Scale),
                "scale",
            )?,
            companion(
                weight_name,
                standard_expert_companion_name(
                    weight_name,
                    eredu_nn::LinearCompanionRole::AffineBias,
                ),
                "affine-bias",
            )?,
        ),
    }?;
    if matches!(format, LinearFormat::E4M3BlockFp8(_) | LinearFormat::NvFp4) {
        declaration.with_row_layout(row_layout)
    } else {
        Ok(declaration)
    }
}

/// Binds a selected encoding to its retained output identities, without naming policy.
pub(crate) fn selected_linear_format(
    weight: &ParameterSpec,
    task: &eredu_runtime::ReplicatedTextMaterializationTask,
    row_layout: eredu_nn::LinearRowLayout,
) -> Result<LinearFormatSpec, Error> {
    if task.name() != weight.id.as_str() {
        return Err(Error::backend(
            "selected format references another primary parameter",
        ));
    }
    let mut scale = None;
    let mut bias = None;
    let mut global = None;
    for output in task.output_companions() {
        let slot = match output.role() {
            eredu_nn::LinearCompanionRole::Scale => &mut scale,
            eredu_nn::LinearCompanionRole::AffineBias => &mut bias,
            eredu_nn::LinearCompanionRole::GlobalScale => &mut global,
        };
        if slot
            .replace(companion(
                task.name(),
                output.name().to_owned(),
                "selected output",
            )?)
            .is_some()
        {
            return Err(Error::backend("selected format repeats a companion role"));
        }
    }
    let required = |slot: Option<ParameterSpec>| {
        slot.ok_or_else(|| Error::backend("selected format omits a required companion"))
    };
    let encoding = task.executable();
    let format = match encoding {
        LinearFormat::Dense | LinearFormat::GgufIQuant { .. } => {
            if scale.is_some() || bias.is_some() || global.is_some() {
                return Err(Error::backend(
                    "unscaled selected format declares companions",
                ));
            }
            LinearFormatSpec::unscaled(encoding)
        }
        LinearFormat::Affine(_) => {
            if global.is_some() {
                return Err(Error::backend(
                    "affine selected format declares a global scale",
                ));
            }
            LinearFormatSpec::affine(encoding, required(scale)?, required(bias)?)
        }
        LinearFormat::MxFp4 | LinearFormat::E4M3BlockFp8(_) => {
            if bias.is_some() || global.is_some() {
                return Err(Error::backend(
                    "scaled selected format declares unexpected companions",
                ));
            }
            LinearFormatSpec::scaled(encoding, required(scale)?)
        }
        LinearFormat::NvFp4 => {
            if bias.is_some() {
                return Err(Error::backend(
                    "NVFP4 selected format declares an affine bias",
                ));
            }
            LinearFormatSpec::nvfp4(required(scale)?, required(global)?)
        }
    }?;
    let format = if matches!(
        encoding,
        LinearFormat::E4M3BlockFp8(_) | LinearFormat::NvFp4
    ) {
        format.with_row_layout(row_layout)?
    } else {
        format
    };
    format.validate_for_weight(weight)?;
    Ok(format)
}

/// Declares one expert projection using the repository's checkpoint convention.
///
/// The returned neutral specification contains the complete topology. Backends
/// consume these identities literally and never reconstruct this convention.
/// The typed projection supplies independent row origins for block-scaled formats.
pub(crate) fn standard_expert_projection(
    weight_name: &str,
    bias_name: Option<&str>,
    format: LinearFormat,
    row_layout: eredu_nn::LinearRowLayout,
) -> Result<GroupedProjectionSpec, Error> {
    GroupedProjectionSpec::new(
        expert_parameter(weight_name)?,
        bias_name.map(expert_parameter).transpose()?,
        standard_expert_format(weight_name, format, row_layout)?,
    )
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
        let LinearFormat::E4M3BlockFp8(format) = format_of(owner.as_str()) else {
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
                ));
            }
        };
        let divisor = match member.global_shape().len().checked_sub(axis) {
            Some(2) => format.block_rows as usize,
            Some(1) => format.block_columns as usize,
            _ => {
                return Err(ParallelPlanError::InvalidTensor(
                    "dense FFN companion axis is not a matrix axis".into(),
                ));
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
