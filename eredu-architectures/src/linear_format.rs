//! Shared physical-format declarations for architecture operators and parallel plans.

use eredu_checkpoint::LinearFormat;
use eredu_nn::{Error, GroupedProjectionSpec, LinearFormatSpec, ParameterSpec};
use eredu_runtime::{ParallelPlanError, ParameterMemberSpec};

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
    let prefix = weight_name.strip_suffix(".weight").ok_or_else(|| {
        Error::backend(format!(
            "encoded ordinary linear parameter {weight_name:?} must end in .weight"
        ))
    });
    match format {
        LinearFormat::Dense | LinearFormat::GgufIQuant { .. } => LinearFormatSpec::unscaled(format),
        LinearFormat::E4M3BlockFp8(_) => {
            let prefix = prefix?;
            LinearFormatSpec::scaled(
                format,
                companion(weight_name, format!("{prefix}.weight_scale_inv"), "scale")?,
            )
        }
        LinearFormat::MxFp4 => {
            let prefix = prefix?;
            LinearFormatSpec::scaled(
                format,
                companion(weight_name, format!("{prefix}.scales"), "scale")?,
            )
        }
        LinearFormat::Affine(_) => {
            let prefix = prefix?;
            LinearFormatSpec::affine(
                format,
                companion(weight_name, format!("{prefix}.scales"), "scale")?,
                companion(weight_name, format!("{prefix}.biases"), "affine-bias")?,
            )
        }
    }
}

fn standard_expert_format(
    weight_name: &str,
    format: LinearFormat,
) -> Result<LinearFormatSpec, Error> {
    match format {
        LinearFormat::Dense | LinearFormat::GgufIQuant { .. } => LinearFormatSpec::unscaled(format),
        LinearFormat::MxFp4 | LinearFormat::E4M3BlockFp8(_) => LinearFormatSpec::scaled(
            format,
            companion(weight_name, format!("{weight_name}_scales"), "scale")?,
        ),
        LinearFormat::Affine(_) => LinearFormatSpec::affine(
            format,
            companion(weight_name, format!("{weight_name}_scales"), "scale")?,
            companion(weight_name, format!("{weight_name}_biases"), "affine-bias")?,
        ),
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
