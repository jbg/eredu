//! Shared checkpoint companion declarations for architecture parallel plans.

use eredu_checkpoint::LinearFormat;
use eredu_nn::{
    Error, ExpertProjectionSpec, ExpertQuantizationSpec, ParameterSpec, ParameterTopologyError,
};
use eredu_runtime::{LinearFormatParameter, ParameterMemberSpec};

fn expert_parameter(name: &str) -> Result<ParameterSpec, Error> {
    ParameterSpec::trainable(name).map_err(Error::backend)
}

fn expert_companion(
    weight: &ParameterSpec,
    component: &str,
) -> Result<ParameterSpec, ParameterTopologyError> {
    let mut companion = ParameterSpec::trainable(format!("{}_{component}", weight.id.as_str()))?;
    companion.trainable = weight.trainable;
    companion.group = Some(weight.id.as_str().to_owned());
    Ok(companion)
}

/// Declares one expert projection using the repository's checkpoint convention.
///
/// The returned neutral specification contains the complete topology. Backends
/// consume these identities literally and never reconstruct this convention.
pub(crate) fn standard_expert_projection(
    weight_name: &str,
    bias_name: Option<&str>,
    format: LinearFormat,
) -> Result<ExpertProjectionSpec, Error> {
    let weight = expert_parameter(weight_name)?;
    let quantization = match format {
        LinearFormat::Dense | LinearFormat::GgufIQuant { .. } => None,
        LinearFormat::MxFp4 | LinearFormat::E4M3BlockFp8(_) => Some(ExpertQuantizationSpec {
            scales: expert_companion(&weight, "scales").map_err(Error::backend)?,
            biases: None,
        }),
        LinearFormat::Affine(_) => Some(ExpertQuantizationSpec {
            scales: expert_companion(&weight, "scales").map_err(Error::backend)?,
            biases: Some(expert_companion(&weight, "biases").map_err(Error::backend)?),
        }),
    };
    Ok(ExpertProjectionSpec {
        weight,
        bias: bias_name.map(expert_parameter).transpose()?,
        format,
        quantization,
    })
}

/// Declares the canonical dense-matrix and packed-expert companion convention.
pub(crate) fn standard_linear_format_parameter(
    member: &ParameterMemberSpec,
    format: LinearFormat,
) -> Option<LinearFormatParameter> {
    if format == LinearFormat::Dense || member.global_shape().len() < 2 {
        return None;
    }
    let name = member.target();
    let (prefix, expert_bank) = if let Some(prefix) = name.strip_suffix(".weight") {
        (prefix, false)
    } else if name.ends_with(".gate_up_proj") || name.ends_with(".down_proj") {
        (name, true)
    } else {
        return None;
    };
    match format {
        LinearFormat::Dense => None,
        LinearFormat::GgufIQuant { .. } => Some(LinearFormatParameter::unscaled(format)),
        LinearFormat::E4M3BlockFp8(_) => Some(LinearFormatParameter::scaled(
            format,
            if expert_bank {
                format!("{prefix}_scales")
            } else {
                format!("{prefix}.weight_scale_inv")
            },
        )),
        LinearFormat::MxFp4 => Some(LinearFormatParameter::scaled(
            format,
            if expert_bank {
                format!("{prefix}_scales")
            } else {
                format!("{prefix}.scales")
            },
        )),
        LinearFormat::Affine(_) => Some(LinearFormatParameter::affine(
            format,
            if expert_bank {
                format!("{prefix}_scales")
            } else {
                format!("{prefix}.scales")
            },
            if expert_bank {
                format!("{prefix}_biases")
            } else {
                format!("{prefix}.biases")
            },
        )),
    }
}
