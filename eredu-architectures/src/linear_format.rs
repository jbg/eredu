//! Shared checkpoint companion declarations for architecture parallel plans.

use eredu_checkpoint::LinearFormat;
use eredu_runtime::{LinearFormatParameter, ParameterMemberSpec};

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
