//! Qwen hybrid parameter identities and formats for the shared recurrent mixer.
use super::{HybridPolicy, HybridVariant};
use crate::gated_delta::GatedDeltaMixerSpec;
use eredu_nn::{
    CausalDepthwiseConvolutionSpec, ConvolutionActivation, Error, LinearSpec, OutputGateActivation,
    ParameterSpec,
};

/// Declares the rank-local mixer using the family's normalized configuration.
pub(super) fn recurrent_spec(
    config: &HybridPolicy,
    root: &str,
) -> Result<GatedDeltaMixerSpec, Error> {
    let geometry = config.recurrent_head_geometry()?;
    let value_width = geometry.value_width();
    let qkv_width = geometry.qkv_width()?;
    let prefix = format!("{root}.linear_attn");
    let parameter = |suffix: &str| {
        ParameterSpec::trainable(format!("{prefix}.{suffix}")).map_err(Error::backend)
    };
    let linear = |suffix: &str, input: i32, output: i32, force_dense: bool| {
        let weight = format!("{prefix}.{suffix}.weight");
        Ok::<_, Error>(LinearSpec {
            input,
            output,
            weight: ParameterSpec::trainable(&weight).map_err(Error::backend)?,
            bias: None,
            format: crate::linear_format::standard_linear_format(
                &weight,
                if force_dense {
                    eredu_checkpoint::LinearFormat::Dense
                } else {
                    config.linear_format(&weight)
                },
            )?,
        })
    };
    let dense_ba = config.variant == HybridVariant::Qwen3Next && config.fp8.is_some();
    let spec = GatedDeltaMixerSpec {
        geometry,
        input_qkv: linear("in_proj_qkv", config.hidden_size, qkv_width, false)?,
        input_gate: linear("in_proj_z", config.hidden_size, value_width, false)?,
        input_beta: linear(
            "in_proj_b",
            config.hidden_size,
            geometry.value_heads(),
            dense_ba,
        )?,
        input_decay: linear(
            "in_proj_a",
            config.hidden_size,
            geometry.value_heads(),
            dense_ba,
        )?,
        convolution: CausalDepthwiseConvolutionSpec {
            arithmetic: eredu_nn::ConvolutionArithmetic::Native,
            channels: qkv_width,
            kernel_size: config.linear_conv_kernel_dim,
            dilation: 1,
            weight: parameter("conv1d.weight")?,
            bias: None,
            activation: ConvolutionActivation::Silu,
        },
        decay_bias: parameter("dt_bias")?,
        transition_log: parameter("A_log")?,
        normalization_weight: parameter("norm.weight")?,
        output: linear("out_proj", value_width, config.hidden_size, false)?,
        l2_epsilon: 1e-6,
        output_epsilon: 1e-6,
        output_gate: OutputGateActivation::Silu,
        output_arithmetic: eredu_nn::OutputGatedNormArithmetic::RoundedNormalization,
    };
    spec.validate()?;
    Ok(spec)
}
