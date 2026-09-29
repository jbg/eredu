//! Qwen hybrid parameter identities and formats for the shared recurrent mixer.
use super::{HybridConfig, HybridVariant};
use crate::gated_delta::GatedDeltaMixerSpec;
use eredu_nn::{
    CausalDepthwiseConvolutionSpec, ConvolutionActivation, Error, LinearSpec, OutputGateActivation,
    ParameterSpec,
};

/// Declares the rank-local mixer using the family's normalized configuration.
pub fn recurrent_spec(config: &HybridConfig, layer: usize) -> Result<GatedDeltaMixerSpec, Error> {
    let key_heads = config.linear_num_key_heads;
    let value_heads = config.linear_num_value_heads;
    if key_heads <= 0 || value_heads <= 0 || value_heads % key_heads != 0 {
        return Err(Error::backend(format!(
            "invalid rank-local recurrent heads: key={key_heads}, value={value_heads}"
        )));
    }
    let key_width = key_heads
        .checked_mul(config.linear_key_head_dim)
        .ok_or_else(|| Error::backend("recurrent key width overflowed"))?;
    let value_width = value_heads
        .checked_mul(config.linear_value_head_dim)
        .ok_or_else(|| Error::backend("recurrent value width overflowed"))?;
    let qkv_width = key_width
        .checked_mul(2)
        .and_then(|width| width.checked_add(value_width))
        .ok_or_else(|| Error::backend("recurrent QKV width overflowed"))?;
    let prefix = format!("model.layers.{layer}.linear_attn");
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
        key_heads,
        value_heads,
        key_head_dim: config.linear_key_head_dim,
        value_head_dim: config.linear_value_head_dim,
        input_qkv: linear("in_proj_qkv", config.hidden_size, qkv_width, false)?,
        input_gate: linear("in_proj_z", config.hidden_size, value_width, false)?,
        input_beta: linear("in_proj_b", config.hidden_size, value_heads, dense_ba)?,
        input_decay: linear("in_proj_a", config.hidden_size, value_heads, dense_ba)?,
        convolution: CausalDepthwiseConvolutionSpec {
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
