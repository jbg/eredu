//! The same Mamba declarations feed ordinary and retained target construction.
use super::*;
use crate::decoder::{ModuleMetadata, construction_specs::{copy_linear, copy_parameter, copy_convolution}};
use super::super::attention::linear_spec_with_metadata;
#[derive(Debug)]
pub(crate) struct MambaSpec {
    input: LinearSpec,
    convolution: CausalDepthwiseConvolutionSpec,
    dt_bias: ParameterSpec,
    a_log: ParameterSpec,
    d: ParameterSpec,
    norm: ParameterSpec,
    output: LinearSpec,
    heads: i32,
    groups: i32,
    head_dim: i32,
    state_dim: i32,
    intermediate: i32,
    convolution_width: i32,
    chunk_size: usize,
    time_step_floor: f32,
    epsilon: f32,
}
impl MambaSpec {
    pub(crate) fn new(args: &ModelArgs, layer: usize, heads: i32, groups: i32) -> Result<Self, Error> {
        Self::new_with_metadata(args, layer, heads, groups, ModuleMetadata::ordinary())
    }
    pub(crate) fn new_with_metadata(
        args: &ModelArgs, layer: usize, heads: i32, groups: i32, metadata: ModuleMetadata<'_>,
    ) -> Result<Self, Error> {
        metadata.controls::<(Self, &ModelArgs, usize, i32, i32, String, LinearSpec, CausalDepthwiseConvolutionSpec)>()?;
        let intermediate = heads.checked_mul(args.mamba_head_dim)
            .ok_or_else(|| metadata.error(format_args!("Mamba intermediate width overflowed")))?;
        let convolution_width = groups.checked_mul(args.ssm_state_size)
            .and_then(|width| width.checked_mul(2)).and_then(|width| intermediate.checked_add(width))
            .ok_or_else(|| metadata.error(format_args!("Mamba convolution width overflowed")))?;
        let projection = intermediate.checked_add(convolution_width)
            .and_then(|width| width.checked_add(heads))
            .ok_or_else(|| metadata.error(format_args!("Mamba input projection width overflowed")))?;
        let prefix = metadata.text(format_args!("model.layers.{layer}.mamba"))?;
        let parameter = |field: &str| metadata.named_parameter(format_args!("{prefix}.{field}"));
        metadata.borrowed_controls(&parameter)?;
        Ok(Self {
            input: linear_spec_with_metadata(args, &metadata.text(format_args!("{prefix}.in_proj"))?, args.hidden_size, projection, args.use_bias, metadata)?,
            convolution: CausalDepthwiseConvolutionSpec {
                channels: convolution_width, kernel_size: args.conv_kernel,
                weight: parameter("conv1d.weight")?,
                bias: args.use_conv_bias.then(|| parameter("conv1d.bias")).transpose()?,
                activation: ConvolutionActivation::Silu,
            },
            dt_bias: parameter("dt_bias")?, a_log: parameter("A_log")?, d: parameter("D")?, norm: parameter("norm.weight")?,
            output: linear_spec_with_metadata(args, &metadata.text(format_args!("{prefix}.out_proj"))?, intermediate, args.hidden_size, args.use_bias, metadata)?,
            heads, groups, head_dim: args.mamba_head_dim, state_dim: args.ssm_state_size,
            intermediate, convolution_width,
            chunk_size: usize::try_from(args.chunk_size).map_err(|cause| metadata.error(format_args!("{cause}")))?,
            time_step_floor: args.time_step_min, epsilon: args.layer_norm_epsilon,
        })
    }
    pub(crate) fn instantiate<B: NeuralBackend>(&self, context: &<B::Tensor as Tensor>::Context) -> Result<Mamba2<B>, Error> {
        ModuleMetadata::new::<B>(context).controls::<(&Self, Mamba2<B>, [i32; 1])>()?;
        Ok(Mamba2 {
            in_proj: B::linear(copy_linear::<B>(&self.input, context)?, context)?,
            conv1d: CausalDepthwiseConvolution::new(copy_convolution::<B>(&self.convolution, context)?, context)?,
            dt_bias: Parameter::unloaded(copy_parameter::<B>(&self.dt_bias, context)?, &[self.heads], context)?,
            a_log: Parameter::unloaded(copy_parameter::<B>(&self.a_log, context)?, &[self.heads], context)?,
            d: Parameter::unloaded(copy_parameter::<B>(&self.d, context)?, &[self.heads], context)?,
            norm_weight: Parameter::unloaded(copy_parameter::<B>(&self.norm, context)?, &[self.intermediate], context)?,
            out_proj: B::linear(copy_linear::<B>(&self.output, context)?, context)?,
            heads: self.heads, groups: self.groups, head_dim: self.head_dim, state_dim: self.state_dim,
            intermediate: self.intermediate, convolution_width: self.convolution_width, chunk_size: self.chunk_size,
            time_step_floor: self.time_step_floor, epsilon: self.epsilon,
        })
    }
}
