//! Nemotron-H Mamba2 over neutral convolution and selective-scan contracts.

use eredu_core::cache::StateTensorRole;
use eredu_nn::{
    CausalDepthwiseConvolution, CausalDepthwiseConvolutionSpec, ConvolutionActivation, Error,
    Index, LinearOperator, LinearSpec, NeuralBackend, Parameter, ParameterSpec, Parameterized,
    SelectiveStateSpaceScanInput, Tensor,
};
use eredu_runtime::RuntimeStateComponents;

use super::ModelArgs;

/// One unloaded Mamba2 mixer with architecture-authored parameter identity.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Mamba2<B: NeuralBackend> {
    /// Joint gate, convolution-input, and timestep projection.
    pub in_proj: B::Linear,
    /// Shared neutral causal depthwise convolution.
    pub conv1d: CausalDepthwiseConvolution<B>,
    /// Per-head timestep bias.
    pub dt_bias: Parameter<B::Tensor>,
    /// Per-head logarithmic transition magnitude.
    pub a_log: Parameter<B::Tensor>,
    /// Per-head direct skip coefficient.
    pub d: Parameter<B::Tensor>,
    /// Scale for gated grouped RMS normalization.
    pub norm_weight: Parameter<B::Tensor>,
    /// Output projection.
    pub out_proj: B::Linear,
    #[parameter(skip)]
    heads: i32,
    #[parameter(skip)]
    groups: i32,
    #[parameter(skip)]
    head_dim: i32,
    #[parameter(skip)]
    state_dim: i32,
    #[parameter(skip)]
    intermediate: i32,
    #[parameter(skip)]
    convolution_width: i32,
    #[parameter(skip)]
    chunk_size: usize,
    #[parameter(skip)]
    time_step_floor: f32,
    #[parameter(skip)]
    epsilon: f32,
}

impl<B: NeuralBackend> Mamba2<B> {
    /// Builds one global-geometry mixer.
    pub fn new(
        args: &ModelArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_with_geometry(args, layer, args.mamba_num_heads, args.n_groups, context)
    }

    /// Builds one placement-resolved mixer.
    pub fn new_with_geometry(
        args: &ModelArgs,
        layer: usize,
        heads: i32,
        groups: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let intermediate = heads
            .checked_mul(args.mamba_head_dim)
            .ok_or_else(|| Error::backend("Mamba intermediate width overflowed"))?;
        let convolution_width = intermediate
            .checked_add(2 * groups * args.ssm_state_size)
            .ok_or_else(|| Error::backend("Mamba convolution width overflowed"))?;
        let projection = intermediate
            .checked_add(convolution_width)
            .and_then(|width| width.checked_add(heads))
            .ok_or_else(|| Error::backend("Mamba input projection width overflowed"))?;
        let prefix = format!("model.layers.{layer}.mamba");
        let parameter = |field: &str| {
            ParameterSpec::trainable(format!("{prefix}.{field}")).map_err(Error::backend)
        };
        let linear = |field: &str, input, output, bias: bool| {
            let weight = format!("{prefix}.{field}.weight");
            B::linear(
                LinearSpec {
                    input,
                    output,
                    weight: ParameterSpec::trainable(&weight).map_err(Error::backend)?,
                    bias: bias
                        .then(|| ParameterSpec::trainable(format!("{prefix}.{field}.bias")))
                        .transpose()
                        .map_err(Error::backend)?,
                    format: crate::linear_format::standard_linear_format(
                        &weight,
                        args.weight_quantization_for(&weight).into(),
                    )?,
                },
                context,
            )
        };
        Ok(Self {
            in_proj: linear("in_proj", args.hidden_size, projection, args.use_bias)?,
            conv1d: CausalDepthwiseConvolution::new(
                CausalDepthwiseConvolutionSpec {
                    channels: convolution_width,
                    kernel_size: args.conv_kernel,
                    weight: parameter("conv1d.weight")?,
                    bias: args
                        .use_conv_bias
                        .then(|| parameter("conv1d.bias"))
                        .transpose()?,
                    activation: ConvolutionActivation::Silu,
                },
                context,
            )?,
            dt_bias: Parameter::unloaded(parameter("dt_bias")?, &[heads], context)?,
            a_log: Parameter::unloaded(parameter("A_log")?, &[heads], context)?,
            d: Parameter::unloaded(parameter("D")?, &[heads], context)?,
            norm_weight: Parameter::unloaded(parameter("norm.weight")?, &[intermediate], context)?,
            out_proj: linear("out_proj", intermediate, args.hidden_size, args.use_bias)?,
            heads,
            groups,
            head_dim: args.mamba_head_dim,
            state_dim: args.ssm_state_size,
            intermediate,
            convolution_width,
            chunk_size: usize::try_from(args.chunk_size).map_err(Error::backend)?,
            time_step_floor: args.time_step_min,
            epsilon: args.layer_norm_epsilon,
        })
    }

    /// Executes Mamba2 and atomically replaces convolution and recurrent state.
    pub fn forward<S>(
        &mut self,
        input: &B::Tensor,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: RuntimeStateComponents<B>,
    {
        self.forward_inner(input, state, None, context)
    }

    /// Executes the same recurrence with a row-parallel output projection.
    pub fn forward_parallel<S>(
        &mut self,
        input: &B::Tensor,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: RuntimeStateComponents<B>,
    {
        self.forward_inner(input, state, Some(parallel), context)
    }

    fn forward_inner<S>(
        &mut self,
        input: &B::Tensor,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: RuntimeStateComponents<B>,
    {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(Error::backend(format!(
                "Mamba2 expects [batch, sequence, hidden], got {shape:?}"
            )));
        }
        let (batch, sequence) = (shape[0], shape[1]);
        let projected = self.in_proj.forward(input, context)?;
        let gate_end = self.intermediate;
        let convolution_end = gate_end + self.convolution_width;
        let gate = projected.index(
            &[Index::Full, Index::Full, Index::Range(0, gate_end)],
            context,
        )?;
        let convolution_input = projected.index(
            &[
                Index::Full,
                Index::Full,
                Index::Range(gate_end, convolution_end),
            ],
            context,
        )?;
        let time_step = projected.index(
            &[
                Index::Full,
                Index::Full,
                Index::Range(convolution_end, convolution_end + self.heads),
            ],
            context,
        )?;
        let convolved = {
            let history = state
                .fixed_component(StateTensorRole::Convolution { slot: 0 })
                .map_err(Error::backend)?;
            self.conv1d
                .forward(&convolution_input, history.as_ref(), context)?
        };
        *state
            .fixed_component(StateTensorRole::Convolution { slot: 0 })
            .map_err(Error::backend)? = convolved.history;
        let values = convolved.output.index(
            &[Index::Full, Index::Full, Index::Range(0, self.intermediate)],
            context,
        )?;
        let input_start = self.intermediate;
        let output_start = input_start + self.groups * self.state_dim;
        let input_state = convolved.output.index(
            &[
                Index::Full,
                Index::Full,
                Index::Range(input_start, output_start),
            ],
            context,
        )?;
        let output_state = convolved.output.index(
            &[
                Index::Full,
                Index::Full,
                Index::Range(output_start, self.convolution_width),
            ],
            context,
        )?;
        let expand_groups = |value: B::Tensor| -> Result<B::Tensor, Error> {
            let repeats = self.heads / self.groups;
            let grouped =
                value.reshape(&[batch, sequence, self.groups, self.state_dim], context)?;
            if repeats == 1 {
                return Ok(grouped);
            }
            grouped
                .expand_dims(3, context)?
                .broadcast_to(
                    &[batch, sequence, self.groups, repeats, self.state_dim],
                    context,
                )?
                .reshape(&[batch, sequence, self.heads, self.state_dim], context)
        };
        let values = values.reshape(&[batch, sequence, self.heads, self.head_dim], context)?;
        let input_state = expand_groups(input_state)?;
        let output_state = expand_groups(output_state)?;
        let scan = {
            let recurrent = state
                .fixed_component(StateTensorRole::Recurrent)
                .map_err(Error::backend)?;
            B::selective_state_space_scan(
                SelectiveStateSpaceScanInput {
                    values: &values,
                    input_state: &input_state,
                    output_state: &output_state,
                    time_step: &time_step,
                    time_step_bias: self.dt_bias.as_ref(),
                    transition_log: self.a_log.as_ref(),
                    skip: self.d.as_ref(),
                    initial_state: recurrent.as_ref(),
                    time_step_floor: self.time_step_floor,
                    chunk_size: self.chunk_size,
                },
                context,
            )?
        };
        *state
            .fixed_component(StateTensorRole::Recurrent)
            .map_err(Error::backend)? = Some(scan.state);
        state.advance_fixed(sequence).map_err(Error::backend)?;
        let scan = scan
            .output
            .reshape(&[batch, sequence, self.intermediate], context)?;
        let normalized = B::gated_group_rms_norm(
            &scan,
            &gate,
            self.norm_weight.as_ref(),
            self.groups,
            self.epsilon,
            context,
        )?;
        match parallel {
            Some(parallel) => {
                B::row_parallel_linear(&mut self.out_proj, &normalized, parallel, context)
            }
            None => self.out_proj.forward(&normalized, context),
        }
    }
}
