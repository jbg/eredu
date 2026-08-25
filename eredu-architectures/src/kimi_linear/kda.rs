//! Kimi Delta Attention composed from neutral projection, convolution, and scan contracts.

use eredu_core::cache::StateTensorRole;
use eredu_nn::{
    CausalDepthwiseConvolution, CausalDepthwiseConvolutionSpec, ConvolutionActivation, Error,
    GatedDeltaScanInput, LinearOperator, LinearSpec, NeuralBackend, NormalizationConstructionSpec,
    NormalizationOperator, Parameter, ParameterSpec, Tensor,
};
use eredu_runtime::RuntimeStateComponents;

use super::ModelArgs;

/// Backend-neutral Kimi Delta Attention operator.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct KimiDeltaAttention<B: NeuralBackend> {
    #[parameter(skip)]
    num_heads: i32,
    #[parameter(skip)]
    head_dim: i32,
    q_proj: B::Linear,
    k_proj: B::Linear,
    v_proj: B::Linear,
    q_conv1d: CausalDepthwiseConvolution<B>,
    k_conv1d: CausalDepthwiseConvolution<B>,
    v_conv1d: CausalDepthwiseConvolution<B>,
    f_a_proj: B::Linear,
    f_b_proj: B::Linear,
    b_proj: B::Linear,
    g_a_proj: B::Linear,
    g_b_proj: B::Linear,
    /// Log transition-rate parameter.
    pub a_log: Parameter<B::Tensor>,
    /// Per-channel decay bias.
    pub dt_bias: Parameter<B::Tensor>,
    o_norm: B::Normalization,
    o_proj: B::Linear,
}

impl<B: NeuralBackend> KimiDeltaAttention<B> {
    /// Creates unloaded KDA parameters for one physical layer.
    pub fn new(
        args: &ModelArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_with_heads(args, layer, args.kda_config.num_heads, context)
    }

    /// Creates unloaded KDA parameters for a placement-resolved head count.
    pub fn new_with_heads(
        args: &ModelArgs,
        layer: usize,
        heads: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let head_dim = args.kda_config.head_dim;
        let projection = heads
            .checked_mul(head_dim)
            .ok_or_else(|| Error::backend("Kimi KDA projection width overflowed"))?;
        let prefix = format!("model.layers.{layer}.self_attn");
        let parameter = |name: String| ParameterSpec::trainable(name).map_err(Error::backend);
        let linear = |name: &str, input, output| {
            let weight = format!("{prefix}.{name}.weight");
            B::linear(
                LinearSpec {
                    input,
                    output,
                    weight: parameter(weight.clone())?,
                    bias: None,
                    format: crate::linear_format::standard_linear_format(
                        &weight,
                        args.weight_quantization_for(&weight).into(),
                    )?,
                },
                context,
            )
        };
        let convolution = |name: &str| {
            CausalDepthwiseConvolution::new(
                CausalDepthwiseConvolutionSpec {
                    channels: projection,
                    kernel_size: args.kda_config.short_conv_kernel_size,
                    weight: parameter(format!("{prefix}.{name}.weight"))?,
                    bias: None,
                    activation: ConvolutionActivation::Silu,
                },
                context,
            )
        };
        Ok(Self {
            num_heads: heads,
            head_dim,
            q_proj: linear("q_proj", args.hidden_size, projection)?,
            k_proj: linear("k_proj", args.hidden_size, projection)?,
            v_proj: linear("v_proj", args.hidden_size, projection)?,
            q_conv1d: convolution("q_conv1d")?,
            k_conv1d: convolution("k_conv1d")?,
            v_conv1d: convolution("v_conv1d")?,
            f_a_proj: linear("f_a_proj", args.hidden_size, head_dim)?,
            f_b_proj: linear("f_b_proj", head_dim, projection)?,
            b_proj: linear("b_proj", args.hidden_size, heads)?,
            g_a_proj: linear("g_a_proj", args.hidden_size, head_dim)?,
            g_b_proj: linear("g_b_proj", head_dim, projection)?,
            a_log: Parameter::unloaded(
                parameter(format!("{prefix}.A_log"))?,
                &[1, 1, heads, 1],
                context,
            )?,
            dt_bias: Parameter::unloaded(
                parameter(format!("{prefix}.dt_bias"))?,
                &[projection],
                context,
            )?,
            o_norm: B::normalization(
                NormalizationConstructionSpec::learned(
                    head_dim,
                    args.rms_norm_eps,
                    parameter(format!("{prefix}.o_norm.weight"))?,
                ),
                context,
            )?,
            o_proj: linear("o_proj", projection, args.hidden_size)?,
        })
    }

    /// Executes KDA and replaces all bounded histories and recurrent state.
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

    /// Executes the same KDA recurrence with a row-parallel output projection.
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
                "Kimi KDA expects [batch, sequence, hidden], got {shape:?}"
            )));
        }
        let (batch, sequence) = (shape[0], shape[1]);
        let projection = self.num_heads * self.head_dim;
        let projected = [
            self.q_proj.forward(input, context)?,
            self.k_proj.forward(input, context)?,
            self.v_proj.forward(input, context)?,
        ];
        let mut convolved = Vec::with_capacity(3);
        for (slot, (conv, value)) in [
            (&self.q_conv1d, &projected[0]),
            (&self.k_conv1d, &projected[1]),
            (&self.v_conv1d, &projected[2]),
        ]
        .into_iter()
        .enumerate()
        {
            let role = StateTensorRole::Convolution { slot: slot as u32 };
            let output = {
                let history = state.fixed_component(role).map_err(Error::backend)?;
                conv.forward(value, history.as_ref(), context)?
            };
            *state.fixed_component(role).map_err(Error::backend)? = output.history;
            convolved.push(output.output);
        }
        let head_shape = [batch, sequence, self.num_heads, self.head_dim];
        let q = B::rms_norm_without_weight(
            &convolved[0].reshape(&head_shape, context)?,
            1e-6,
            context,
        )?
        .multiply_scalar(1.0 / self.head_dim as f32, context)?;
        let k = B::rms_norm_without_weight(
            &convolved[1].reshape(&head_shape, context)?,
            1e-6,
            context,
        )?
        .multiply_scalar((self.head_dim as f32).sqrt().recip(), context)?;
        let value = convolved[2].reshape(&head_shape, context)?;
        let decay_logits = self
            .f_b_proj
            .forward(&self.f_a_proj.forward(input, context)?, context)?
            .reshape(&head_shape, context)?;
        let dt_bias = self
            .dt_bias
            .as_ref()
            .reshape(&[1, 1, self.num_heads, self.head_dim], context)?;
        let rate = B::exp(self.a_log.as_ref().clone(), context)?.multiply_scalar(-1.0, context)?;
        let log_decay =
            B::softplus(decay_logits.add(&dt_bias, context)?, context)?.multiply(&rate, context)?;
        let beta = B::sigmoid(
            self.b_proj
                .forward(input, context)?
                .reshape(&[batch, sequence, self.num_heads], context)?,
            context,
        )?;
        let scan = {
            let recurrent = state
                .fixed_component(StateTensorRole::Recurrent)
                .map_err(Error::backend)?;
            B::gated_delta_scan(
                GatedDeltaScanInput {
                    query: &q,
                    key: &k,
                    value: &value,
                    log_decay: &log_decay,
                    beta: &beta,
                    initial_state: recurrent.as_ref(),
                },
                context,
            )?
        };
        *state
            .fixed_component(StateTensorRole::Recurrent)
            .map_err(Error::backend)? = Some(scan.state);
        state.advance_fixed(sequence).map_err(Error::backend)?;
        let gate = B::sigmoid(
            self.g_b_proj
                .forward(&self.g_a_proj.forward(input, context)?, context)?
                .reshape(&head_shape, context)?,
            context,
        )?;
        let normalized = self
            .o_norm
            .forward(&scan.output, context)?
            .multiply(&gate, context)?;
        let normalized = normalized.reshape(&[batch, sequence, projection], context)?;
        match parallel {
            Some(parallel) => {
                B::row_parallel_linear(&mut self.o_proj, &normalized, parallel, context)
            }
            None => self.o_proj.forward(&normalized, context),
        }
    }
}
