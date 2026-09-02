//! Neutral gated-delta recurrent layer shared by every hybrid variant.

use eredu_core::cache::StateTensorRole;
use eredu_nn::{
    CausalDepthwiseConvolution, CausalDepthwiseConvolutionSpec, ConvolutionActivation, Error,
    GatedDeltaScanInput, HeadExpansion, LinearOperator, LinearSpec, NeuralBackend, Parameter,
    ParameterSpec, Tensor,
};
use eredu_runtime::RuntimeStateComponents;

use super::{HybridConfig, HybridVariant};

/// One recurrent gated-delta attention operator.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct LinearAttention<B: NeuralBackend> {
    #[parameter(skip)]
    key_heads: i32,
    #[parameter(skip)]
    value_heads: i32,
    #[parameter(skip)]
    key_head_dim: i32,
    #[parameter(skip)]
    value_head_dim: i32,
    #[parameter(skip)]
    key_width: i32,
    #[parameter(skip)]
    value_width: i32,
    input_qkv: B::Linear,
    input_gate: B::Linear,
    input_beta: B::Linear,
    input_decay: B::Linear,
    convolution: CausalDepthwiseConvolution<B>,
    /// Per-value-head decay bias.
    pub decay_bias: Parameter<B::Tensor>,
    /// Per-value-head logarithmic transition magnitude.
    pub transition_log: Parameter<B::Tensor>,
    /// Learned gated-normalization scale.
    pub normalization_weight: Parameter<B::Tensor>,
    output: B::Linear,
}

impl<B: NeuralBackend> LinearAttention<B> {
    /// Creates unloaded parameters for one physical decoder layer.
    pub fn new(
        config: &HybridConfig,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_with_heads(
            config,
            layer,
            config.linear_num_key_heads,
            config.linear_num_value_heads,
            context,
        )
    }

    /// Creates rank-local recurrent heads while preserving global parameter
    /// identities and row-parallel output semantics.
    pub fn new_with_heads(
        config: &HybridConfig,
        layer: usize,
        key_heads: i32,
        value_heads: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
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
            B::linear(
                LinearSpec {
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
                },
                context,
            )
        };
        let dense_ba = config.variant == HybridVariant::Qwen3Next && config.fp8.is_some();
        Ok(Self {
            key_heads,
            value_heads,
            key_head_dim: config.linear_key_head_dim,
            value_head_dim: config.linear_value_head_dim,
            key_width,
            value_width,
            input_qkv: linear("in_proj_qkv", config.hidden_size, qkv_width, false)?,
            input_gate: linear("in_proj_z", config.hidden_size, value_width, false)?,
            input_beta: linear("in_proj_b", config.hidden_size, value_heads, dense_ba)?,
            input_decay: linear("in_proj_a", config.hidden_size, value_heads, dense_ba)?,
            convolution: CausalDepthwiseConvolution::new(
                CausalDepthwiseConvolutionSpec {
                    channels: qkv_width,
                    kernel_size: config.linear_conv_kernel_dim,
                    weight: parameter("conv1d.weight")?,
                    bias: None,
                    activation: ConvolutionActivation::Silu,
                },
                context,
            )?,
            decay_bias: Parameter::unloaded(parameter("dt_bias")?, &[value_heads], context)?,
            transition_log: Parameter::unloaded(parameter("A_log")?, &[value_heads], context)?,
            normalization_weight: Parameter::unloaded(
                parameter("norm.weight")?,
                &[config.linear_value_head_dim],
                context,
            )?,
            output: linear("out_proj", value_width, config.hidden_size, false)?,
        })
    }

    /// Executes recurrent attention and replaces convolution/recurrent state.
    pub fn forward<S>(
        &mut self,
        input: &B::Tensor,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: RuntimeStateComponents<B>,
    {
        self.forward_inner(input, state, context, |projection, value, context| {
            projection.forward(value, context)
        })
    }

    fn forward_inner<S, F>(
        &mut self,
        input: &B::Tensor,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        project: F,
    ) -> Result<B::Tensor, Error>
    where
        S: RuntimeStateComponents<B>,
        F: FnOnce(
            &mut B::Linear,
            &B::Tensor,
            &<B::Tensor as Tensor>::Context,
        ) -> Result<B::Tensor, Error>,
    {
        let shape = input.shape();
        if shape.len() != 3 || shape[0] <= 0 || shape[1] <= 0 {
            return Err(Error::backend(format!(
                "linear attention expects [batch, sequence, hidden], got {shape:?}"
            )));
        }
        let (batch, sequence) = (shape[0], shape[1]);
        let projected = self.input_qkv.forward(input, context)?;
        let projected = {
            let history = state
                .fixed_component(StateTensorRole::Convolution { slot: 0 })
                .map_err(Error::backend)?;
            let output = self
                .convolution
                .forward(&projected, history.as_ref(), context)?;
            *history = output.history;
            output.output
        };
        let query = projected
            .index(
                &[
                    eredu_nn::Index::Full,
                    eredu_nn::Index::Full,
                    eredu_nn::Index::Range(0, self.key_width),
                ],
                context,
            )?
            .reshape(
                &[batch, sequence, self.key_heads, self.key_head_dim],
                context,
            )?;
        let key = projected
            .index(
                &[
                    eredu_nn::Index::Full,
                    eredu_nn::Index::Full,
                    eredu_nn::Index::Range(self.key_width, 2 * self.key_width),
                ],
                context,
            )?
            .reshape(
                &[batch, sequence, self.key_heads, self.key_head_dim],
                context,
            )?;
        let value = projected
            .index(
                &[
                    eredu_nn::Index::Full,
                    eredu_nn::Index::Full,
                    eredu_nn::Index::Range(
                        2 * self.key_width,
                        2 * self.key_width + self.value_width,
                    ),
                ],
                context,
            )?
            .reshape(
                &[batch, sequence, self.value_heads, self.value_head_dim],
                context,
            )?;
        let expansion = HeadExpansion {
            axis: 2,
            source_heads: self.key_heads,
            target_heads: self.value_heads,
        };
        let query = B::expand_heads(&B::l2_normalize(&query, 1e-6, context)?, expansion, context)?
            .multiply_scalar((self.key_head_dim as f32).sqrt().recip(), context)?;
        let key = B::expand_heads(&B::l2_normalize(&key, 1e-6, context)?, expansion, context)?;
        let beta = B::sigmoid(self.input_beta.forward(input, context)?, context)?;
        let decay_bias = self
            .decay_bias
            .as_ref()
            .reshape(&[1, 1, self.value_heads], context)?;
        let decay = B::softplus(
            self.input_decay
                .forward(input, context)?
                .add(&decay_bias, context)?,
            context,
        )?
        .multiply(
            &B::exp(self.transition_log.as_ref().clone(), context)?
                .multiply_scalar(-1.0, context)?,
            context,
        )?;
        let scan = {
            let recurrent = state
                .fixed_component(StateTensorRole::Recurrent)
                .map_err(Error::backend)?;
            B::gated_delta_scan(
                GatedDeltaScanInput {
                    query: &query,
                    key: &key,
                    value: &value,
                    log_decay: &decay,
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
        let gate = self.input_gate.forward(input, context)?.reshape(
            &[batch, sequence, self.value_heads, self.value_head_dim],
            context,
        )?;
        let normalized = B::silu_gated_group_rms_norm(
            &scan.output,
            &gate,
            self.normalization_weight.as_ref(),
            1,
            1e-6,
            context,
        )?
        .reshape(&[batch, sequence, self.value_width], context)?;
        project(&mut self.output, &normalized, context)
    }
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> LinearAttention<B> {
    /// Executes the same recurrence with one row-parallel output reduction.
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
        self.forward_inner(input, state, context, |projection, value, context| {
            B::row_parallel_linear(projection, value, parallel, context)
        })
    }
}
