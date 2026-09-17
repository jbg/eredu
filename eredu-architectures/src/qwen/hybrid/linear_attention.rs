//! Neutral gated-delta recurrent layer shared by every hybrid variant.

use eredu_core::cache::StateTensorRole;
use eredu_nn::{
    CausalDepthwiseConvolution, CausalDepthwiseConvolutionSpec, ConvolutionActivation, Error,
    GatedDeltaScanInput, HeadExpansion, LinearOperator, LinearSpec, NeuralBackend, Parameter,
    ParameterSpec, Tensor,
};
use eredu_runtime::RuntimeStateComponents;

use super::{HybridConfig, HybridVariant};
use crate::decoder::ComponentInstrumentation;

/// One recurrent gated-delta attention operator.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct LinearAttention<B: NeuralBackend> {
    #[parameter(skip, metadata)]
    key_heads: i32,
    #[parameter(skip, metadata)]
    value_heads: i32,
    #[parameter(skip, metadata)]
    key_head_dim: i32,
    #[parameter(skip, metadata)]
    value_head_dim: i32,
    #[parameter(skip, metadata)]
    key_width: i32,
    #[parameter(skip, metadata)]
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
        let metadata = crate::decoder::ModuleMetadata::new::<B>(context);
        metadata.controls::<(Self, &HybridConfig, usize, i32, i32, String, LinearSpec,
            CausalDepthwiseConvolutionSpec, ParameterSpec, [i32; 1],
            &<B::Tensor as Tensor>::Context)>()?;
        if key_heads <= 0 || value_heads <= 0 || value_heads % key_heads != 0 {
            return Err(metadata.error(format_args!(
                "invalid rank-local recurrent heads: key={key_heads}, value={value_heads}"
            )));
        }
        let key_width = key_heads
            .checked_mul(config.linear_key_head_dim)
            .ok_or_else(|| metadata.error(format_args!("recurrent key width overflowed")))?;
        let value_width = value_heads
            .checked_mul(config.linear_value_head_dim)
            .ok_or_else(|| metadata.error(format_args!("recurrent value width overflowed")))?;
        let qkv_width = key_width
            .checked_mul(2)
            .and_then(|width| width.checked_add(value_width))
            .ok_or_else(|| metadata.error(format_args!("recurrent QKV width overflowed")))?;
        let prefix = metadata.text(format_args!("model.layers.{layer}.linear_attn"))?;
        let parameter = |suffix: &str| {
            metadata.named_parameter(format_args!("{prefix}.{suffix}"))
        };
        let linear = |suffix: &str, input: i32, output: i32, force_dense: bool| {
            let weight = metadata.text(format_args!("{prefix}.{suffix}.weight"))?;
            metadata.controls::<LinearSpec>()?;
            B::linear(
                LinearSpec {
                    input,
                    output,
                    weight: metadata.plain_parameter(&weight)?,
                    bias: None,
                    format: metadata.format(
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
        metadata.borrowed_controls(&parameter)?;
        metadata.borrowed_controls(&linear)?;
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
        self.forward_instrumented(
            input,
            state,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Executes the same recurrence with component hooks on consumed channels.
    pub fn forward_instrumented<S>(
        &mut self,
        input: &B::Tensor,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        S: RuntimeStateComponents<B>,
    {
        self.forward_inner(input, state, context, instrumentation, None)
    }

    fn forward_inner<S>(
        &mut self,
        input: &B::Tensor,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
        parallel: Option<&B::ParallelContext>,
    ) -> Result<B::Tensor, Error>
    where
        S: RuntimeStateComponents<B>,
    {
        let shape = input.shape();
        if shape.len() != 3 || shape[0] <= 0 || shape[1] <= 0 {
            return Err(Error::backend(format!(
                "linear attention expects [batch, sequence, hidden], got {shape:?}"
            )));
        }
        let (batch, sequence) = (shape[0], shape[1]);
        let projected = self.input_qkv.forward(input, context)?;
        instrumentation.observe("mixer.qkv.projected", &projected)?;
        let projected = if self.convolution.history_len() == 0 {
            // The declared width-one state has no convolution history role.
            // Recurrence replacement and frontier advance below are unchanged.
            self.convolution.forward(&projected, None, context)?.output
        } else {
            let history = state
                .fixed_component(StateTensorRole::Convolution { slot: 0 })
                .map_err(Error::backend)?;
            let output = self
                .convolution
                .forward(&projected, history.as_ref(), context)?;
            *history = output.history;
            output.output
        };
        instrumentation.observe("mixer.qkv.convolved", &projected)?;
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
        let beta = {
            let projected = self.input_beta.forward(input, context)?;
            instrumentation.observe("mixer.update.projected", &projected)?;
            B::sigmoid(projected, context)?
        };
        let decay_bias = self
            .decay_bias
            .as_ref()
            .reshape(&[1, 1, self.value_heads], context)?;
        let decay = B::softplus(
            {
                let projected = self.input_decay.forward(input, context)?;
                instrumentation.observe("mixer.decay.projected", &projected)?;
                projected.add(&decay_bias, context)?
            },
            1.0,
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
        let gate = {
            let projected = self.input_gate.forward(input, context)?;
            instrumentation.observe("mixer.gate.projected", &projected)?;
            projected.reshape(
                &[batch, sequence, self.value_heads, self.value_head_dim],
                context,
            )?
        };
        let normalized = B::silu_gated_group_rms_norm(
            &scan.output,
            &gate,
            self.normalization_weight.as_ref(),
            1,
            1e-6,
            context,
        )?
        .reshape(&[batch, sequence, self.value_width], context)?;
        let channels = instrumentation.apply("mixer.channels", normalized)?;
        instrumentation.project::<B>(
            "mixer.write_input",
            &mut self.output,
            &channels,
            parallel,
            context,
        )
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
        self.forward_parallel_instrumented(
            input,
            state,
            parallel,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Executes observed recurrent channels with one row-parallel reduction.
    pub fn forward_parallel_instrumented<S>(
        &mut self,
        input: &B::Tensor,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        S: RuntimeStateComponents<B>,
    {
        self.forward_inner(input, state, context, instrumentation, Some(parallel))
    }
}
