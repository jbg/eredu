//! Released recurrent parameter construction and gated residual composition.
use super::{config::Config, residual::mixer_spec};
use crate::{
    decoder::ComponentInstrumentation,
    gated_delta::{GatedDeltaMixer, GatedDeltaMixerSpec},
};
use eredu_nn::{
    residual_streams::{GatedResidual, GatedResidualInput, GatedResidualSpec},
    CausalDepthwiseConvolutionSpec, ConvolutionActivation, DistributedNeuralBackend, Error,
    LinearFormatSpec, LinearSpec, NeuralBackend, ParameterSpec, Tensor,
};
use eredu_runtime::RuntimeStateComponents;

/// Exact target or prediction recurrent parameters. Formats come from retained
/// preparation; neither the operator nor a backend interprets checkpoint names.
pub fn recurrent_spec(
    config: &Config,
    root: &str,
    mut format: impl FnMut(&str) -> Result<LinearFormatSpec, Error>,
) -> Result<GatedDeltaMixerSpec, Error> {
    let r = &config.recurrent;
    let key = r
        .key_heads
        .checked_mul(r.key_dim)
        .ok_or_else(|| Error::backend("qwen4_exp recurrent key width overflowed"))?;
    let value = r
        .value_heads
        .checked_mul(r.value_dim)
        .ok_or_else(|| Error::backend("qwen4_exp recurrent value width overflowed"))?;
    let qkv = key
        .checked_mul(2)
        .and_then(|k| k.checked_add(value))
        .ok_or_else(|| Error::backend("qwen4_exp recurrent projection width overflowed"))?;
    let parameter =
        |suffix: &str| ParameterSpec::trainable(format!("{root}.{suffix}")).map_err(Error::backend);
    let mut linear = |suffix: &str, input, output| {
        let name = format!("{root}.{suffix}.weight");
        Ok::<_, Error>(LinearSpec {
            input,
            output,
            weight: ParameterSpec::trainable(&name).map_err(Error::backend)?,
            bias: None,
            format: format(&name)?,
        })
    };
    let spec = GatedDeltaMixerSpec {
        key_heads: r.key_heads,
        value_heads: r.value_heads,
        key_head_dim: r.key_dim,
        value_head_dim: r.value_dim,
        input_qkv: linear("in_proj_qkv", config.hidden_size, qkv)?,
        input_gate: linear("in_proj_z", config.hidden_size, value)?,
        input_beta: linear("in_proj_b", config.hidden_size, r.value_heads)?,
        input_decay: linear("in_proj_a", config.hidden_size, r.value_heads)?,
        convolution: CausalDepthwiseConvolutionSpec {
            dilation: 1,
            channels: qkv,
            kernel_size: r.kernel,
            weight: parameter("conv1d.weight")?,
            bias: None,
            activation: ConvolutionActivation::Silu,
        },
        decay_bias: parameter("dt_bias")?,
        transition_log: parameter("A_log")?,
        normalization_weight: parameter("norm.weight")?,
        output: linear("out_proj", value, config.hidden_size)?,
        l2_epsilon: 1e-6,
        output_epsilon: config.norm_epsilon,
        output_gate: r.gate,
        output_arithmetic: eredu_nn::OutputGatedNormArithmetic::RoundedNormalization,
    };
    spec.validate()?;
    Ok(spec)
}

/// Gated residual ingress, recurrent update and injection as one sublayer.
#[derive(Debug, Clone)]
pub struct RecurrentSublayerSpec {
    /// The small residual mixer is replicated in tensor-parallel execution.
    pub residual: GatedResidualSpec,
    /// May describe globally sized or rank-local recurrent heads/projections.
    pub mixer: GatedDeltaMixerSpec,
}
impl RecurrentSublayerSpec {
    /// Constructs released identities beneath one decoder layer root.
    pub fn from_config(
        config: &Config,
        root: &str,
        mut format: impl FnMut(&str) -> Result<LinearFormatSpec, Error>,
    ) -> Result<Self, Error> {
        let spec = Self {
            residual: mixer_spec(
                config,
                &format!("{root}.attn_hyper_connection"),
                true,
                &mut format,
            )?,
            mixer: recurrent_spec(config, &format!("{root}.linear_attn"), format)?,
        };
        spec.validate()?;
        Ok(spec)
    }
    /// Rejects inconsistent composition before allocating any operator.
    pub fn validate(&self) -> Result<(), Error> {
        self.residual.validate()?;
        self.mixer.validate()?;
        if self.residual.injection.is_none()
            || self.residual.geometry.hidden_size() != self.mixer.input_qkv.input
        {
            return Err(Error::backend(
                "qwen4_exp recurrent residual geometry mismatch",
            ));
        }
        Ok(())
    }
}

/// Complete residual streams remain intact across both sublayer boundaries.
/// Lexical injection executes before this unit with independently owned state.
#[derive(Debug, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct RecurrentSublayer<B: NeuralBackend> {
    residual: GatedResidual<B>,
    mixer: GatedDeltaMixer<B>,
}
impl<B: NeuralBackend> RecurrentSublayer<B> {
    /// Constructs only this sublayer's parameters; state is owned by the session.
    pub fn new(
        spec: RecurrentSublayerSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        spec.validate()?;
        Ok(Self {
            residual: GatedResidual::new(spec.residual, context)?,
            mixer: GatedDeltaMixer::new(spec.mixer, context)?,
        })
    }
    fn mix_input(
        &mut self,
        hidden: &B::Tensor,
        padding: Option<&B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(GatedResidualInput<B::Tensor>, B::Tensor), Error> {
        if padding
            .is_some_and(|mask| hidden.shape().len() != 4 || mask.shape() != &hidden.shape()[..2])
        {
            return Err(Error::backend(
                "recurrent padding mask must have [batch, sequence] shape",
            ));
        }
        let residual = self.residual.forward(hidden, context)?;
        // Padding zeros the recurrent input, while original residual streams and
        // the causal time step are preserved, including retained convolution taps.
        let input = match padding {
            Some(mask) => {
                let dtype = residual
                    .mixed
                    .element_type()
                    .ok_or_else(|| Error::backend("recurrent input dtype is unavailable"))?;
                residual
                    .mixed
                    .multiply(
                        &mask
                            .expand_dims(-1, context)?
                            .broadcast_to(residual.mixed.shape(), context)?,
                        context,
                    )?
                    .cast_float(dtype, context)?
            }
            None => residual.mixed.clone(),
        };
        Ok((residual, input))
    }
    /// Shared ordinary/controlled equation over explicitly supplied mutable state.
    /// `padding` is a zero/one mask; masked tokens still advance causal state.
    pub fn forward<S: RuntimeStateComponents<B>>(
        &mut self,
        hidden: &B::Tensor,
        padding: Option<&B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let (residual, input) = self.mix_input(hidden, padding, context)?;
        let input = instrumentation.apply("mixer.input", input)?;
        let output = self
            .mixer
            .forward_instrumented(&input, state, context, instrumentation)?;
        let output = instrumentation.apply("mixer.write", output)?;
        let output = instrumentation.apply("mixer.output", output)?;
        instrumentation.apply("mixer.residual", residual.inject(&output, context)?)
    }
}
impl<B: NeuralBackend + DistributedNeuralBackend> RecurrentSublayer<B> {
    /// Replicated residual mixing with rank-local recurrence and one output reduction.
    pub fn forward_parallel<S: RuntimeStateComponents<B>>(
        &mut self,
        hidden: &B::Tensor,
        padding: Option<&B::Tensor>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let (residual, input) = self.mix_input(hidden, padding, context)?;
        let input = instrumentation.apply("mixer.input", input)?;
        let output = self.mixer.forward_parallel_instrumented(
            &input,
            state,
            parallel,
            context,
            instrumentation,
        )?;
        let output = instrumentation.apply("mixer.write", output)?;
        let output = instrumentation.apply("mixer.output", output)?;
        instrumentation.apply("mixer.residual", residual.inject(&output, context)?)
    }
}
