//! Reusable gated-delta equations with architecture-authored construction specs.

pub(crate) mod checkpoint;
mod state;
pub use state::GatedDeltaStateGeometry;

use crate::decoder::ComponentInstrumentation;
use eredu_core::cache::StateTensorRole;
use eredu_nn::{
    CausalDepthwiseConvolution, CausalDepthwiseConvolutionSpec, Error, GatedDeltaScanInput,
    HeadExpansion, LinearOperator, LinearSpec, NeuralBackend, OutputGateActivation, Parameter,
    ParameterSpec, Tensor, TensorElementType,
};
use eredu_runtime::RuntimeStateComponents;

/// Optional mechanisms needed by the recurrent equation, including FP32 boundaries.
pub const REQUIRED_OPERATORS: eredu_nn::NeuralOperatorCapabilities = {
    use eredu_nn::NeuralOperatorCapabilities as C;
    C::SIGMOID
        .union(C::SOFTPLUS)
        .union(C::EXP)
        .union(C::L2_NORMALIZE)
        .union(C::OUTPUT_GATED_GROUP_RMS_NORM)
        .union(C::GATED_DELTA_SCAN)
        .union(C::BROADCAST_TO)
        .union(C::CAST_FLOAT)
};

/// Exact parameter identities, formats, geometry and arithmetic for a mixer.
#[derive(Debug, Clone)]
pub struct GatedDeltaMixerSpec {
    /// Query and key heads before repetition to value heads.
    pub key_heads: i32,
    /// Value heads and recurrent state matrices.
    pub value_heads: i32,
    /// Dimensions per query/key head.
    pub key_head_dim: i32,
    /// Dimensions per value head.
    pub value_head_dim: i32,
    /// Component-major query, key and value projection.
    pub input_qkv: LinearSpec,
    /// Per-value-channel output gate projection.
    pub input_gate: LinearSpec,
    /// Per-value-head update projection, followed by sigmoid.
    pub input_beta: LinearSpec,
    /// Per-value-head log-decay projection, followed by biased softplus.
    pub input_decay: LinearSpec,
    /// Causal convolution over the component-major projection.
    pub convolution: CausalDepthwiseConvolutionSpec,
    /// Per-value-head softplus bias.
    pub decay_bias: ParameterSpec,
    /// Per-value-head logarithmic transition magnitude.
    pub transition_log: ParameterSpec,
    /// Per-value-dimension RMS scale shared by heads.
    pub normalization_weight: ParameterSpec,
    /// Projection from value heads back to the residual width.
    pub output: LinearSpec,
    /// Additive epsilon for query/key L2 normalization.
    pub l2_epsilon: f32,
    /// Additive epsilon for output RMS normalization.
    pub output_epsilon: f32,
    /// Gate activation applied after output normalization.
    pub output_gate: OutputGateActivation,
    /// Rounding boundary between normalization and learned scaling.
    pub output_arithmetic: eredu_nn::OutputGatedNormArithmetic,
}

impl GatedDeltaMixerSpec {
    /// Declares exactly the rank-local mutable components consumed by this mixer.
    pub fn state_policy(&self) -> Result<eredu_core::cache::LayerCachePolicy, Error> {
        self.validate()?;
        GatedDeltaStateGeometry {
            key_heads: self.key_heads,
            value_heads: self.value_heads,
            key_dim: self.key_head_dim,
            value_dim: self.value_head_dim,
            kernel: self.convolution.kernel_size,
            dilation: self.convolution.dilation,
        }
        .state_policy()
    }

    /// Validates the full construction before backend parameter allocation.
    pub fn validate(&self) -> Result<(), Error> {
        if self.key_heads <= 0
            || self.value_heads <= 0
            || self.value_heads % self.key_heads != 0
            || self.key_head_dim <= 0
            || self.value_head_dim <= 0
        {
            return Err(Error::backend("invalid gated-delta head geometry"));
        }
        for epsilon in [self.l2_epsilon, self.output_epsilon] {
            if !epsilon.is_finite() || epsilon <= 0.0 {
                return Err(Error::backend(
                    "gated-delta normalization epsilon must be finite and positive",
                ));
            }
        }
        let key = self
            .key_heads
            .checked_mul(self.key_head_dim)
            .ok_or_else(|| Error::backend("gated-delta key width overflowed"))?;
        let value = self
            .value_heads
            .checked_mul(self.value_head_dim)
            .ok_or_else(|| Error::backend("gated-delta value width overflowed"))?;
        let qkv = key
            .checked_mul(2)
            .and_then(|k| k.checked_add(value))
            .ok_or_else(|| Error::backend("gated-delta QKV width overflowed"))?;
        let hidden = self.input_qkv.input;
        if hidden <= 0 {
            return Err(Error::backend("gated-delta input width must be positive"));
        }
        for (spec, input, output) in [
            (&self.input_qkv, hidden, qkv),
            (&self.input_gate, hidden, value),
            (&self.input_beta, hidden, self.value_heads),
            (&self.input_decay, hidden, self.value_heads),
            (&self.output, value, hidden),
        ] {
            if spec.input != input || spec.output != output {
                return Err(Error::backend("gated-delta projection geometry mismatch"));
            }
            spec.format.validate_for_weight(&spec.weight)?;
        }
        self.convolution.validate()?;
        if self.convolution.channels != qkv {
            return Err(Error::backend(
                "gated-delta convolution channels must match QKV width",
            ));
        }
        Ok(())
    }
}

/// One recurrent gated-delta attention operator.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct GatedDeltaMixer<B: NeuralBackend> {
    #[parameter(skip)]
    l2_epsilon: f32,
    #[parameter(skip)]
    output_epsilon: f32,
    #[parameter(skip)]
    output_gate: OutputGateActivation,
    #[parameter(skip)]
    output_arithmetic: eredu_nn::OutputGatedNormArithmetic,
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

impl<B: NeuralBackend> GatedDeltaMixer<B> {
    /// Constructs one mixer from exact family-authored parameter specifications.
    pub fn new(
        spec: GatedDeltaMixerSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        spec.validate()?;
        B::require_operator_capabilities("gated-delta mixer", REQUIRED_OPERATORS)?;
        Ok(Self {
            key_heads: spec.key_heads,
            value_heads: spec.value_heads,
            key_head_dim: spec.key_head_dim,
            value_head_dim: spec.value_head_dim,
            key_width: spec.key_heads * spec.key_head_dim,
            value_width: spec.value_heads * spec.value_head_dim,
            l2_epsilon: spec.l2_epsilon,
            output_epsilon: spec.output_epsilon,
            output_gate: spec.output_gate,
            output_arithmetic: spec.output_arithmetic,
            input_qkv: B::linear(spec.input_qkv, context)?,
            input_gate: B::linear(spec.input_gate, context)?,
            input_beta: B::linear(spec.input_beta, context)?,
            input_decay: B::linear(spec.input_decay, context)?,
            convolution: CausalDepthwiseConvolution::new(spec.convolution, context)?,
            decay_bias: Parameter::unloaded(spec.decay_bias, &[spec.value_heads], context)?,
            transition_log: Parameter::unloaded(spec.transition_log, &[spec.value_heads], context)?,
            normalization_weight: Parameter::unloaded(
                spec.normalization_weight,
                &[spec.value_head_dim],
                context,
            )?,
            output: B::linear(spec.output, context)?,
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
        let activation_dtype = projected
            .element_type()
            .ok_or_else(|| Error::backend("gated-delta activation dtype is unavailable"))?;
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
            )?
            .cast_float(TensorElementType::F32, context)?;
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
            )?
            .cast_float(TensorElementType::F32, context)?;
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
        let query = B::expand_heads(
            &B::l2_normalize(&query, self.l2_epsilon, context)?,
            expansion,
            context,
        )?
        .multiply_scalar((self.key_head_dim as f32).sqrt().recip(), context)?;
        let key = B::expand_heads(
            &B::l2_normalize(&key, self.l2_epsilon, context)?,
            expansion,
            context,
        )?;
        let beta = {
            let projected = self.input_beta.forward(input, context)?;
            instrumentation.observe("mixer.update.projected", &projected)?;
            B::sigmoid(projected, context)?
        };
        let decay_bias = self
            .decay_bias
            .as_ref()
            .cast_float(TensorElementType::F32, context)?
            .reshape(&[1, 1, self.value_heads], context)?;
        let decay = B::softplus(
            {
                let projected = self.input_decay.forward(input, context)?;
                instrumentation.observe("mixer.decay.projected", &projected)?;
                projected
                    .cast_float(TensorElementType::F32, context)?
                    .add(&decay_bias, context)?
            },
            1.0,
            context,
        )?
        .multiply(
            &B::exp(
                self.transition_log
                    .as_ref()
                    .cast_float(TensorElementType::F32, context)?,
                context,
            )?
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
        let normalized = B::output_gated_group_rms_norm(
            &scan.output.cast_float(activation_dtype, context)?,
            &gate,
            self.normalization_weight.as_ref(),
            1,
            self.output_epsilon,
            self.output_gate,
            self.output_arithmetic,
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

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> GatedDeltaMixer<B> {
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
