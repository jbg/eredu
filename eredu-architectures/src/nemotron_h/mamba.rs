//! Nemotron-H Mamba2 over neutral convolution and selective-scan contracts.

use eredu_core::cache::StateTensorRole;
use eredu_nn::{
    CausalDepthwiseConvolution, CausalDepthwiseConvolutionSpec, ConvolutionActivation,
    DeclareLogicalParameters, Error, Index, LinearOperator, LinearSpec,
    LogicalParameterDeclaration, LogicalParameterRole, NeuralBackend, Parameter, ParameterSpec,
    Parameterized, SelectiveStateSpaceScanInput, Tensor,
};
use eredu_runtime::RuntimeStateComponents;

use super::ModelArgs;

/// Checked head/group axes and component-major input rows of a Mamba2 mixer.
#[derive(Debug, Clone)]
pub(crate) struct Mamba2Geometry {
    heads: i32,
    groups: i32,
    head_dim: i32,
    state_dim: i32,
    input: eredu_nn::FusedProjectionLayout,
}

impl Mamba2Geometry {
    pub(crate) fn new(
        heads: i32,
        groups: i32,
        head_dim: i32,
        state_dim: i32,
    ) -> Result<Self, Error> {
        if heads <= 0 || groups <= 0 || heads % groups != 0 || head_dim <= 0 || state_dim <= 0 {
            return Err(Error::backend("invalid Mamba head or state geometry"));
        }
        let intermediate = heads
            .checked_mul(head_dim)
            .ok_or_else(|| Error::backend("Mamba intermediate width overflowed"))?;
        let grouped = groups
            .checked_mul(state_dim)
            .ok_or_else(|| Error::backend("Mamba grouped state width overflowed"))?;
        let input = eredu_nn::FusedProjectionLayout::new(
            [
                ("gate", intermediate),
                ("value", intermediate),
                ("input_state", grouped),
                ("output_state", grouped),
                ("time_step", heads),
            ]
            .into_iter()
            .map(|(name, width)| eredu_nn::FusedProjectionSegment::new(name, width))
            .collect::<Result<Vec<_>, _>>()?,
        )?;
        Ok(Self {
            heads,
            groups,
            head_dim,
            state_dim,
            input,
        })
    }
    pub(crate) fn groups(&self) -> i32 {
        self.groups
    }
    pub(crate) fn heads(&self) -> i32 {
        self.heads
    }
    pub(crate) fn state_dim(&self) -> i32 {
        self.state_dim
    }
    fn intermediate(&self) -> i32 {
        self.input.segments()[0].width()
    }
    fn projection_width(&self) -> i32 {
        self.input.output_width()
    }
    fn convolution_width(&self) -> i32 {
        self.projection_width() - self.intermediate() - self.heads
    }
    fn input_range(&self, component: usize) -> std::ops::Range<i32> {
        let start = self.input.segments()[..component]
            .iter()
            .map(|s| s.width())
            .sum::<i32>();
        start..start + self.input.segments()[component].width()
    }
    fn input_segments(&self) -> Vec<std::ops::Range<usize>> {
        (0..self.input.segments().len())
            .map(|component| {
                let range = self.input_range(component);
                range.start as usize..range.end as usize
            })
            .collect()
    }
    fn convolution_segments(&self) -> Vec<std::ops::Range<usize>> {
        let offset = self.intermediate() as usize;
        self.input_segments()[1..4]
            .iter()
            .map(|range| range.start - offset..range.end - offset)
            .collect()
    }
}

/// Typed construction and logical geometry of one global or rank-local Mamba2 mixer.
#[derive(Debug, Clone)]
pub(crate) struct Mamba2Spec {
    pub(crate) in_proj: LinearSpec,
    pub(crate) convolution: CausalDepthwiseConvolutionSpec,
    pub(crate) dt_bias: ParameterSpec,
    pub(crate) a_log: ParameterSpec,
    pub(crate) d: ParameterSpec,
    pub(crate) norm_weight: ParameterSpec,
    pub(crate) out_proj: LinearSpec,
    pub(crate) geometry: Mamba2Geometry,
    pub(crate) chunk_size: usize,
    pub(crate) time_step_floor: f32,
    pub(crate) epsilon: f32,
}

impl Mamba2Spec {
    pub(crate) fn local(mut self, heads: i32, groups: i32) -> Result<Self, Error> {
        if heads > self.geometry.heads || groups > self.geometry.groups {
            return Err(Error::backend(
                "local Mamba scope exceeds retained geometry",
            ));
        }
        self.geometry = Mamba2Geometry::new(
            heads,
            groups,
            self.geometry.head_dim,
            self.geometry.state_dim,
        )?;
        self.in_proj.output = self.geometry.projection_width();
        self.convolution.channels = self.geometry.convolution_width();
        self.out_proj.input = self.geometry.intermediate();
        self.logical_parameters()?;
        Ok(self)
    }

    fn validate_geometry(&self) -> Result<(), Error> {
        self.convolution.validate()?;
        if self.in_proj.output != self.geometry.projection_width()
            || self.convolution.channels != self.geometry.convolution_width()
            || self.out_proj.input != self.geometry.intermediate()
            || self.in_proj.input != self.out_proj.output
            || self.convolution.dilation != 1
            || self.convolution.activation != ConvolutionActivation::Silu
            || self.convolution.arithmetic != eredu_nn::ConvolutionArithmetic::Native
            || self.chunk_size == 0
        {
            return Err(Error::backend(
                "incompatible Mamba projection, convolution or execution geometry",
            ));
        }
        Ok(())
    }
    /// Joint projection row boundaries retain gate, value, B, C and timestep axes.
    pub(crate) fn input_segments(&self) -> Vec<std::ops::Range<usize>> {
        self.geometry.input_segments()
    }
    /// Convolution channels carry values and the two grouped state projections.
    pub(crate) fn convolution_segments(&self) -> Vec<std::ops::Range<usize>> {
        self.geometry.convolution_segments()
    }
    /// Exact mutable state associated with this invocation, independent of parameter sharing.
    pub(crate) fn state_policy(&self) -> Result<eredu_core::cache::LayerCachePolicy, Error> {
        use eredu_core::cache::{
            LayerCachePolicy, MutableStateResidency, StateTensorDimension as D, StateTensorDtype,
            StateTensorPolicy,
        };
        self.validate_geometry()?;
        let fixed = |n| D::fixed(n).map_err(Error::backend_source);
        let mut tensors = Vec::new();
        let history = self.convolution.history_len()?;
        if history > 0 {
            tensors.push(
                StateTensorPolicy::new(
                    StateTensorRole::Convolution { slot: 0 },
                    vec![
                        D::Batch,
                        fixed(history)?,
                        fixed(self.geometry.convolution_width())?,
                    ],
                    StateTensorDtype::Floating,
                    MutableStateResidency::AlwaysDeviceMutable,
                )
                .map_err(Error::backend_source)?,
            );
        }
        tensors.push(
            StateTensorPolicy::new(
                StateTensorRole::Recurrent,
                vec![
                    D::Batch,
                    fixed(self.geometry.heads)?,
                    fixed(self.geometry.head_dim)?,
                    fixed(self.geometry.state_dim)?,
                ],
                StateTensorDtype::Float32,
                MutableStateResidency::LayerScopedOffloadable,
            )
            .map_err(Error::backend_source)?,
        );
        LayerCachePolicy::fixed_only(tensors).map_err(Error::backend_source)
    }
}

impl DeclareLogicalParameters for Mamba2Spec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.validate_geometry()?;
        use LogicalParameterDeclaration as D;
        use LogicalParameterRole as R;
        let mut declarations = self.in_proj.logical_parameters()?;
        declarations.extend(self.convolution.logical_parameters()?);
        declarations.extend([
            D::new(&self.dt_bias, &[self.geometry.heads], R::RecurrentBias)?,
            D::new(&self.a_log, &[self.geometry.heads], R::RecurrentTransition)?,
            D::new(&self.d, &[self.geometry.heads], R::RecurrentSkip)?,
            D::new(&self.norm_weight, &[self.out_proj.input], R::Normalization)?,
        ]);
        declarations.extend(self.out_proj.logical_parameters()?);
        eredu_nn::validate_logical_parameter_topology(&declarations)
            .map_err(Error::backend_source)?;
        Ok(declarations)
    }
}

/// Declares parameter identities and exact geometry at an explicit target or prediction path.
pub(crate) fn mixer_spec(
    args: &ModelArgs,
    prefix: &str,
    heads: i32,
    groups: i32,
) -> Result<Mamba2Spec, Error> {
    let geometry = Mamba2Geometry::new(heads, groups, args.mamba_head_dim, args.ssm_state_size)?;
    let parameter =
        |field: &str| ParameterSpec::trainable(format!("{prefix}.{field}")).map_err(Error::backend);
    let spec = Mamba2Spec {
        in_proj: super::attention::linear_spec(
            args,
            &format!("{prefix}.in_proj"),
            args.hidden_size,
            geometry.projection_width(),
            args.use_bias,
        )?,
        convolution: CausalDepthwiseConvolutionSpec {
            arithmetic: eredu_nn::ConvolutionArithmetic::Native,
            dilation: 1,
            channels: geometry.convolution_width(),
            kernel_size: args.conv_kernel,
            weight: parameter("conv1d.weight")?,
            bias: args
                .use_conv_bias
                .then(|| parameter("conv1d.bias"))
                .transpose()?,
            activation: ConvolutionActivation::Silu,
        },
        dt_bias: parameter("dt_bias")?,
        a_log: parameter("A_log")?,
        d: parameter("D")?,
        norm_weight: parameter("norm.weight")?,
        out_proj: super::attention::linear_spec(
            args,
            &format!("{prefix}.out_proj"),
            geometry.intermediate(),
            args.hidden_size,
            args.use_bias,
        )?,
        geometry,
        chunk_size: usize::try_from(args.chunk_size)
            .ok()
            .filter(|&n| n > 0)
            .ok_or_else(|| Error::backend("invalid Mamba chunk size"))?,
        time_step_floor: args.time_step_min,
        epsilon: args.layer_norm_epsilon,
    };
    spec.logical_parameters()?;
    Ok(spec)
}

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
    geometry: Mamba2Geometry,
    #[parameter(skip)]
    chunk_size: usize,
    #[parameter(skip)]
    time_step_floor: f32,
    #[parameter(skip)]
    epsilon: f32,
}

impl<B: NeuralBackend> Mamba2<B> {
    /// Constructs a mixer from its validated declaration before allocating backend parameters.
    pub(crate) fn from_spec(
        spec: Mamba2Spec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        spec.logical_parameters()?;
        let intermediate = spec.geometry.intermediate();
        Ok(Self {
            in_proj: B::linear(spec.in_proj, context)?,
            conv1d: CausalDepthwiseConvolution::new(spec.convolution, context)?,
            dt_bias: Parameter::unloaded(spec.dt_bias, &[spec.geometry.heads], context)?,
            a_log: Parameter::unloaded(spec.a_log, &[spec.geometry.heads], context)?,
            d: Parameter::unloaded(spec.d, &[spec.geometry.heads], context)?,
            norm_weight: Parameter::unloaded(spec.norm_weight, &[intermediate], context)?,
            out_proj: B::linear(spec.out_proj, context)?,
            geometry: spec.geometry,
            chunk_size: spec.chunk_size,
            time_step_floor: spec.time_step_floor,
            epsilon: spec.epsilon,
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
        if shape.len() != 3 {
            return Err(Error::backend(format!(
                "Mamba2 expects [batch, sequence, hidden], got {shape:?}"
            )));
        }
        let (batch, sequence) = (shape[0], shape[1]);
        let projected = self.in_proj.forward(input, context)?;
        let gate_end = self.geometry.input_range(0).end;
        let time_step_range = self.geometry.input_range(4);
        let convolution_end = time_step_range.start;
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
                Index::Range(time_step_range.start, time_step_range.end),
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
            &[
                Index::Full,
                Index::Full,
                Index::Range(0, self.geometry.intermediate()),
            ],
            context,
        )?;
        let input_start = self.geometry.input_range(2).start - gate_end;
        let output_start = self.geometry.input_range(3).start - gate_end;
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
                Index::Range(output_start, self.geometry.convolution_width()),
            ],
            context,
        )?;
        let expand_groups = |value: B::Tensor| -> Result<B::Tensor, Error> {
            let repeats = self.geometry.heads / self.geometry.groups;
            let grouped = value.reshape(
                &[
                    batch,
                    sequence,
                    self.geometry.groups,
                    self.geometry.state_dim,
                ],
                context,
            )?;
            if repeats == 1 {
                return Ok(grouped);
            }
            grouped
                .expand_dims(3, context)?
                .broadcast_to(
                    &[
                        batch,
                        sequence,
                        self.geometry.groups,
                        repeats,
                        self.geometry.state_dim,
                    ],
                    context,
                )?
                .reshape(
                    &[
                        batch,
                        sequence,
                        self.geometry.heads,
                        self.geometry.state_dim,
                    ],
                    context,
                )
        };
        let values = values.reshape(
            &[batch, sequence, self.geometry.heads, self.geometry.head_dim],
            context,
        )?;
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
            .reshape(&[batch, sequence, self.geometry.intermediate()], context)?;
        let normalized = B::gated_group_rms_norm(
            &scan,
            &gate,
            self.norm_weight.as_ref(),
            self.geometry.groups,
            self.epsilon,
            context,
        )?;
        project(&mut self.out_proj, &normalized, context)
    }
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> Mamba2<B> {
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
        self.forward_inner(input, state, context, |projection, value, context| {
            B::row_parallel_linear(projection, value, parallel, context)
        })
    }
}
