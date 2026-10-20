//! EAR codec's causal ConvNeXt latent-to-spectrogram decoder.
//!
//! Learned parameters retain canonical PyTorch convolution layouts. Execution
//! transforms them to shared tensor layouts; snapshots contain only convolution
//! carry. Magnitude/phase conversion and inverse STFT are separate components.

use eredu_core::{execution_control::SnapshotEstimate, BackendFailure};
use eredu_runtime::execution_control::{
    snapshot_value_estimate, SnapshotReservation, SnapshotValue,
};

use eredu_nn::RetainedTensorRoots;

use eredu_nn::{
    LayerNorm, NeuralBackend, PadMode, Parameter, ParameterMetadata, ParameterSpec,
    ParameterVisitor, Parameterized, Tensor, TensorElementType,
};

use crate::{
    streaming::{Conv1dSpec, Conv1dState, StreamPhase},
    Error,
};

/// Decoder-order geometry; codec encoder configuration reverses rates/multipliers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecoderConfig {
    /// Latent input width.
    pub latent_size: i32,
    /// Base channel count multiplied at each stage.
    pub base_hidden_size: i32,
    /// Multipliers in decoder execution order.
    pub channel_mult: Vec<i32>,
    /// Upsampling rates in decoder execution order.
    pub rates: Vec<i32>,
    /// ConvNeXt residual blocks following each upsampling convolution.
    pub blocks_per_stage: usize,
    /// Causal depthwise convolution width.
    pub kernel_size: i32,
    /// Final spectral projection emits `fft_size + 2` channels.
    pub fft_size: i32,
    /// Groups for upsampling convolutions.
    pub groups: i32,
}

impl SnapshotValue for DecoderConfig {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        snapshot_value_estimate::<Self>([
            self.channel_mult.estimate_snapshot_value(),
            self.rates.estimate_snapshot_value(),
        ])
    }
    fn copy_snapshot_value(&self, retention: &SnapshotReservation) -> Result<Self, BackendFailure> {
        Ok(Self {
            latent_size: self.latent_size,
            base_hidden_size: self.base_hidden_size,
            channel_mult: self.channel_mult.copy_snapshot_value(retention)?,
            rates: self.rates.copy_snapshot_value(retention)?,
            blocks_per_stage: self.blocks_per_stage,
            kernel_size: self.kernel_size,
            fft_size: self.fft_size,
            groups: self.groups,
        })
    }
}

impl DecoderConfig {
    /// Validates complete unit geometry before native parameter allocation.
    pub fn validate(&self) -> Result<(), Error> {
        self.unit_specs().map(|_| ())
    }

    /// Ordered units with canonical `layers.N` identities and independent parameters.
    pub fn unit_specs(&self) -> Result<Vec<DecoderUnitSpec>, Error> {
        if self.latent_size <= 0
            || self.base_hidden_size <= 0
            || self.channel_mult.is_empty()
            || self.channel_mult.len() != self.rates.len()
            || self.blocks_per_stage == 0
            || self.kernel_size <= 0
            || self.fft_size < 4
            || self.fft_size % 2 != 0
            || self.fft_size.checked_add(2).is_none()
            || self.groups <= 0
        {
            return Err(invalid("invalid EAR decoder geometry"));
        }
        let count = self
            .blocks_per_stage
            .checked_add(1)
            .and_then(|n| n.checked_mul(self.rates.len()))
            .and_then(|n| n.checked_add(1))
            .filter(|n| *n <= i32::MAX as usize)
            .ok_or_else(|| invalid("EAR decoder layer count overflows"))?;
        let mut specs = Vec::with_capacity(count);
        let mut input = self.latent_size;
        let mut expansion = 1i32;
        for (&multiple, &rate) in self.channel_mult.iter().zip(&self.rates) {
            let output = self
                .base_hidden_size
                .checked_mul(multiple)
                .filter(|n| *n > 0)
                .ok_or_else(|| invalid("EAR decoder channel count overflows"))?;
            let intermediate = output
                .checked_mul(4)
                .ok_or_else(|| invalid("EAR decoder intermediate width overflows"))?;
            expansion = expansion
                .checked_mul(rate)
                .filter(|n| *n > 0)
                .ok_or_else(|| invalid("EAR decoder frame expansion overflows"))?;
            if input % self.groups != 0 || output % self.groups != 0 {
                return Err(invalid(
                    "EAR decoder channels do not divide convolution groups",
                ));
            }
            specs.push(DecoderUnitSpec::Upsample {
                layer: specs.len(),
                input,
                output,
                rate,
                groups: self.groups,
            });
            for _ in 0..self.blocks_per_stage {
                specs.push(DecoderUnitSpec::Block {
                    layer: specs.len(),
                    width: output,
                    intermediate,
                    kernel: self.kernel_size,
                });
            }
            input = output;
        }
        specs.push(DecoderUnitSpec::Output {
            layer: specs.len(),
            input,
            output: self.fft_size + 2,
        });
        Ok(specs)
    }
}

/// Independently materializable execution-unit geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum DecoderUnitSpec {
    Upsample {
        layer: usize,
        input: i32,
        output: i32,
        rate: i32,
        groups: i32,
    },
    Block {
        layer: usize,
        width: i32,
        intermediate: i32,
        kernel: i32,
    },
    Output {
        layer: usize,
        input: i32,
        output: i32,
    },
}
impl DecoderUnitSpec {
    /// Canonical zero-based position in decoder execution order.
    pub fn layer(self) -> usize {
        match self {
            Self::Upsample { layer, .. }
            | Self::Block { layer, .. }
            | Self::Output { layer, .. } => layer,
        }
    }
    /// Constructs empty unit carry using geometry alone.
    pub fn initial_state<T: Tensor>(self) -> Result<DecoderUnitState<T>, Error> {
        self.validate()?;
        let convolution = match self {
            Self::Block { width, kernel, .. } => Some(Conv1dState::new(Conv1dSpec {
                input_channels: width,
                output_channels: width,
                kernel_size: kernel,
                stride: 1,
                dilation: 1,
                groups: width,
                bias: true,
                pad_mode: PadMode::Constant,
            })?),
            _ => None,
        };
        Ok(DecoderUnitState {
            spec: self,
            convolution,
            phase: StreamPhase::AcceptingInput,
            batch: None,
        })
    }
    fn validate(self) -> Result<(), Error> {
        match self {
            Self::Upsample {
                input,
                output,
                rate,
                groups,
                ..
            } if input > 0
                && output > 0
                && rate > 0
                && groups > 0
                && input % groups == 0
                && output % groups == 0 =>
            {
                Ok(())
            }
            Self::Block {
                width,
                intermediate,
                kernel,
                ..
            } if width > 0 && intermediate > 0 && kernel > 0 => Ok(()),
            Self::Output { input, output, .. } if input > 0 && output > 0 => Ok(()),
            _ => Err(invalid("invalid EAR decoder unit geometry")),
        }
    }
}

/// State belonging to one unit; no learned parameter handles are retained.
#[derive(Debug, Clone)]
pub struct DecoderUnitState<T: Tensor> {
    pub(crate) spec: DecoderUnitSpec,
    pub(crate) convolution: Option<Conv1dState<T>>,
    pub(crate) phase: StreamPhase,
    pub(crate) batch: Option<i32>,
}

impl<T: Tensor + SnapshotValue> SnapshotValue for DecoderUnitState<T> {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        snapshot_value_estimate::<Self>([self.convolution.estimate_snapshot_value()])
    }
    fn copy_snapshot_value(&self, retention: &SnapshotReservation) -> Result<Self, BackendFailure> {
        Ok(Self {
            convolution: self.convolution.copy_snapshot_value(retention)?,
            spec: self.spec,
            phase: self.phase,
            batch: self.batch,
        })
    }
}

impl<T: Tensor> RetainedTensorRoots<T> for DecoderUnitState<T> {
    fn visit_retained_tensors<'a>(&'a self, visitor: &mut dyn FnMut(&'a T)) {
        if let Some(convolution) = &self.convolution {
            convolution.visit_retained_tensors(visitor);
        }
    }
}
impl<T: Tensor> DecoderUnitState<T> {
    /// Finite-stream lifecycle of this independently executable unit.
    pub fn phase(&self) -> StreamPhase {
        self.phase
    }
    /// Causal convolution input history `[batch, channels, kernel - 1]`.
    pub fn carry(&self) -> Option<&T> {
        self.convolution.as_ref().and_then(Conv1dState::carry)
    }
}

/// Complete neural state, independent of parameter residency.
#[derive(Debug, Clone)]
pub struct DecoderState<T: Tensor> {
    pub(crate) units: Vec<DecoderUnitState<T>>,
    pub(crate) batch: Option<i32>,
    pub(crate) phase: StreamPhase,
}

impl<T: Tensor + SnapshotValue> SnapshotValue for DecoderState<T> {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        snapshot_value_estimate::<Self>([self.units.estimate_snapshot_value()])
    }
    fn copy_snapshot_value(&self, retention: &SnapshotReservation) -> Result<Self, BackendFailure> {
        Ok(Self {
            units: self.units.copy_snapshot_value(retention)?,
            batch: self.batch,
            phase: self.phase,
        })
    }
}

impl<T: Tensor> RetainedTensorRoots<T> for DecoderState<T> {
    fn visit_retained_tensors<'a>(&'a self, visitor: &mut dyn FnMut(&'a T)) {
        for unit in &self.units {
            unit.visit_retained_tensors(visitor);
        }
    }
}
impl<T: Tensor> DecoderState<T> {
    /// Ordered states matching the decoder's individually materializable units.
    pub fn units(&self) -> &[DecoderUnitState<T>] {
        &self.units
    }
    /// Finite-stream lifecycle.
    pub fn phase(&self) -> StreamPhase {
        self.phase
    }
}

/// Biasless transposed convolution with equal kernel and stride, hence no overlap.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "T")]
pub struct Upsample<T: Tensor> {
    weight: Parameter<T>,
    #[parameter(skip)]
    input: i32,
    #[parameter(skip)]
    output: i32,
    #[parameter(skip)]
    rate: i32,
    #[parameter(skip)]
    groups: i32,
}
impl<T: Tensor> Upsample<T> {
    fn forward(&self, input: &T, context: &T::Context) -> Result<T, Error> {
        validate_input(input, self.input)?;
        require_f32(self, input)?;
        if self.weight.as_ref().shape() != [self.input, self.output / self.groups, self.rate] {
            return Err(invalid(
                "EAR decoder transpose kernel has invalid canonical geometry",
            ));
        }
        input
            .dim(2)
            .checked_mul(self.rate)
            .ok_or_else(|| invalid("EAR decoder output length overflows"))?;
        let weight = self
            .weight
            .as_ref()
            .reshape(
                &[
                    self.groups,
                    self.input / self.groups,
                    self.output / self.groups,
                    self.rate,
                ],
                context,
            )?
            .transpose_axes(&[0, 2, 3, 1], context)?
            .reshape(&[self.output, self.rate, self.input / self.groups], context)?;
        Ok(T::conv_transpose1d(
            &input.swap_axes(1, 2, context)?,
            &weight,
            self.rate,
            0,
            1,
            0,
            self.groups,
            context,
        )?
        .swap_axes(1, 2, context)?)
    }
}

/// Canonical OIK pointwise convolution, with optional released bias.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "T")]
pub struct Pointwise<T: Tensor> {
    weight: Parameter<T>,
    bias: Option<Parameter<T>>,
    #[parameter(skip)]
    input: i32,
    #[parameter(skip)]
    output: i32,
}
impl<T: Tensor> Pointwise<T> {
    pub(crate) fn new(
        prefix: &str,
        group: &str,
        input: i32,
        output: i32,
        bias: bool,
        context: &T::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            weight: slot(
                &format!("{prefix}.weight"),
                group,
                &[output, input, 1],
                context,
            )?,
            bias: bias
                .then(|| slot(&format!("{prefix}.bias"), group, &[output], context))
                .transpose()?,
            input,
            output,
        })
    }
    pub(crate) fn forward(&self, input: &T, context: &T::Context) -> Result<T, Error> {
        self.forward_with_reduction(input, context, Ok)
    }
    fn forward_with_reduction(
        &self,
        input: &T,
        context: &T::Context,
        reduce: impl FnOnce(T) -> Result<T, Error>,
    ) -> Result<T, Error> {
        validate_input(input, self.input)?;
        if self.weight.as_ref().shape() != [self.output, self.input, 1]
            || self
                .bias
                .as_ref()
                .is_some_and(|bias| bias.as_ref().shape() != [self.output])
        {
            return Err(invalid(
                "EAR decoder pointwise parameters have invalid geometry",
            ));
        }
        let mut output = reduce(T::conv1d(
            &input.swap_axes(1, 2, context)?,
            &self.weight.as_ref().swap_axes(1, 2, context)?,
            1,
            0,
            1,
            1,
            context,
        )?)?;
        if let Some(bias) = &self.bias {
            output = output.add(bias.as_ref(), context)?;
        }
        Ok(output.swap_axes(1, 2, context)?)
    }
}

/// Causal depthwise convolution, affine channel normalization and exact-GELU residual MLP.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "T")]
pub struct ConvNextBlock<T: Tensor> {
    depthwise: Parameter<T>,
    bias: Parameter<T>,
    norm: LayerNorm<T>,
    expand: Pointwise<T>,
    project: Pointwise<T>,
    #[parameter(skip)]
    width: i32,
    #[parameter(skip)]
    kernel: i32,
}
impl<T: Tensor> ConvNextBlock<T> {
    pub(crate) fn new(
        prefix: &str,
        group: &str,
        width: i32,
        intermediate: i32,
        kernel: i32,
        context: &T::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            depthwise: slot(
                &format!("{prefix}.dwconv.weight"),
                group,
                &[width, 1, kernel],
                context,
            )?,
            bias: slot(&format!("{prefix}.dwconv.bias"), group, &[width], context)?,
            norm: LayerNorm::unloaded(
                width,
                1e-6,
                Some(parameter(&format!("{prefix}.norm.weight"), group)?),
                Some(parameter(&format!("{prefix}.norm.bias"), group)?),
                context,
            )?,
            expand: Pointwise::new(
                &format!("{prefix}.pwconv1"),
                group,
                width,
                intermediate,
                true,
                context,
            )?,
            project: Pointwise::new(
                &format!("{prefix}.pwconv2"),
                group,
                intermediate,
                width,
                true,
                context,
            )?,
            width,
            kernel,
        })
    }

    pub(crate) fn forward(&self, input: &T, context: &T::Context) -> Result<T, Error> {
        self.step(input, &Conv1dState::new(self.convolution_spec())?, context)
            .map(|(output, _)| output)
    }

    pub(crate) fn forward_with_reduction(
        &self,
        input: &T,
        context: &T::Context,
        reduce: impl FnOnce(T) -> Result<T, Error>,
    ) -> Result<T, Error> {
        self.step_with_reduction(
            input,
            &Conv1dState::new(self.convolution_spec())?,
            context,
            reduce,
        )
        .map(|(output, _)| output)
    }

    fn convolution_spec(&self) -> Conv1dSpec {
        Conv1dSpec {
            input_channels: self.width,
            output_channels: self.width,
            kernel_size: self.kernel,
            stride: 1,
            dilation: 1,
            groups: self.width,
            bias: true,
            pad_mode: PadMode::Constant,
        }
    }
    fn step(
        &self,
        input: &T,
        state: &Conv1dState<T>,
        context: &T::Context,
    ) -> Result<(T, Conv1dState<T>), Error> {
        self.step_with_reduction(input, state, context, Ok)
    }
    fn step_with_reduction(
        &self,
        input: &T,
        state: &Conv1dState<T>,
        context: &T::Context,
        reduce: impl FnOnce(T) -> Result<T, Error>,
    ) -> Result<(T, Conv1dState<T>), Error> {
        require_f32(self, input)?;
        validate_input(input, self.width)?;
        if self.depthwise.as_ref().shape() != [self.width, 1, self.kernel] {
            return Err(invalid(
                "EAR decoder depthwise kernel has invalid canonical geometry",
            ));
        }
        let mut next = state.clone();
        let convolved = next
            .step(
                input,
                &self.depthwise.as_ref().swap_axes(1, 2, context)?,
                Some(self.bias.as_ref()),
                context,
            )?
            .ok_or_else(|| invalid("nonempty unit produced no causal convolution output"))?;
        let normalized = self
            .norm
            .forward(&convolved.swap_axes(1, 2, context)?, context)?
            .swap_axes(1, 2, context)?;
        let activated = T::gelu(&self.expand.forward(&normalized, context)?, context)?;
        let result = input.add(
            &self
                .project
                .forward_with_reduction(&activated, context, reduce)?,
            context,
        )?;
        Ok((result, next))
    }
}

/// One execution and residency unit in canonical decoder layer order.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "T")]
#[allow(clippy::large_enum_variant)]
enum DecoderOperation<T: Tensor> {
    /// Nonoverlapping learned upsampling.
    Upsample(Upsample<T>),
    /// Independently cached ConvNeXt block.
    Block(ConvNextBlock<T>),
    /// Raw magnitude/phase projection.
    Output(Pointwise<T>),
}
/// One independently materializable unit with stable layer identity.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "T")]
pub struct DecoderUnit<T: Tensor> {
    operation: DecoderOperation<T>,
    #[parameter(skip)]
    spec: DecoderUnitSpec,
    #[parameter(skip)]
    prefix: String,
    #[parameter(skip)]
    local: Option<crate::ear_decoder_layered::DecoderUnitLocalDimensions>,
}
impl<T: Tensor> DecoderUnit<T> {
    /// Declares one unit with canonical `prefix.layers.N` parameter IDs.
    pub fn new(spec: DecoderUnitSpec, prefix: &str, context: &T::Context) -> Result<Self, Error> {
        Self::new_selected(spec, prefix, None, context)
    }
    pub(crate) fn new_selected(
        spec: DecoderUnitSpec,
        prefix: &str,
        local: Option<crate::ear_decoder_layered::DecoderUnitLocalDimensions>,
        context: &T::Context,
    ) -> Result<Self, Error> {
        spec.validate()?;
        if prefix.is_empty() {
            return Err(invalid("empty EAR decoder parameter prefix"));
        }
        let name = format!("{prefix}.layers.{}", spec.layer());
        let group = name.clone();
        let physical = match (spec, local) {
            (
                DecoderUnitSpec::Upsample {
                    layer,
                    input,
                    output: _,
                    rate,
                    groups,
                },
                Some(local),
            ) => DecoderUnitSpec::Upsample {
                layer,
                input,
                output: local.width * groups,
                rate,
                groups,
            },
            (
                DecoderUnitSpec::Block {
                    layer,
                    width,
                    intermediate: _,
                    kernel,
                },
                Some(local),
            ) => DecoderUnitSpec::Block {
                layer,
                width,
                intermediate: local.width,
                kernel,
            },
            (
                DecoderUnitSpec::Output {
                    layer,
                    input,
                    output: _,
                },
                Some(local),
            ) => DecoderUnitSpec::Output {
                layer,
                input,
                output: local.width,
            },
            _ => spec,
        };
        let operation = match physical {
            DecoderUnitSpec::Upsample {
                input,
                output,
                rate,
                groups,
                ..
            } => DecoderOperation::Upsample(Upsample {
                weight: slot(
                    &format!("{name}.weight"),
                    &group,
                    &[input, output / groups, rate],
                    context,
                )?,
                input,
                output,
                rate,
                groups,
            }),
            DecoderUnitSpec::Block {
                width,
                intermediate,
                kernel,
                ..
            } => DecoderOperation::Block(ConvNextBlock::new(
                &name,
                &group,
                width,
                intermediate,
                kernel,
                context,
            )?),
            DecoderUnitSpec::Output { input, output, .. } => DecoderOperation::Output(
                Pointwise::new(&name, &group, input, output, false, context)?,
            ),
        };
        Ok(Self {
            spec,
            operation,
            prefix: prefix.into(),
            local,
        })
    }

    pub(crate) fn local(&self) -> Option<crate::ear_decoder_layered::DecoderUnitLocalDimensions> {
        self.local
    }

    /// Canonical decoder namespace retained for exact resident assembly.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// Declared geometry and layer identity.
    pub fn spec(&self) -> DecoderUnitSpec {
        self.spec
    }

    /// Initial cache with no learned tensor handles.
    pub fn initial_state(&self) -> Result<DecoderUnitState<T>, Error> {
        self.spec.initial_state()
    }

    /// Runs one unit against an unpublished candidate cache.
    pub fn step(
        &self,
        input: &T,
        state: &DecoderUnitState<T>,
        context: &T::Context,
    ) -> Result<(T, DecoderUnitState<T>), Error> {
        if self.local.is_some() {
            return Err(invalid(
                "rank-local EAR decoder unit requires parallel execution",
            ));
        }
        self.step_selected(input, state, context, Ok, Ok)
    }
    pub(crate) fn step_selected(
        &self,
        input: &T,
        state: &DecoderUnitState<T>,
        context: &T::Context,
        reduce: impl FnOnce(T) -> Result<T, Error>,
        gather: impl FnOnce(T) -> Result<T, Error>,
    ) -> Result<(T, DecoderUnitState<T>), Error> {
        if state.spec != self.spec {
            return Err(invalid("EAR decoder unit state identity mismatch"));
        }
        if state.phase == StreamPhase::Finished {
            return Err(Error::StreamFinished);
        }
        if input.shape().len() != 3 || state.batch.is_some_and(|batch| batch != input.dim(0)) {
            return Err(invalid("EAR decoder unit state batch changed"));
        }
        require_f32(self, input)?;
        let mut next = state.clone();
        let output = match (&self.operation, &state.convolution) {
            (DecoderOperation::Upsample(unit), None) => gather(unit.forward(input, context)?)?,
            (DecoderOperation::Output(unit), None) => gather(unit.forward(input, context)?)?,
            (DecoderOperation::Block(unit), Some(cache)) => {
                let (output, cache) = unit.step_with_reduction(input, cache, context, reduce)?;
                next.convolution = Some(cache);
                output
            }
            _ => {
                return Err(invalid(
                    "EAR decoder unit state kind does not match parameters",
                ))
            }
        };
        next.batch = Some(input.dim(0));
        Ok((output, next))
    }
    /// Completes this unit without emitting a neural tail; repeated finish is idempotent.
    pub fn finish(&self, state: &DecoderUnitState<T>) -> Result<DecoderUnitState<T>, Error> {
        if state.spec != self.spec {
            return Err(invalid("EAR decoder unit state identity mismatch"));
        }
        let mut next = self.initial_state()?;
        next.batch = state.batch;
        next.phase = StreamPhase::Finished;
        Ok(next)
    }
}

/// Resident decoder using the same layered traversal as bounded providers.
pub struct NeuralDecoder<B: NeuralBackend> {
    runtime: eredu_runtime::ResidentRuntime<
        eredu_runtime::ComponentAdapter<
            eredu_runtime::ComponentDefinition,
            crate::ear_decoder_layered::DecoderKernel<B>,
        >,
        B,
        crate::ear_decoder_layered::DecoderRuntimeState<B::Tensor>,
    >,
}
impl<B: NeuralBackend> std::fmt::Debug for NeuralDecoder<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NeuralDecoder")
            .field("config", self.config())
            .finish()
    }
}
impl<B: NeuralBackend> Parameterized<B::Tensor> for NeuralDecoder<B> {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: eredu_nn::ParameterVisitor<'a, B::Tensor>,
    {
        for unit in self.units() {
            unit.visit_parameters(visitor);
        }
    }
    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: eredu_nn::ParameterVisitorMut<'a, B::Tensor>,
    {
        for unit in self.units_mut() {
            unit.visit_parameters_mut(visitor);
        }
    }
    fn set_trainable(&mut self, value: bool) {
        for unit in self.units_mut() {
            unit.set_trainable(value);
        }
    }
}
impl<B: NeuralBackend> NeuralDecoder<B> {
    /// Declares each canonical unit exactly once after geometry validation.
    pub fn new(
        config: DecoderConfig,
        prefix: &str,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let units = config
            .unit_specs()?
            .into_iter()
            .map(|spec| DecoderUnit::new(spec, prefix, context))
            .collect::<Result<_, _>>()?;
        Self::from_units(config, units)
    }
    /// Adopts exact ordered units without allocating or copying learned parameters.
    pub fn from_units(
        config: DecoderConfig,
        units: Vec<DecoderUnit<B::Tensor>>,
    ) -> Result<Self, Error> {
        let specs = config.unit_specs()?;
        let prefix = units
            .first()
            .ok_or_else(|| invalid("EAR decoder has no units"))?
            .prefix()
            .to_owned();
        if units.len() != specs.len()
            || units.iter().zip(specs).any(|(unit, spec)| {
                unit.spec() != spec || unit.prefix() != prefix || unit.local().is_some()
            })
        {
            return Err(invalid(
                "EAR decoder units do not match configured order, namespace and geometry",
            ));
        }
        let architecture =
            crate::ear_decoder_layered::DecoderKernel::new(config, &prefix)?.into_component()?;
        let runtime = eredu_runtime::ResidentRuntime::from_units(architecture, vec![units])
            .map_err(|error| match error {
                eredu_runtime::ResidentUnitsError::Architecture(error) => error,
                _ => invalid("EAR decoder resident unit count mismatch"),
            })?;
        Ok(Self { runtime })
    }
    /// Exact decoder-order geometry.
    pub fn config(&self) -> &DecoderConfig {
        self.runtime.architecture().kernel().config()
    }
    /// Ordered independently materializable units.
    pub fn units(&self) -> &[DecoderUnit<B::Tensor>] {
        &self.runtime.units()[0]
    }
    /// Ordered independently bindable units.
    pub fn units_mut(&mut self) -> &mut [DecoderUnit<B::Tensor>] {
        &mut self.runtime.units_mut()[0]
    }
    /// Constructs empty state without learned parameter access.
    pub fn initial_state(&self) -> Result<DecoderState<B::Tensor>, Error> {
        crate::ear_decoder_layered::DecoderExecution::initial_state(self)
    }
    /// Executes ordinary decoding through the same candidate-state transition.
    pub fn forward(
        &mut self,
        latent: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        crate::ear_decoder_layered::DecoderExecution::forward(self, latent, context)
    }
    /// Executes all units against an unpublished copy of the immutable parent.
    pub fn step(
        &mut self,
        latent: &B::Tensor,
        state: &DecoderState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, DecoderState<B::Tensor>), Error> {
        crate::ear_decoder_layered::DecoderExecution::step(self, latent, state, context)
    }
    /// Releases neural carry without inventing a neural output tail.
    pub fn finish(
        &self,
        state: &DecoderState<B::Tensor>,
    ) -> Result<DecoderState<B::Tensor>, Error> {
        crate::ear_decoder_layered::DecoderExecution::finish(self, state)
    }
}
impl<B: NeuralBackend> crate::ear_decoder_layered::DecoderExecution<B> for NeuralDecoder<B> {
    fn config(&self) -> &DecoderConfig {
        self.config()
    }
    fn step(
        &mut self,
        latent: &B::Tensor,
        parent: &DecoderState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, DecoderState<B::Tensor>), Error> {
        use eredu_runtime::LayeredInvocation;
        let mut state =
            crate::ear_decoder_layered::DecoderRuntimeState::from_state(self.config(), parent)?;
        let output = self.runtime.invoke(latent, &mut state, context)?;
        Ok((output, state.into_state()))
    }
}

fn parameter(name: &str, group: &str) -> Result<ParameterSpec, Error> {
    let mut spec = ParameterSpec::trainable(name).map_err(|error| invalid(&error.to_string()))?;
    spec.trainable = false;
    spec.group = Some(group.into());
    Ok(spec)
}
pub(crate) fn slot<T: Tensor>(
    name: &str,
    group: &str,
    shape: &[i32],
    context: &T::Context,
) -> Result<Parameter<T>, Error> {
    Ok(Parameter::unloaded(
        parameter(name, group)?,
        shape,
        context,
    )?)
}
fn validate_input<T: Tensor>(input: &T, channels: i32) -> Result<(), Error> {
    if !matches!(input.shape(), [batch, width, time] if *batch > 0 && *width == channels && *time > 0)
    {
        return Err(invalid(
            "EAR decoder unit expects nonempty [batch, channels, time]",
        ));
    }
    Ok(())
}
pub(crate) fn require_f32<T: Tensor>(
    module: &impl Parameterized<T>,
    input: &T,
) -> Result<(), Error> {
    struct Check(bool);
    impl<'a, T: Tensor> ParameterVisitor<'a, T> for Check {
        fn visit(&mut self, _: ParameterMetadata, value: &'a T) {
            self.0 &= value.element_type() == Some(TensorElementType::F32);
        }
    }
    let mut check = Check(input.element_type() == Some(TensorElementType::F32));
    module.visit_parameters(&mut check);
    if check.0 {
        Ok(())
    } else {
        Err(invalid(
            "EAR codec neural execution requires F32 input and parameters",
        ))
    }
}
pub(crate) fn invalid(message: &str) -> Error {
    Error::InvalidShape(message.into())
}
