//! EAR codec's finite spectrogram-to-latent ConvNeXt encoder.
//!
//! The released encoder consumes a complete finite spectrogram. Each canonical
//! layer is independently materializable; ConvNeXt equations are shared with
//! the decoder. No streaming encoder cache is implied by this interface.

use eredu_nn::{NeuralBackend, Parameter, Parameterized, Tensor};

use crate::{
    ear_decoder::{require_f32, slot, ConvNextBlock, Pointwise},
    Error,
};

/// Geometry in encoder execution order, from spectra toward latent vectors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncoderConfig {
    /// Latent output width.
    pub latent_size: i32,
    /// Base channel count multiplied at each stage.
    pub base_hidden_size: i32,
    /// Multipliers in encoder execution order.
    pub channel_mult: Vec<i32>,
    /// Downsampling factors in encoder execution order.
    pub rates: Vec<i32>,
    /// ConvNeXt residual blocks preceding each downsampler.
    pub blocks_per_stage: usize,
    /// Causal depthwise convolution width.
    pub kernel_size: i32,
    /// Input spectrum contains `fft_size + 2` real/imaginary channels.
    pub fft_size: i32,
    /// Groups for downsampling convolutions.
    pub groups: i32,
}

impl EncoderConfig {
    /// Validates all layers before allocating parameter handles.
    pub fn validate(&self) -> Result<(), Error> {
        self.unit_specs().map(|_| ())
    }

    /// Number of input spectral frames per latent vector.
    pub fn frame_reduction(&self) -> Result<i32, Error> {
        self.rates.iter().try_fold(1i32, |frames, rate| {
            if *rate <= 0 {
                return Err(invalid("EAR encoder rates must be positive"));
            }
            frames
                .checked_mul(*rate)
                .ok_or_else(|| invalid("EAR encoder frame reduction overflows"))
        })
    }

    /// Ordered independently materializable layers with canonical `layers.N` IDs.
    pub fn unit_specs(&self) -> Result<Vec<EncoderUnitSpec>, Error> {
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
            return Err(invalid("invalid EAR encoder geometry"));
        }
        self.frame_reduction()?;
        let count = self
            .blocks_per_stage
            .checked_add(1)
            .and_then(|n| n.checked_mul(self.rates.len()))
            .and_then(|n| n.checked_add(1))
            .filter(|n| *n <= i32::MAX as usize)
            .ok_or_else(|| invalid("EAR encoder layer count overflows"))?;
        let widths = self
            .channel_mult
            .iter()
            .map(|multiple| {
                self.base_hidden_size
                    .checked_mul(*multiple)
                    .filter(|width| *width > 0)
                    .ok_or_else(|| invalid("EAR encoder channel count overflows"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut specs = Vec::with_capacity(count);
        specs.push(EncoderUnitSpec::Input {
            layer: 0,
            input: self.fft_size + 2,
            output: widths[0],
        });
        for (stage, (&width, &rate)) in widths.iter().zip(&self.rates).enumerate() {
            let intermediate = width
                .checked_mul(4)
                .ok_or_else(|| invalid("EAR encoder intermediate width overflows"))?;
            for _ in 0..self.blocks_per_stage {
                specs.push(EncoderUnitSpec::Block {
                    layer: specs.len(),
                    width,
                    intermediate,
                    kernel: self.kernel_size,
                });
            }
            let output = widths.get(stage + 1).copied().unwrap_or(self.latent_size);
            if width % self.groups != 0 || output % self.groups != 0 {
                return Err(invalid(
                    "EAR encoder channels do not divide convolution groups",
                ));
            }
            specs.push(EncoderUnitSpec::Downsample {
                layer: specs.len(),
                input: width,
                output,
                rate,
                groups: self.groups,
            });
        }
        Ok(specs)
    }
}

/// Geometry of one finite encoder execution and residency unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum EncoderUnitSpec {
    Input {
        layer: usize,
        input: i32,
        output: i32,
    },
    Block {
        layer: usize,
        width: i32,
        intermediate: i32,
        kernel: i32,
    },
    Downsample {
        layer: usize,
        input: i32,
        output: i32,
        rate: i32,
        groups: i32,
    },
}

impl EncoderUnitSpec {
    /// Canonical zero-based encoder execution position.
    pub fn layer(self) -> usize {
        match self {
            Self::Input { layer, .. }
            | Self::Block { layer, .. }
            | Self::Downsample { layer, .. } => layer,
        }
    }
    fn input(self) -> i32 {
        match self {
            Self::Input { input, .. } | Self::Downsample { input, .. } => input,
            Self::Block { width, .. } => width,
        }
    }
    fn validate(self) -> Result<(), Error> {
        let valid = match self {
            Self::Input { input, output, .. } => input > 0 && output > 0,
            Self::Block {
                width,
                intermediate,
                kernel,
                ..
            } => width > 0 && intermediate > 0 && kernel > 0,
            Self::Downsample {
                input,
                output,
                rate,
                groups,
                ..
            } => {
                input > 0
                    && output > 0
                    && rate > 0
                    && groups > 0
                    && input % groups == 0
                    && output % groups == 0
            }
        };
        if valid {
            Ok(())
        } else {
            Err(invalid("invalid EAR encoder unit geometry"))
        }
    }
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "T")]
#[allow(clippy::large_enum_variant)]
enum EncoderOperation<T: Tensor> {
    Input(Pointwise<T>),
    Block(ConvNextBlock<T>),
    Downsample(Parameter<T>),
}

/// One independently bindable layer in the finite encoder.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "T")]
pub struct EncoderUnit<T: Tensor> {
    operation: EncoderOperation<T>,
    #[parameter(skip)]
    spec: EncoderUnitSpec,
    #[parameter(skip)]
    prefix: String,
    #[parameter(skip)]
    local: Option<crate::ear_encoder_layered::EncoderUnitLocalDimensions>,
}

impl<T: Tensor> EncoderUnit<T> {
    /// Declares one canonical `prefix.layers.N` layer with frozen parameters.
    pub fn new(spec: EncoderUnitSpec, prefix: &str, context: &T::Context) -> Result<Self, Error> {
        Self::new_selected(spec, prefix, None, context)
    }
    pub(crate) fn new_selected(
        spec: EncoderUnitSpec,
        prefix: &str,
        local: Option<crate::ear_encoder_layered::EncoderUnitLocalDimensions>,
        context: &T::Context,
    ) -> Result<Self, Error> {
        spec.validate()?;
        if prefix.is_empty() {
            return Err(invalid("empty EAR encoder parameter prefix"));
        }
        let name = format!("{prefix}.layers.{}", spec.layer());
        let physical = local_spec(spec, local);
        let operation = match physical {
            EncoderUnitSpec::Input { input, output, .. } => EncoderOperation::Input(
                Pointwise::new(&name, &name, input, output, false, context)?,
            ),
            EncoderUnitSpec::Block {
                width,
                intermediate,
                kernel,
                ..
            } => EncoderOperation::Block(ConvNextBlock::new(
                &name,
                &name,
                width,
                intermediate,
                kernel,
                context,
            )?),
            EncoderUnitSpec::Downsample {
                input,
                output,
                rate,
                groups,
                ..
            } => EncoderOperation::Downsample(slot(
                &format!("{name}.weight"),
                &name,
                &[output, input / groups, rate],
                context,
            )?),
        };
        Ok(Self {
            operation,
            spec,
            prefix: prefix.into(),
            local,
        })
    }

    pub(crate) fn local(&self) -> Option<crate::ear_encoder_layered::EncoderUnitLocalDimensions> {
        self.local
    }

    /// Canonical source namespace for exact consuming assembly.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// Canonical layer identity and exact geometry.
    pub const fn spec(&self) -> EncoderUnitSpec {
        self.spec
    }

    /// Runs a finite `[batch, channels, frames]` activation through this layer.
    /// Downsampling uses valid convolution and drops a partial trailing stride.
    pub fn forward(&self, input: &T, context: &T::Context) -> Result<T, Error> {
        if self.local.is_some() {
            return Err(invalid(
                "rank-local EAR encoder unit requires parallel execution",
            ));
        }
        self.forward_selected(input, context, Ok, Ok)
    }
    pub(crate) fn forward_selected(
        &self,
        input: &T,
        context: &T::Context,
        reduce: impl FnOnce(T) -> Result<T, Error>,
        gather: impl FnOnce(T) -> Result<T, Error>,
    ) -> Result<T, Error> {
        if !matches!(input.shape(), [batch, channels, time] if *batch > 0 && *channels == self.spec.input() && *time > 0)
        {
            return Err(invalid(
                "EAR encoder unit expects nonempty [batch, channels, frames]",
            ));
        }
        require_f32(self, input)?;
        match (&self.operation, local_spec(self.spec, self.local)) {
            (EncoderOperation::Input(unit), _) => gather(unit.forward(input, context)?),
            (EncoderOperation::Block(unit), _) => {
                if self.local.is_some() {
                    unit.forward_with_reduction(input, context, reduce)
                } else {
                    unit.forward(input, context)
                }
            }
            (
                EncoderOperation::Downsample(weight),
                EncoderUnitSpec::Downsample {
                    input: channels,
                    output,
                    rate,
                    groups,
                    ..
                },
            ) => {
                if input.dim(2) < rate
                    || weight.as_ref().shape() != [output, channels / groups, rate]
                {
                    return Err(invalid(
                        "EAR encoder downsampling input or canonical kernel has invalid geometry",
                    ));
                }
                gather(
                    T::conv1d(
                        &input.swap_axes(1, 2, context)?,
                        &weight.as_ref().swap_axes(1, 2, context)?,
                        rate,
                        0,
                        1,
                        groups,
                        context,
                    )?
                    .swap_axes(1, 2, context)?,
                )
            }
            _ => Err(invalid("EAR encoder operation does not match its geometry")),
        }
    }
}

fn local_spec(
    spec: EncoderUnitSpec,
    local: Option<crate::ear_encoder_layered::EncoderUnitLocalDimensions>,
) -> EncoderUnitSpec {
    match (spec, local) {
        (EncoderUnitSpec::Input { layer, input, .. }, Some(local)) => EncoderUnitSpec::Input {
            layer,
            input,
            output: local.width,
        },
        (
            EncoderUnitSpec::Block {
                layer,
                width,
                kernel,
                ..
            },
            Some(local),
        ) => EncoderUnitSpec::Block {
            layer,
            width,
            kernel,
            intermediate: local.width,
        },
        (
            EncoderUnitSpec::Downsample {
                layer,
                input,
                rate,
                groups,
                ..
            },
            Some(local),
        ) => EncoderUnitSpec::Downsample {
            layer,
            input,
            rate,
            groups,
            output: local.width * groups,
        },
        _ => spec,
    }
}

/// Finite resident encoder using the shared ordered-unit traversal.
pub struct NeuralEncoder<B: NeuralBackend> {
    runtime: eredu_runtime::ResidentRuntime<
        crate::ear_encoder_layered::EncoderArchitecture<B>,
        B,
        crate::ear_encoder_layered::EncoderRuntimeState,
    >,
}
impl<B: NeuralBackend> std::fmt::Debug for NeuralEncoder<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NeuralEncoder")
            .field("config", self.config())
            .finish()
    }
}
impl<B: NeuralBackend> Parameterized<B::Tensor> for NeuralEncoder<B> {
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
impl<B: NeuralBackend> NeuralEncoder<B> {
    /// Declares each canonical unloaded unit exactly once after geometry validation.
    pub fn new(
        config: EncoderConfig,
        prefix: &str,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let units = config
            .unit_specs()?
            .into_iter()
            .map(|spec| EncoderUnit::new(spec, prefix, context))
            .collect::<Result<_, _>>()?;
        Self::from_units(config, units)
    }
    /// Adopts complete canonical units without copying learned tensors.
    pub fn from_units(
        config: EncoderConfig,
        units: Vec<EncoderUnit<B::Tensor>>,
    ) -> Result<Self, Error> {
        let specs = config.unit_specs()?;
        let prefix = units
            .first()
            .ok_or_else(|| invalid("EAR encoder has no units"))?
            .prefix()
            .to_owned();
        if units.len() != specs.len()
            || units.iter().zip(specs).any(|(unit, spec)| {
                unit.spec() != spec || unit.prefix() != prefix || unit.local().is_some()
            })
        {
            return Err(invalid(
                "EAR encoder units do not match configured order, namespace and geometry",
            ));
        }
        let architecture = crate::ear_encoder_layered::EncoderArchitecture::new(config, &prefix)?;
        let runtime = eredu_runtime::ResidentRuntime::from_units(architecture, vec![units])
            .map_err(|error| match error {
                eredu_runtime::ResidentUnitsError::Architecture(error) => error,
                _ => invalid("EAR encoder resident unit count mismatch"),
            })?;
        Ok(Self { runtime })
    }
    /// Exact finite encoder geometry.
    pub fn config(&self) -> &EncoderConfig {
        self.runtime.architecture().config()
    }
    /// Individually executable finite layers in canonical order.
    pub fn units(&self) -> &[EncoderUnit<B::Tensor>] {
        &self.runtime.units()[0]
    }
    /// Individually bindable finite layers.
    pub fn units_mut(&mut self) -> &mut [EncoderUnit<B::Tensor>] {
        &mut self.runtime.units_mut()[0]
    }
    /// Converts `[batch, fft+2, frames]` spectra to `[batch, latent_frames, latent_size]`.
    pub fn forward(
        &mut self,
        spectrum: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        crate::ear_encoder_layered::EncoderExecution::forward(self, spectrum, context)
    }
}
impl<B: NeuralBackend> crate::ear_encoder_layered::EncoderExecution<B> for NeuralEncoder<B> {
    fn config(&self) -> &EncoderConfig {
        self.config()
    }
    fn forward(
        &mut self,
        spectrum: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        use eredu_runtime::LayeredInvocation;
        let mut state = crate::ear_encoder_layered::EncoderRuntimeState::new()?;
        self.runtime.invoke(spectrum, &mut state, context)
    }
}

pub(crate) fn invalid(message: &str) -> Error {
    Error::InvalidShape(message.into())
}
